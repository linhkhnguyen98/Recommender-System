#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import numpy as np
import tensorflow as tf
from copy import copy
from sklearn.metrics import mean_squared_error
import random
import pandas as pd
from sklearn.metrics import confusion_matrix


# In[2]:


def readJSON(path):
  for l in gzip.open(path, 'rt', encoding="utf-8"):
    d = eval(l)
    u = d['userID']
    try:
      g = d['gameID']
    except Exception as e:
      g = None
    yield u,g,d


# In[3]:


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)
    
for user,game,review in allHours:
    review["played"] = 1


# In[4]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)

for u,g,d in hoursTrain:
    r = d['hours_transformed']
    hoursPerUser[u].append((g,r))
    hoursPerItem[g].append((u,r))


# In[5]:


def validation_set(data, sample):
    games_set = set()
    games_by_user = defaultdict(set)

    # Collect all games and user-game associations
    for user, game, review in data:
        game_id = review.get("gameID")  
        user_id = review.get("userID")  
        if game_id is not None and user_id is not None:
            games_set.add(game_id)
            games_by_user[user_id].add(game_id)

    unplayed_games = []
    

    # Generate negative samples for unplayed games
    for user, game, review in sample:
        user_id = review.get("userID")  # Get userID from review
        if user_id is not None:
            games_not_played = games_set - games_by_user.get(user_id, set())
            if games_not_played:
                selected_game = random.choice(tuple(games_not_played))
                unplayed_games.append((user, selected_game, {"played": 0}))

    # Combine original and generated negative samples
    balanced_sample = sample + unplayed_games
    return balanced_sample


# In[6]:


class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb, itemIDs, userIDs):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI))
    
    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))


# In[7]:


### Play Prediction


# In[8]:


class GamePlayPrediction:
    
    def train_play_prediction(self, data, K, iters):
        self.userIDs = {}
        self.itemIDs = {}
        interactions = []

        for u,i,r in data:
            if not u in self.userIDs: 
                self.userIDs[u] = len(self.userIDs)
            if not i in self.itemIDs: 
                self.itemIDs[i] = len(self.itemIDs)
            interactions.append((u,i,r["played"]))
        
        items = list(self.itemIDs.keys())
        
        itemsPerUser = defaultdict(list)
        usersPerItem = defaultdict(list)
        for u,i,r in interactions:
            itemsPerUser[u].append(i)
            usersPerItem[i].append(u)
        
        optimizer = tf.keras.optimizers.Adam(0.1)
        self.modelBPR = BPRbatch(K, 0.00001, self.itemIDs, self.userIDs)
        
        def trainingStep(model, interactions):
            Nsamples = 50000
            with tf.GradientTape() as tape:
                sampleU, sampleI, sampleJ = [], [], []
                for _ in range(Nsamples):
                    u,i,_ = random.choice(interactions) # positive sample
                    j = random.choice(items) # negative sample
                    while j in itemsPerUser[u]:
                        j = random.choice(items)
                    sampleU.append(self.userIDs[u])
                    sampleI.append(self.itemIDs[i])
                    sampleJ.append(self.itemIDs[j])

                loss = model(sampleU,sampleI,sampleJ)
                loss += model.reg()
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients((grad, var) for
                                    (grad, var) in zip(gradients, model.trainable_variables)
                                    if grad is not None)
            return loss.numpy()

        for i in range(iters):
            obj = trainingStep(self.modelBPR, interactions)
            if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))
            
    def predict(self, user, game, threshold=0.5):
        uid = self.userIDs.get(user)
        gid = self.itemIDs.get(game)

        if uid is None or gid is None:
            # Handle the case where user or game is not found in the IDs
            return 0

        pred = self.modelBPR.predict(uid, gid).numpy()
        return int(pred > threshold)


# In[9]:


model = GamePlayPrediction()
model.train_play_prediction(hoursTrain, 7, 500)


# In[10]:


# Initialize confusion matrix
CM = np.array([[0, 0], [0, 0]])

# Create lists to store actual and predicted values
actual_values = []
predicted_values = []

# Create a balanced validation dataset
balanced_valid = validation_set(allHours, hoursValid)

# Iterate over the balanced dataset to collect actual and predicted values
for user, game, review in balanced_valid:
    actual = review["played"]
    pred = model.predict(user, game)
    actual_values.append(actual)
    predicted_values.append(pred)

    
# Calculate the confusion matrix
CM = confusion_matrix(actual_values, predicted_values)

# Print confusion matrix and accuracy
# print("Confusion Matrix:\n", CM)
true_positives = CM[1][1]
true_negatives = CM[0][0]
false_positives = CM[0][1]
false_negatives = CM[1][0]
accuracy = 1 - (false_negatives + false_positives) / len(balanced_valid)
print(f"accuracy: {accuracy}")


# In[11]:


predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    pred = model.predict(u,g)
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# In[12]:


### Time Prediction


# In[13]:


class GameTimeEstimator:
    
    def __init__(self):
        self.avgUserTime = {}
        self.avgGameTime = {}
        self.overallAverage = 0

    def train_model(self, dataset, regularization=5.0, iterations=200):
        userReviewTime = defaultdict(list)
        gameReviewTime = defaultdict(list)

        totalHours = 0

        for user, game, review in dataset:
            userReviewTime[user].append(review)
            gameReviewTime[game].append(review)
            totalHours += review["hours_transformed"]

        self.overallAverage = totalHours / len(dataset)

        for user in userReviewTime:
            avgHours = np.mean([r["hours_transformed"] for r in userReviewTime[user]])
            self.avgUserTime[user] = avgHours

        for game in gameReviewTime:
            avgHours = np.mean([r["hours_transformed"] for r in gameReviewTime[game]])
            self.avgGameTime[game] = avgHours

        for _ in range(iterations):
            tempOverallAvg = 0
            for user, game, review in dataset:
                tempOverallAvg += review["hours_transformed"] - (self.avgUserTime[user] + self.avgGameTime[game])
            self.overallAverage = tempOverallAvg / len(dataset)

            for user in userReviewTime:
                totalDiff = sum(review["hours_transformed"] - (self.overallAverage + self.avgGameTime[review["gameID"]]) for review in userReviewTime[user])
                self.avgUserTime[user] = totalDiff / (regularization + len(userReviewTime[user]))

            for game in gameReviewTime:
                totalDiff = sum(review["hours_transformed"] - (self.overallAverage + self.avgUserTime[review["userID"]]) for review in gameReviewTime[game])
                self.avgGameTime[game] = totalDiff / (regularization + len(gameReviewTime[game]))

    def predict(self, user, game):
        userBias = self.avgUserTime.get(user, 0)
        gameBias = self.avgGameTime.get(game, 0)
        return self.overallAverage + userBias + gameBias


# In[14]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[15]:


def calculate_mse(model, data):
    y = []
    y_pred = []
    for user, game, review in data:
        y_pred.append(model.predict(user, game))
        y.append(review["hours_transformed"])
        
    mse = MSE(y_pred, y)
    return mse


# In[16]:


model = GameTimeEstimator()
model.train_model(hoursTrain)

mse = calculate_mse(model, hoursValid)
print(f"GameTimeEstimator MSE: {mse}")


# In[18]:


predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    pred = model.predict(u,g)
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# In[ ]:




