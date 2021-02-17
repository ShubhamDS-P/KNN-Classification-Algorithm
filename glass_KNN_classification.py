# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:48:48 2021

@author: Shubham
"""

import pandas as pd
import numpy as np

glass = pd.read_csv('D:\\Data Science study\\Assignment of Data Science\\Sent\\11 KNN\\glass.csv')
glass

# data is already in the wprking format and there no need to make any changes

#Let's split the data into train and test data for model building 
from sklearn.model_selection import train_test_split

glass_train,glass_test = train_test_split(glass,test_size = 0.2)  #splitting data into 80% and 20% ratio.

#Now we will use KNN
#first we will import KNN algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

glass.shape

# Now we will decide the value of K as 3 and apply the KNN funtion
# For 3 nearest neighbors
glass_neighbors = KNC(n_neighbors = 3)

#fitting the zoo_neighbors with training data
glass_neighbors.fit(glass_train.iloc[:,0:9],glass_train.iloc[:,9])

#getting the training accuracy
train_accuracy = np.mean(glass_neighbors.predict(glass_train.iloc[:,0:9])==glass_train.iloc[:,9])
train_accuracy   #0.8245614035087719%

#Getting test accuracy
test_accuracy = np.mean(glass_neighbors.predict(glass_test.iloc[:,0:9])==glass_test.iloc[:,9])
test_accuracy   # 0.7441860465116279%


#To get more results with the differnt K values we will make a for loop so that we can get the values we want
# without repeating the same steps

# First create an empty list variable

accuracy = []

# NOw we will run the KNN algorithm for 3 to 50 nearest neighbors and store the results which are predicted accuracy
#values.

for i in range(3,50):
    glass_neighbors = KNC(n_neighbors = i)
    glass_neighbors.fit(glass_train.iloc[:,0:9],glass_train.iloc[:,9])
    train_accuracy = np.mean(glass_neighbors.predict(glass_train.iloc[:,0:9])==glass_train.iloc[:,9])
    test_accuracy = np.mean(glass_neighbors.predict(glass_test.iloc[:,0:9])==glass_test.iloc[:,9])
    accuracy.append([train_accuracy,test_accuracy])
    
import matplotlib.pyplot as plt

#Plot for training accuracy
plt.plot(np.arange(3,50),[i[0] for i in accuracy], "bo-")
plt.legend(["glass_train","glass_test"])

#Testing accuracy plot
plt.plot(np.arange(3,50),[i[0] for i in accuracy],"ro-")
plt.legend(["glass_train","glass_test"])


# graph shows that k value of 3 has the highest accuracy and hence we will consider 3 for our final k value
