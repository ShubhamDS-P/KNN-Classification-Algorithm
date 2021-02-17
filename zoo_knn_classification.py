# -*- coding: utf-8 -*-
"""Zoo KNN Classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vZjG6ekfBQKjSlqCx3mCpEqCilri7auE
"""

import pandas as pd
import numpy as np

from google.colab import files

zoo = files.upload()

zoo = pd.read_csv('Zoo.csv')

zoo

knn_zoo = zoo.iloc[:,1:]    #Removing unwanted data and creating the dataframe with only usable data
knn_zoo

# lets split the data into train and test for the model building and testing purposes
from sklearn.model_selection import train_test_split

zoo_train,zoo_test = train_test_split(knn_zoo,test_size = 0.2)  # splitting the data into 80% and 20% ratio, 20% test data.

# Now we will use KNN 
# frist we import KNN algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

zoo.shape

# Now we will decide the value of the k as 3 and apply the KNN function
# For 3 nearest neighbors
zoo_neighbors = KNC(n_neighbors= 3)

#fitting the zoo_neighbors with training data
zoo_neighbors.fit(zoo_train.iloc[:,0:16],zoo_train.iloc[:,16])

# Getting train accuracy
train_accuracy = np.mean(zoo_neighbors.predict(zoo_train.iloc[:,0:16])==zoo_train.iloc[:,16])
train_accuracy    # 98.75%

#getting test accuracy
test_accuracy = np.mean(zoo_neighbors.predict(zoo_test.iloc[:,0:16])==zoo_test.iloc[:,16])
test_accuracy   #85.71%

"""We will try different different k values so we can get the better grasp on the accuracy changes of the data. So we will create a for loop where we create outputs with multiple values of the k and plot them in the graph for visualization."""

# First create an empty list variable
accuracy = []

# Now we will run the KNN algorithm for 3 to 50 nearest neighbors (odd numbers) and store the 
# predicted accuracy values

for i in range(3,50):
  zoo_neighbors = KNC(n_neighbors=i)
  zoo_neighbors.fit(zoo_train.iloc[:,0:16],zoo_train.iloc[:,16])
  train_accuracy = np.mean(zoo_neighbors.predict(zoo_train.iloc[:,0:16])==zoo_train.iloc[:,16])
  test_accuracy = np.mean(zoo_neighbors.predict(zoo_test.iloc[:,0:16])==zoo_test.iloc[:,16])
  accuracy.append([train_accuracy,test_accuracy])

import matplotlib.pyplot as plt  # Library for visualizatuon

# Plot for training accuracy
plt.plot(np.arange(3,50),[i[0] for i in accuracy],"bo-")
plt.legend(["zoo_train","zoo_test"])
plt.show()

# Testing accuracy plot
plt.plot(np.arange(3,50),[i[0] for i in accuracy],"ro-")
plt.legend(["zoo_train","zoo_test"])

"""From the graph we can say that we get the highest accuracy at the k value of 2.
so we will decide that the best value of the K for this data set is the 2.
"""