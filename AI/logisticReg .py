# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 19:42:23 2025
logastic Regression
@author: msi h
"""
# import librarries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read dataset
dataset = pd.read_csv('Social_Network_ads.csv')

x=dataset.iloc[ : ,2:4].values
y=dataset.iloc[ : ,[4]].values

#spilitting the dataset as training dataset and test dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.3)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

# fitting logestic regression to the train dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

# prediction the test set results
y_pred=classifier.predict(x_test)

#evalation ... confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)

acc=(cm[0][0] + cm[1][1])/ len(x_test)
print('the accuracy is :',acc*100,"%")