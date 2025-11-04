# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:57:08 2025

@author: msi h
"""

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

# fitting Decision tree to the train dataset
from sklearn.svm import SVC
classifier = SVC(kernel='linear')
classifier.fit(x_train,y_train)

# prediction the test set results
y_pred=classifier.predict(x_test)


#evalation ... confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_test,y_pred)

acc= accuracy_score(y_test,y_pred)
