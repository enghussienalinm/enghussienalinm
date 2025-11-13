# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 19:03:02 2025

@author: msi h
"""
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
dataset = pd.read_csv('diabetes.csv')
x=dataset.iloc[ : ,0:8].values
y=dataset.iloc[ : ,8].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.3)


#model creation
model= Sequential()
model.add(Dense(10,input_dim=8,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


#model compiler

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

#model fit 
model.fit(x_train,y_train, epochs=50 ,batch_size=10)

#evalte model
scores=model.evaluate(x_test,y_test)
print("\n%s:%.2f%%" % (model.metrics_names[1], scores[1]* 100))