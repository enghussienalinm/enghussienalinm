# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression

dataset = pd.read_csv("salary_data.csv")

x=dataset.iloc[ : ,[0]]
x=dataset.iloc[ : ,[0]].values

y=dataset.iloc[ : ,[1]].values

x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2)
# class < object
regressor= LinearRegression()
# fit function for training
regressor.fit(x_train,y_train)
# test < predict
ypred=regressor.predict(x_test)

#evaluuation
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,ypred))

A=regressor.coef_
b=regressor.intercept_
print("y=",A,"x+",b)

print("the salary of employee",regressor.predict([[10.5]]))

#visalization
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regressor.predict(x_train),'blue')
#plt.plot(x_train,y_train,'red')
plt.title("Salary vs experince years")
plt.xlabel("Years of experince")
plt.ylabel("Salary")
plt.show()