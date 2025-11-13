# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 20:42:53 2025

@author: msi h
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read dataset
dataset = pd.read_csv('Mall_Customers.csv')
print(dataset.shape)
X=dataset.iloc[ : ,[3,4]].values
x=  pd.DataFrame(X)


from sklearn.cluster import KMeans
WCSS=[]

for i in range(1,21):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,21),WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of classes')
plt.ylabel('WCSS')
plt.show()