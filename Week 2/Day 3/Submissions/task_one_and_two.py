#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:10:18 2020

@author: yashmadhwal
"""
# Importing all the required libraries
import pandas as pd
import numpy as np
import os
import fnmatch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from import_module import GridSearch
import dask.array as da
import joblib
import dask.distributed

   
     
'''
path_fullname_list = []
for path,dirs,files in os.walk('.'):
    for f in fnmatch.filter(files,'*.csv'):
        path_fullname = os.path.abspath(os.path.join(path,f))
        #print(path_fullname)
        path_fullname_list.append(path_fullname)
        
path_fullname_list.sort()
'''
  
    #Importing Json
    #data = pd.read_csv('/Users/yashmadhwal/Desktop/yashmadhwal/data/nycflights/1991.csv',nrows=10000)

data = pd.read_csv('/Users/yashmadhwal/Desktop/yashmadhwal/data/nycflights/1991.csv',nrows=10) #User
data = data.fillna(0)
print(data)

train, test = train_test_split(data,train_size=0.5, test_size=0.5)

train_x = train.drop(['DepDelay','UniqueCarrier','Origin','Dest'], axis=1)
train_y = train['DepDelay']
test_x = test.drop(['DepDelay','UniqueCarrier','Origin','Dest'], axis=1)
test_y = test['DepDelay']

# Support Vector Machines
GridSearch.support_vector_machine(train_x,train_y,test_x,test_y)


    
import time
start_time = time.time()
GridSearch.sklearn_grid_search(train_x, train_y)
print("--- %s seconds ---" % (time.time() - start_time))



#____DASK____


from dask.distributed import Client, progress
c = dask.distributed.Client()
client = Client(processes=False, threads_per_worker=4, n_workers=1, memory_limit='2GB')
print(client)


import time
start_time = time.time()
GridSearch.dask_grid_search(train_x, train_y)
print(f"--- {time.time() - start_time}seconds ---")


#DASK DELAY
import dask




output = []
#for x in data:
a = dask.delayed(GridSearch.support_vector_machine)(train_x,train_y,test_x,test_y)
print(a)
start_time = time.time()
a.compute()
print("--- %s seconds ---" % (time.time() - start_time))
output.append(a)
b = dask.delayed(GridSearch.sklearn_grid_search)(train_x, train_y)
print(b)
output.append(b)
start_time = time.time()
b.compute()
print("--- %s seconds ---" % (time.time() - start_time))
c = dask.delayed(GridSearch.dask_grid_search)(train_x, train_y)
print(c)
output.append(c)
start_time = time.time()
c.compute()
print("--- %s seconds ---" % (time.time() - start_time))

total = dask.delayed(sum)(output)
#Visaualize
total.visualize() 



#Other Code:
import pandas as pd

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

clean_dataset(train_x)
train_x = train_x.values
train_x

train_y = train_y.values
train_y

from sklearn.preprocessing import Normalizer
x = train_x
transformer = Normalizer().fit(x)

transformer

transformer.transform(x)

train_x = transformer.transform(x)

train_x = train_x.round(decimals=2)
train_x

train_x, train_y =  make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=1, n_samples=1000)
train_x[:5]

train_y


# Scale up: increase N, the number of times we replicate the data.
N = 2
X_large = da.concatenate([da.from_array(train_x, chunks=train_x.shape)
                          for _ in range(N)])
y_large = da.concatenate([da.from_array(train_y, chunks=train_y.shape)
                          for _ in range(N)])
print(X_large)


from sklearn.linear_model import LogisticRegressionCV
from dask_ml.wrappers import ParallelPostFit
clf = ParallelPostFit(LogisticRegressionCV(cv=3), scoring="r2")
y_pred = clf.predict(X_large)
print(y_pred)
    
    
