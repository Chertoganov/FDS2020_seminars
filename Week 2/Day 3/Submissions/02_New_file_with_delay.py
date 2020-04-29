#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:12:27 2020

@author: yashmadhwal
"""

import pandas as pd
import os
import fnmatch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def support_vector_machine(train_x, train_y):
    svm_model = SVC(C=1.0, gamma=0.5)
    svm_model.fit(train_x, train_y)
    train_prediction = svm_model.predict(train_x)
    test_prediction = svm_model.predict(test_x)
    train_accuracy = accuracy_score(train_y, train_prediction)
    test_accuracy = accuracy_score(test_y, test_prediction)
    
    print('Train Accuracy:', train_accuracy)
    print('Test Accuracy:', test_accuracy)    

def estimator_calculation():
    #return SVC(gamma='auto', random_state=0, probability=True),{'C': [0.001, 1.0],}
    return SVC(C=0.001, shrinking=False, random_state=0),{'C': [0.001, 1.0],}
    
    

def execution(train_x, train_y):
    estimator, param_grid = estimator_calculation()
    grid_search = GridSearchCV(estimator, param_grid, verbose=2, cv=2)
    
    print(grid_search.fit(train_x, train_y))


def execution_2(train_x, train_y):
    estimator, param_grid = estimator_calculation()
    grid_search = GridSearchCV(estimator, param_grid, verbose=2, cv=2, n_jobs=-1)
    with joblib.parallel_backend("dask", scatter=[train_x, train_y]):
        grid_search.fit(train_x, train_y)



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
data = pd.read_json('/Users/yashmadhwal/Desktop/yashmadhwal/data/nycflights/1990.json') #User
data = data.fillna(0)
print(data)

train, test = train_test_split(data,train_size=0.5, test_size=0.5)

train_x = train.drop(['DepDelay','UniqueCarrier','Origin','Dest'], axis=1)
train_y = train['DepDelay']
test_x = test.drop(['DepDelay','UniqueCarrier','Origin','Dest'], axis=1)
test_y = test['DepDelay']


    
import time
start_time = time.time()
execution(train_x, train_y)
print("--- %s seconds ---" % (time.time() - start_time))


#____DASK____
import joblib
import dask.distributed

c = dask.distributed.Client()



import time
start_time = time.time()
execution_2(train_x, train_y)
print("--- %s seconds ---" % (time.time() - start_time))