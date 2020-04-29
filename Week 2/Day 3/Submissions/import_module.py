#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 01:20:13 2020

@author: yashmadhwal
"""
import pandas as pd
import numpy as np
import os
import fnmatch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
import dask.array as da
import joblib
import dask.distributed

class GridSearch:
    def support_vector_machine(train_x,train_y,test_x,test_y):

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
    
    #Function about GridsearchCV
    #by sklearn
    def sklearn_grid_search(train_x, train_y):
        estimator, param_grid = GridSearch.estimator_calculation()
        grid_search = GridSearchCV(estimator, param_grid, verbose=2, cv=2)
        
        print(grid_search.fit(train_x, train_y))
    
    #by Dask_grid_search
    def dask_grid_search(train_x, train_y):
        estimator, param_grid = GridSearch.estimator_calculation()
        grid_search = GridSearchCV(estimator, param_grid, verbose=2, cv=2, n_jobs=-1)
        
        with joblib.parallel_backend("dask", scatter=[train_x, train_y]):
            print(grid_search.fit(train_x, train_y))