#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:58:10 2020

@author: yashmadhwal
"""

#Dataframe Part
import dask.dataframe as dd
df = dd.read_csv(os.path.join('data', 'nycflights', '*.csv'), parse_dates={'Date': [0, 1, 2]})

df = dd.read_csv(os.path.join('data', 'nycflights', '*.csv'),parse_dates={'Date': [0, 1, 2]},dtype={'TailNum': str,'CRSElapsedTime': float,'Cancelled': bool})

start_time = time.time()
temp = df.DepDelay.max().compute()
print("--- %s seconds ---" % (time.time() - start_time))

print(temp)



#XG_Regressor:
print("\n\nXG_Regressor")
print("\n\n\nSample XG Regressor from The example")
from dask_ml.datasets import make_classification
print("\nXG_Regressor Client at..",Client(n_workers=4, threads_per_worker=1),"\n")
X, y = make_classification(n_samples=100000, n_features=20,
                           chunks=1000, n_informative=4,
                           random_state=0)
print(X)

