#!/usr/bin/env python3

#    MIT License

#    COPYRIGHT (C) 2024 MERCK SHARP & DOHME CORP. A SUBSIDIARY OF MERCK & CO., 
#    INC., RAHWAY, NJ, USA. ALL RIGHTS RESERVED

#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:

#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import glob     
import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from scipy import stats
import logging
import argparse

k = 18
p = joblib.load('models/efs_best_knn.pkl')
# get parameters of sklearn model 
params = p.get_params()
print(params)
#os.chdir('')
# load data
tv = pd.read_csv('rf_efs.csv')
#t = tv.values
#t = t[:,:].astype(float)
#Xt, t = t[:,1:-1], t[:,-1]

test = pd.read_csv('holdout.csv')
cols = tv.columns.tolist()
cols = cols[1:]
n = test[cols]
print(n)
d = n.values
X, y = d[:, :-1], d[:, -1]

# list of 100 random non overlaping  numbers 
rand = np.random.randint(0, 1000, size=1)
rand = rand.tolist()
r2_scores = []
rp_scores = []
for rand_st in rand:
    kf = KFold(n_splits=k, shuffle=True, random_state=rand_st)
    for train, val in kf.split(tv):
        # set knn regressor parameters to params
        knn = KNeighborsRegressor(**params)
        # fit model
        knn.fit(tv.iloc[train, 1:-1], tv.iloc[train, -1])
        # predict on test set
        y_pred = knn.predict(X)
        r2 = r2_score(y, y_pred)
        r2_scores.append(r2)
        pearson = stats.pearsonr(y,y_pred).statistic
        pearson = pearson**2
        rp_scores.append(pearson)

r2 = np.array(r2_scores)
#print(r2_scores)
print('R2 score: ', r2.mean())
print('R2 95 err: ', 1.96*r2.std()/np.sqrt(k*1))
print('R2 90 err: ', 1.64*r2.std()/np.sqrt(k*1))

rp = np.array(rp_scores)
#print(rp_scores)
print('Pearson sq score: ', rp.mean())
print('Pearson 95 sq err: ', 1.96*rp.std()/np.sqrt(k*1))
print('Pearson 90 sq err: ', 1.64*rp.std()/np.sqrt(k*1))

