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

#import data from train.csv
import pandas as pd
import numpy as np
import argparse
import ray
import os
import joblib
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('-input', type=str, default="train.csv", help= '{default = "train.csv"}; csv file for rfs, recursive feature selection (features[:,1:-1], target[:,-1])')
parser.add_argument('-corr_max', type=float, default=0.95, help= '{default = 0.95}; maximum feature cross correlation')
parser.add_argument('-cv_splits', type=int, default=5, help='{default = 5}; number of random cross validation folds')
parser.add_argument('-cv_repeats', type=int, default=3, help='{default = 3}; repeats of cv_splits')
parser.add_argument('-n_jobs', type=int, default=None, help='{default = None}; number of parallelized jobs (max = cv_splits * cv_repeats * n_grid_search_hp)')
parser.add_argument('-n_gpus', type=int, default=0, help='{default = 0}; RAPIDS cuML GPU acceleration')
args = parser.parse_args()

ray.init(address='auto', _redis_password='')
assert ray.is_initialized()
pprint(ray.nodes())

features3 = pd.read_csv(args.input)
features2 = features3.iloc[:,:-1]

# drop highly correlated physiochemical descriptors
corr = features2.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > args.corr_max)]
features = features2.drop(to_drop, axis=1)
features.insert(len(features.columns), features3.columns.tolist()[-1], features3.iloc[:,-1].tolist() ) 
d = features.values

# drop rows with nans
f = features3.columns.tolist()[-1]
nan = features.isna().sum().sum()
features = features.dropna(subset=[f])

# recursive feature selection
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from numpy import arange
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFECV
from datetime import datetime
import logging

# write console output to log file
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(r'log/rf_rfs.log', 'a'))
print = logger.info
print("nan count: %s" % nan)
print('============================================================================================================')
start_time = datetime.now()

d = features.values

X, y = d[:,1:-1], d[:,-1]

# define grid
grid =[{'estimator__n_estimators': arange(10,1000,50).tolist()},
        {'estimator__criterion':['squared_error','absolute_error','poisson']},
        {'estimator__max_features':['sqrt','log2']}]

# cross validation method                                          
cv = RepeatedKFold(n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=1)

# recursive feature selection
reg = RandomForestRegressor()
min_ft = 1
rfecv = RFECV(
    estimator = reg,
    step = 1,
    cv = cv,
    scoring = 'neg_mean_absolute_error',
    n_jobs=args.n_jobs)

# define search
clf = GridSearchCV(rfecv, grid, scoring = 'neg_mean_absolute_error', cv = cv)

from ray.util.joblib import register_ray
register_ray()
with joblib.parallel_backend('ray'):
    clf.fit(X, y)

    # plot number of features vs  cross validation scores (grid scores deprecated in sklearn >1.0 needs reworking)
    #plt.figure()
    #plt.xlabel("Number of features selected")
    #plt.ylabel("R\u00b2")
    #plt.plot(
    #    range(min_ft, len(clf.best_estimator_.grid_scores_) + min_ft),
    #    clf.best_estimator_.grid_scores_,
    #)
    #plt.show()
    #plt.savefig(r'rf_rfs.png', dpi=300, bbox_inches='tight')

    # reorder columns to recursive ranking
    rnk_msk = clf.best_estimator_.ranking_.tolist()
    ranked = features.iloc[:,1:-1]
    ranked.loc[len(ranked.index)]=rnk_msk
    ranked = ranked.sort_values(by = len(features), axis=1) #check row number
    ranked = ranked[:-1]
    ranked[f] = features[f]
    ranked.to_csv("rf_ranked_features.csv")

    # print summary to log file
    feat_cnt = clf.best_estimator_.n_features_
    feat_lst = ranked.columns.tolist()
    print("optimal feature count: %d" % clf.best_estimator_.n_features_)
    print("R\u00b2: %f" % clf.best_score_)
    print("std: %f" % clf.cv_results_['std_test_score'][clf.best_index_])
    print('configuration: %s' % clf.best_params_)
    print('recursive features:')
    print(feat_lst[:feat_cnt])
    end_time = datetime.now()
    print('duration: {}'.format(end_time - start_time))

    # output recursive only features
    nf = clf.best_estimator_.n_features_
    rf_ranked = pd.DataFrame()
    rf_ranked = ranked.iloc[:,:nf]
    rf_ranked[f] = features[f]
    rf_ranked.to_csv("rf_rfs.csv")

    print('============================================================================================================')
    print('ranked recursive features:')
    print(ranked.columns.tolist())
    print('============================================================================================================')

ray.shutdown()
assert not ray.is_initialized()
