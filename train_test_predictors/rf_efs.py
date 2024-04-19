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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from numpy import arange
from sklearn.model_selection import RepeatedKFold
from datetime import datetime
import logging
import argparse
import ray
import os
import joblib
from pprint import pprint


# argument parser for user customization
parser = argparse.ArgumentParser()
parser.add_argument('-input', type=str, default="rf_ranked_features.csv", help= '{default = "rf_ranked_features.csv"}; csv file for efs, exhaustive feature selection (features[:,1:-1], target[:,-1])')
parser.add_argument('-n_efs', type=int, default=10, help= '{default = 10}; number of top ranked features to exhaustively search')
parser.add_argument('-cv_splits', type=int, default=5, help='{default = 5}; number of random cross validation folds')
parser.add_argument('-cv_repeats', type=int, default=3, help='{default = 3}; repeats of cv_splits')
parser.add_argument('-n_jobs', type=int, default=None, help='{default = None} number of parallelized jobs (max = cv_splits * cv_repeats * n_grid_search_hp)')
parser.add_argument('-n_gpus', type=int, default=0, help='{default = 0}; RAPIDS cuML GPU acceleration')
args = parser.parse_args()

ray.init(address='auto', _redis_password='')
assert ray.is_initialized()
pprint(ray.nodes())

# write console output to log file
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(r'log/rf_efs.log', 'a'))
print = logger.info
start_time = datetime.now()

# exhaustive feature selection on max n_efs features
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.pipeline import make_pipeline

# import ranked data from recursive feature selection
topf = args.n_efs + 1
ranked = pd.read_csv(args.input)

if args.input == 'train.csv':
    ranked.reset_index(inplace=True)
    ranked.drop(0, axis=1, inplace=True)

X_names = list(ranked.columns)
X_names = X_names[1:topf]

ranked = ranked.values
X, y = ranked[:,1:topf], ranked[:,-1]

# define grid
grid_efs = [{'exhaustivefeatureselector__estimator__n_estimators':[arange(10,1000,50)]},
        {'exhaustivefeatureselector__estimator__criterion':['squared_error','absolute_error','poisson']},
       {'exhaustivefeatureselector__estimator__max_features':['sqrt','log2']}]

# cross validation
cv = RepeatedKFold(n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=1)
reg = RandomForestRegressor()

# exhaustive feature selection
efs = EFS(estimator=reg,
           min_features=1,
           max_features=args.n_efs,
           scoring='neg_mean_absolute_error',
           print_progress=False,
           clone_estimator=False,
           cv=cv,
           n_jobs=args.n_jobs)

pipe = make_pipeline(efs, reg)

# define search
search = GridSearchCV(estimator=pipe,
                      param_grid=grid_efs,
                      scoring='neg_mean_absolute_error',
                      cv=cv,
                      n_jobs=args.n_jobs)

# perform the search & exhaustive feature selection
from ray.util.joblib import register_ray
register_ray()
with joblib.parallel_backend('ray'):
        results = search.fit(X, y)

        fidx = list(results.best_estimator_.steps[0][1].best_idx_)
        features = [X_names[i] for i in fidx]
        feat_cnt = len(features)
        print('optimal feature count: %s' % feat_cnt )
        print('R\u00b2: %.3f' % results.best_score_)
        print('std: %f ' % results.cv_results_['std_test_score'][results.best_index_]) 
        print('configuration: %s' % results.best_params_)
        print('exhaustive features:')
        print(features)
        end_time = datetime.now()
        print('duration: {}'.format(end_time - start_time))

        df = pd.DataFrame(results.cv_results_)
        df2 = pd.DataFrame(results.best_estimator_.steps[0][1].subsets_)
        df3 =pd.DataFrame(features)

        # output gridsearch data
        df.to_csv('cv/rf_efs_cv.csv')
        df2.to_csv('cv/rf_efs_best_estimators.csv')
        df3.to_csv('cv/rf_efs_best_features.csv')

        # output exhaustive feature ranked list
        ranked = pd.read_csv("rf_ranked_features.csv")
        ex = pd.DataFrame()
        for i in range(len(features)):
                ex[features[i]] = ranked[features[i]]
        trgt = ranked.columns.tolist()
        trgt_nm = trgt[-1]
        ex[trgt_nm] = ranked.iloc[:,-1]
        ex.to_csv("rf_efs.csv")

ray.shutdown()
assert not ray.is_initialized()
