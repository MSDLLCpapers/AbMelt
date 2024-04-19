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

import numpy as np
from numpy import mean
from numpy import std
from numpy import absolute
import pandas as pd
import matplotlib.pyplot as plt
from numpy import arange
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import joblib
from joblib import Parallel, delayed
from datetime import datetime
import logging
import warnings
import argparse
warnings.filterwarnings("ignore")
import ray
import os
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('-input', type=str, default="train.csv", help= '{default = "train.csv"}; csv file for model prediction (features[:,1:-1], target[:,-1])')
parser.add_argument('-mode', type=str, default="afs", help= '{default = "afs"}; mode for building predictors (afs, rfs, efs, or any prefix for naming output files)')
parser.add_argument('-scoring', type=str, default="r2", help= '{default = "r2"}; scoring metric for model evaluation (r2, mae, mse)')
parser.add_argument('-scale_features', type=str, default=None, help= '{default = None}; scale features for prediction (avg_std, med_iqr, min_max)')
parser.add_argument('-cv_splits', type=int, default=5, help='{default = 5}; number of random cross validation folds')
parser.add_argument('-cv_repeats', type=int, default=3, help='{default = 3}; repeats of cv_splits')
parser.add_argument('-bopt_iters', type=int, default=50, help='{default = 50}; iterations of skopt bayesian optimization on model hyperparameter space')
parser.add_argument('-bopt_points', type=int, default=4, help='{default = 4}; number of hyperparameter points per bopt_iter')
parser.add_argument('-n_jobs', type=int, default=None, help='{default = None}; number of parallelized jobs (max per iteration = cv_splits * cv_repeats * bopt_points)')
parser.add_argument('-n_gpus', type=int, default=0, help='{default = 0}; RAPIDS cuML GPU acceleration (1 for acceleration)')
args = parser.parse_args()

ray.init(address='auto', _redis_password='5241590000000000')
assert ray.is_initialized()
pprint(ray.nodes())

# define scoring metric for cpu/gpu computed models
if args.n_gpus != 0:
    # RAPIDS cuML GPU acceleration
    import cudf, cupy, cuml
    from cuml import LinearRegression as cuML_LinearRegression
    from cuml import ElasticNet as cuML_ElasticNet
    from cuml.neighbors import KNeighborsRegressor as cuML_KNeighborsRegressor
    from cuml.svm import SVR as cuML_SVR
    from cuml.ensemble import RandomForestRegressor as cuML_RandomForestRegressor
    from cuml.metrics.regression import mean_absolute_error
    from cuml.metrics.regression import mean_squared_error
    from cuml.metrics.regression import r2_score
    from sklearn.metrics import make_scorer
    cuml.set_global_output_type('numpy')
    if args.scoring == 'r2':
        score_metric = make_scorer(r2_score, convert_dtype=True)
    elif args.scoring == 'mae':
        score_metric = make_scorer(mean_absolute_error)
    elif args.scoring == 'mse':
        score_metric = make_scorer(mean_squared_error)
else:
    if args.scoring == 'r2':
        score_metric = "r2"
    elif args.scoring == 'mae':
        score_metric = "neg_mean_absolute_error"
    elif args.scoring == 'mse':
        score_metric = "neg_mean_squared_error"
 
# bayesian optimization parameters
ITERATIONS = args.bopt_iters
POINTS = args.bopt_points
JOBS = args.n_jobs
MODE = args.mode

# output log
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('log/%s_p.log' %MODE, 'a'))
print = logger.info

# import recursively selected features

if args.mode == 'afs' or args.input == 'train.csv':
    data = pd.read_csv(args.input)
    data.reset_index(inplace=True)
    data.drop('name', axis=1, inplace=True)
if args.mode == 'rfs':
    data = pd.read_csv('rf_rfs.csv')
if args.mode == 'efs':
    data = pd.read_csv('rf_efs.csv')

    

# select top ranked features
d = data.values
d = d[:,:].astype(float)
Xp, y = d[:,1:-1], d[:,-1]

# feature scaling options
if args.scale_features == 'avg_std':
    X = StandardScaler().fit_transform(Xp)
elif args.scale_features == 'med_iqr':
    X = RobustScaler().fit_transform(Xp)
elif args.scale_features == 'min_max':
    X = MinMaxScaler().fit_transform(Xp)
else:
    X = Xp

# define cross validation
cv = RepeatedKFold(n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=1)

# define hyperparameter space

if args.n_gpus != 0:
    model_params_gpu = {
    'LinearRegression':{
            'model': cuML_LinearRegression(),
            'params':{
                'algorithm':['eig', 'svd']
            }
        },
        'ElasticNet':{
            'model':cuML_ElasticNet(),
            'params':{
            'alpha':(1e-6, 1e6,'uniform'),
            'l1_ratio':(0, 1, 'uniform')
            }
        },
        'kNN':{
            'model':cuML_KNeighborsRegressor(),
            'params':{
                'n_neighbors':(1,2, 'uniform')
            }
        },
        'SVM':{
            'model': cuML_SVR(),
            'params':{
                'C': (1e-1, 1e3, 'uniform'),
                'gamma': (1e-4, 1e3, 'uniform')
            }
        },
        'RandomForest':{
            'model':cuML_RandomForestRegressor(),
            'params':{
                'n_estimators':(10,1000, 'uniform'),
                'max_features':['sqrt','log2']
            }
        },
        'XGBoost':{
            'model':XGBRegressor(),
            'params':{
                'max_depth': (1, 10, 'uniform'),
                'n_estimators': (10, 1000, 'uniform'),
                'tree_method':['gpu_hist']
            }
        }
    }

else:
    model_params = {
    'LinearRegression':{
            'model': linear_model.TweedieRegressor(),
            'params':{
                'alpha':(1e-6, 1e6, 'uniform')
            }
        },
        'ElasticNet':{
            'model':ElasticNet(),
            'params':{
            'alpha':(1e-6, 1e6,'uniform'),
            'l1_ratio':(0, 1, 'uniform')

            }
        },
        'kNN':{
            'model':KNeighborsRegressor(),
            'params':{
                'n_neighbors':(1,2, 'uniform'),
                'leaf_size': (1,2, 'uniform'),
                'p':(1,2, 'uniform')
            }
        },
        'SVM':{
            'model': svm.SVR(),
            'params':{
                'C': (1e-1, 1e3, 'uniform'),
                'gamma': (1e-4, 1e3, 'uniform')
            }
        },
        'DecisionTree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'splitter':['best','random'],
                'max_depth' : (1,10, 'uniform') ,
                'max_features':['auto', 'sqrt','log2'],
                'max_leaf_nodes': (2,10, 'uniform')
            }
        },
        'RandomForest':{
            'model':RandomForestRegressor(),
            'params':{
                'n_estimators':(10,1000, 'uniform'),
                'criterion': ['squared_error','absolute_error'],
                'max_features':['sqrt','log2']
            }
        },
        'AdaBoost':{
            'model':AdaBoostRegressor(),
            'params':{
                'n_estimators': (10,1000, 'uniform'),
                'loss':['linear','square','exponential']
            }
        },
        'XGBoost':{
            'model':XGBRegressor(),
            'params':{
                'max_depth': (1, 10, 'uniform'),
                #'n_estimators': (10, 1000, 'uniform'),
                'tree_method':['approx']
            }
        }
    }

scores = []
cv_results = []
bayes_opt = []
cwd = os.getcwd()

def search_models (X, y, score_metric, model_name, mp):
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(cwd + '/log/%s_p.log' %MODE, 'a'))
    print = logger.info
    
    search =  BayesSearchCV(mp['model'], mp['params'], scoring=score_metric, cv=cv, n_iter= ITERATIONS,  n_points = POINTS ,refit=True, n_jobs=JOBS)
    # numpy to cupy
    if args.n_gpus > 0:
        X = cupy.asarray(X)
        y = cupy.asarray(y)
    start_time = datetime.now()
    results = search.fit(X, y)
    end_time = datetime.now()
    mdn = model_name.strip("()").lower()
    itrs = search.total_iterations
    print("model: %s" % mdn)
    print("total iterations: %s" % itrs)
    score = search.cv_results_['mean_test_score'][search.best_index_]
    print("R\u00b2: %s" % score)
    print("std: %f " % search.cv_results_['std_test_score'][search.best_index_])
    print('configuration: %s' % search.best_params_)
    print('duration: {}'.format(end_time - start_time))
    print('============================================================================================================')
    cv_results.append({
        'model': mdn,
        'means': results.cv_results_['mean_test_score'],
        'stds': results.cv_results_['std_test_score'],
        'params': results.cv_results_['params'],
    })
    scores.append({
        'model': mdn,
        'best_score': results.best_score_,
        'best_stds':results.cv_results_['std_test_score'][results.best_index_],
        'best_params': results.best_params_,
    })
    bayes_opt.append({
        'model': mdn,
        'hyp_space': results.optimizer_results_
    })
    sv_dir = cwd +'/models/%s_best_%s.pkl' %(MODE,mdn)
    joblib.dump(results.best_estimator_, sv_dir, compress = 1)
    return cv_results, scores, bayes_opt

from ray.util.joblib import register_ray
register_ray()

if args.n_gpus != 0:
    with joblib.parallel_backend('ray', ray_remote_args=dict(num_gpus = 1)):
        final_results = Parallel()(delayed(search_models)(X, y, score_metric, model_name, mp) for model_name, mp in model_params_gpu.items())
        
        # remove joblib nesting of lists
        scores_final = []
        cv_results_final = []
        bayes_opt_final = []
        for i in range(len(final_results)):
            scores_final.append(final_results[i][1])
            cv_results_final.append(final_results[i][0])
            bayes_opt_final.append(final_results[i][2])

        scores_final = [item for sublist in scores_final for item in sublist]
        cv_results_final = [item for sublist in cv_results_final for item in sublist]
        bayes_opt_final = [item for sublist in bayes_opt_final for item in sublist]

        df = pd.DataFrame(scores_final,columns=['model','best_score','best_stds','best_params'])
        cv_results2 = pd.DataFrame(cv_results_final,columns=['model','means','stds','params'])
        b_opt = pd.DataFrame(bayes_opt_final, columns=['model','hyp_space'])

        # colormap
        data_color = [x / max(df['best_score']) for x in df['best_score']]
        my_cmap = plt.cm.get_cmap('cool')
        colors = my_cmap(data_color)

        # plot
        ax = df.plot(x='model',y='best_score', xerr='best_stds',kind='barh', color=colors)
        ax.set_xlabel('Repeated 5-Fold Test Score (R\u00b2)')
        ax.set_title('Prediction')
        ax.get_legend().remove()
        plt.savefig("%s_p.png" %MODE, dpi=300, bbox_inches='tight')

        #save data
        df.to_csv("cv/%s_best_p.csv"%MODE) 
        cv_results2.to_csv("cv/%s_cv_p.csv"%MODE) 
        b_opt.to_csv("cv/%s_bopt_p.csv"%MODE) 
else:
    with joblib.parallel_backend('ray'):
        final_results = Parallel()(delayed(search_models)(X, y, score_metric, model_name, mp) for model_name, mp in model_params.items())
        
        # remove joblib nesting of lists
        scores_final = []
        cv_results_final = []
        bayes_opt_final = []
        for i in range(len(final_results)):
            scores_final.append(final_results[i][1])
            cv_results_final.append(final_results[i][0])
            bayes_opt_final.append(final_results[i][2])

        scores_final = [item for sublist in scores_final for item in sublist]
        cv_results_final = [item for sublist in cv_results_final for item in sublist]
        bayes_opt_final = [item for sublist in bayes_opt_final for item in sublist]

        df = pd.DataFrame(scores_final,columns=['model','best_score','best_stds','best_params'])
        cv_results2 = pd.DataFrame(cv_results_final,columns=['model','means','stds','params'])
        b_opt = pd.DataFrame(bayes_opt_final, columns=['model','hyp_space'])

        # colormap
        data_color = [x / max(df['best_score']) for x in df['best_score']]
        my_cmap = plt.cm.get_cmap('cool')
        colors = my_cmap(data_color)

        # plot
        ax = df.plot(x='model',y='best_score', xerr='best_stds',kind='barh', color=colors)
        ax.set_xlabel('CV Test Score (R\u00b2)')
        ax.set_title('Prediction')
        ax.get_legend().remove()
        plt.savefig("%s_p.png" %MODE, dpi=300, bbox_inches='tight')

        #save data
        df.to_csv("cv/%s_best_p.csv" %MODE)
        cv_results2.to_csv("cv/%s_cv_p.csv"%MODE)
        b_opt.to_csv("cv/%s_bopt_p.csv"%MODE)
    
ray.shutdown()
assert not ray.is_initialized()
