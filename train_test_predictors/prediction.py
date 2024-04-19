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

# output log
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('log/prediction.log', 'a'))
print = logger.info

parser = argparse.ArgumentParser()
parser.add_argument('-holdout', type=str, default='holdout.csv', help='{default: "holdout.csv"}; holdout dataset')
parser.add_argument('-models', type=str, default='all', help='{default: "all"}; models to predict on holdout set ("rfs", "efs", "afs")')
parser.add_argument('-rescaled', type=int, default=0, help='{default: 0}; holdout data is rescaled relative to train data (0 or 1)')
args = parser.parse_args()

# collect models (pkl files)
path = "models/*pkl"
models = glob.glob(path)

# collect selected features
r = pd.read_csv("rf_rfs.csv")
e = pd.read_csv("rf_efs.csv")
a = pd.read_csv("tagg_final.csv")

# model acronyms
rfs = "rfs"
efs = "efs"
afs = "afs"
all = "all"

# collect holdout dataset
holdout = pd.read_csv(args.holdout)

# holdout descriptor
h_des = args.holdout
h_des = h_des.split('.')[0]

# loop through models and plot predictions on holdout set
for i in range(len(models)):
    if rfs in models[i] and args.models == rfs or rfs in models[i] and args.models == all:
        p = joblib.load(models[i])
        model = models[i].split('.')[0]
        model = model.split('/')[1]
        cols = r.columns.tolist()
        cols = cols[1:]
        n = holdout[cols]
        d = n.values
        X, y = d[:,:-1], d[:,-1]
        predicted = p.predict(X)
        lsq = LinearRegression(fit_intercept=True)
        y = y.reshape(-1,1)
        lsq.fit(y, predicted)
        r3 = lsq.score(y, predicted)
        predicted_fit = lsq.predict(y)
        y = np.squeeze(y)
        predicted = np.squeeze(predicted)
        r2=r2_score(y,predicted)
        pearson = stats.pearsonr(y,predicted).statistic
        spearman = stats.spearmanr(y,predicted).correlation
        print(args.holdout)
        if args.rescaled == 1:
            print('%s (LR R\u00b2: ' %model+ "{:.2f}".format(r3) + ')')
        elif   args.rescaled == 0:
            print('%s (R\u00b2: ' %model + "{:.2f}".format(r2) + ')')
        print('%s (Rp: ' %model+ "{:.2f}".format(pearson) + ')')
        print('%s (Rs: ' %model+ "{:.2f}".format(spearman) + ')')
        fig, ax = plt.subplots()
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        ax.scatter(y, predicted, c='black',edgecolors=(0, 0, 0))
        if args.rescaled == 1:
            ax.plot(y, predicted_fit, 'b--', lw=4)
        elif args.rescaled == 0:
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax.set_title('%s' %model)
        if args.rescaled == 1:
            text_box = AnchoredText('LR R\u00b2: ' + "{:.2f}".format(r3) , frameon=True,prop=dict(color = 'blue'), loc='lower center', pad=0.5)
            text_box2 = AnchoredText('R$_{p}$: ' + "{:.2f}".format(pearson) + "\n" + 'R$_{s}$: ' + "{:.2f}".format(spearman), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)    
            plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box2)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box)
        elif args.rescaled == 0:
            text_box2 = AnchoredText('R\u00b2: ' "{:.2f}".format(r2) + "\n" + 'r$_{p}$\u00b2: ' + "{:.2f}".format(pearson**2), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)
            #text_box2 = AnchoredText('R\u00b2: ' "{:.2f}".format(r2) + "\n" + 'R$_{p}$: ' + "{:.2f}".format(pearson) + "\n" + 'R$_{s}$: ' + "{:.2f}".format(spearman), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)    
            plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box2)
        plt.savefig(r'%s_%s.png'%(model, h_des), dpi=300, bbox_inches='tight')
    elif efs in models[i] and args.models == efs or efs in models[i] and args.models == all:
        p = joblib.load(models[i])
        model = models[i].split('.')[0]
        model = model.split('/')[1]
        cols = e.columns.tolist()
        cols = cols[1:]
        n = holdout[cols]
        d = n.values
        X, y = d[:,:-1], d[:,-1]
        predicted = p.predict(X)
        lsq = LinearRegression(fit_intercept=True)
        y = y.reshape(-1,1)
        lsq.fit(y, predicted)
        r3 = lsq.score(y, predicted)
        predicted_fit = lsq.predict(y)
        y = np.squeeze(y)
        predicted = np.squeeze(predicted)
        r2=r2_score(y,predicted)
        pearson = stats.pearsonr(y,predicted).statistic
        spearman = stats.spearmanr(y,predicted).correlation
        print(args.holdout)
        if args.rescaled == 1:
            print('%s (LR R\u00b2: ' %model+ "{:.2f}".format(r3) + ')')
        elif   args.rescaled ==0:
            print('%s (R\u00b2: ' %model + "{:.2f}".format(r2) + ')')
        print('%s (Rp: ' %model+ "{:.2f}".format(pearson) + ')')
        print('%s (Rs: ' %model+ "{:.2f}".format(spearman) + ')')
        fig, ax = plt.subplots()
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        ax.scatter(y, predicted, c='black',edgecolors=(0, 0, 0))
        if args.rescaled == 1:
            ax.plot(y, predicted_fit, 'b--', lw=4)
        elif args.rescaled == 0:
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax.set_title('%s' %model)
        if args.rescaled == 1:
            text_box = AnchoredText('LR R\u00b2: ' + "{:.2f}".format(r3) , frameon=True,prop=dict(color = 'blue'), loc='lower center', pad=0.5)
            text_box2 = AnchoredText('R$_{p}$: ' + "{:.2f}".format(pearson) + "\n" + 'R$_{s}$: ' + "{:.2f}".format(spearman), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)    
            plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box2)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box)
        elif args.rescaled == 0:
            text_box2 = AnchoredText('R\u00b2: ' "{:.2f}".format(r2) + "\n" + 'r$_{p}$\u00b2: ' + "{:.2f}".format(pearson**2), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)
            #text_box2 = AnchoredText('R\u00b2: ' "{:.2f}".format(r2) + "\n" + 'R$_{p}$: ' + "{:.2f}".format(pearson) + "\n" + 'R$_{s}$: ' + "{:.2f}".format(spearman), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)    
            plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box2)
        plt.savefig(r'%s_%s.png'%(model, h_des), dpi=300, bbox_inches='tight')
    elif afs in models[i] and args.models == afs or afs in models[i] and args.models == all:
        p = joblib.load(models[i])
        model = models[i].split('.')[0]
        model = model.split('/')[1]
        cols = a.columns.tolist()
        cols = cols[1:]
        n = holdout[cols]
        d = n.values
        X, y = d[:,:-1], d[:,-1]
        predicted = p.predict(X)
        lsq = LinearRegression(fit_intercept=True)
        y = y.reshape(-1,1)
        lsq.fit(y, predicted)
        r3 = lsq.score(y, predicted)
        predicted_fit = lsq.predict(y)
        y = np.squeeze(y)
        predicted = np.squeeze(predicted)
        r2=r2_score(y,predicted)
        pearson = stats.pearsonr(y,predicted).statistic
        spearman = stats.spearmanr(y,predicted).correlation
        print(args.holdout)
        if args.rescaled == 1:
            print('%s (LR R\u00b2: ' %model+ "{:.2f}".format(r3) + ')')
        elif   args.rescaled ==0:
            print('%s (R\u00b2: ' %model + "{:.2f}".format(r2) + ')')
        print('%s (Rp: ' %model+ "{:.2f}".format(pearson) + ')')
        print('%s (Rs: ' %model+ "{:.2f}".format(spearman) + ')')
        fig, ax = plt.subplots()
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        ax.scatter(y, predicted, c='black',edgecolors=(0, 0, 0))
        if args.rescaled == 1:
            ax.plot(y, predicted_fit, 'b--', lw=4)
        elif args.rescaled == 0:
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax.set_title('%s' %model)
        if args.rescaled == 1:
            text_box = AnchoredText('LR R\u00b2: ' + "{:.2f}".format(r3) , frameon=True,prop=dict(color = 'blue'), loc='lower center', pad=0.5)
            text_box2 = AnchoredText('R$_{p}$: ' + "{:.2f}".format(pearson) + "\n" + 'R$_{s}$: ' + "{:.2f}".format(spearman), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)    
            plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box2)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box)
        elif args.rescaled == 0:
            text_box2 = AnchoredText('R\u00b2: ' "{:.2f}".format(r2) + "\n" + 'r$_{p}$\u00b2: ' + "{:.2f}".format(pearson**2), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)
            #text_box2 = AnchoredText('R\u00b2: ' "{:.2f}".format(r2) + "\n" + 'R$_{p}$: ' + "{:.2f}".format(pearson) + "\n" + 'R$_{s}$: ' + "{:.2f}".format(spearman), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)    
            plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box2)
        plt.savefig(r'%s_%s.png'%(model, h_des), dpi=300, bbox_inches='tight')
    
