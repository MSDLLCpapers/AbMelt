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
import os
import pandas as pd
import mdtraj as md
from sklearn.linear_model import LinearRegression as reg

def core_surface(temp):
    temp = str(temp)
    topology = 'md_final_' + temp + '.gro'
    trajectory = 'md_final_' + temp + '.xtc'
    traj = md.load_xtc(trajectory, top= topology)
    sasa = md.shrake_rupley(traj, mode = 'residue')
    np.savetxt(fname = 'res_sasa_{}.np'.format(temp), X = sasa, fmt = '%d')

def get_core_surface(sasa_dict, temp, k = 20, start = 20):
    data = np.loadtxt('res_sasa_{}.np'.format(temp))
    traj_length = data.shape[0]
    traj_length = int(round(traj_length/100, 0))
    per_res = data[0,:]
    # highest 20 & lowest 20 SASA residues are considered surface and core residues, respectively
    ns = traj_length - start # last (traj_length - start) ns of trajectory
    core = np.mean(data[:-ns*100,np.argpartition(per_res, k)[:k]], axis = 0)
    surface = np.mean(data[:-ns*100,np.argpartition(per_res, -k)[-k:]], axis = 0)
    total = np.mean(data[:-ns*100,:], axis = 1)
    sasa_dict[temp]['total_mean'] = np.mean(total)
    sasa_dict[temp]['core_mean'] = np.mean(core)
    sasa_dict[temp]['surface_mean'] = np.mean(surface)
    sasa_dict[temp]['total_std'] = np.std(total)
    sasa_dict[temp]['core_std'] = np.std(core)
    sasa_dict[temp]['surface_std'] = np.std(surface)
    return sasa_dict

def get_slope(data):
    lin_reg = reg()
    x, y = np.array([np.log(xy[0]) for xy in data]).reshape(-1,1) ,[xy[1] for xy in data]
    lin_reg.fit(x,y)
    return lin_reg.coef_[0]
