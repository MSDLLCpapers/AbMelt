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


import MDAnalysis as mda
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression as reg
import subprocess
from collections import OrderedDict
import freesasa
from MDAnalysis.analysis import rms
from matplotlib import pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import argparse


def get_corr(data):
    lin_reg = reg()
    x, y = [xy[0] for xy in data] ,[xy[1] for xy in data]
    lin_reg.fit(x,y)
    return lin_reg.score(x, y)

def no_dash(msa):
    ctr = 0
    for char in msa:
        if char != '-':
            ctr +=1 
    return ctr

def pad_short_lists(dic):
    max_len = max([len(val) for val in dic.values()])
    for key, val in dic.items():
        leng = len(val)
        if leng != max_len:
            dic[key] = np.array(list(val) + [None]*(max_len - leng))
    return dic

def get_lambda(master_dict, temps = [310,350,373]):
    #temps = [int(x) for x in temp.split(",")]
    residues = master_dict[temps[0]].keys()
    lambda_dict = {}
    r_dict = {}
    for resid in residues:
        log_temps = np.array([np.log(temp) for temp in temps]).reshape(-1,1)
        log_s = np.array([np.log(1 - np.sqrt(master_dict[temp][resid])) for temp in temps])
        lin_reg = reg()
        lin_reg.fit(X = log_temps, y = log_s)
        lambda_dict[resid], r_dict[resid] = lin_reg.coef_[0], lin_reg.score(log_temps, log_s)
    return lambda_dict, r_dict

def get_df(master_s2_dict, lambda_dict, r_dict, temps):
    residues = master_s2_dict[temps[0]].keys()
    df_dict = {resid:{} for resid in residues}
    for resid in residues:
        for temp in temps:
            df_dict[resid][temp] = master_s2_dict[temp][resid]
        if lambda_dict != None and r_dict != None:
            df_dict[resid]['lamda'] = lambda_dict[resid]
            df_dict[resid]['r'] = r_dict[resid]
    return pd.DataFrame(df_dict).T

def get_h_n_coords(u):
    coords_dict = OrderedDict()
    #we ignore first residue and Prolines because they don't have n-h bond
    backbone_residues = u.select_atoms('backbone and not resname PRO').residues[1:]
    num_res = backbone_residues.n_residues
    for resid in range(num_res):
        coords_dict[resid + 1] = backbone_residues[resid].atoms[0:2].positions
    return coords_dict

def get_vectors(coords_dict):
    vector_dict = {}
    for resid, coords in coords_dict.items():
        vec = (coords[0,0] - coords[1,0], coords[0,1] - coords[1,1], coords[0,2] - coords[1,2])
        norm  = np.linalg.norm(vec)
        vector_dict[resid] = [comp/norm for comp in vec]
    return vector_dict

def multiply_comps(comps):
    products = []
    for ui in comps:
        for uj in comps:
            products.append(ui*uj)
    return products

def get_s2_df(s2_blocks_dict):
    df_dict = {}
    for block, s2_dic in s2_blocks_dict.items():
        df_dict[block] = sum([s2 for s2 in s2_dic.values()])/len(s2_dic.values())
    return pd.DataFrame([df_dict]).T
        
def get_products(vector_dict):
    product_dict = {}
    for resid, comps in vector_dict.items():
        product_dict[resid] = multiply_comps(comps)
    return product_dict

def get_s2(product_dict):
    s2_dict = {}
    for resid, products in product_dict.items():
        s2 =  (1.5*sum([product**2 for product in products]) - 0.5)
        s2_dict[resid] = 0.89*s2
    return s2_dict
  
def get_range(a,b):
    return [i for i in range(int(a),int(b))]

def get_blocks(traj_length, block_length):
    block_length *= 100
    remainder = traj_length%block_length
    num_blocks = int(traj_length // (block_length))
    if remainder < 0.5*block_length:
        num_blocks = num_blocks - 1
    blocks = {block_num: get_range(block_num*block_length, (block_num+1)*block_length) for block_num in range(num_blocks)}
    blocks[num_blocks] = get_range(num_blocks*block_length, traj_length)
    return blocks
                      

def update_average(dict1, dict2, ts):
    new_dict = {}
    if ts == -1:
        return dict1
    else:
        for pair1, pair2 in zip(dict1.items(), dict2.items()):
            key1, val1 = pair1
            key2, val2 = pair2
            new_val = []
            for prod1, prod2 in zip(val1, val2):
                new_val.append((prod1*ts + prod2)/(ts + 1))
            new_dict[key1] = new_val
    return new_dict

def avg_s2_blocks(s2_dic):
    avg_dict = {}
    blocks = s2_dic.keys()
    resids = s2_dic[0].keys()
    for resid in resids:
        avg_dict[resid] = sum([s2_dic[block][resid] for block in blocks])/len(blocks)
    return avg_dict

def order_s2(mab='mab01', temp='310', block_length=10, start=20):
    temp = str(temp)
    topology = 'md_final_' + temp + '.gro'
    trajectory = 'md_final_' + temp + '.xtc'
    u = mda.Universe(topology,trajectory)
    protein = u.select_atoms("protein")
    prealigner = align.AlignTraj(u, u, select="protein and name CA", in_memory=True).run()
    ref_coordinates = u.trajectory.timeseries(asel=protein).mean(axis=1)
    ref = mda.Merge(protein).load_new(ref_coordinates[:, None, :], order="afc")
    aligner = align.AlignTraj(u, ref, select="protein and name CA", in_memory=True).run()
    traj_length = len(u.trajectory)
    print("starting order parameter calculations...")
    print("trajectory length: {} ns".format(int(round(traj_length/100, 0))))
    print("start: {} ns".format(start))
    print("block length: {} ns".format(block_length))
    print("-----------------------------------------")
    blocks = get_blocks(traj_length - start*100, block_length)
    print("{} blocks for {}K temperature".format(len(blocks.keys()), temp))
    s2_blocks_dict = {block: None for block in blocks.keys()}
    for block, timesteps in blocks.items():
            print("    block {} order parameter calculation successful".format(block))
            block_product_dict = None
            
            #we will iterate over all timesteps in the block, getting the average vector products acorss the block
            for ts in timesteps:
                
                #setting the ts then getting the vector products for the ts
                u.trajectory[start + ts]
                ts_product_dict = get_products(get_vectors(get_h_n_coords(u)))

                #if this is the first ts in the block, the avg product is just the product for the ts
                #else we update the average
                if block_product_dict != None:
                    block_product_dict = update_average(block_product_dict, ts_product_dict, ts)
                else:
                    block_product_dict = ts_product_dict

            #using the average vector products across the block, we get the s2 value for the block
            #and store it in the dictionary
            if block_product_dict == None:
                continue
 
            s2_blocks_dict[block] = get_s2(block_product_dict)
    print("saving order parameter values...")
    for block, timesteps in blocks.items():
        block_product_dict = None
        
        #we will iterate over all timesteps in the block, getting the average vector products acorss the block
        for ts in timesteps:
            
            #setting the ts then getting the vector products for the ts
            u.trajectory[start + ts]
            ts_product_dict = get_products(get_vectors(get_h_n_coords(u)))

            #if this is the first ts in the block, the avg product is just the product for the ts
            #else we update the average
            if block_product_dict != None:
                block_product_dict = update_average(block_product_dict, ts_product_dict, ts)
            else:
                block_product_dict = ts_product_dict

        #using the average vector products across the block, we get the s2 value for the block
        #and store it in the dictionary
        if block_product_dict == None:
            continue

        s2_blocks_dict[block] = get_s2(block_product_dict)
    get_s2_df(s2_blocks_dict).to_csv('order_s2_{}K_{}block_{}start.csv'.format(str(temp),str(block_length),str(start)))
    print("order parameter values saved.")
    print("")
    return s2_blocks_dict

def order_lambda(master_dict= None, mab='mab01', temps=[310, 350, 373], block_length='10', start='20000'):

    temps = [int(x) for x in temps]
    #master_s2_dict = {temp:{} for temp in temps}
    for temp in temps:
        lambda_dict, r_dict = get_lambda(master_dict=master_dict, temps=temps)        
        df = get_df(master_dict, lambda_dict, r_dict, temps)
        df.to_csv('order_lambda_{}block_{}start.csv'.format(block_length, start))
