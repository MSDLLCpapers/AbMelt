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
import os, glob
from pymbar import timeseries
from os.path import basename
os.chdir('pathway/to/executable/.py')
from src.res_sasa import get_core_surface
from src.res_sasa import get_slope

# aggregate descriptors into csv for further analysis
os.chdir('pathway/to/data/AbMelt')

os.chdir('pathway/to/data/AbMelt')
dirs = glob.glob('*/', recursive=True)
cwd = os.getcwd()
temps = ['300','350','400']

eq_parameters = {'mAb':[],'TEMP':[],'metric':[],'eq_time':[],'eq_mu':[],'eq_std':[]}
# iterate through dirs and collect gromacs analysis data
for dir in range(len(dirs)):
    os.chdir(dirs[dir])
    xvgs = glob.glob('*.xvg')
    for xvg in range(len(xvgs)):
        temps = ['310','350','373']
        metric = xvgs[xvg].split('.')[0]
        t, x, y, z, r= [], [], [], [] , []
        try:
            temp = [temp for temp in temps if temp in xvgs[xvg]][0]
        except:
            print('skip %s' % xvgs[xvg])
            continue
        with open("%s" % xvgs[xvg]) as f:
            #lines = (line for line in f if not line.startsiwith('#') if not line.startswith('$'))
            for line in f:
                if line.startswith('#'):
                    continue
                if line.startswith('@'):
                    continue
                cols = line.split()

                if len(cols) == 2:
                    t.append(float(cols[0]))
                    x.append(float(cols[1]))
                elif len(cols) == 3:
                    t.append(float(cols[0]))
                    x.append(float(cols[1]))
                    y.append(float(cols[2]))
                elif len(cols) == 4:
                    t.append(float(cols[0]))
                    x.append(float(cols[1]))
                    y.append(float(cols[2]))
                    z.append(float(cols[3]))
                elif len(cols) == 5:
                    t.append(float(cols[0]))
                    r.append(float(cols[1]))
                    x.append(float(cols[2]))
                    y.append(float(cols[3]))
                    z.append(float(cols[4]))
                else:
                    raise ValueError("Invalid number of columns: %d" % len(cols))

            # if metric contains substring then
        if "bonds" in metric:
            x = np.array(x)
            #[t0, g, Neff_max] = timeseries.detect_equilibration(x)
            t0 = 2000
            x_equlibrated = x[t0:]
            eqt_x = (len(t)-len(x_equlibrated))
            x_mu = np.mean(x_equlibrated)
            x_std = np.std(x_equlibrated)
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric + '_' + 'hbonds','eq_time':eqt_x,'eq_mu':x_mu,'eq_std':x_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

            y = np.array(y)
            #[t0, g, Neff_max] = timeseries.detect_equilibration(y)
            t0 = 2000
            y_equlibrated = y[t0:]
            eqt_y = (len(t)-len(y_equlibrated))
            y_mu = np.mean(y_equlibrated)
            y_std = np.std(y_equlibrated)
            # append multiple values to eq_parameters dictionary
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric + '_' + 'contacts','eq_time':eqt_y,'eq_mu':y_mu,'eq_std':y_std}
            for key, value in params.items():
                eq_parameters[key].append(value)
        elif "gyr" in metric:
            r = np.array(r)
            #[t0, g, Neff_max] = timeseries.detect_equilibration(r)
            t0 = 2000
            r_equlibrated = r[t0:]
            eqt_r = (len(t)-len(r_equlibrated))
            r_mu = np.mean(r_equlibrated)
            r_std = np.std(r_equlibrated)
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric + '_' + 'Rg','eq_time':eqt_r,'eq_mu':r_mu,'eq_std':r_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

            x = np.array(x)
            #[t0, g, Neff_max] = timeseries.detect_equilibration(x)
            t0 = 2000
            x_equlibrated = x[t0:]
            eqt_x = (len(t)-len(x_equlibrated))
            x_mu = np.mean(x_equlibrated)
            x_std = np.std(x_equlibrated)
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric + '_' + 'Rx','eq_time':eqt_x,'eq_mu':x_mu,'eq_std':x_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

            y = np.array(y)
            #[t0, g, Neff_max] = timeseries.detect_equilibration(y)
            t0 = 2000
            y_equlibrated = y[t0:]
            eqt_y = (len(t)-len(y_equlibrated))
            y_mu = np.mean(y_equlibrated)
            y_std = np.std(y_equlibrated)
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric + '_' + 'Ry','eq_time':eqt_y,'eq_mu':y_mu,'eq_std':y_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

            z = np.array(z)
            #[t0, g, Neff_max] = timeseries.detect_equilibration(z)
            t0 = 2000
            z_equlibrated = z[t0:]
            eqt_z = (len(t)-len(z_equlibrated))
            z_mu = np.mean(z_equlibrated)
            z_std = np.std(z_equlibrated)
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric + '_' + 'Rz','eq_time':eqt_z,'eq_mu':z_mu,'eq_std':z_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

        elif "rmsd" in metric:
            plt.plot(t, x)
            plt.axvline(20000, color='black')
            plt.show()
            x = np.array(x)
            #[t0, g, Neff_max] = timeseries.detect_equilibration(x)
            t0 = 2000
            x_equlibrated = x[t0:]
            eqt_x = (len(t)-len(x_equlibrated))
            x_mu = np.mean(x_equlibrated)
            x_std = np.std(x_equlibrated)
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric,'eq_time':eqt_x,'eq_mu':x_mu,'eq_std':x_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

        elif "rmsf" in metric:
            x = np.array(x)
            x_mu = np.mean(x)
            x_std = np.std(x)
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric,'eq_time':0,'eq_mu':x_mu,'eq_std':x_std}
            for key, value in params.items():
                eq_parameters[key].append(value)
        elif "sasa_res" == metric:
            continue
        elif "sasa" in metric:
            x = np.array(x)
            #[t0, g, Neff_max] = timeseries.detect_equilibration(x)
            t0 = 2000
            x_equlibrated = x[t0:]
            eqt_x = (len(t)-len(x_equlibrated))
            x_mu = np.mean(x_equlibrated)
            x_std = np.std(x_equlibrated)
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric,'eq_time':eqt_x,'eq_mu':x_mu,'eq_std':x_std}
            for key, value in params.items():
                eq_parameters[key].append(value)
        elif "potential" in metric:
            t = np.array(t)
            x = np.array(x)
            if "cdrs" in metric:
                radius = 5
            else:
                radius = 2
            x_mu = x[radius]
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric,'eq_time':0,'eq_mu':x_mu,'eq_std':0}
            for key, value in params.items():
                eq_parameters[key].append(value)

        elif "dipole" in metric:
            z = np.array(z)
            #[t0, g, Neff_max] = timeseries.detect_equilibration(z)
            t0 = 2000
            z_equlibrated = z[t0:]
            eqt_z = (len(t)-len(z_equlibrated))
            z_mu = np.mean(z_equlibrated)
            z_std = np.std(z_equlibrated)
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':temp,'metric':metric ,'eq_time':eqt_z,'eq_mu':z_mu,'eq_std':z_std}
            for key, value in params.items():
                eq_parameters[key].append(value)
        else:
            continue
    # add core/surface sasa metrics 
    nps = glob.glob('*.np')
    res_sasa = [np for np in nps if "res_sasa" in np]
    temps = [np.split('res_sasa_')[1] for np in res_sasa]
    temps = [temp.split('.np')[0] for temp in temps]
    temps = [int(x) for x in temps]
    sasa_dict = {temp: {} for temp in temps}
    sasa_dict['dSASA/dT'] = {}
    sections = ['total_mean','core_mean','surface_mean','total_std','core_std','surface_std']
    for temp in temps:
        sasa_dict = get_core_surface(sasa_dict, temp, k = 20, start = 20)
    for sec in sections:
        sasa_dict['dSASA/dT'][sec] = get_slope([(temp, sasa_dict[temp][sec]) for temp in temps])
    # iterate through sasa_dict and append values to eq_parameters
    for key, value in sasa_dict.items():
        if key == 'dSASA/dT':
            for sec, val in value.items():
                params = {'mAb':dirs[dir].split('/')[0],'TEMP':'all','metric':sec + '_dSASA_dT','eq_time':2000,'eq_mu':val,'eq_std':0}
                for k, value in params.items():
                    eq_parameters[k].append(value)
        elif key != 'dSASA/dT':
            for sec, val in value.items():
                params = {'mAb':dirs[dir].split('/')[0],'TEMP':str(key),'metric':sec + '_' + str(key),'eq_time':2000,'eq_mu':val,'eq_std':0}
                for k, value in params.items():
                    eq_parameters[k].append(value)

    # add order parameter metrics
    csvs = glob.glob('*.csv')
    ops = [csv for csv in csvs if "order_" in csv]
    # iterate through csvs and collect order parameter data
    for op in ops:
        if 'order_lambda' in op:
            df = pd.read_csv(op).drop(columns = ['Unnamed: 0'])
            # split left of csv
            name = op.split('.csv')[0]
            # mean of lambda and mean of r 
            lambda_mean = np.mean(df['lamda'])
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':'all','metric':name,'eq_time':2000,'eq_mu':lambda_mean,'eq_std':0}
            for key, value in params.items():
                eq_parameters[key].append(value)
            r_mean = np.mean(df['r'])
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':'all','metric':name + '_r','eq_time':2000,'eq_mu':r_mean,'eq_std':0}
            for key, value in params.items():
                eq_parameters[key].append(value)
        elif 'order_s2' in op:
            df = pd.read_csv(op).drop(columns = ['Unnamed: 0'])
            # split left of csv
            name = op.split('.csv')[0]
            # mean and std of first column by index
            s2_mean = np.mean(df.iloc[:,0])
            s2_std = np.std(df.iloc[:,0])
            # split name to get value before K
            temp = name.split('order_s2_')[1]
            temp = temp.split('K_')[0]
            params = {'mAb':dirs[dir].split('/')[0],'TEMP':str(temp),'metric':name,'eq_time':2000,'eq_mu':s2_mean,'eq_std':s2_std}
    # read output_analyze.log and find lines containing "Entropy"
    # if file contains output_analyze.507708.* then read file
    log_list = glob.glob('output_analyze.524677.*.log')
    with open(log_list[0]) as f:
        for line in f:
            if 'gmx_mpi anaeig -f md_final_covar_' in line:
                line = line.split()
                temp = line[8]
            if 'Entropy' and 'J/mol K' and 'Schlitter' in line:
                # split line on spaces
                line = line.split()
                # get value after Entropy
                entropy = line[8]
                params = {'mAb':dirs[dir].split('/')[0],'TEMP':str(temp),'metric':'sconf_schlitter','eq_time':2000,'eq_mu':entropy,'eq_std':0}
                for key, value in params.items():
                    eq_parameters[key].append(value)
            elif 'Entropy' and 'J/mol K' and 'Quasiharmonic' in line:
                line = line.split()
                entropy = line[8]
                params = {'mAb':dirs[dir].split('/')[0],'TEMP':str(temp),'metric':'sconf_quasiharmonic','eq_time':2000,'eq_mu':entropy,'eq_std':0}
                for key, value in params.items():
                    eq_parameters[key].append(value)
    os.chdir(cwd)
os.chdir(cwd)
df = pd.DataFrame(eq_parameters)
print(df)
df.to_csv('_abmelt_eq_20ns_parameters.csv')
