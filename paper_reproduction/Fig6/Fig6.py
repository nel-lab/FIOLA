#!/usr/bin/env python
import caiman as cm
from caiman.base.rois import com
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.utilities import fast_prct_filt
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy import random
import os
import pandas as pd
from pynwb import NWBHDF5IO
import pyximport
pyximport.install()
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
from sklearn.preprocessing import StandardScaler

#from fiola.config import load_fiola_config, load_caiman_config
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import bin_median, to_2D
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False})

import sys
sys.path.append('/home/nel/CODE/VIOLA/paper_reproduction')
from time import time
from paper_reproduction.utilities import multiple_dfs
from Fig6.Fig6_caiman_pipeline import run_caiman_fig6
from Fig6.Fig6_utilities import *

#base_folder = '/media/nel/storage/fiola/F2_20190415'
#base_folder = '/media/nel/storage/fiola/R6_20200210T2100'
base_folder = '/media/nel/storage/fiola/R2_20190219'
savef = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig6/'
for file in os.listdir(base_folder):
    if '.nwb' in file:
        path = os.path.join(base_folder, file)
        print(path)

io = NWBHDF5IO(path, 'r')
nwbfile_in = io.read()
pos = nwbfile_in.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['pos'].data[:]
speed = nwbfile_in.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['speed'].data[:]


#%% Behavior
s_t = (pos >= 0)  # remove reward time and preparation time
s_t[np.where(np.diff(pos)<-1)[0]] = False
s_t[np.where(pos==-0.5)[0]] = False
#s_t[s_t==False] = True
s_t[24710:25510] = False # remove bad trial
#s_t = (pos >= -20)
pos_n = pos.copy()
for i in range(len(pos_n)):
    if pos_n[i] < -2:
        j = i
        flag = 1
        while flag:
            j += 1
            try:
                if pos_n[j] >= -0.5:
                    pos_n[i] = pos_n[j]
                    flag = 0
            except:
                pos_n[i] = 0
                flag = 0
                
pos_s = StandardScaler().fit_transform(pos_n[:, None])[:, 0]
spd_s = StandardScaler().fit_transform(speed[:, None])[:, 0]
pos_ss = pos_s.copy()
pos_ss[s_t==False] = np.nan
#plt.plot(pos); plt.plot(pos_s[s_t])
print(s_t.sum())
#%% Fig 6c
lag = ['lag1', 'lag3', 'lag5'][2]
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff+f'Fig6c_result_{lag}_v4.0.npy', allow_pickle=True).item()
p = np.load(ff+f'Fig6c_prediction_{lag}_v4.0.npy', allow_pickle=True).item()
fig = plt.figure(figsize=(4.8, 6.4)) 
ax1 = plt.subplot()
rr = list(r.values())
methods = list(r.keys())
#num = [tt.shape[1] for tt in t_g.values()]
num = [1307, 1065, 907, 549, 1952, 1788, 1990]
r_mean = [np.mean(x) for x in rr]
r_std = [np.std(x) for x in rr]
colors = ['C0', 'C0', 'C0', 'C0', 'C1', 'C2', 'C3', 'C6']

p_value = {}
labels = methods.copy()
labels[-1] = 'CaImAn_Batch'


for idx in range(len(list(r.keys()))):
    ax1.errorbar(num[idx], r_mean[idx], yerr=r_std[idx], fmt='o', capsize=5, color=colors[idx], label=methods[idx])
    ax1.scatter(rand_jitter([num[idx]]*5, dev=3), list(r.values())[idx], color=colors[idx], alpha=0.6, s=15, facecolor='none')
    method = methods[idx]
        
    for k in range(len(methods)):
        dat = ttest_rel(r[methods[idx]], r[methods[k]], alternative='two-sided').pvalue 
        if f'{labels[idx]}' not in p_value:
            p_value[f'{labels[idx]}'] = []
        if idx < k:
            p_value[f'{labels[idx]}'].append(float(f'{dat:.2e}'))
        else:
            p_value[f'{labels[idx]}'].append('-')
        
    if 'FIOLA' in method:
        dat = ttest_rel(r[method], r['Suite2p'], alternative='two-sided').pvalue 
        #print(method)
        #print(dat)
        #print(r[method])
        #print(r['Suite2p'])
        barplot_annotate_brackets(dat, num[idx], num[5], 
                                  height = 0.003+ 0.005 * idx + np.max([max(r[method]), max(r['Suite2p'])]), 
                                  dy=0.003)

ax1.locator_params(axis='y', nbins=8)
ax1.locator_params(axis='x', nbins=4)
ax1.set_ylabel('Decoding R square')
ax1.set_xlabel('Number of neurons')
ax1.legend()
plt.tight_layout()
#plt.savefig(savef + f'Fig6d_{lag}_v4.0.pdf')
#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/source_data'
results = r.copy()

df1 = pd.DataFrame({})
for idx1, s1 in enumerate(labels):
    rr = r[methods[idx1]].copy()
    #rr = results[idx1].item()['result'][idx2]['F1'].copy()
    df1[s1] = rr

dfs = [df1]
text = 'Decoding R square of different algorithms'
fig_name = 'Fig 6c'
excel_name = os.path.join(excel_folder, 'Fig6.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/p_value'

df1 = pd.DataFrame({})
df1['methods'] = labels
for mmm in p_value:
    df1[mmm] = p_value[mmm]

dfs = [df1]
fig_name = 'Sheet'
excel_name = os.path.join(excel_folder, f'Fig 6c.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 0, text=None, row=0)

#%% Fig 6d
t_s = 3000
t_e = 31930
s_tt = s_t[t_s:t_e]
speed_s = speed[t_s:t_e][s_tt]
low_spd = np.percentile(speed_s, 33)
mid_spd = np.percentile(speed_s, 66)
t1 = np.where(speed_s <= low_spd)[0]
t2 = np.where(np.logical_and(speed_s > low_spd, speed_s < mid_spd))[0]
t3 = np.where(speed_s > mid_spd)[0]
plt.plot(speed_s)
plt.hlines([low_spd, mid_spd], 0, 30000, linestyles='dashed', color='black')

start = time()
#t_test = {key:t_g[key] for key in ('FIOLA3000', 'CaImAn_Online', 'Suite2p', 'CaImAn') if key in t_g}
tt = [t1, t2, t3]
spd_group = ['low', 'mid', 'high']

#%%
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff+f'Fig6e_result_{lag}_v3.8.npy', allow_pickle=True).item()
amps = [0]
spk_amps = [0]
colors = ['C0', 'C1', 'C2', 'C3']
methods = ['FIOLA3000', 'CaImAn_Online', 'Suite2p', 'CaImAn']
labels = methods.copy()
labels[-1] = 'CaImAn_Batch'

for amp in amps:
    r_all = []
    rr = {}
    batches = [0, 1, 2]
    n_batch = len(batches)
    
    for batch in batches:
        for jj, method in enumerate(methods):
            for idx, key in enumerate(r[method].keys()):    
                if idx == batch:
                    rr[method] =  r[method][key]
        r_all.append(rr)
        rr = {}
    
    fig = plt.figure(figsize=(8, 10))
    ax = plt.subplot(111)
    p_value = barplot_pvalue(r_all, methods, colors, ax, dev=0.01, capsize=5)
                        
    ax.set_xlabel('Speed group')
    ax.set_ylabel('Decoding R^2')
    ax.set_ylim([0.4,1.2])
    
    ff = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig6'
#plt.savefig(savef + f'Fig6e_{lag}_v3.10.pdf')

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/source_data'
results = r.copy()

df1 = pd.DataFrame({})
for idx1, s1 in enumerate(methods):
    for idx2, s2 in enumerate(['low', 'mid', 'high']):
        #try:
        rr = results[s1][s2].copy()
        #rr = results[idx1].item()['result'][idx2]['F1'].copy()
        df1[labels[idx1] + '_' + str(s2)] = rr

dfs = [df1]
text = 'Decoding R square of different algorithms'
fig_name = 'Fig 6d'
excel_name = os.path.join(excel_folder, 'Fig6.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/p_value'
results = r_all.copy()

df1 = pd.DataFrame({})
df1['speed group'] = spd_group
for mmm in p_value:
    df1[mmm] = p_value[mmm]
dfs = [df1]
fig_name = 'Sheet'
excel_name = os.path.join(excel_folder, f'Fig 6d.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 0, text=None, row=0)


#%% Fig 6e 
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff+f'/Fig6d_result_lag5_v3.8.npy', allow_pickle=True).item()
xx = np.array(list(range(13000, 28000, 2000)))
#xx = np.array(xx) - 13000
#fig = plt.figure(figsize=(8, 6)) 
methods = list(r.keys())
labels = methods.copy()
labels[-1] = 'CaImAn_Batch'
fig = plt.figure() 
ax1 = plt.subplot()
colors = ['C0', 'C3', 'C2', 'pink']

for idx in [0, 1, 2, 3]:
    ax1.plot(xx, list(r.values())[idx], label=list(r.keys())[idx], color=colors[idx])
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Decoding R^2')
ax1.locator_params(axis='y', nbins=8)
plt.xticks(xx, labels=[float(f'{xxx:.2e}') for xxx in xx/15.46/60])
#ax1.locator_params(axis='x', nbins=8)
plt.savefig(savef + f'Fig6d_{lag}_v3.10.pdf')

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/source_data'

df1 = pd.DataFrame({})
df1['starting frames'] = xx
for idx1, s1 in enumerate(labels):
    df1[s1] = r[methods[idx1]]
dfs = [df1]
text = 'Decoding R square of different algorithms with time'
fig_name = 'Fig 6e'
excel_name = os.path.join(excel_folder, 'Fig6.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%% Compute slope
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

slopes = []
intercepts = []

#xx = np.array(range(8))
xx = np.array(range(13000, 29000, 2000)) / 15.46/ 60
for key in ['FIOLA3000', 'CaImAn_Online', 'Suite2p', 'CaImAn']:
    X = np.array(xx)[:, None]
    #y = np.array(r['Suite2p']) - np.array(r['FIOLA3000'])
    y = np.array(r[key])# - np.array(r['FIOLA3000'])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xx,y)
    slopes.append(slope)
    intercepts.append(intercept)
    print(key)
    print(f'p_value:{p_value}')
    print(f'slope:{slope}')
    print(f'intercept:{intercept}')

plt.scatter(slopes, intercepts, color=['C0', 'C1', 'C2', 'C3'])

#%% Fig 6f
data_fr_custom = {}
for idx, num_frames in enumerate([500, 1000, 1500, 3000]):
    for i in [1]:
        file = f'/media/nel/storage/fiola/R2_20190219/{num_frames}/fiola_result_init_frames_{num_frames}_iteration_{i}_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_v3.15.npy'
        #file = f'/media/nel/storage/fiola/R2_2019import numpy 0219/{num_frames}/fiola_result_init_frames_{num_frames}_iteration_{i}_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_3_test_v3.21.npy'
        r = np.load(file, allow_pickle=True).item()
        data_fr_custom[num_frames] = np.diff(list(r.timing.values())[3])[1:] * 1000

for key in data_fr_custom.keys():
    print(np.mean(data_fr_custom[key]))
#%%
import matplotlib.cbook as cbook
stats = {}
count = 0
for key in data_fr_custom.keys():
    print(key)
    stats[key] = cbook.boxplot_stats(data_fr_custom[key], labels=str(count))[0]
    stats[key]["q1"], stats[key]["q3"] = np.percentile(
        data_fr_custom[key], [5, 95])
    stats[key]["whislo"], stats[key]["whishi"] = np.percentile(
        data_fr_custom[key], [0.1, 99.9])

    # stats[key]["whishi"] = stats[key]["q3"] + 1.5*(stats[key]["q3"]-stats[key]["q1"])
    # stats[key]["whislo"] = stats[key]["q1"] - 1.5*(stats[key]["q3"]-stats[key]["q1"])
    outliers = []
    for val in stats[key]["fliers"]:
        if val >= stats[key]['whishi'] or val <= stats[key]["whislo"]:
            outliers.append(val)
    stats[key]["fliers"] = outliers
    count += 1

colors = ["C0", "C0", "C0", "C0"]
fig, ax = plt.subplots(1, 1)
bplot = ax.bxp(stats.values(),  positions=range(4),  patch_artist=True)
ax.set_yscale("log")
ax.set_xticklabels(data_fr_custom.keys())

for patch, color in zip(bplot["boxes"], colors):
    print(color)
    patch.set_facecolor(color)
    
ax.hlines(1/15.46*1000, -0.5, 3, linestyle='dashed', color='black', label='frame rate')
ax.legend()
#ax.set_xticks([])
ax.set_xlabel('Init frames')
ax.set_ylabel('Time per frame (ms)')    
#plt.savefig(savef+'Fig6h_tpf_v3.72.pdf')

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/source_data'

df1 = pd.DataFrame({})
for idx1, s1 in enumerate(list(['FIOLA500', 'FIOLA1000', 'FIOLA1500', 'FIOLA3000'])):
    temp = data_fr_custom[[500, 1000, 1500, 3000][idx1]]
    if idx1 > 0:
        df1[s1] = np.append(temp, [np.nan]*(31431-temp.shape[0]))
    else:
        df1[s1] = temp
dfs = [df1]
text = 'FIOLA time per frame'
fig_name = 'Fig 6f'
excel_name = os.path.join(excel_folder, 'Fig6.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%% Fig 6g timing for init + acquisition + online exp
f_data = []
for init_frames in [3000, 1500, 1000, 500]:
    ff = f'/media/nel/storage/fiola/R2_20190219/{init_frames}'
    files = os.listdir(ff)
    file = [os.path.join(ff, file) for file in files if 'lag_3_test_v3.21' in file][0]
    print(len(files))
    r = np.load(file, allow_pickle=True)
    f_data.append((r.item().timing['init'] - r.item().timing['start']))
f_data = np.array(f_data)[:, None]

#%%
t_3000 = - np.diff(list(cnm3000.estimates.timing.values()))[0]
t_1500 = - np.diff(list(cnm1500.estimates.timing.values()))[0]
t_1000 = - np.diff(list(cnm1000.estimates.timing.values()))[0]
t_500 = - np.diff(list(cnm500.estimates.timing.values()))[0]
data = np.array([t_3000, t_1500, t_1000, t_500])[:, None]
data = np.hstack([np.array([3000/15.46, 1500/15.46, 1000/15.46, 500/15.46])[:, None], data, f_data,
                 np.array([1800, 1800, 1800, 1800])[:, None]])

# np.array([(31932 - 3000)/15.46, (31932 - 1500)/15.46, (31932 - 1000)/15.46, (31932 - 500)/15.46])[:, None]
data = data/60
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
ax1.bar(range(4), data[:, 0], label='acquision for init')
ax1.bar(range(4), data[:, 1] + data[:, 2], bottom=data[:, 0], label='init time')
ax1.bar(range(4), data[:, 3], bottom=data[:, 2] + data[:, 1] + data[:, 0], label='online exp time')
ax1.legend(frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_xlabel('init batch size (frames)')
ax1.set_ylabel('time (mins)')

ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(['3000', '1500', '1000', '500' ])
ax1.set_yticks([0, 10, 20, 30, 40])
#ax1.set_yticks([])
#ax1.set_ylim([0,1])
#plt.show()
#plt.savefig(savef + 'Fig6f_init_time_v3.7.pdf')

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/source_data'

df1 = pd.DataFrame({})
df1['init frames'] = [3000, 1500, 1000, 500]
df1['acquisition'] = data[:, 0] 
df1['init time'] = data[:, 1] + data[:, 2]  
df1['online exp time'] = data[:, 3] 

dfs = [df1]
text = 'FIOLA timing for acquision, initialization and online exp'
fig_name = 'Fig 6g'
excel_name = os.path.join(excel_folder, 'Fig6.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%% Fig 6h 
t_init = []
for init_frames in [500, 1000, 1500, 3000]:
    caiman_file = f'/media/nel/storage/fiola/R2_20190219/{init_frames}/memmap__d1_796_d2_512_d3_1_order_C_frames_{init_frames}__v3.7.hdf5'
    cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
    t_init.append(-np.diff(list(cnm2.estimates.timing.values()))[0])
t_init = np.array(t_init)

t_init1 = []
for init_frames in [500, 1000, 1500, 3000]:
    ff = f'/media/nel/storage/fiola/R2_20190219/{init_frames}'
    files = os.listdir(ff)
    file = [os.path.join(ff, file) for file in files if 'test_v3.15' in file][0]
    r = np.load(file, allow_pickle=True)
    t_init1.append((r.item().timing['init'] - r.item().timing['start']))
t_init1 = np.array(t_init1)

t_online = []
for idx, num_frames in enumerate([500, 1000, 1500, 3000]):
    for i in [1, 2, 3]:
        file = f'/media/nel/storage/fiola/R2_20190219/{num_frames}/fiola_result_init_frames_{num_frames}_iteration_{i}_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_v3.15.npy'
        r = np.load(file, allow_pickle=True).item()
        t_online.append(np.diff(list(r.timing.values())[1:3])[0])
tt = np.array(t_online).reshape((3,4), order='F').copy()
t_online = np.array(t_online).reshape((3,4), order='F')[2]

#%%
onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_caiman_online_results_v3.13.hdf5')
t_onacid = onacid.time_spend.sum()
t_s2p = 1836.44
t_s2p_rigid = 970
caiman = load_CNMF('/media/nel/storage/fiola/R2_20190219/memmap__d1_796_d2_512_d3_1_order_C_frames_31933__v3.7.hdf5')
t_caiman = -np.diff(list(caiman.estimates.timing.values()))[0]

data = np.zeros((2, 4))
data[0, :4] = t_init + t_init1
data[1, :4] = t_online

#%%
colors = [ 'C0', 'C1', 'C2', 'C3', 'C4', 'C6']
fig, ax1 = plt.subplots()
ax1.bar(range(4), data[0,:4]/60, label='FIOLA_init', color=colors[4])
ax1.bar(range(4), data[1,:4]/60, bottom=data[0, :4]/60, label='FIOLA_online', color=colors[0])
ax1.bar(4, t_onacid/60, label='CaImAn_Online', color=colors[1])
ax1.bar(5, t_s2p/60, label='Suite2p', color=colors[2])
ax1.bar(6, t_caiman/60, label='CaImAn', color=colors[3])
#ax1.bar(7, t_s2p_rigid/60, label='Suite2p_rigid', color=colors[5])

ax1.legend()
#ax1.set_xlabel('method')
ax1.set_ylabel('time (mins)')
ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([])
#ax1.set_xticklabels(['Fiola 3000', 'Fiola 1500', 'Fiola 1000', 'Fiola 500' ])
#ax1.set_yticks([])
#ax1.set_ylim([0,1])
#plt.savefig(savef+'Fig6g_total_time_v3.9.pdf')
#%%
data = data / 60
t_onacid /=  60
t_s2p /= 60
t_caiman /=60

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/source_data'

df1 = pd.DataFrame({})
df1['mode'] = ['init', 'online', 'offline']
df1['FIOLA500'] = [data[0, 0], data[1, 0], '-']  
df1['FIOLA1000'] = [data[0, 1], data[1, 1], '-']  
df1['FIOLA1500'] = [data[0, 2], data[1, 2], '-']  
df1['FIOLA3000'] = [data[0, 3], data[1, 3], '-']  
df1['CaImAn_Online'] = ['-', '-', t_onacid]  
df1['Suite2p'] = ['-', '-', t_s2p]  
df1['CaImAn_Batch'] = ['-', '-', t_caiman]  


dfs = [df1]
text = 'Timing for all algorithms'
fig_name = 'Fig 6h'
excel_name = os.path.join(excel_folder, 'Fig6.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%% Fig supp 16
files = ['/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_30_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_5_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_3_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_1_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy']
layers = [30, 10, 5, 3, 1]
#r = np.load('/media/nel/storage/fiola/R2_20190219/result/Fig6_supp_layers_result_v3.7.npy', allow_pickle=True).item()
r = np.load('/media/nel/storage/fiola/R2_20190219/result/Fig6_supp_layers_result_lag_5_v3.8.npy', allow_pickle=True).item()
mean = []
std = []
tpf = []
for idx, layer in enumerate(layers):
    for j in np.arange(1, 4):
        file = f'/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_{j}_num_layers_{layer}_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy'
        re = np.load(file, allow_pickle=True).item()
        tpf.append(np.diff(list(re.timing.values())[3]).mean())
    mean.append(np.mean(r['FIOLA'+str(layers[idx])]))
    std.append(np.std(r['FIOLA'+str(layers[idx])]))
tpf = np.array(tpf) * 1000
#
tpfm = np.array(tpf).reshape(5, 3).mean(1)
tpfs = np.array(tpf).reshape(5, 3).std(1)

#%%
ff = '/media/nel/storage/fiola/R2_20190219/result/'
rr = np.load(ff+'Fig6c_result_lag5_v4.0.npy', allow_pickle=True).item()

r['CaImAn_Online'] = rr['CaImAn_Online']
r['Suite2p'] = rr['Suite2p']
r['CaImAn'] = rr['CaImAn']

for key in ['CaImAn_Online', 'Suite2p', 'CaImAn']:
    mean.append(np.mean(r[key]))
    std.append(np.std(r[key]))

#%%
#onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_caiman_online_results_v3.16.hdf5')
onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_caiman_online_results_final_v3.14.hdf5')
t_onacid = onacid.time_spend.sum()
t_s2p = 1836.44
t_s2p_rigid = 970
caiman = load_CNMF('/media/nel/storage/fiola/R2_20190219/memmap__d1_796_d2_512_d3_1_order_C_frames_31933__v3.7.hdf5')
t_caiman = -np.diff(list(caiman.estimates.timing.values()))[0]

tpfm = np.append(tpfm, np.array([t_onacid / 31932, t_s2p/31932, t_caiman/31932]) * 1000)

#%%
ff = '/media/nel/storage/fiola/R2_20190219/result/'
fig = plt.figure() 
ax1 = plt.subplot()
colors = ['C5', 'C6', 'C7', 'C8', 'C9', 'C1', 'C2', 'C3', 'C4']

for idx in range(len(list(r.keys()))):
    #if idx < 5:
        try:
            ax1.errorbar(tpfm[idx], mean[idx], yerr=std[idx], fmt='o', capsize=5, color=colors[idx], label=layers[idx])
            ax1.scatter(rand_jitter([tpfm[idx]]*5, dev=0.001), list(r.values())[idx], color=colors[idx], alpha=0.6, s=15, facecolor='none')
        except:
            ax1.errorbar(tpfm[idx], mean[idx], yerr=std[idx], fmt='o', capsize=5, color=colors[idx], label=list(r.keys())[idx])
            ax1.scatter(rand_jitter([tpfm[idx]]*5, dev=1), list(r.values())[idx], color=colors[idx], alpha=0.6, s=15, facecolor='none')

ax1.set_ylabel('decoding R square')
ax1.set_xlabel('time per frame (ms)')
ax1.set_ylim([0.75, 0.96])
ax1.legend()
#plt.savefig(savef + 'Fig6_supp_num_layers_lag5_b_v3.10.pdf')
plt.savefig(savef + 'Fig6_supp_num_layers_lag5_a_v3.11.pdf')

#%% Fig supp 17
lag = ['lag1', 'lag3', 'lag5'][2]
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff + f'Fig6_supp_decoding_frames_result_{lag}_v3.8.npy', allow_pickle=True)

fig = plt.figure() 
ax1 = plt.subplot()
rr = np.array([[rrr['FIOLA3000'][0], rrr['Suite2p'][0]] for rrr in r])
methods = ['FIOLA3000', 'Suite2p']
num = np.array(list(range(1000, 16000, 1000)))
colors = ['C0', 'C3']

for idx in range(len(methods)):
    ax1.bar(num+200*idx, rr[:, idx], width=200, label=methods[idx])

ax1.set_ylim([0.5, 1])
ax1.locator_params(axis='y', nbins=8)
ax1.locator_params(axis='x', nbins=4)
ax1.set_ylabel('Decoding R square')
ax1.set_xlabel('Number of frames')
ax1.legend()
plt.tight_layout()
plt.savefig(savef + f'Fig6_supp_decoding_frames_{lag}_v3.10.pdf')

#%% Fig supp onacid result
lag = ['lag1', 'lag3', 'lag5'][2]
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff+f'Fig6_supp_onacid_result_lag5_v3.10.npy', allow_pickle=True).item()
fig = plt.figure(figsize=(10, 10)) 
ax1 = plt.subplot()
rr = list(r.values())
methods = list(r.keys())
num = [1788, 256, 530, 714, 1041, 1305, 1538, 1952]
r_mean = [np.mean(x) for x in rr]
r_std = [np.std(x) for x in rr]

colors = ['C2', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1']
p_value = []

for idx in range(len(list(r.keys()))):
    ax1.errorbar(num[idx], r_mean[idx], yerr=r_std[idx], fmt='o', capsize=5, color=colors[idx], label=methods[idx])
    ax1.scatter(rand_jitter([num[idx]]*5, dev=3), list(r.values())[idx], color=colors[idx], alpha=0.6, s=15, facecolor='none')
    method = methods[idx]
    
    if 'CaImAn_Online' in method:
        dat = ttest_rel(r[method], r['Suite2p'], alternative='two-sided').pvalue 
        
        barplot_annotate_brackets(dat, num[idx], num[0], 
                              height = 0.003+ 0.005 * idx + np.max([max(r[method]), max(r['Suite2p'])]), 
                              dy=0.003)
        p_value.append(float(f'{dat:.2e}'))
ax1.locator_params(axis='y', nbins=8)
ax1.locator_params(axis='x', nbins=4)
ax1.set_ylabel('Decoding R square')
ax1.set_xlabel('Number of neurons')
ax1.legend()
plt.tight_layout()

#plt.savefig(savef + f'Fig_supp_onacid_result_{lag}_v3.10.pdf')

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/p_value'
#results = r_all.copy()

df1 = pd.DataFrame({})
df1['CaImAn Online number of frames'] = [500, 1000, 1500, 3000, 5000, 7000, 10000]
df1['p_value'] = p_value
dfs = [df1]
fig_name = 'Sheet'
excel_name = os.path.join(excel_folder, f'Supp Fig 17b.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 0, text=None, row=0)




#%%

# #%%
# fig, ax = plt.subplots(figsize=(8,8))
# mean = [[np.mean(list(r.values())[idx][:4]), np.mean(list(r.values())[idx][4:])] for idx in [0, 1, 2, 3]] 
# std = [[np.std(list(r.values())[idx][:4]), np.std(list(r.values())[idx][4:])] for idx in [0, 1, 2, 3]] 
# for idx in [0, 1, 2, 3]:
#     ax.bar(np.array([0, 0.6]) + idx * 0.1, mean[idx], width=0.1, yerr=std[idx], color=colors[idx], label=list(r.keys())[idx])
#     method = methods[idx]
#     dat = ttest_ind(r[method][:4], r[method][-4:], alternative='greater').pvalue 
#     print(dat) 
#     barplot_annotate_brackets(dat, idx*0.1, idx*0.1+0.6, 
#                               height = 0.95+ 0.03 * idx, 
#                               dy=0.003)

# ax.set_ylim([0.3,1.1])
# #ax.set_xlabel('Frame')
# ax.set_ylabel('Decoding R^2')
# ax.legend()
# plt.tight_layout()
# #plt.savefig(savef + f'Fig6d_small_{lag}_v3.8.pdf')
# # 0.010530810321843241
# # 0.009157310787153716
# # 0.014374410454194287
# # 0.009964330324046627
# #%%
# ff = '/media/nel/storage/fiola/R2_20190219/result/'
# #r = np.load(ff+f'Fig6d_result_lag3_v3.8.npy', allow_pickle=True).item()
# r = np.load(ff+f'Fig6d_result_lag5_v3.8.npy', allow_pickle=True).item()
# #r['FIOLA3000'] = r1['FIOLA3000']
# #r['CaImAn_Online'] = r1['CaImAn_Online']
# amps = [0]
# spk_amps = [0]
# colors = ['C0', 'C1', 'C2', 'C3']
# methods = ['FIOLA3000', 'CaImAn_Online', 'Suite2p', 'CaImAn']

# for amp in amps:
#     r_all = []
#     rr = {}
#     batches = [0, 1]
#     n_batch = len(batches)
    
#     for batch in batches:
#         for jj, method in enumerate(methods):
#             # for idx, key in enumerate(r[method].keys()):    
#             #     if idx == batch:
#             rr[method] =  r[method][batch * 4 : (batch + 1) * 4]
#         r_all.append(rr)
#         rr = {}
    
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     barplot_pvalue(r_all, methods, colors, ax)
                    
#     ax.set_xlabel('Speed group')
#     ax.set_ylabel('Decoding R^2')
#     #ax.set_ylim([0.4,1.1])


#%% supp timing s2p
colors = [ 'C0', 'C3', 'C7', 'C4']
fig, ax1 = plt.subplots()
ax1.bar(range(1), data[0,-1]/60, label='FIOLA_init', color=colors[-1])
ax1.bar(range(1), data[1,-1]/60, bottom=data[0, -1]/60, label='FIOLA_online', color=colors[0])
ax1.bar(1, t_s2p/60, label='Suite2p', color=colors[1])
ax1.bar(2, t_s2p_rigid/60, label='Suite2p_rigid', color=colors[2])
#ax1.bar(7, t_s2p_rigid/60, label='Suite2p_rigid', color=colors[5])

ax1.legend()
#ax1.set_xlabel('method')
ax1.set_ylabel('time (mins)')
ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([])
#ax1.set_xticklabels(['Fiola 3000', 'Fiola 1500', 'Fiola 1000', 'Fiola 500' ])
#ax1.set_yticks([])
#ax1.set_ylim([0,1])
#plt.savefig(savef+'Fig6_supp_s2p_rigid_timing_v3.8.pdf')
