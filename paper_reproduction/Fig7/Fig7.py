#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:37:36 2021

@author: nel
"""
import caiman as cm
from caiman.base.rois import com
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.utilities import fast_prct_filt
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy import random
import os
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
from Fig7.Fig7_caiman_pipeline import run_caiman_fig7
from Fig7.Fig7_utilities import *

#base_folder = '/media/nel/storage/fiola/F2_20190415'
#base_folder = '/media/nel/storage/fiola/R6_20200210T2100'
base_folder = '/media/nel/storage/fiola/R2_20190219'
savef = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig7/'
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

#%%
#aa = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_2_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', allow_pickle=True).item()
# t3000 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_v3.15.npy', allow_pickle=True).item().trace_deconvolved
# t1500 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_result_init_frames_1500_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_v3.15.npy', allow_pickle=True).item().trace_deconvolved
# t1000 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_result_init_frames_1000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_v3.15.npy', allow_pickle=True).item().trace_deconvolved
# t500 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_result_init_frames_500_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_v3.15.npy', allow_pickle=True).item().trace_deconvolved
# t3000 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_1_test_v3.21.npy', allow_pickle=True).item().online_trace_deconvolved
# t1500 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_result_init_frames_1500_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_1_test_v3.21.npy', allow_pickle=True).item().online_trace_deconvolved
# t1000 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_result_init_frames_1000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_1_test_v3.21.npy', allow_pickle=True).item().online_trace_deconvolved
# t500 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_result_init_frames_500_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_1_test_v3.21.npy', allow_pickle=True).item().online_trace_deconvolved

t3000 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_5_test_v3.21.npy', allow_pickle=True).item().online_trace_deconvolved
t1500 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_result_init_frames_1500_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_5_test_v3.21.npy', allow_pickle=True).item().online_trace_deconvolved
t1000 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_result_init_frames_1000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_5_test_v3.21.npy', allow_pickle=True).item().online_trace_deconvolved
t500 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_result_init_frames_500_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_5_test_v3.21.npy', allow_pickle=True).item().online_trace_deconvolved

cnm3000 = load_CNMF('/media/nel/storage/fiola/R2_20190219/3000/memmap__d1_796_d2_512_d3_1_order_C_frames_3000__v3.7.hdf5')
cnm1500 = load_CNMF('/media/nel/storage/fiola/R2_20190219/1500/memmap__d1_796_d2_512_d3_1_order_C_frames_1500__v3.7.hdf5')
cnm1000 = load_CNMF('/media/nel/storage/fiola/R2_20190219/1000/memmap__d1_796_d2_512_d3_1_order_C_frames_1000__v3.7.hdf5')
cnm500 = load_CNMF('/media/nel/storage/fiola/R2_20190219/500/memmap__d1_796_d2_512_d3_1_order_C_frames_500__v3.7.hdf5')

cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/memmap__d1_796_d2_512_d3_1_order_C_frames_31933__v3.13.hdf5')
tracec = cnm2.estimates.S
traces = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p_result_non_rigid_v3.1/plane0/spks.npy')
#onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mov_R2_20190219T210000_caiman_online_results_v3.0_new_params_min_snr_2.0.hdf5')
#onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_caiman_online_results_v3.13.hdf5')
#onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_caiman_online_results_final_v3.14.hdf5')
#onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_caiman_online_results_v3.16.hdf5')
#onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_caiman_online_results_final_v3.20.hdf5')
onacid_comp = onacid.estimates.idx_components
traceo = onacid.estimates.online_deconvolved_trace[onacid_comp]
onacid.estimates.A = onacid.estimates.A[:, onacid_comp]
onacid.time_neuron_added = onacid.time_neuron_added[onacid_comp]
#cnm = load_CNMF('/media/nel/storage/fiola/R2_20190219/3000/memmap__d1_796_d2_512_d3_1_order_C_frames_3000__v3.7.hdf5')
stats = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p_result_non_rigid_v3.1/plane0/stat.npy', allow_pickle=True)

#%%
sp_raw = {'FIOLA3000': cnm3000, 'FIOLA1500': cnm1500, 'FIOLA1000': cnm1000, 'FIOLA500': cnm500, 
          'CaImAn_Online': onacid, 'Suite2p': stats, 'CaImAn':cnm2}
#sp_raw = {'FIOLA3000': cnm3000, 'CaImAn_Online': onacid}
selection = select_neurons_within_regions(sp_raw, y_limit=[30, 777])
iscell = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p_result_non_rigid_v3.1/plane0/iscell.npy')
s_selected = np.where(iscell[:, 1] >=0.05)[0]
selection['Suite2p'] = np.intersect1d(selection['Suite2p'], s_selected)
#o_selected = np.intersect1d(np.where(onacid.time_neuron_added[:, 1] < 10000)[0], onacid.estimates.idx_components)
o_selected = np.where(onacid.time_neuron_added[:, 1] < 10000)[0]
selection['CaImAn_Online'] = np.intersect1d(selection['CaImAn_Online'], o_selected)

sp_processed = run_spatial_preprocess(sp_raw, selection)
#sp_processed = {key:values    for key,values in sp_raw.items()}
t_raw = {'FIOLA3000': t3000.T, 'FIOLA1500': t1500.T, 'FIOLA1000': t1000.T, 'FIOLA500': t500.T , 
          'CaImAn_Online': traceo.T, 'Suite2p': traces.T, 'CaImAn':tracec.T}
#t_raw = {'FIOLA3000': t3000.T, 'CaImAn_Online': traceo.T}

t_g, t_rs, t_rm = run_trace_preprocess(t_raw, selection, sigma=12)

#%%
#[print(tt.shape) for tt in t_raw.values()]
t_g1 = t_g.copy()
lag = 5
for idx, key in enumerate(t_g.keys()):
    print(key)
    if 'FIOLA' in key:
        ttest = np.zeros((31933, t_g[key].shape[1]))
        ttest[31933-t_g[key].shape[0]-lag-1:-lag-1, :] = t_g[key]
        t_g1[key] = ttest
    if 'CaImAn_Online' in key:
        ttest = np.zeros((31933, t_g[key].shape[1]))
        ttest[0:-lag-1, :] = t_g[key][lag+1:, :]
        t_g1[key] = ttest
t_g = t_g1.copy()

#%% number of iterations
t_raw = {}
files = ['/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_30_trace_with_neg_False_center_dims_(398, 256)_lag_5_test_v3.21.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_5_test_v3.21.npy',
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_5_trace_with_neg_False_center_dims_(398, 256)_lag_5_test_v3.21.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_3_trace_with_neg_False_center_dims_(398, 256)_lag_5_test_v3.21.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_1_trace_with_neg_False_center_dims_(398, 256)_lag_5_test_v3.21.npy']

num = [30, 10, 5, 3, 1]
for idx, file in enumerate(files):
    t_raw[f'FIOLA{num[idx]}'] = np.load(file, allow_pickle=True).item().online_trace_deconvolved.T
[print(tt.shape) for tt in t_raw.values()]

sp_raw = {'FIOLA30': cnm3000, 'FIOLA10': cnm3000, 'FIOLA5': cnm3000, 'FIOLA3': cnm3000, 
          'FIOLA1': cnm3000}
selection = select_neurons_within_regions(sp_raw, y_limit=[30, 777])
t_g, t_rs, t_rm = run_trace_preprocess(t_raw, selection, sigma=12)

#%%
#[print(tt.shape) for tt in t_raw.values()]
t_g1 = t_g.copy()
lag = 5
for idx, key in enumerate(t_g.keys()):
    print(key)
    if 'FIOLA' in key:
        ttest = np.zeros((31933, t_g[key].shape[1]))
        ttest[31933-t_g[key].shape[0]-lag-1:-lag-1, :] = t_g[key]
        t_g1[key] = ttest
t_g = t_g1.copy()

#%%
# #%% s2p rigid
# t_s2p_rigid = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p/plane0/spks.npy')
# stats_rigid = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p/plane0/stat.npy', allow_pickle=True)
# sp_raw = {'Suite2p_rigid':stats_rigid}
# selection = select_neurons_within_regions(sp_raw, y_limit=[30, 777])

# t_raw = {'Suite2p_rigid':t_s2p_rigid.T}
# t_g, t_rs, t_rm = run_trace_preprocess(t_raw, selection, sigma=12)

# t_raw = {}
# select = selection['FIOLA3000']
# for lag in [0, 1, 3, 5, 10]:
#     tt = np.load(f'/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_{lag}_test_v3.21.npy', allow_pickle=True).item().online_trace_deconvolved
#     t_raw[f'FIOLA_lag{lag}'] = tt.T[:, select]
# t_raw['FIOLA3000'] = t3000.T[:, select]
# t_g, t_rs, t_rm = run_trace_preprocess(t_raw, None, sigma=12)

# t_g1 = t_g.copy()
# for lag in [0, 1, 3, 5, 10]:
#     ttest = np.zeros(t_raw['FIOLA3000'].shape)
#     ttest[3000-lag-1:-lag-1, :] = t_g[f'FIOLA_lag{lag}']
#     t_g1[f'FIOLA_lag{lag}'] = ttest
# t_g = t_g1.copy()
#%% Fig 7c decoding performance through cross-validation
start = time()
t_test = t_g.copy()
t_test = {'FIOLA3000': t_g['FIOLA3000'], 'CaImAn_Online': t_g['CaImAn_Online']}
#t_test = {'Suite2p': t_g['Suite2p'], 'CaImAn':t_g['CaImAn']}
#t_test = {'FIOLA_10': t_g['FIOLA_10']}
t_s = 3000
t_e = 31900
dec = [pos_s, spd_s][0].copy()

r = {}
p = {}
alpha_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
for key, tr in t_test.items():
    print(key)
    s_tt = s_t[t_s:t_e]
    X = tr[t_s:t_e][s_tt]
    Y = dec[t_s:t_e][s_tt]
    r[key], p[key] = cross_validation_ridge(X, Y, normalize=True, n_splits=5, alpha_list=alpha_list)
    print(f'average decoding performance: {np.mean(r[key])}')
end = time()
print(f'processing time {end-start}')

ff = '/media/nel/storage/fiola/R2_20190219/result/'
#np.save(ff+'Fig7c_result_lag5_v3.9.npy', r); np.save(ff+'Fig7c_prediction_lag5_v3.9.npy', p)
#np.save(ff+'Fig7_supp_dec_lag_v3.8.npy', r); np.save(ff+'Fig7_supp_dec_lag_prediction_v3.8.npy', p)
#np.save(ff+'Fig7_supp_layers_result_lag_5_v3.9.npy', r); np.save(ff+'Fig7_supp_layers_prediction_lag_5_v3.9.npy', p)
#np.save(ff+'Fig7_supp_s2p_rigid_result_v3.8.npy', r)

#%% Fig 7d decoding performance across time
train = np.array([3000, 13000])
flag = 0
from copy import deepcopy
t_test = {key:t_g[key] for key in ('FIOLA3000', 'CaImAn_Online', 'Suite2p', 'CaImAn') if key in t_g}
#t_test = {key:t_g[key] for key in (['FIOLA3000', 'CaImAn_Online']) if key in t_g}
#t_test = t_g.copy()
t_s = 3000
t_e = 31500
dec = [pos_s, spd_s][0].copy()
alpha_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

r = {}
for key, tr in deepcopy(t_test).items():
    s = []
    print(key)
    s_tt = s_t[train[0]:train[1]]
    X = tr[train[0]:train[1]][s_tt]
    Y = dec[train[0]:train[1]][s_tt]       
    X = StandardScaler().fit_transform(X)
    Y = StandardScaler().fit_transform(Y[:, None])[:, 0]
    alpha_best = cross_validation_regularizer_strength(x=X, y=Y, normalize=True, n_splits=5, alpha_list=alpha_list)
    clf = Ridge(alpha=alpha_best)
    clf.fit(X, Y)  
    
    for test in [[i, i+4000] for i in range(13000, 28000, 2000)]:
        print(test)
        s_tt = s_t[test[0]:test[1]]
        x = tr[test[0]:test[1]][s_tt]
        y = dec[test[0]:test[1]][s_tt]   
        x = StandardScaler().fit_transform(x)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]        
        s.append(clf.score(x, y))
        if test[0] == 25000:
            plt.figure(); plt.plot(clf.predict(x)); plt.plot(y); plt.title(f'{test}'); plt.show()
    r[key] = s        
    print(f'average decoding performance: {np.mean(r[key])}')
ff = '/media/nel/storage/fiola/R2_20190219/result/'
np.save(ff+'Fig7d_result_lag5_v3.9.npy', r)

#%% Fig 7e Decoding position at different speed group
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
t_test = {key:t_g[key] for key in ('FIOLA3000', 'CaImAn_Online', 'Suite2p', 'CaImAn') if key in t_g}
tt = [t1, t2, t3]
spd_group = ['low', 'mid', 'high']
t_s = 3000
t_e = 31900
dec = [pos_s, spd_s][0].copy()

r = {}
p = {}
alpha_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
for key, tr in t_test.items():
    print(key)
    r[key] = {}; p[key] = {}
    for i, t in enumerate(tt):
        g = spd_group[i]
        print(g)
        s_tt = s_t[t_s:t_e]
        X = tr[t_s:t_e][s_tt][t]
        Y = dec[t_s:t_e][s_tt][t]
        r[key][g], p[key][g] = cross_validation_ridge(X, Y, normalize=True, n_splits=5, alpha_list=alpha_list)
        print(f'average decoding performance: {np.mean(r[key][g])}')
end = time()
print(f'processing time {end-start}')

np.save(ff+'Fig7e_result_lag5_v3.9.npy', r)
np.save(ff+'Fig7e_prediction_lag5_v3.9.npy', p)

#%% Figure 7c
lag = ['lag1', 'lag3', 'lag5'][2]
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff+f'Fig7c_result_{lag}_v3.8.npy', allow_pickle=True).item()
p = np.load(ff+f'Fig7c_prediction_{lag}_v3.8.npy', allow_pickle=True).item()
fig = plt.figure(figsize=(4.8, 6.4)) 
ax1 = plt.subplot()
rr = list(r.values())
methods = list(r.keys())
#num = [tt.shape[1] for tt in t_g.values()]
num = [1307, 1065, 907, 549, 2019, 1788, 1990]
r_mean = [np.mean(x) for x in rr]
r_std = [np.std(x) for x in rr]

colors = ['C0', 'C0', 'C0', 'C0', 'C1', 'C2', 'C3', 'C6']


for idx in range(len(list(r.keys()))):
    ax1.errorbar(num[idx], r_mean[idx], yerr=r_std[idx], fmt='o', capsize=5, color=colors[idx], label=methods[idx])
    ax1.scatter(rand_jitter([num[idx]]*5, dev=3), list(r.values())[idx], color=colors[idx], alpha=0.6, s=15, facecolor='none')
    method = methods[idx]
    
    for mm in ['CaImAn', 'CaImAn_Online', 'Suite2p']:
        d1 = ttest_rel(r[method], r[mm], alternative='two-sided').pvalue 
        print(method)
        print(mm)
        print(d1)
    
    
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

plt.savefig(savef + f'Fig7c_pos_{lag}_v3.10.pdf')
#plt.savefig(savef + 'Fig_supp_spd_v3.1.pdf')
# 0.17898542457533081
# 0.0055860117488064406
# 0.023733082648987967
# 0.0035924052250248485

#%% Fig 7d 
r = np.load(ff+f'Fig7d_result_lag5_v3.8.npy', allow_pickle=True).item()
xx = np.array(list(range(13000, 28000, 2000)))
#xx = np.array(xx) - 13000
#fig = plt.figure(figsize=(8, 6)) 
methods = list(r.keys())
fig = plt.figure() 
ax1 = plt.subplot()
colors = ['C0', 'C1', 'C2', 'C3']

for idx in [0, 1, 2, 3]:
    #if 'Fiola' in methods[idx]:
    ax1.plot(xx, list(r.values())[idx], label=list(r.keys())[idx], color=colors[idx])
    ax1.scatter(rand_jitter([num[idx]]*5, dev=0.1), list(r.values())[idx], color=colors[idx], alpha=0.5, s=15)
#[ax1.plot(xx, x) for x in r_all]
#ax1.plot(xx, np.array(r['Suite2p']) - np.array(r['FIOLA3000']), label='diff between Suite2p and FIOLA', color='purple') 
#ax1.plot(xx, np.array(r['CaImAn_Online']) - np.array(r['FIOLA3000']), label='diff between CaImAn Online and FIOLA', color='pink') 
ax1.set_xlabel('Frame')
ax1.set_ylabel('Decoding R^2')
ax1.locator_params(axis='y', nbins=6)
ax1.locator_params(axis='x', nbins=4)
#ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
#ax1.set_xticks([])
#ax1.set_yticks([])
#ax1.set_ylim([-0.1,1])
#plt.savefig(savef + f'Fig7d_{lag}_v3.9.pdf')

#%%
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
# #plt.savefig(savef + f'Fig7d_small_{lag}_v3.8.pdf')
# # 0.010530810321843241
# # 0.009157310787153716
# # 0.014374410454194287
# # 0.009964330324046627
# #%%
# ff = '/media/nel/storage/fiola/R2_20190219/result/'
# #r = np.load(ff+f'Fig7d_result_lag3_v3.8.npy', allow_pickle=True).item()
# r = np.load(ff+f'Fig7d_result_lag5_v3.8.npy', allow_pickle=True).item()
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
    

#%%
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
r = np.load(ff+f'Fig7e_result_{lag}_v3.8.npy', allow_pickle=True).item()
amps = [0]
spk_amps = [0]
colors = ['C0', 'C1', 'C2', 'C3']
methods = ['FIOLA3000', 'CaImAn_Online', 'Suite2p', 'CaImAn']

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
    barplot_pvalue(r_all, methods, colors, ax, dev=0.01, capsize=5)
                        
    ax.set_xlabel('Speed group')
    ax.set_ylabel('Decoding R^2')
    ax.set_ylim([0.4,1.2])
    
    ff = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig7'
plt.savefig(savef + f'Fig7e_{lag}_v3.10.pdf')


#%% Fig 7g timing for init + acquisition + online exp
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
plt.savefig(savef + 'Fig7f_init_time_v3.7.pdf')

#%% Fig 7h timing for all methods
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
onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_caiman_online_results_v3.16.hdf5')
t_onacid = onacid.time_spend.sum()
t_s2p = 1836.44
t_s2p_rigid = 970
caiman = load_CNMF('/media/nel/storage/fiola/R2_20190219/memmap__d1_796_d2_512_d3_1_order_C_frames_31933__v3.7.hdf5')
t_caiman = -np.diff(list(caiman.estimates.timing.values()))

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
plt.savefig(savef+'Fig7g_total_time_v3.9.pdf')

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
plt.savefig(savef+'Fig7_supp_s2p_rigid_timing_v3.8.pdf')


#%% Fig h
data_fr_custom = {}
for idx, num_frames in enumerate([500, 1000, 1500, 3000]):
    for i in [1]:
        file = f'/media/nel/storage/fiola/R2_20190219/{num_frames}/fiola_result_init_frames_{num_frames}_iteration_{i}_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_v3.15.npy'
        #file = f'/media/nel/storage/fiola/R2_20190219/{num_frames}/fiola_result_init_frames_{num_frames}_iteration_{i}_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_3_test_v3.21.npy'
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
#plt.savefig(savef+'Fig7h_tpf_v3.72.pdf')


#%% Fig supp timing vs accuracy trade off
files = ['/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_30_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_5_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_3_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy', 
         '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_1_trace_with_neg_False_center_dims_(398, 256)_test_num_layers_v3.15.npy']
layers = [30, 10, 5, 3, 1]
#r = np.load('/media/nel/storage/fiola/R2_20190219/result/Fig7_supp_layers_result_v3.7.npy', allow_pickle=True).item()
r = np.load('/media/nel/storage/fiola/R2_20190219/result/Fig7_supp_layers_result_lag_5_v3.8.npy', allow_pickle=True).item()
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
rr = np.load(ff+'Fig7c_result_lag5_v3.8.npy', allow_pickle=True).item()

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
    if idx < 5:
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
plt.savefig(savef + 'Fig7_supp_num_layers_lag5_b_v3.10.pdf')
#plt.savefig(savef + 'Fig7_supp_num_layers_lag5_a_v3.10.pdf')


#%%
data = {}
for idx, method in enumerate(list(r.keys())):
    print(method)
    data[f'{method}_R_sq_mean'] = mean[idx]
    data[f'{method}_R_sq_std'] = std[idx]
    data[f'{method}_time_mean'] = tpfm[idx]
    try:
        data[f'{method}_time_std'] = tpfs[idx]
    except:
        print('skip')
np.save(savef +  'Fig7_supp_num_layers_v3.72_data.npy', data)
        

#%%
for j in layers:
    method = 'FIOLA'+str(j)
    m30 = 'FIOLA'+str(30)
    dat = ttest_rel(r[method], r[m30], alternative='two-sided').pvalue 
    print(dat) 


#%% preprocessing to find the best lag and gaussian filter size
#for sigma in [4*i + 2 for i in range(5)]:
#sigma = 12
#sigma = 0
#lag = 5
start = time()
result = {}
for lag in range(0, 2, 2):#(-6, 8, 2): # positive means shift to the right, minus means shift to the left
    result[lag] = {}
    for sigma in range(0, 16, 2):
        #print(lag); print(sigma)
        t_g = {}
        for key in t_rs.keys():
            if key == 'FIOLA3000':
                t_g[key] = t_rs[key].copy()
                if sigma > 0:
                    for i in range(t_g[key].shape[1]):
                        t_g[key][:, i] = gaussian_filter1d(t_g[key][:, i], sigma=sigma)
                else:
                    pass
                t_g[key] = np.roll(t_g[key], shift=lag, axis=0)
            
        t_test = t_g.copy()
        t_s = 3000
        t_e = 31930
        dec = [pos_s, spd_s][1].copy()
        r = {}
        p = {}
        alpha_list = [1000, 5000, 10000]
        for key, tr in t_test.items():
            print(f'{key}, {sigma}, {lag}')
            s_tt = s_t[t_s:t_e]
            X = tr[t_s:t_e][s_tt]
            Y = dec[t_s:t_e][s_tt]
            r[key], p[key] = cross_validation_ridge(X, Y, normalize=False, n_splits=5, alpha_list=alpha_list)
            print(f'average decoding performance: {np.mean(r[key])}')
        result[lag][sigma] = np.mean(r[key])
end = time()
print(f'processing time {end-start}')

aa = np.array([list(result[key].values()) for key in result.keys()])
plt.plot(list(range(-6, 8, 2)), aa); plt.legend(list(range(0, 16, 2))); plt.xlabel('lag'); plt.ylabel('R2'); plt.savefig(savef + 'Fig7_lag_gaussian1.pdf')
plt.plot(list(range(0, 16, 2)), aa.T); plt.legend(list(range(-6, 8, 2))); plt.xlabel('gaussian filter size'); plt.ylabel('R2'); plt.savefig(savef + 'Fig7_lag_gaussian2.pdf')

#%% Decoding vs threshold
plt.plot([1389, 1311, 1225, 1116, 932, 817], [np.mean(r[rr]) for rr in r.keys()]); plt.ylabel('R square'); plt.xlabel('num neurons'); 
plt.savefig(savef+'Fig7_Rsquare_num_neurons.pdf')

#for i in range(len(method)):
#    ax1.annotate(method[i], (xx[i] + 10, yy[i] - 0.02))    
#%%
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff+'Fig7c_result_v3.7.npy', allow_pickle=True).item()
p = np.load(ff+'Fig7c_prediction_v3.7.npy', allow_pickle=True).item()
r1 = np.load(ff + 'Fig7_supp_dec_lag_v3.8.npy', allow_pickle=True).item()
fig = plt.figure() 
ax1 = plt.subplot()
rr = list(r.values())
methods = list(r.keys())
#num = [tt.shape[1] for tt in t_g.values()]
num = [1307, 1065, 907, 549, 1952, 1788, 1990]
r_mean = [np.mean(x) for x in rr]
r_std = [np.std(x) for x in rr]

colors = ['C0', 'C0', 'C0', 'C0', 'C1', 'C2', 'C3', 'C6']

for idx in range(len(list(r.keys()))):
    ax1.errorbar(num[idx], r_mean[idx], yerr=r_std[idx], fmt='o', capsize=5, color=colors[idx], label=methods[idx])

    method = methods[idx]
    if 'FIOLA' in method:
        dat = ttest_rel(r[method], r['Suite2p'], alternative='two-sided').pvalue 
        print(method)
        print(dat) 
        print(r[method])
        print(r['Suite2p'])
        barplot_annotate_brackets(dat, num[idx], num[5], 
                                  height = 0.003+ 0.005 * idx + np.max([max(r[method]), max(r['Suite2p'])]), 
                                  dy=0.003)

ax1.locator_params(axis='y', nbins=8)
ax1.locator_params(axis='x', nbins=4)
ax1.set_ylabel('Decoding R square')
ax1.set_xlabel('Number of neurons')
ax1.legend()

#plt.savefig(savef + 'Fig7c_pos_v3.7.pdf')
#plt.savefig(savef + 'Fig_supp_spd_v3.1.pdf')
# 0.17898542457533081
# 0.0055860117488064406
# 0.023733082648987967
# 0.0035924052250248485

#%% supp suite2p rigid
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff+'Fig7c_result_v3.7.npy', allow_pickle=True).item()
r1 = np.load(ff + 'Fig7_supp_s2p_rigid_result_v3.8.npy', allow_pickle=True).item()
fig = plt.figure() 
ax1 = plt.subplot()
rrr = {}
rrr['FIOLA3000'] = r['FIOLA3000']
rrr['Suite2p'] = r['Suite2p']
rrr['Suite2p_rigid'] = r1['Suite2p_rigid']
methods = list(rrr.keys())
rr = list(rrr.values())
num = [1307, 1788, 1856]
r_mean = [np.mean(x) for x in rr]
r_std = [np.std(x) for x in rr]

colors = ['C0', 'C2', 'C7']

for idx in range(len(list(rrr.keys()))):
    ax1.errorbar(num[idx], r_mean[idx], yerr=r_std[idx], fmt='o', capsize=5, color=colors[idx], label=methods[idx])

    # method = methods[idx]
    # if 'FIOLA' in method:
    #     dat = ttest_rel(r[method], r['Suite2p'], alternative='two-sided').pvalue 
    #     print(method)
    #     print(dat) 
    #     print(r[method])
    #     print(r['Suite2p'])
    #     barplot_annotate_brackets(dat, num[idx], num[5], 
    #                               height = 0.003+ 0.005 * idx + np.max([max(r[method]), max(r['Suite2p'])]), 
    #                               dy=0.003)
    method = methods[idx]
    
    dat = ttest_rel(rrr[method], rrr['Suite2p'], alternative='two-sided').pvalue 
    print(method)
    print(dat)
    
ax1.locator_params(axis='y', nbins=8)
ax1.locator_params(axis='x', nbins=4)
ax1.set_ylabel('Decoding R square')
ax1.set_xlabel('Number of neurons')
ax1.set_ylim([0.88, 0.95])
ax1.legend()
plt.savefig(savef + 'Fig7_supp_s2p_rigid_v3.8.pdf')

#%%
r = np.load(ff+'Fig7c_result_v3.7.npy', allow_pickle=True).item()
r1 = np.load(ff + 'Fig7_supp_dec_lag_v3.8.npy', allow_pickle=True).item()
for method in r1.keys():
    print(method)
    for compared in ['Suite2p', 'CaImAn', 'CaImAn_Online']:
        dat = ttest_rel(r1[method], r[compared], alternative='two-sided').pvalue 
        print(compared)
        print(dat) 
        
#%%
for method in r.keys():
    dat = ttest_rel(r[method], r['Suite2p'], alternative='two-sided').pvalue 
    print(method)
    print(dat) 
    print(np.array(r[method]).mean())
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% timing new
files = ['/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_2_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_result_init_frames_1500_iteration_2_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_result_init_frames_1500_iteration_1_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', '/media/nel/storage/fiola/R2_20190219/1000/fiola_result_init_frames_1000_iteration_2_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', '/media/nel/storage/fiola/R2_20190219/1000/fiola_result_init_frames_1000_iteration_1_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', '/media/nel/storage/fiola/R2_20190219/500/fiola_result_init_frames_500_iteration_2_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', '/media/nel/storage/fiola/R2_20190219/500/fiola_result_init_frames_500_iteration_1_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy']
for file in files:
    print(file)
    tt = np.load(file, allow_pickle=True).item().timing['all_online']
    print(np.mean(np.diff(tt)))

#%% drift analysis
import caiman as cm
import matplotlib.pyplot as plt
import numpy as np
# non-rigid
name = '/media/nel/storage/fiola/R2_20190219/full_nonrigid/mov_R2_20190219T210000._els__d1_796_d2_512_d3_1_order_F_frames_31933_.mmap'
m1 = cm.load(name, subindices=slice(0, 3000), in_memory=True)
m2 = cm.load(name, subindices=slice(28000, 31000), in_memory=True)
mm1 = m1.mean((1,2))
mm2 = m2.mean((1,2))

plt.plot(mm1); plt.plot(mm2)


#%%
mm3 = m1.mean((0))
mm4 = m2.mean((0))
plt.figure(); plt.imshow(mm3); 
plt.figure(); plt.imshow(mm4)

#%%
name = '/media/nel/storage/fiola/R2_20190219/full/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_.mmap'
m = cm.load(name, in_memory=True)
mm3 = m[:3000].mean((0))
mm4 = m[-3000:].mean((0))

#%%
mm3 = (mm3 - mm3.mean())/mm3.std()
mm4 = (mm4 - mm4.mean())/mm4.std()

#%%
plt.figure(); plt.imshow(mm3); 
plt.figure(); plt.imshow(mm4)
plt.imshow(mm3-mm4, cmap='bwr')

#%%
path = '/media/nel/storage/fiola/R2_20190219/memmap__d1_796_d2_512_d3_1_order_C_frames_31933__v3.7.hdf5'
cnm2 = load_CNMF(path)

#%%
files = ['/media/nel/storage/fiola/R2_20190219/3000/fiola_result_v3.5_init_frames_3000_iteration_5.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_v3.5_init_frames_3000_iteration_4.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_v3.5_init_frames_3000_iteration_3.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_v3.5_init_frames_3000_iteration_2.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_result_v3.5_init_frames_3000_iteration_1.npy'][::-1]

for idx, file in enumerate(files):
    a = np.load(file, allow_pickle=True).item().trace_deconvolved
    print(a.shape)
    #print(a.max())
    #plt.figure(); plt.plot(a[:30].T); plt.show()

#%% Decoding performance with different number of neurons
import numpy as np
from sklearn.model_selection import KFold
def cross_validation_ridge(X, y, n_splits=5, alpha=500):
    score = []
    kf = KFold(n_splits=5)
    for cv_train, cv_test in kf.split(X):
        x_train = X[cv_train]
        y_train = y[cv_train]
        x_test = X[cv_test]
        y_test = y[cv_test]
    
        clf = Ridge(alpha=alpha)
        clf.fit(x_train, y_train)  
        score.append(clf.score(x_test, y_test))
    return score

method = ['Suite2p', 'CaImAn', 'Fiola_3000', 'Fiola_1500','Fiola_1000', 'Fiola_500']
X_list = [t_g, tc_g, t1_g, t2_g, t3_g, t4_g]
# method = ['Fiola_500']
# X_list = [t1_g]

t_s = 3000
t_e = 31500
r = {}
dec = [pos_s, spd_s][0]
alpha_list = [100, 500, 1000, 5000, 10000]
for idx, X in enumerate(X_list):
    r[method[idx]] = []
    print(method[idx])
    for num in range(100, 2000, 200):
        print(num)        
        xx = X[t_s:t_e]
        y = dec[t_s:t_e]
        xx = xx[:, :num]
        if num > X.shape[1] + 200:
            r[method[idx]].append(r[method[idx]][-1])
        else:
            score_all = {}
            for alpha in alpha_list:
                print(f'{alpha:} alpha')
                score = cross_validation_ridge(xx, y, n_splits=5, alpha=alpha)
                score_all[alpha] = score
            #print(score_all)
            score_m = [np.mean(s) for s in score_all.values()]
            print(f'max score:{max(score_m)}')
            print(f'alpha:{alpha_list[np.argmax(score_m)]}')
            r[method[idx]].append(np.mean(score_all[alpha_list[np.argmax(score_m)]]))
        
#%%
xx = list(range(100, 2000, 200))
# plt.figure(); plt.plot(xx, r[method[0]]); plt.plot(xx, r[method[1]]); 
# plt.plot(xx, r[method[2]]); plt.plot(xx, r[method[3]])
# #plt.plot(xx, np.array(r) - np.array(r1)); 
# plt.legend(['Suite2p', 'Fiola', 'MeanROI','CaImAn', 'Difference Suite2p FIOLA'])
# plt.title('Performance vs different number of neurons')
fig = plt.figure() 
ax1 = plt.subplot()
r_all = list(r.values())
[ax1.plot(xx, x) for x in r_all]
# ax1.plot(xx, np.array(r_all[0]) - np.array(r_all[1])); 
ax1.legend(method)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('number of neurons')
ax1.set_ylabel('Decoding R^2')
#ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
#ax1.set_xticks([])
#ax1.set_yticks([])
ax1.set_ylim([0.5,0.9])

#np.save(savef + 'decoding_vs_num_neurons.npy', r)
plt.savefig(savef + 'decoding_vs_num_neurons.pdf')

#%%
r_all = []
for k_n in list(range(0, 2100, 200)):
    r_temp = []
    for it in range(5):
        trace = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries'].data[:]
        idx = np.random.permutation(list(range(trace.shape[1])))[:k_n]
        trace[:, idx] = 1
        trace_s = StandardScaler().fit_transform(trace)
        
        t_g = trace_s.copy()
        for i in range(t_g.shape[1]):
            t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)
        
        r = []
        train = [0, 10000]
        flag = 1
        test = [10000, 12000]
        for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
            #test = [10000, 12000]
            clf = Ridge(alpha=800)
            #clf = LinearRegression()
            #clf = Lasso(alpha=0.000001)
            clf.fit(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
            pos_pre_test = clf.predict(t_g[test[0]:test[1]])     
            pos_pre_train = clf.predict(t_g[train[0]:train[1]])     
            print(f'test: {clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
            print(f'train: {clf.score(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
            r.append(clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
            if flag == 0:
                flag = 1
                plt.figure()
                plt.plot(pos_s[train[0]:train[1]])
                plt.plot(pos_pre_train)
                plt.figure()
                plt.plot(pos_s[test[0]:test[1]])
                plt.plot(pos_pre_test)
        print(f'average test: {np.mean(r)}')
        r_temp.append(np.mean(r))
    r_all.append(r_temp)
    
rr = [np.mean(r) for r in r_all]
plt.plot(list(range(50, 2250, 200)), rr[::-1])
plt.hlines(np.mean(r1), xmin=0, xmax=2000, color='orange')
plt.legend(['suite2p number of remaining', 'fiola 749 neurons with bg'])
plt.ylabel('R2')
plt.xlabel('number of neurons')


#%%
corr = []
for idx in range(len(tracem.T)):
    corr.append(np.corrcoef(tracem_s[:-1, idx], pos_n)[0, 1])



#%%
coef = clf.coef_.copy()
plt.plot(clf.coef_)
#idx = np.argsort(clf.coef_)[::-1][:40]
#idx = np.argsort(coef)[::-1][:300]
#idx1 = np.argsort(coef)[:300]
#idx = np.concatenate([idx, idx1])

trace.shape
idx = np.random.permutation(list(range(trace.shape[1])))[:1300]
trace[:, idx] = 1
plt.plot(t_g[:, idx])
plt.plot(pos_s)
plt.plot(trace_s[:, idx])
plt.plot(trace0[:, idx])


#%%
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g[train[0]:train[1]], spd_s[train[0]:train[1]])  
spd_pre_test = clf.predict(t_g[test[0]:test[1]])     
spd_pre_train = clf.predict(t_g[train[0]:train[1]])     
print(f'test: {clf.score(t_g[test[0]:test[1]], spd_s[test[0]:test[1]])}')
print(f'train: {clf.score(t_g[train[0]:train[1]], spd_s[train[0]:train[1]])}')
plt.figure()
plt.plot(spd_s[train[0]:train[1]])
plt.plot(spd_pre_train)
plt.figure()
plt.plot(spd_s[test[0]:test[1]])
plt.plot(spd_pre_test)




#%%
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t1_g[train[0]:train[1]], spd_s[train[0]:train[1]])  
spd_pre_test = clf.predict(t1_g[test[0]:test[1]])     
spd_pre_train = clf.predict(t1_g[train[0]:train[1]])     
print(f'test: {clf.score(t1_g[test[0]:test[1]], spd_s[test[0]:test[1]])}')
print(f'train: {clf.score(t1_g[train[0]:train[1]], spd_s[train[0]:train[1]])}')
plt.figure()
plt.plot(spd_s[train[0]:train[1]])
plt.plot(spd_pre_train)
plt.figure()
plt.plot(spd_s[test[0]:test[1]])
plt.plot(spd_pre_test)

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(trace_s, pos_s)  
pos_pre = clf.predict(trace_s)     
clf.score(trace_s, pos_s)

#%%
plt.figure()
plt.plot(pos_s) 
plt.plot(pos_pre, alpha=0.9)        
plt.title('pos')
plt.legend(['pos', 'fit'])

#%%
plt.plot(pos_n) 
plt.plot(spd_s)

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(trace_s, spd_s)  
spd_pre = clf.predict(trace_s)     
clf.score(trace_s, spd_s)

#%%
plt.figure()
plt.plot(spd_s) 
plt.plot(spd_pre, alpha=0.9)        
plt.title('spd')
plt.legend(['spd', 'fit'])

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(trace_s[2500:12500], pos_s[2500:12500])  
pos_pre = clf.predict(trace_s[2500:12500])     
clf.score(trace_s[2500:12500], pos_s[2500:12500])

#%%
plt.plot(pos_s[2500:12500])
plt.plot(pos_pre)

#%%
from scipy.ndimage import gaussian_filter1d
t_g = trace_s.copy()
for i in range(t_g.shape[1]):
    t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)

#%%
plt.plot(trace_s[2500:12500, 0])
plt.plot(t_g[:, 0])

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g, pos_s[2500:12500])  
pos_pre = clf.predict(t_g)     
clf.score(t_g, pos_s[2500:12500])

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g, pos_s)  
pos_pre = clf.predict(t_g)     
clf.score(t_g, pos_s)


#%%
plt.plot(pos_s)
plt.plot(pos_pre)
# #%%
# trace1_s = trace1.copy()
# for idx in range(len(trace1.T)):
#     t = trace1[:, idx]
#     window = 3000
#     weights = np.repeat(1.0, window)/window
#     t_pad = np.concatenate([t[:1500][::-1], t, t[-1499:]])
#     ma = np.convolve(t_pad, weights, 'valid')
#     # plt.plot(t)
#     # plt.plot(ma)
#     t = t - ma
#     trace1_s[:, idx] = t
#     if idx % 100 == 0:
#         print(idx)
        
# trace1_s = StandardScaler().fit_transform(trace1_s)

#%%
dims = (796, 512)
li = []
li_matrix = np.zeros((len(range(0, dims[0], 30)), len(range(0, dims[1], 30))))
for i in range(0, dims[0], 30):
    for j in range(0, dims[1], 30):
        num = 0
        for idx in range(centers.shape[0]):
            if centers[idx, 1] >= i and centers[idx, 1] < i + 30:
                if centers[idx, 0] >= j and centers[idx, 0] < j + 30:
                   num = num + 1
        li.append(num)
        li_matrix[i//30, j//30] = num
#li_matrix[li_matrix<] = 0
plt.imshow(li_matrix); plt.colorbar
#plt.hist(li_matrix.flatten())#; plt.colorbar()

#%%
from caiman.base.rois import com
centers_cm = com(cnm2.estimates.A[:, cnm2.estimates.idx_components], 796, 512)
centers_cm = centers_cm[:, np.array([1, 0])]
centers = centers_cm.copy()
#%%
dims = (796, 512)
A = cnm2.estimates.A.toarray()
A = A.reshape([dims[0], dims[1], -1], order='F')
A = A.transpose([2, 0, 1])

aa = np.load('/media/nel/storage/fiola/R2_20190219/3000/caiman_masks_3000_749.npy', allow_pickle=True)

#%%
cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/7000/memmap__d1_796_d2_512_d3_1_order_C_frames_7000_.hdf5')
A = cnm2.estimates.A.toarray()
A = A.reshape([dims[0], dims[1], -1], order='F')
A = A.transpose([2, 0, 1])

#%%
r = []
train = [0, 10000]
flag = 1
test = [10000, 30000]
#test = [10000, 12000]
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
pos_pre_test = clf.predict(t_g[test[0]:test[1]])     
pos_pre_train = clf.predict(t_g[train[0]:train[1]])     
print(f'test: {clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
print(f'train: {clf.score(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
r.append(clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
if flag == 0:
    #flag = 1
    # plt.figure()
    # plt.plot(pos_s[train[0]:train[1]])
    # plt.plot(pos_pre_train)
    plt.figure()
    plt.plot(pos_s[test[0]:test[1]])
    plt.plot(pos_pre_test)
print(f'average test: {np.mean(r)}')

r1 = []
flag = 1
#test = [10000, 12000]
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
pos_pre_test = clf.predict(t1_g[test[0]:test[1]])     
pos_pre_train = clf.predict(t1_g[train[0]:train[1]])     
print(f'test: {clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
print(f'train: {clf.score(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
r1.append(clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
if flag == 0:
    #flag = 1
    #plt.figure()
    # plt.plot(pos_s[train[0]:train[1]])
    # plt.plot(pos_pre_train)
    #plt.figure()
    #plt.plot(pos_s[test[0]:test[1]])
    plt.plot(pos_pre_test)
print(f'average test: {np.mean(r1)}')


rc = []
flag = 1
#test = [10000, 12000]
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
pos_pre_test = clf.predict(tc_g[test[0]:test[1]])     
pos_pre_train = clf.predict(tc_g[train[0]:train[1]])     
print(f'test: {clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
print(f'train: {clf.score(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
rc.append(clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
if flag == 0:
    #flag = 1
    #plt.figure()
    # plt.plot(pos_s[train[0]:train[1]])
    # plt.plot(pos_pre_train)
    #plt.figure()
    #plt.plot(pos_s[test[0]:test[1]])
    plt.plot(pos_pre_test)
print(f'average test: {np.mean(rc)}')

#%%

trace_s = StandardScaler().fit_transform(trace)
tracem_s = StandardScaler().fit_transform(tracem)
pos_s = StandardScaler().fit_transform(pos_n[:, None])[:, 0]
spd_s = StandardScaler().fit_transform(speed[:, None])[:, 0]
trace1_s = StandardScaler().fit_transform(trace1)
trace1_s = trace1_s / trace1_s.max(0)

#%%
tracec_s = StandardScaler().fit_transform(tracec)

#%%
#tracem_s = tracem_s / tracem_s.max(0)
# tracem_s= tracem_s - tracem_s.min(0)
# tracem_s = tracem_s / tracem_s.max(0)
# tracem_s = tracem_s/tracem_s.max(0)
# for i in range(tracem_s.shape[1]):
#     tracem_s[:, i] = gaussian_filter1d(tracem_s[:, i], sigma=5)
ii = 101
plt.plot(trace_s[:, ii])
print((trace_s[:3000, ii] > 5).sum())
#%%
n = []
for ii in range(trace_s.shape[1]):
    n.append((trace_s[:3000, ii] > 5).sum())
    
n = np.array(n)
(n > 5).sum()
select = np.where(n>5)[0]
#%%
# from caiman.source_extraction.cnmf.temporal import constrained_foopsi
# dec = []
# for idx in range(len(tracem_s.T)):
#     c_full, bl, c1, g, sn, s_full, lam = constrained_foopsi(tracem_s[:, idx], p=1, s_min=0.1)
#     dec.append(s_full)
#     if idx in [125, 130, 135, 140, 145, 150]:
#         plt.title(f'neuron {idx}'); plt.plot(tracem_s[:, idx]); plt.plot(c_full-3); 
#         plt.plot(s_full-6, c='black'); plt.legend(['calcium', 'fit', 'dec']); plt.show()
# trace2 = np.array(dec).T

# trace2 = np.array(dec).T
trace2 = trace1_s.copy()
trace2_s = StandardScaler().fit_transform(trace2)
t1_g = trace2_s.copy()

from caiman.source_extraction.volpy.spikepursuit import signal_filter
tt = signal_filter(trace1, freq=1, fr=15)

# plt.plot(trace1[:, 30]); plt.plot(tt[:, 30])
plt.plot(trace1[:, :].sum(1)); plt.plot(tt[:, :].sum(1))
trace1 = tt
#trace1 = trace1[1:]#%%
r = []
train = [0, 10000]
flag = 0
#test = [10000, 12000]
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    #test = [10000, 12000]
    clf = Ridge(alpha=800)
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.000001)
    clf.fit(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(t_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(t_g[train[0]:train[1]])     
    print(f'test: {clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    r.append(clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        flag = 1
        plt.figure()
        plt.title(f'{test}')
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(r)}')
    
#%%
r1 = []
flag = 0
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    clf = Ridge(alpha=800)
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.000001)
    clf.fit(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(t1_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(t1_g[train[0]:train[1]])     
    print(f'test: {clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    r1.append(clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        #flag = 1
        plt.figure()
        plt.title(f'{test}')
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(r1)}')

#%%
rm = []
flag = 0
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    clf = Ridge(alpha=800)
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.000001)
    clf.fit(tm_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(tm_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(tm_g[train[0]:train[1]])     
    print(f'test: {clf.score(tm_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(tm_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    rm.append(clf.score(tm_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        flag = 1
        plt.figure()
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(rm)}')

#%%
rc = []
flag = 0
train = [0, 10000]
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    clf = Ridge(alpha=800)    
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.05)
    clf.fit(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(tc_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(tc_g[train[0]:train[1]])     
    print(f'test: {clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    rc.append(clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        #flag = 1
        plt.figure()
        plt.title(f'{test}')
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(rc)}')

#%%
rc = []
flag = 0
sets = [[0, 10000], [10000, 20000], [20000, 30000]]
for train in sets:
    for test in [[i, i+10000] for i in range(0, 30000, 10000)]:
        clf = Ridge(alpha=800)    
        #clf = LinearRegression()
        #clf = Lasso(alpha=0.05)
        clf.fit(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
        pos_pre_test = clf.predict(tc_g[test[0]:test[1]])     
        pos_pre_train = clf.predict(tc_g[train[0]:train[1]])     
        print(f'test: {clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
        print(f'train: {clf.score(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
        rc.append(clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
        if flag == 0:
            #flag = 1
            plt.figure()
            plt.title(f'{test}')
            plt.plot(pos_s[train[0]:train[1]])
            plt.plot(pos_pre_train)
            plt.figure()
            plt.plot(pos_s[test[0]:test[1]])
            plt.plot(pos_pre_test)
    print(f'average test: {np.mean(rc)}')
    
    
#%%
#%%
import numpy as np
from sklearn.model_selection import KFold
def cross_validation_ridge(X, y, n_splits=5, alpha=500):
    score = []
    kf = KFold(n_splits=5)
    for cv_train, cv_test in kf.split(X):
        x_train = X[cv_train]
        y_train = y[cv_train]
        x_test = X[cv_test]
        y_test = y[cv_test]
    
        clf = Ridge(alpha=alpha)
        clf.fit(x_train, y_train)  
        score.append(clf.score(x_test, y_test))
    return score

#X = t_g[:30000]
method = ['Suite2p', 'Fiola', 'MeanROI', 'CaImAn'][0:2]
#X_list = [t_g, t1_g, tm_g, tc_g][0:3]
#X_list = [t_g, t1_g, tm_g][0:3]
X_list = [t_g, t1_g]
r = {}
alpha_list = [500, 1000, 5000]
for idx, X in enumerate(X_list):
    r[method[idx]] = []
    print(method[idx])
    for num in range(100, 2800, 400):
        print(num)        
        X = X[:30000]
        y = pos_s[:30000]
        xx = X[:, :num]
        score_all = {}
        for alpha in alpha_list:
            print(f'{alpha:} alpha')
            score = cross_validation_ridge(xx, y, n_splits=5, alpha=alpha)
            score_all[alpha] = score
        #print(score_all)
        score_m = [np.mean(s) for s in score_all.values()]
        print(f'max score:{max(score_m)}')
        print(f'alpha:{alpha_list[np.argmax(score_m)]}')
        r[method[idx]].append(np.mean(score_all[alpha_list[np.argmax(score_m)]]))
        
#%%
xx = list(range(100, 2200, 200))
plt.figure(); plt.plot(xx, r[method[0]]); plt.plot(xx, r[method[1]]); 
plt.plot(xx, r[method[2]]); plt.plot(xx, r[method[3]])
#plt.plot(xx, np.array(r) - np.array(r1)); 
plt.legend(['Suite2p', 'Fiola', 'MeanROI','CaImAn', 'Difference Suite2p FIOLA'])
plt.title('Performance vs different number of neurons')

        
#%%
for key in r.keys():
    print(np.mean(r[key]))

#%%
def rand_jitter(arr):
    stdev = .01 #* (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), y, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)
for i in range(4):
    jitter([i]*5, r[method[i]])
    
plt.ylim([0, 1])
plt.xticks(list(range(4)), method)
plt.title('R^2 for prediction with cross validation')

#plt.scatter([1, 1], [1, 2])#%%
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
ax1.bar([0, 1, 2, 3], [np.mean(r), np.mean(r1), np.mean(rm), np.mean(rc)], 
        yerr=[np.std(r), np.std(r1), np.std(rm), np.std(rc)], label=['Suite2p', 'Fiola', 'MeanROI', 'CaImAn'])
ax1.legend()
#ax1.bar(0, np.mean(r), yerr=np.std(r))
#ax1.bar(0, np.mean(r1), yerr=np.std(r1))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Average')
ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([])
#ax1.set_yticks([])
ax1.set_ylim([0,1])

print(np.mean(r))
print(np.mean(r1))
print(np.mean(rm))
print(np.mean(rc))# T = [trace, trace1, trace2, trace3, trace4, tracec, traceo]
# T = [fast_prct_filt(t,20).T for t in T]
# T = [remove_neurons_with_sparse_activities(t) for t in T]
# T = for i in range(t_g.shape[1]):
#     t_g[:, i] = gaussian_filter1d(t[:, i], sigma=5) for ] for t in T
# trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_v3.1.npy')
# trace1 = trace1.T
# #trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_10_False.npy')
# #trace1 = trace1.T
# trace2 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_result_v3.1.npy')
# trace2 = trace2.T
# trace3 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_result_v3.1.npy')
# trace3 = trace3.T
# trace4 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_result_v3.1.npy')
# trace4 = trace4.T

# trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_30.npy')
# trace1 = trace1.T
# trace2 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_10_False.npy')
# trace2 = trace2.T
# trace3 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_1_False.npy')
# trace3 = trace3.T
# trace4 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_1.npy')
# trace4 = trace4.T



#tracem = np.load('/media/nel/storage/fiola/R2_20190219/3000/mean_roi_99.98.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_all_masks.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/3000/mean_roi_99.98_non_rigid.npy')
#tracem = tracem.T
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks_selected_1543.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks_selected_2365.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks_selected_1285.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/3000/mean_roi_99.98_non_rigid_init_non_rigid_1285.npy')
#tracem = np.load('/media/nel/storage/fiola/R6_20200210T2100/3000/mean_roi_99.98_non_rigid_init.npy')
#tracem = np.load('/media/nel/storage/fiola/F2_20190415/3000/mean_roi_99.98_non_rigid_init.npy')
#tracem = tracem.T
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_30000.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_3000_nonrigid.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_30000_K_8.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_1285_nonrigid.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_1285_non_rigid_init_non_rigid_movie.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_1285_non_rigid_init_non_rigid_movie_nnls_only.npy')
#trace1 = np.load('/media/nel/storage/fiola/R6_20200210T2100/3000/fiola_non_rigid_init.npy')
#trace1 = np.load('/media/nel/storage/fiola/F2_20190415/3000/fiola_non_rigid_init.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_do_hals.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_do_hals_new.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/mov_R2_20190219T210000._fiola_result_new.npy')
#%%
[print(t.shape) for t in [trace, trace1, trace2, trace3, trace4, tracec, traceo]]
trace = fast_prct_filt(trace,20).T
trace1 = fast_prct_filt(trace1,20).T
trace2 = fast_prct_filt(trace2,20).T
trace3 = fast_prct_filt(trace3,20).T
trace4 = fast_prct_filt(trace4,20).T
tracec = fast_prct_filt(tracec,20).T
traceo = fast_prct_filt(traceo,20).T

# remove atificial feature for onacid due to filtering
for idx in range(traceo.shape[1]):
    if onacid.time_neuron_added[idx, 1] > 200:
        traceo[:onacid.time_neuron_added[idx, 1]-100, idx] = np.mean(traceo[onacid.time_neuron_added[idx, 1]-100:, idx])

# trace1 = trace1.T
# traceo = traceo.T

#%%
[print(t.shape) for t in [trace, trace1, trace2, trace3, trace4, tracec, traceo]]
trace_s, _ = remove_neurons_with_sparse_activities(trace)
trace1_s, _ = remove_neurons_with_sparse_activities(trace1)
trace2_s, _ = remove_neurons_with_sparse_activities(trace2)
trace3_s, _ = remove_neurons_with_sparse_activities(trace3)
trace4_s, _ = remove_neurons_with_sparse_activities(trace4)
tracec_s, _ = remove_neurons_with_sparse_activities(tracec)
traceo_s, _ = remove_neurons_with_sparse_activities(traceo)

#%%
first_act = []
std_level = 5
for idx in range(len(traceo_s.T)):
    t = traceo_s[:, idx]
    first_act.append(np.where(t>t.std() * std_level)[0][0])

select = np.where(np.array(first_act)<10000)[0]
traceo_s = traceo_s[:, select]
#%%
t_g = trace_s.copy()
for i in range(t_g.shape[1]):
    t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)
t1_g = trace1_s.copy()
for i in range(t1_g.shape[1]):
    t1_g[:, i] = gaussian_filter1d(t1_g[:, i], sigma=5)
t2_g = trace2_s.copy()
for i in range(t2_g.shape[1]):
    t2_g[:, i] = gaussian_filter1d(t2_g[:, i], sigma=5)
t3_g = trace3_s.copy()
for i in range(t3_g.shape[1]):
    t3_g[:, i] = gaussian_filter1d(t3_g[:, i], sigma=5)    
t4_g = trace4_s.copy()
for i in range(t4_g.shape[1]):
    t4_g[:, i] = gaussian_filter1d(t4_g[:, i], sigma=5)
tc_g = tracec_s.copy()
for i in range(tc_g.shape[1]):
    tc_g[:, i] = gaussian_filter1d(tc_g[:, i], sigma=5)
to_g = traceo_s.copy()
for i in range(to_g.shape[1]):
    to_g[:, i] = gaussian_filter1d(to_g[:, i], sigma=5)
[print(t.shape) for t in [t_g, t1_g, t2_g, t3_g, t4_g, tc_g, to_g]]    
# tm_g = tracem_s.copy()
# for i in range(tm_g.shape[1]):
#     tm_g[:, i] = gaussian_filter1d(tm_g[:, i], sigma=5)

#%%
r_onacid = {}
for tt in [500, 1000, 1500, 3000, 5000, 8000, 10000, 15000, 20000, 25000, 30000]:
    traceo_s, indexes = remove_neurons_with_sparse_activities(traceo)
    first_act = []
    std_level = 5
    for idx in range(len(traceo_s.T)):
        t = traceo_s[:, idx]
        first_act.append(np.where(t>t.std() * std_level)[0][0])
        
    #plt.plot(first_act)
    select = np.where(np.array(first_act)<tt)[0]
    #select = (onacid.time_neuron_added[:, 1] < tt)[np.array(indexes)]
    traceo_s = traceo_s[:, select]
    
    to_g = traceo_s.copy()
    for i in range(to_g.shape[1]):
        to_g[:, i] = gaussian_filter1d(to_g[:, i], sigma=5)
    
    dec = [pos_s, spd_s][0]
    
    X = to_g; idx=1
    print(method[idx])
    X = X[t_s:t_e]
    y = dec[t_s:t_e]
    score_all = {}
    for alpha in alpha_list:
        print(alpha)
        score = cross_validation_ridge(X, y, n_splits=5, alpha=alpha)
        score_all[alpha] = score
    #print(score_all)
    score_m = [np.mean(s) for s in score_all.values()]
    print(f'max score:{max(score_m)}')
    print(f'alpha:{alpha_list[np.argmax(score_m)]}')
    
    r_onacid[tt] = [to_g.shape[1], score_m]
#%%
# rr = list(r.values())
# fig = plt.figure(figsize=(8, 6)) 
# ax1 = plt.subplot()
# ax1.bar(list(range(len(method))), [np.mean(x) for x in rr], yerr=[np.std(x) for x in rr])
# ax1.legend()
# #ax1.bar(0, np.mean(r), yerr=np.std(r))
# #ax1.bar(0, np.mean(r1), yerr=np.std(r1))
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# #ax1.spines['left'].set_visible(False)
# ax1.set_xlabel('Method')
# ax1.set_ylabel('Decoding R^2')

# ax1.xaxis.set_ticks_position('none') 
# #ax1.yaxis.set_ticks_position('none') 
# #ax1.set_xticks([0, 1, 2, 3])
# ax1.set_xticklabels(method)
# ax1.set_xticks(list(range(len(method))))
# #ax1.set_yticks([])
# ax1.set_ylim([0,1])
#plt.savefig(savef + 'decoding_cross_validation.pdf')
#%% number of neurons detected
fig = plt.figure() 
ax1 = plt.subplot()
ax1.bar(t_g.keys(), [tt.shape[1] for tt in t_g.values()])
ax1.set_xlabel('Method')
ax1.set_ylabel('Number of neurons detected')

ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
#ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(method)
ax1.set_xticks(list(range(len(method))))
#ax1.set_yticks([])
#%%
rrr = list(r_onacid.values())
fig = plt.figure() 
ax1 = plt.subplot()
xx = [rr[0] for rr in rrr]
yy = [np.max(rr[1]) for rr in rrr]
ax1.scatter(xx, yy)
for i in range(len(xx)):
    ax1.annotate(list(r_onacid.keys())[i], (xx[i] + 10, yy[i] - 0.03))

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.locator_params(axis='y', nbins=8)
ax1.locator_params(axis='x', nbins=4)
#ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_ylabel('Decoding R square')
ax1.set_xlabel('Number of neurons selected')
ax1.set_ylim([0.55, 0.95])

plt.savefig(savef + 'Fig_supp_caiman_online_pos_v3.1.pdf')

#%% Wilcoxon test
from scipy.stats import wilcoxon
wilcoxon(rr[0], rr[6], alternative='greater')

#%% timing for init + online
data = np.array([t_3000, t_1500, t_1000, t_500])
data = data/60
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
ax1.bar(range(4), data[:, 0], label='init')
ax1.bar(range(4), data[:, 1], bottom=data[:, 0], label='online')
ax1.legend()
ax1.set_xlabel('method')
ax1.set_ylabel('time (mins)')
#ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(['Fiola 3000', 'Fiola 1500', 'Fiola 1000', 'Fiola 500' ])
#ax1.set_yticks([])
#ax1.set_ylim([0,1])
plt.savefig(savef + 'init_time_and_online.pdf')

#%%
#%% timing
t_3000 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_timing.npy', allow_pickle=True)
t_1500 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_timing.npy', allow_pickle=True)
t_1000 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_timing.npy', allow_pickle=True)
t_500 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_timing.npy', allow_pickle=True)

t_3000 = np.diff(list(t_3000.item().values()))
t_1500 = np.diff(list(t_1500.item().values()))
t_1000 = np.diff(list(t_1000.item().values()))
t_500 = np.diff(list(t_500.item().values()))

t_500 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_timing_v3.4_num_layers_10_trace_with_neg_False_with_dec_1_init_frames_500.npy', allow_pickle=True)
t_1000 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_timing_v3.4_num_layers_10_trace_with_neg_False_with_dec_1_init_frames_1000.npy', allow_pickle=True)
t_1500 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.4_num_layers_10_trace_with_neg_False_with_dec_1_init_frames_1500.npy', allow_pickle=True)
t_3000 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.4_num_layers_10_trace_with_neg_False_with_dec_2.npy', allow_pickle=True)

for tt in [t_500, t_1000, t_1500, t_3000]:
    ttt = list(tt.item().values())
    print(ttt[1]-ttt[0])
    print(ttt[2]-ttt[1])


t_500 = list(t_500.item().values())
t_500 = t_500[2] - t_500[0]
t_1000 = list(t_1000.item().values())
t_1000 = t_1000[2] - t_1000[0]
t_1500 = list(t_1500.item().values())
t_1500 = t_1500[2] - t_1500[0]
t_3000 = list(t_3000.item().values())
t_3000 = t_3000[2] - t_3000[0]

#%%
# files = ["/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.3_num_layers_10_trace_with_neg_False_with_dec_5.npy","/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.3_num_layers_10_trace_with_neg_False_with_dec_4.npy","/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.3_num_layers_10_trace_with_neg_False_with_dec_3.npy","/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.3_num_layers_10_trace_with_neg_False_with_dec_2.npy","/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.3_num_layers_10_trace_with_neg_False_with_dec_1.npy"]
# files = ['/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.2_num_layers_10_trace_with_neg_False_with_dec_2.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.2_num_layers_10_trace_with_neg_False_with_dec_1.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.2_num_layers_10_trace_with_neg_False_with_dec_0.npy']
# files = ['/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.4_num_layers_10_trace_with_neg_False_with_dec_5.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.4_num_layers_10_trace_with_neg_False_with_dec_4.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.4_num_layers_10_trace_with_neg_False_with_dec_3.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.4_num_layers_10_trace_with_neg_False_with_dec_2.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.4_num_layers_10_trace_with_neg_False_with_dec_1.npy']
import numpy as np
ttt = []
#fig, ax = plt.subplots(5, 1, sharey=True)



files = ['/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_5_init_frames_3000_iteration_5.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_4_init_frames_3000_iteration_4.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_3_init_frames_3000_iteration_3.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_2_init_frames_3000_iteration_2.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_1_init_frames_3000_iteration_1.npy'][::-1]
#files = ['/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_1_init_frames_1500_iteration_1_no_init.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_2_init_frames_1500_iteration_2_no_init.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_3_init_frames_1500_iteration_3_no_init.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_4_init_frames_1500_iteration_4_no_init.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_5_init_frames_1500_iteration_5_no_init.npy']
#files = ['/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.6_num_layers_10_trace_with_neg_False_with_dec_1_init_frames_1500_iteration_1_no_init.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.6_num_layers_10_trace_with_neg_False_with_dec_2_init_frames_1500_iteration_2_no_init.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.6_num_layers_10_trace_with_neg_False_with_dec_3_init_frames_1500_iteration_3_no_init.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.6_num_layers_10_trace_with_neg_False_with_dec_4_init_frames_1500_iteration_4_no_init.npy', '/media/nel/storage/fiola/R2_20190219/1500/fiola_timing_v3.6_num_layers_10_trace_with_neg_False_with_dec_5_init_frames_1500_iteration_5_no_init.npy']
#files = ['/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_5_init_frames_3000_iteration_5_no_init.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_4_init_frames_3000_iteration_4_no_init.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_3_init_frames_3000_iteration_3_no_init.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_2_init_frames_3000_iteration_2_no_init.npy', '/media/nel/storage/fiola/R2_20190219/3000/fiola_timing_v3.5_num_layers_10_trace_with_neg_False_with_dec_1_init_frames_3000_iteration_1_no_init.npy']
for idx, file in enumerate(files):

    mm = list(np.load(file, allow_pickle=True).item().values())
    #print(mm)
    print(np.diff(mm[1:3]))        
    #print(np.diff(mm[0:2]))        
    #print(np.mean(np.diff(mm[3])))
    #print(np.max(np.diff(mm[3])))
    #print(np.std(np.diff(mm[3])))
    plt.figure()
    plt.plot(np.diff(mm[3]), '.'); plt.title(f'trial {idx}')
    plt.show()
    #ax[idx].plot(np.diff(mm[3]), '.')
    #ttt.append(np.diff(mm[:2])[0])
    #import matplotlib.pyplot as plt
    #plt.plot(np.diff(mm[3]), '.')
#plt.tight_layout()tt = []
num_neurons = []
fig, ax = plt.subplots(3, 4)
for idx, num_frames in enumerate([500, 1000, 1500, 3000]):
    for i in [1, 2, 3]:
        file = f'/media/nel/storage/fiola/R2_20190219/{num_frames}/fiola_result_init_frames_{num_frames}_iteration_{i}_no_init_v3.9.npy'
        r = np.load(file, allow_pickle=True).item()
        tt.append(np.mean(np.diff(r.timing['all_online'])))
        ax[i-1, idx].plot(np.diff(r.timing['all_online']), '.')
        ax[i-1, idx].set_title(f'num_frames_{num_frames}_iteration_{i}')
        # plt.figure(); 
        # plt.plot(np.diff(r.timing['all_online']))
        # plt.title(f'num_frames_{num_frames}_iteration_{i}')
        # plt.show()
    num_neurons.append(r.trace.shape[0])
plt.tight_layout()


def run_fiola(mov, cnm, select, fnames='/'):
    # load movie and masks
    mode_idx = 1
    mode = ['voltage', 'calcium'][mode_idx]
    mask = np.hstack([cnm.estimates.A[:, select].toarray(), cnm.estimates.b])    
    #mask = np.hstack([cnm.estimates.A[:, cnm.estimates.idx_components].toarray(), cnm.estimates.b])    
    #mask = mask.reshape([dims[0], dims[1], -1], order='C').transpose([2, 0, 1])
    
    mask = mask.reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1])
    Ab = to_2D(mask, order='F').T
    Ab = Ab / norm(Ab, axis=0)

    options = load_fiola_config(fnames, num_frames_total=30000, mode=mode, mask=mask) 
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    fio.Ab = Ab        

    min_mov = mov[:3000].min()
    template = bin_median(mov[:3000])
    plt.imshow(template)
    trace = fio.fit_gpu_motion_correction_nnls(mov[:30000], template=template, batch_size=5, 
                                min_mov=min_mov, Ab=fio.Ab)
    #trace = fio.fit_gpu_nnls(mov[:31930], fio.Ab, batch_size=5) 
    plt.plot(trace[:-2].T)
    return trace    

# plt.figure(); 
# plt.plot(pos)
# plt.title('pos')
# plt.plot(pos_n) 
# plt.figure()
# plt.plot(speed)
# plt.title('spd')
# 
# mov_100 = nwbfile_in.acquisition['TwoPhotonSeries'].data[:1000]
# #m1 = mov_100.mo
# mov_l_100 = nwbfile_in.acquisition['TwoPhotonSeries'].data[-5000:-4000]
# me = np.median(mov_100, axis=0)
# me1 = np.median(mov_l_100, axis=0)
# mm = cm.concatenate([cm.movie(me)[None, :, :], cm.movie(me1)[None, :, :]])
# mov = cm.movie(mm)
#mov = nwbfile_in.acquisition['TwoPhotonSeries'].data[:]
#cm.movie(mov).save('/media/nel/storage/fiola/R6_20200210T2100/mov_R6__20200210T2100.hdf5')
#mov.save('/media/nel/storage/fiola/mov_R2_20190219T210000.hdf5')
# mov[:1000].save('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_1000.hdf5')
# mov[:5000].save('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_5000.hdf5')
# mov[:7000].save('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_7000.hdf5')

# mov.save('/media/nel/storage/fiola/mov_R2_20190219T210000.hdf5')

# masks = nwbfile_in.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].columns[0][:]
# centers = nwbfile_in.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].columns[1][:]
# #accepted = nwbfile_in.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].columns[2][:]
# plt.imshow(masks.sum(0))
# plt.scatter(centers[:, 0], centers[:, 1],  s=0.5, color='red')

#path = '/media/nel/storage/fiola/sub-R2_ses-20190206T210000_obj-16d9pmi_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/R2_20190219/sub-R2_ses-20190219T210000_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/sub-R6_ses-20200210T210000_obj-10qdhbr_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/sub-R6_ses-20200206T210000_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/R4_20190904/sub-R4_ses-20190904T210000_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/F2_20190415/sub-F2_ses-20190415T210000_behavior+ophys.nwb'
# mean roi


#%% init
fnames_init = os.path.join(base_folder, '5000', 'mov_raw_5000.hdf5')
fnames_all = os.path.join(base_folder, 'mov_raw.hdf5')
run_caiman_fig7(fnames_init, pw_rigid=True, motion_correction_only=False, K=5)
run_caiman_fig7(fnames_all, pw_rigid=False, motion_correction_only=True)
run_caiman_fig7(fnames_all, pw_rigid=True, motion_correction_only=False, K=8)


t_test = t_g.copy()
#t_test = {'Suite2p_rigid': t_g['Suite2p_rigid']}
#t_test = {'FIOLA_10': t_g['FIOLA_10']}
t_s = 3000
t_e = 31500
dec = [pos_s, spd_s][0].copy()

r = {}
alpha_list = [100, 500, 1000, 5000, 10000]
for key, X in t_test.items():
    print(key)
    s_tt = s_t[t_s:t_e]
    X = X[t_s:t_e][s_tt]
    y = dec[t_s:t_e][s_tt]
    score_all = {}
    for alpha in alpha_list:
        print(alpha)
        score = cross_validation_ridge(X, y, n_splits=5, alpha=alpha)
        score_all[alpha] = score
    #print(score_all)
    score_m = [np.mean(s) for s in score_all.values()]
    print(f'max score:{max(score_m)}')
    print(f'alpha:{alpha_list[np.argmax(score_m)]}')
    r[key] = score_all[alpha_list[np.argmax(score_m)]]
#%%
def cross_validation_ridge(X, y, n_splits=5, alpha=500):
    score = []
    kf = KFold(n_splits=5)
    for cv_train, cv_test in kf.split(X):
        x_train = X[cv_train]
        y_train = y[cv_train]
        x_test = X[cv_test]
        y_test = y[cv_test]
        clf = Ridge(alpha=alpha)
        clf.fit(x_train, y_train)  

        # if alpha == 5000:
        #     import pdb
        #     pdb.set_trace()
        #     print(cv_train)
        #     print(cv_test)
            
        #     plt.figure()
        #     plt.plot(y[cv_train])
        #     plt.plot(clf.predict(X[cv_train]))
        #     plt.title(f'{cv_train[0]}, {clf.score(x_train, x_train)}')
        #     plt.show()
            
        #     plt.figure()
        #     plt.plot(y[cv_test])
        #     plt.plot(clf.predict(X[cv_test]))
        #     plt.title(f'{cv_test[0]}, {clf.score(x_test, y_test)}')
        #     plt.show()
        score.append(clf.score(x_test, y_test))
    return score
#%%
for idx, X in t_test.items():
    score_all = {}
    for alpha in alpha_list:
        print(alpha)
        r = []
        for test in [[i, i+2000] for i in range(13000, 31000, 2000)]:
            #test = [10000, 12000]
            clf = Ridge(alpha=alpha)
            #clf = LinearRegression()
            #clf = Lasso(alpha=0.000001)
            clf.fit(X[train[0]:train[1]], dec[train[0]:train[1]])  
            pos_pre_test = clf.predict(X[test[0]:test[1]])     
            pos_pre_train = clf.predict(X[train[0]:train[1]])     
            #print(f'test: {clf.score(X[test[0]:test[1]], dec[test[0]:test[1]])}')
            #print(f'train: {clf.score(X[train[0]:train[1]], dec[train[0]:train[1]])}')
            r.append(clf.score(X[test[0]:test[1]], dec[test[0]:test[1]]))
            if flag == 0:
                flag = 1
                plt.figure()
                plt.title(f'{test}')
                plt.plot(dec[train[0]:train[1]])
                plt.plot(pos_pre_train)
                plt.figure()
                plt.plot(dec[test[0]:test[1]])
                plt.plot(pos_pre_test)
        score_all[alpha] = r
        score_m = [np.mean(s) for s in score_all.values()]
    print(f'max score:{max(score_m)}')
    print(f'alpha:{alpha_list[np.argmax(score_m)]}')
    r_all[method[idx]] = score_all[alpha_list[np.argmax(score_m)]]
r_all = list(r_all.values())

#%%
from sklearn.preprocessing import StandardScaler
#caiman_output_path = os.path.join(base_folder, '5000', [file for file in os.listdir(base_folder+'/5000') if 'memmap_pw_rigid_True' in file and '.hdf5' in file][0])
caiman_output_path = '/media/nel/storage/fiola/R2_20190219/3000/memmap_pw_rigid_True_d1_796_d2_512_d3_1_order_C_frames_3000_non_rigid_K_5.hdf5'
cnm = load_CNMF(caiman_output_path)
dims = (796, 512)
A = cnm.estimates.A[:, cnm.estimates.idx_components].toarray()
#A = cnm.estimates.A[:, :2000].toarray()
noisy_C = cnm.estimates.C + cnm.estimates.YrA
noisy_C = noisy_C.T
noisy_C = StandardScaler().fit_transform(noisy_C)
# select = []
# high_act = []
# std_level = 5; timepoints = 10
# for idx in range(len(noisy_C.T)):
#     t = noisy_C[:, idx]
#     select.append(len(t[t>t.std() * std_level]) > timepoints)
#     high_act.append(len(t[t>t.std() * std_level]))
noisy_C_selected, select = remove_neurons_with_sparse_activities(noisy_C)
select = np.zeros(cnm.estimates.C.shape[0])
select[cnm.estimates.idx_components] = 1
select = select.astype(bool)
if np.array(select).sum()>1300:
    raise Exception('Too many components for fiola')

#%%
trace = run_fiola(mov.astype(np.float32), cnm=cnm, select=select)
np.save(os.path.join(base_folder, '3000/fiola_non_rigid_init.npy'), trace)

#%%
mmap_file = os.path.join(base_folder, [file for file in os.listdir(base_folder) if '.mmap' in file][0])
mov = cm.load(mmap_file, in_memory=True)
#mov = mov.reshape([mov.shape[0], 796, 512], order='F')
mov = mov.reshape([mov.shape[0], -1], order='F')
tracem = run_meanroi(mov, cnm, select)
np.save(os.path.join(base_folder,'3000/mean_roi_99.98_non_rigid_init.npy'), tracem.T)

#%%

plt.imshow(aaa, vmax=np.percentile(aaa, 99), vmin=np.percentile(aaa, 5), cmap='gray')
plt.savefig(savef + 'median_img_3000.pdf')
plt.imshow(cnm.estimates.Cn, vmax=np.percentile(cnm.estimates.Cn, 99), vmin=np.percentile(cnm.estimates.Cn, 5), cmap='gray')
plt.savefig(savef + 'corr_img_3000.pdf')




#%% save movie 
mov = nwbfile_in.acquisition['TwoPhotonSeries'].data[:]
mov = cm.movie(mov)
try:
    os.mkdir(os.path.join(base_folder, '3000'))
    print('make folder')
except:
    print('fail to make folder')
mov.save(os.path.join(base_folder, 'mov_raw.hdf5'))
mov[:3000].save(os.path.join(base_folder, '3000', 'mov_raw_3000.hdf5'))

#%% test on detrending
fio = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_2_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', allow_pickle=True).item()
#fio = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_True_with_detrending_v3.11.npy', allow_pickle=True).item()
dc_param = 0.995
tt = fio.trace[:, :].copy()
#tt = tt - tt[:, 0][:, None]
t_d = np.zeros(tt.shape)
#t_d[:, 0] = tt[:, 0]
for tp in range(tt.shape[1]):
    if tp > 0:
        t_d[:, tp] = tt[:, tp] - tt[:, tp - 1] + dc_param * t_d[:, tp - 1]

t_d = t_d - 50000

#%%
#t_d = tt.copy()
fig, ax = plt.subplots(2,2)
plt.suptitle(f'dc_param:{dc_param}')
ii = 120
ax[0,0].plot(tt[ii]); ax[0,0].plot(t_d[ii])
ii += 10
ax[1,0].plot(tt[ii]); ax[1,0].plot(t_d[ii])
ii += 10
ax[0,1].plot(tt[ii]); ax[0,1].plot(t_d[ii])
ii += 10
ax[1,1].plot(tt[ii]); ax[1,1].plot(t_d[ii])

        
#%%
ii = 120
plt.plot(tt[ii]); plt.plot(t_d[ii])
print(np.median(tt[ii, 4000:7000]) - np.median(tt[ii, -3000:]))
print(np.median(t_d[ii, 4000:7000]) - np.median(t_d[ii, -3000:]))
#%%
t1 = np.median(tt[:, 4000:7000], axis=1)[:-2]
t2 = np.median(tt[:, -3000:], axis=1)[:-2]
#plt.scatter(t1, t2)
print(np.mean(np.abs(t1 - t2)))
#%%
t3 = np.median(t_d[:, 4000:7000], axis=1)[:-2]
t4 = np.median(t_d[:, -3000:], axis=1)[:-2]
#plt.scatter(t3, t4)
print(np.mean(np.abs(t3 - t4)))

#%%
plt.scatter((t1-t2), (t3-t4))
#%%
imageio.mimwrite('output_filename.mp4', m, fps =100.0)
#%%
key = 'FIOLA3000'#'FIOLA3000'
fig, ax = plt.subplots(2,2)
fig.suptitle(f'{key}')
#ax[0, 0].plot(t_raw[key][:, :-2].mean(0)[3000:])
#ax[0, 0].plot(t_b[key].mean(0)[3000:])
#ax[0, 1].plot(t_det[key].mean(1)[3000:])
#ax[1, 0].plot(t_rs[key].mean(1)[3000:])
#ax[1, 1].plot(t_g[key].mean(1)[3000:])

#ii = np.random.randint(0, 1300)
print(ii)
ax[0, 0].plot(t_raw[key][ii][3000:])
ax[0, 0].plot(t_b[key][ii][3000:])
ax[0, 1].plot(t_det[key][:, ii][3000:])
# ax[1, 0].plot(t_rs[key][:, ii][3000:])
# ax[1, 1].plot(t_g[key][:, ii][3000:])

# ax[0, 0].set_title('raw')
# ax[0, 1].set_title('detrend with signal filter')
# ax[0, 1].set_title('detrend with percentile filter')
# ax[1, 0].set_title('normalize and remove neurons')
# ax[1, 1].set_title('gaussian filter')
plt.tight_layout()

# 353, 1206, 1274, 893, 474

#%%
fig, ax = plt.subplots(3,3)
for i in range(3):
    for j in range(3):
        try:
            jj = i * 3 + j
            key = list(t_raw.keys())[jj]
            # t1 = np.median(t_raw[key][:, 3000:6000], axis=1)[:-2]
            # t2 = np.median(t_raw[key][:, -3000:], axis=1)[:-2]
            t1 = np.median(t_rs[key].T[:, 4000:7000], axis=1)[:-2]
            t2 = np.median(t_rs[key].T[:, -3000:], axis=1)[:-2]
            print(f'{key}:mean {np.mean(np.abs(t1-t2))}')# std {np.std(t1-t2)}')

            ax[i, j].scatter(t1,t2)
            #ax[i, j].set_xlim(t1.min(), t1.max())
            #ax[i, j].set_ylim(t1.min(), t1.max())
            #ax[i, j].plot([t1.min(),t1.max()], [t1.min(), t1.max()], linestyle='dashed', color='black')
            ax[i, j].set_title(key)
        except:
            pass
plt.tight_layout()
#%%
t1 = np.median(t_raw[key][:, 4000:7000], axis=1)[:-2]
t2 = np.median(t_raw[key][:, -3000:], axis=1)[:-2]

np.argsort(t1-t2)


#%% drifting analysis
#ii = 353
rigid = True
fio = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_True_no_init_v3.9.npy', allow_pickle=True).item()
if not rigid:
    mov = cm.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_.mmap')
else:
    mov = cm.load('/media/nel/storage/fiola/R2_20190219/full/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_.mmap')

spatial = fio.Ab.reshape((796, 512, 1313), order='F')  # 353, 1206, 1274, 893, 474
do_mroi = False
if do_mroi:
    mroi_traces = []
    for ii in range(spatial.shape[2]):
        print(ii)
        a = spatial[:, :, ii].copy()
        a[a>np.percentile(a, 99.98)] = 1
        a[a!=1]=0
        xx, yy = np.where(a>0)
        # l = 0
        # x_min = xx.min()-0; x_max = xx.max()+0; y_min = yy.min()-0; y_max = yy.max()+0
        # mmm = mov[:, x_min:x_max+1, y_min:y_max+1]
        # mmm.resize(1,1,0.01).play(fr=50, magnification=10)
        mroi = mov[:, xx, yy]
        mroi = mroi.mean(1)
        mroi_traces.append(mroi)
    mroi_traces = np.array(mroi_traces)
else:
    wa_traces = []
    for ii in range(spatial.shape[2]-2):
        print(ii)
        a = spatial[:, :, ii].copy()
        xx, yy = np.where(a>0)
        #a[a>np.percentile(a, 99.98)] = 1
        #a[a!=1]=0
        x_min = xx.min()-0; x_max = xx.max()+0; y_min = yy.min()-0; y_max = yy.max()+0
        a = a[x_min:x_max+1, y_min:y_max+1]
        a = a.reshape([-1], order='F')
        mmm = mov[:, x_min:x_max+1, y_min:y_max+1]
        mmm = mmm.reshape([mmm.shape[0], -1], order='F')
        wa = np.matmul(mmm, a[:, None])[:, 0]
        wa_traces.append(wa)
    wa_traces = np.array(wa_traces)

#%%
#mroi = StandardScaler().fit_transform(mroi[:, None])[:, 0]
# fig, ax = plt.subplots()
# ax.imshow(a)
# import matplotlib.patches as patches
# rect = patches.Rectangle((yy.min(), xx.min()), yy.max()-yy.min()+1, xx.max()-xx.min()+1, 
#                          linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# ax.set_xlim([yy.min(), yy.max()+1])
# ax.set_ylim([xx.min(), xx.max()+1])

# f = t_raw[key][ii]
# f = StandardScaler().fit_transform(f[:, None])[:, 0]
# plt.plot(f)
# plt.plot(mroi-3)
key = 'FIOLA3000'#'Suite2p'#'FIOLA3000'
#f = t_raw[key]#[:-2]
#f = StandardScaler().fit_transform(f.T).T

tt0 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_True_include_bg_True_no_init_v3.9.npy', allow_pickle=True).item().trace
tt0 = tt0[:-2]
#tt0 = tt0 - tt0.mean(1)[:, None]
tt1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_True_include_bg_False_no_init_v3.9.npy', allow_pickle=True).item().trace
tt1 = tt1 - tt1.mean(1)[:, None]
#%%
fig, ax = plt.subplots(2,1, figsize=(8, 16))
ax[0].scatter(np.median(tt0[:, 3000:6000], axis=1), np.median(tt0[:, -3000:], axis=1))
ax[0].plot([0, 30000], [0, 30000], color='black', linestyle='dashed')
ax[0].set_title('fiola with background')
ax[0].set_xlabel('first 3000 frames')
ax[0].set_ylabel('last 3000 frames')
ax[1].scatter(np.median(tt1[:, 3000:6000], axis=1), np.median(tt1[:, -3000:], axis=1))
ax[1].plot([0, 30000], [0, 30000], color='black', linestyle='dashed')
ax[1].set_title('fiola without background')

# plt.scatter(np.median(f[:, 3000:6000], axis=1), np.median(f[:, -3000:], axis=1))
# plt.title()

#plt.xlim([-1, 1]); plt.ylim([-1, 1])
#plt.scatter(np.median(mroi_traces[:, 3000:6000], axis=1), np.median(mroi_traces[:, -3000:], axis=1))
#plt.scatter(np.median(wa_traces[:, 3000:6000], axis=1), np.median(wa_traces[:, -3000:], axis=1))



# print(f[3000:6000].mean() - f[-3000:].mean())
# print(mroi[3000:6000].mean() - mroi[-3000:].mean())
#%% detrend baseline
# t_b = {}
# t_det = {}
# for key in t_raw.keys():
#     t_b[key] = signal_filter(t_raw[key], freq=0.005, fr=15, order=3, mode='low')
#     #t_det[key] = fast_prct_filt(t_raw[key],20).T
#     #t_det[key] = (t_raw[key] - t_b[key]).T
#     t_det[key] = (t_raw[key]).T

# remove atificial feature for onacid due to filtering
# for idx in range(t_det['CaImAn_Online'].shape[1]):
#     if onacid.time_neuron_added[idx, 1] > 200:
#         t_det['CaImAn_Online'][:onacid.time_neuron_added[idx, 1]-100, idx] = np.mean(t_det['CaImAn_Online'][onacid.time_neuron_added[idx, 1]-100:, idx])
#%% remove sparse activities
# t_rs = {}
# for key in t_det.keys():
#     # if 'Suite2p_rigid' == key:
#     #     print('pass')
#     #     t_rs[key] = t_det[key]
#     # else:
#     t_rs[key], _ = remove_neurons_with_sparse_activities(t_det[key])
# [print(tt.shape) for tt in t_rs.values()]

#%% remove neurons in onacid that are detected after 10000 frames
# first_act = []
# std_level = 5
# for idx in range(len(t_rs['CaImAn_Online'].T)):
#     t = t_rs['CaImAn_Online'][:, idx]
#     first_act.append(np.where(t>t.std() * std_level)[0][0])
# select = np.where(np.array(first_act)<10000)[0]
# t_rs['CaImAn_Online'] = t_rs['CaImAn_Online'][:, select]
#%%
# for idx in range(t_raw['CaImAn_Online'].shape[1]):
#     if onacid.time_neuron_added[idx, 1] > 200:
#         t_raw['CaImAn_Online'][:onacid.time_neuron_added[idx, 1]-100, idx] = np.mean(t_raw['CaImAn_Online'][onacid.time_neuron_added[idx, 1]-100:, idx])
#%%
F = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p_result_non_rigid_v3.1/plane0/F.npy')
F_neu = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p_result_non_rigid_v3.1/plane0/Fneu.npy')
FF = F - 0.7 * F_neu
spikes = []
from caiman.source_extraction.cnmf.temporal import constrained_foopsi
for idx, f in enumerate(FF):
    print(idx)
    fff = StandardScaler().fit_transform(f[:, None])
    c_full, bl, c1, g, sn, s_full, lam = constrained_foopsi(fff[:, None], p=1, s_min=0)
    spikes.append(s_full)
spikes = np.array(spikes)

c_full, bl, c1, g, sn, s_full, lam = constrained_foopsi(FF[1242], p=1, s_min=0)
#t_s2p = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p_result_non_rigid_v3.1/plane0/spks.npy')
#t_s2p_rigid = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p/plane0/F.npy')
#%%
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
methods = ['FIOLA', 'Suite2p', 'MeanROI']
colors = ['C0', 'C2', 'C4']

for i, method in enumerate(['FIOLA', 'Suite2p', 'MeanROI']):
    ax1.bar(np.array(range(len(r[method]))) + (i-1) * 0.2, [np.mean(x) for x in np.array(list(r[method].values()))], 
            yerr=[np.std(x) for x in np.array(list(r[method].values()))], width=0.2, label=method, color=colors[i])
ax1.legend(frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xlabel('FOV size')
ax1.set_ylabel('Decoding R^2')
ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
#ax1.set_xticks(np.array(range(len(np.array(num)[::2]))))  
ax1.set_xticklabels(areas)
ax1.set_xticks(list(range(len(areas))))
#ax1.set_yticks([])
ax1.set_ylim([0,1])
plt.savefig(savef+ 'Fig7_fiola_mroi_remove_frames_with_dec.pdf')
#plt.savefig(savef + 'Fig7_fiola_mroi_remove_frames.pdf')

# r['FIOLA']['FIOLA3000_0.125']
# r['MeanROI']['MeanROI_0.125']

# def label_diff(i,j,text,X,Y):
#     x = (X[i]+X[j])/2
#     y = 1.1*max(Y[i], Y[j])
#     dx = abs(X[i]-X[j])

#     props = {'connectionstyle':'bar','arrowstyle':'-',\
#                  'shrinkA':20,'shrinkB':20,'linewidth':2}
#     ax.annotate(text, xy=(X[i],y+7), zorder=10)
#     ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)

for area in areas:
    print(area)
    print(ttest_rel(r['FIOLA'][f'FIOLA3000_{area}'], r['MeanROI'][f'MeanROI_{area}'], alternative='two-sided')) # pvalue 0.014
    print(ttest_rel(r['FIOLA'][f'FIOLA3000_{area}'], r['Suite2p'][f'Suite2p_{area}'], alternative='two-sided')) # pvalue 0.014
# label_diff()


#%%

#trace0 = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['Deconvolved'].data[:]
#trace = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries'].data[:].T
                
F = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p/plane0/F.npy')
Fneu = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p/plane0/Fneu.npy')
trace = F - 0.7 * Fneu
trace = trace[:, 1:]
trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_v3.1.npy', allow_pickle=True).item().trace
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_True_include_bg_True_no_init_v3.9.npy', allow_pickle=True).item().trace
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_True_include_bg_False_no_init_v3.9.npy', allow_pickle=True).item().trace
trace2 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_result_v3.1.npy', allow_pickle=True).item().trace
trace3 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_result_v3.1.npy', allow_pickle=True).item().trace
trace4 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_result_v3.1.npy', allow_pickle=True).item().trace
#fio1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_True_no_init_v3.9.npy', allow_pickle=True).item()
fio = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', allow_pickle=True).item()
#fio1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_True_with_detrending_v3.11.npy', allow_pickle=True).item()
#tt1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_True_no_init_v3.9.npy', allow_pickle=True).item()
#%%
ii = 280
plt.figure()
plt.plot(tt.trace[ii])
plt.plot(tt.trace_deconvolved[ii])
plt.figure()
plt.plot(tt1.trace[ii])
plt.plot(tt1.trace_deconvolved[ii])

#%%
from sklearn.preprocessing import StandardScaler
from caiman.source_extraction.cnmf.cnmf import load_CNMF
#cnm2 = load_CNMF('/media/nel/storage/fiola/R6_20200210T2100/memmap_pw_rigid_True_d1_796_d2_512_d3_1_order_C_frames_31604_non_rigid_K_5.hdf5')
cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8_K_8.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8_cnn_True.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_.hdf5')
tracec = cnm2.estimates.C[cnm2.estimates.idx_components] + cnm2.estimates.YrA[cnm2.estimates.idx_components]
onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mov_R2_20190219T210000_caiman_online_results_v3.0_new_params.hdf5')
#onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/caiman_online_results.hdf5')
traceo = onacid.estimates.C + onacid.estimates.YrA

#%%
t_raw = {}
for num_layers in [1, 3, 5, 10, 30]:
    file = f'/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_{num_layers}_no_init_v3.9.npy'
    aa = np.load(file, allow_pickle=True).item()
    #print(np.mean(np.diff(aa.timing['all_online'])))
    t_raw[f'FIOLA_{num_layers}'] = aa.trace
[print(tt.shape) for tt in t_raw.values()]

#%%
idx = 120;
for ii, num_layers in enumerate([1, 3, 5, 10, 30]):
    tt = t_raw[f'FIOLA_{num_layers}'][idx, 5000:].T
    plt.plot((tt - tt.mean()) / tt.std()-ii*10, label=f'{num_layers}')
    plt.title(f'Neuron_{idx}')
    plt.legend()
    plt.savefig(savef+f'layers_traces_{idx}')

#%%
rr = {}
for key in t_raw.keys():
    rr[key] = np.mean(r[key])
    
#%% detreding
trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_2_num_layers_10_trace_with_neg_False_with_detrending_v3.11.npy', allow_pickle=True).item().trace_deconvolved
trace11 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_no_init_v3.9.npy', allow_pickle=True).item().trace_deconvolved
t_raw = {}
t_raw['FIOLA_n'] = trace1.T
t_raw['FIOLA_d'] = trace11.T

#%% mmotion correction FOV
trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_with_detrending_v3.11.npy', allow_pickle=True).item().trace_deconvolved
trace11 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_None_with_detrending_v3.11.npy', allow_pickle=True).item().trace_deconvolved
t_raw = {}
t_raw['FIOLA_mc_small'] = trace1.T
t_raw['FIOLA_normal'] = trace11.T

#%% different SNR
t_raw = {}
#trace = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_None_test_snr_v3.13.npy', allow_pickle=True).item().trace_deconvolved.T
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_None_with_detrending_v3.12.npy', allow_pickle=True).item().trace_deconvolved.T
trace = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_None_test_snr_orig_movie_v3.13.npy', allow_pickle=True).item().trace_deconvolved.T
#t_raw[f'FIOLA_no_thresh'] = trace.copy()
t_raw[f'FIOLA'] = trace
for snr in [1.1, 1.3, 1.5, 1.8, 2.0]:
    fff = '/media/nel/storage/fiola/R2_20190219/3000/'
    comp = np.load([os.path.join(fff, f) for f in os.listdir(fff) if f'min_snr_{snr}_comp' in f][0])
    t_raw[f'FIOLA_{snr}'] = trace[:, comp].copy()
    
#%%
ii = 10
m = ['FIOLA3000', 'Suite2p'][0]
temp = t_g[m][1:, ii].copy()
temp[s_t==False] = np.nan
plt.plot(t_g[m][:, ii]); plt.plot(temp); plt.plot(pos)  #plt.plot(t_g['FIOLA3000'][1:, 20][s_t])

#%%
plt.figure()
plt.plot(list(p.values())[0][0]['Y'], color='black')
key_list = ['FIOLA3000', 'CaImAn_Online']
for key, pr in p.items():
    if key in key_list:
        print(key)
        plt.plot(pr[0]['Y_pr'])
        #plt.title(f'{cv_test[0]}, {clf.score(x_test, Y_test)}')
        plt.show()
plt.legend(['gt']+[k+f':{round(r[k][0], 2)}' for k in key_list])
plt.xlabel('frames')
plt.ylabel('location after z-score')
plt.savefig(savef + 'Fig7c_supp_postion_prediction_v3.5.pdf')

#%% statistical tests
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

for key in ['CaImAn_Online', 'Suite2p', 'CaImAn']:
    X = np.array(xx)[:, None]
    #y = np.array(r['Suite2p']) - np.array(r['FIOLA3000'])
    y = np.array(r[key]) - np.array(r['FIOLA3000'])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xx,y)
    print(key)
    print(f'p_value:{p_value}')
    print(f'slope:{slope}')
    print(f'intercept:{intercept}')

# CaImAn_Online
# p_value:0.5420357611413338
# slope:-5.284746046288936e-07
# intercept:0.018302775290672736
# Suite2p
# p_value:0.01943630320524657
# slope:2.133904580590397e-06
# intercept:-0.030689514165719252
# CaImAn
# p_value:0.23207276702117616
# slope:1.1313397770836403e-06
# intercept:-0.008672941864403903

from scipy.stats import wilcoxon, ttest_rel
ttest_rel(r['CaImAn'][:4], r['FIOLA3000'][:4], alternative='two-sided')
ttest_rel(r['CaImAn'][4:], r['FIOLA3000'][4:], alternative='two-sided')

# ttest_rel(r['CaImAn'], r['FIOLA3000'], alternative='greater') # pvalue 0.014
# ttest_rel(r['CaImAn_Online'], r['FIOLA3000'], alternative='greater') # pvalue 0.20
# ttest_rel(r['Suite2p'], r['FIOLA3000'], alternative='greater') # 0.07

# trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_None_with_detrending_v3.12.npy', allow_pickle=True).item().trace_deconvolved
tracem = np.load('/media/nel/storage/fiola/R2_20190219/meanroi/traces_dec_v3.13.npy')
# traces = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p_result_non_rigid_v3.1/plane0/spks.npy')
# iscell = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p_result_non_rigid_v3.1/plane0/iscell.npy')
# stat = np.load('/media/nel/storage/fiola/R2_20190219/test/test_full/suite2p_result_non_rigid_v3.1/plane0/stat.npy', allow_pickle=True)
# s_selected = np.where(iscell[:, 1] >=0.05)[0]
# traces = traces[s_selected]
# stat = stat[s_selected]
# s_centers = np.array([ss['med'] for ss in stat])

trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_None_with_detrending_v3.12.npy', allow_pickle=True).item().trace
tracem = np.load('/media/nel/storage/fiola/R2_20190219/meanroi/traces_v3.13.npy')


cnm = load_CNMF('/media/nel/storage/fiola/R2_20190219/3000/memmap__d1_796_d2_512_d3_1_order_C_frames_3000__v3.7.hdf5')
t_raw = {}
centers = com(cnm.estimates.A, 796, 512)
yy = 796
xx = 512
A = cnm.estimates.A.toarray()
dims = [796, 512]
A = A.reshape([dims[0], dims[1], -1], order='F')
A = A.transpose([2, 0, 1])
masks = A
b_masks = masks_to_binary(masks.copy())
num_all = []
s_num_all = []
areas = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
for area in areas:
    yl = int(796 * np.sqrt(area))
    xl = int(512 * np.sqrt(area))
    x_range = [(xx - xl) // 2, (xx - xl) // 2 + xl ]
    y_range = [(yy - yl) // 2, (yy - yl) // 2 + yl ]
    
    select1 = np.where(np.logical_and(centers[:, 1] >= x_range[0], centers[:, 1] <= x_range[1]))[0]
    select2 = np.where(np.logical_and(centers[:, 0] >= y_range[0], centers[:, 0] <= y_range[1]))[0]
    select = np.intersect1d(select1, select2)    
    num = len(select)
    num_all.append(num)
    
    plt.figure()
    plt.imshow(b_masks.sum(0))
    plt.scatter(centers[select, 1], centers[select, 0], s=0.5, color='red')
    plt.plot([x_range[0],x_range[0]], [y_range[0],y_range[1]], 'y-')
    plt.plot([x_range[1],x_range[1]], [y_range[0],y_range[1]], 'y-')
    plt.plot([x_range[0],x_range[1]], [y_range[0],y_range[0]], 'y-')
    plt.plot([x_range[0],x_range[1]], [y_range[1],y_range[1]], 'y-')

    t_raw[f'FIOLA3000_{area}'] = trace1[select, :].T
    t_raw[f'MeanROI_{area}'] = tracem[select, :].T
    
#     ss_select1 = np.where(np.logical_and(s_centers[:, 1] >= x_range[0], s_centers[:, 1] <= x_range[1]))[0]
#     ss_select2 = np.where(np.logical_and(s_centers[:, 0] >= y_range[0], s_centers[:, 0] <= y_range[1]))[0]
#     ss_select = np.intersect1d(ss_select1, ss_select2)
#     s_num = len(ss_select)
#     s_num_all.append(s_num)    

#     t_raw[f'Suite2p_{area}'] = traces[ss_select, :].T

# [print(tt.shape) for tt in t_raw.values()]

#
# t1 = StandardScaler().fit_transform(trace1.T).T
# t2 = StandardScaler().fit_transform(tracem.T).T
t1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_None_with_detrending_v3.12.npy', allow_pickle=True).item().trace
t2 = np.load('/media/nel/storage/fiola/R2_20190219/meanroi/traces_v3.13.npy')

t2 = t2 - t2.mean(1)[:, None]
t2[t2<0] = 0 

# t1 = (t1 - t1.mean(1)[:, None]) / t1.max(1)[:, None]
# t2 = (t2 - t2.mean(1)[:, None]) / t2.max(1)[:, None]
t1 = StandardScaler().fit_transform(t1.T).T
t2 = StandardScaler().fit_transform(t2.T).T

#
view_compare_components(t1=t1, t2=t2, A=cnm.estimates.A.toarray().copy(), img=cnm.estimates.Cn, area=1/8)
#%% Supp fiola v suite2p v meanroi
areas = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
rr1 = np.load('/media/nel/storage/fiola/R2_20190219/result/Fig7_supp_downsampling_result_FIOLA_v3.5.npy', allow_pickle=True).item()
rr2 = np.load('/media/nel/storage/fiola/R2_20190219/result/Fig7_supp_downsampling_result_MeanROI_v3.5.npy', allow_pickle=True).item()
rr3 = np.load('/media/nel/storage/fiola/R2_20190219/result/Fig7_supp_downsampling_result_Suite2p_v3.5.npy', allow_pickle=True).item()
r = {'FIOLA':rr1, 'MeanROI':rr2, 'Suite2p':rr3 }

# significance
for i, area in enumerate(areas):
    temp = []
    temp.append(ttest_rel(r['FIOLA'][f'FIOLA3000_{area}'], r['Suite2p'][f'Suite2p_{area}'], alternative='two-sided')[1])
    temp.append(ttest_rel(r['FIOLA'][f'FIOLA3000_{area}'], r['MeanROI'][f'MeanROI_{area}'], alternative='two-sided')[1])
    temp.append(ttest_rel(r['MeanROI'][f'MeanROI_{area}'], r['Suite2p'][f'Suite2p_{area}'], alternative='two-sided')[1])
    sig.append(temp)

#%%
fig = plt.figure(figsize=(10, 10)) 
ax1 = plt.subplot()
methods = ['FIOLA', 'Suite2p', 'MeanROI']
colors = ['C0', 'C2', 'C4']
mean = np.zeros((3, 7))
std = np.zeros((3, 7))

for i, method in enumerate(['FIOLA', 'Suite2p', 'MeanROI']):
    mean[i] = [np.mean(x) for x in np.array(list(r[method].values()))]
    std[i] = [np.std(x) for x in np.array(list(r[method].values()))]

for i, method in enumerate(['FIOLA', 'Suite2p', 'MeanROI']):
    width = 0.2
    ax1.bar(np.array(range(len(r[method]))) + (i) * width, [np.mean(x) for x in np.array(list(r[method].values()))], 
            yerr=[np.std(x) for x in np.array(list(r[method].values()))], width=width, label=method, color=colors[i])

width=0.2
delta = 0.03
for i in range(len(areas)):
    for j in range(3):
        if j == 0:
            group = [0, 1]
        if j == 1:
            group = [0, 2]
        if j == 2:
            group = [1, 2]
        temp = sig[i][j]
        barplot_annotate_brackets(data=temp, center1=i+width*group[0], center2=i+width*group[1], 
                                  height=np.max(mean[:, i])+delta*j+0.01 , yerr=np.max(std[:, i]))
ax1.legend(frameon=False)
#ax1.spines['bottom'].set_visible(False)
ax1.set_xlabel('FOV size')
ax1.set_ylabel('Decoding R^2')
ax1.xaxis.set_ticks_position('none') 
ax1.set_xticklabels(areas)
ax1.set_xticks(np.array(list(range(len(areas))))+width)
plt.savefig(savef+ 'Fig7_fiola_mroi_significance.pdf')
#ax1.set_yticks([])
#ax1.set_ylim([0.2,1])

#%% neuron downsampling 
t3000 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_v3.15.npy', allow_pickle=True).item().trace_deconvolved
tm = np.load('/media/nel/storage/fiola/R2_20190219/meanroi/traces_dec_v3.13.npy')
sp_raw = {'FIOLA3000': cnm3000}
selection = select_neurons_within_regions(sp_raw, y_limit=[30, 777])

for iteration in range(5):
    t_raw = {}
    select = {}
    random.seed(iteration * 5)
    s1 = list(selection.values())[0]
    sl = random.permutation(s1)
    if iteration == 0:
        num_neurons = [1307, 700, 500, 300, 100]
    else:
        num_neurons = [700, 500, 300, 100]
    select = {}
    for num in num_neurons:
        t_raw[f'FIOLA3000_{num}'] = t3000.T
        t_raw[f'MeanROI_{num}'] = tm.T
        select[f'FIOLA3000_{num}'] = sl[:num]
        select[f'MeanROI_{num}'] = sl[:num]
        print(sl)
        
    t_g, t_rs, t_rm = run_trace_preprocess(t_raw, select, sigma=12)
        
    result = {}
    for method in ['FIOLA', 'MeanROI']:
        t_test = dict((kk, t_g[kk]) for kk in list(t_g.keys()) if method in kk)
        t_s = 3000
        t_e = 31900
        dec = [pos_s, spd_s][0].copy()
        r = {}
        p = {}
        alpha_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        for key, tr in t_test.items():
            print(f'{key}')
            s_tt = s_t[t_s:t_e]
            X = tr[t_s:t_e][s_tt]
            Y = dec[t_s:t_e][s_tt]
            r[key], p[key] = cross_validation_ridge(X, Y, normalize=True, n_splits=5, alpha_list=alpha_list)
            print(f'average decoding performance: {np.mean(r[key])}')
        end = time()
        ff = '/media/nel/storage/fiola/R2_20190219/result/'
        np.save(ff+f'Fig7_supp_downsampling_result_{method}_{iteration}_v3.7.npy', r)

#%%
num_neurons = [1307, 700, 500, 300, 100]
for iteration in range(5):
    fig = plt.figure(figsize=(10, 10)) 
    ax1 = plt.subplot()
    for idx, method in enumerate(['FIOLA', 'MeanROI']):
        file = f'/media/nel/storage/fiola/R2_20190219/result/Fig7_supp_downsampling_result_{method}_{iteration}_v3.7.npy'
        rr = np.load(file, allow_pickle=True).item()
        mean = [np.mean(r) for r in rr.values()]
        std = [np.std(r) for r in rr.values()]
        ax1.bar(np.array(range(len(mean)))[::-1]+idx*0.2, mean, yerr=std, width=0.2)

#%%
num_neurons = [1307, 700, 500, 300, 100]
F_mean = np.zeros((5, 5))
M_mean = np.zeros((5, 5))
for iteration in range(5):
    
    for idx, method in enumerate(['FIOLA', 'MeanROI']):
        file = f'/media/nel/storage/fiola/R2_20190219/result/Fig7_supp_downsampling_result_{method}_{iteration}_v3.7.npy'
        rr = np.load(file, allow_pickle=True).item()
        # mean = [np.mean(r) for r in rr.values()]
        # std = [np.std(r) for r in rr.values()]
        #mean.append([np.mean(r) for r in rr.values()])    
        if method == 'FIOLA':
            if iteration == 0:
                F_mean[iteration, :] = [np.mean(r) for r in rr.values()]
            else:
                F_mean[iteration, 1:5] = [np.mean(r) for r in rr.values()]
        else:
            if iteration == 0:
                M_mean[iteration, :] = [np.mean(r) for r in rr.values()]
            else:
                M_mean[iteration, 1:5] = [np.mean(r) for r in rr.values()]
   
F_mean[:, 0] = F_mean[0, 0] 
M_mean[:, 0] = M_mean[0, 0] 

fig = plt.figure(figsize=(10, 10)) 
ax1 = plt.subplot()
ax1.bar(np.array(range(5))[::-1], F_mean.mean(0), yerr=F_mean.std(0), width=0.2, label='FIOLA3000')
ax1.bar(np.array(range(5))[::-1]+0.2, M_mean.mean(0), yerr=M_mean.std(0), width=0.2, label='MeanROI')
ax1.set_xlabel('num neurons')
ax1.set_ylabel('Decoding R^2')
ax1.xaxis.set_ticks_position('none') 
ax1.set_xticklabels(np.array(num_neurons)[::-1])
ax1.set_xticks(np.array(list(range(len(num_neurons))))+0.1)
ax1.legend()
plt.savefig(savef+ 'Fig7_supp_fiola_mroi_v3.7.pdf')


temp = []
for idx in range(4):
    temp.append(ttest_rel(F_mean[:, idx+1], M_mean[:, idx+1], alternative='two-sided')[1])

#%% Preprocessing
t_g, t_rs = run_preprocess(t_raw, sigma=12)
[print(tt.shape) for tt in t_g.values()]
#%% Quick test
result = {}
for method in ['FIOLA', 'MeanROI']:
    t_test = dict((kk, t_g[kk]) for kk in list(t_g.keys()) if method in kk)
    t_s = 3000
    t_e = 31900
    dec = [pos_s, spd_s][0].copy()
    r = {}
    p = {}
    alpha_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    for key, tr in t_test.items():
        print(f'{key}')
        s_tt = s_t[t_s:t_e]
        X = tr[t_s:t_e][s_tt]
        Y = dec[t_s:t_e][s_tt]
        r[key], p[key] = cross_validation_ridge(X, Y, normalize=True, n_splits=5, alpha_list=alpha_list)
        print(f'average decoding performance: {np.mean(r[key])}')
    end = time()
    ff = '/media/nel/storage/fiola/R2_20190219/result/'
    np.save(ff+f'Fig7_supp_downsampling_result_{method}_no_dec_v3.5.npy', r)
[{rr: np.mean(r[rr])} for rr in r.keys()]

#%%
# r = np.load(ff+'Fig7e_result_v3.7.npy', allow_pickle=True).item()
# methods = list(r.keys())
# fig = plt.figure(figsize=(8, 6)) 
# ax1 = plt.subplot()
# colors = ['C0', 'C1', 'C2', 'C3']
# for idx, i in enumerate([0, 1, 2, 3]):
#     key = list(r.keys())[i]
#     method = methods[idx]
#     for j in range(len(spd_group)):
#         sp = spd_group[j]
#         if 'Suite2p' != method:
#             dat = ttest_rel(r[method][sp], r['Suite2p'][sp], alternative='two-sided').pvalue 
#             print(f'{method}, {sp}: {dat}')
#             barplot_annotate_brackets(dat, j + (idx-1)*0.2, j + (2-1)*0.2, 
#                                       height = 0.01 + 0.01 * idx + 0.95, 
#                                       dy=0.005)
#     #i = r[methodzz][idx]
#     ax1.bar(np.array(range(len(spd_group))) + (idx-1) * 0.2, [np.mean(x) for x in list(r[key].values())], 
#             yerr=[np.std(x) for x in list(r[key].values())], width=0.2, label=key, color=colors[i])
# ax1.legend(frameon=False)
# #ax1.bar(0, np.mean(r), yerr=np.std(r))
# #ax1.bar(0, np.mean(r1), yerr=np.std(r1))
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# #ax1.spines['left'].set_visible(False)
# ax1.set_xlabel('Speed group')
# ax1.set_ylabel('Decoding R^2')
# ax1.xaxis.set_ticks_position('none') 
# #ax1.yaxis.set_ticks_position('none') 
# #ax1.set_xticks([0, 1, 2, 3])  
# ax1.set_xticklabels(spd_group)
# ax1.set_xticks(list(range(len(spd_group))))
# #ax1.set_yticks([])
# ax1.set_ylim([0.4,1])
# #plt.savefig(savef + 'Fig7e_pos_v3.7.pdf')

#%%
fig, ax1 = plt.subplots()
ax1.bar(range(4), np.array(tpf*1000), label='FIOLA')
ax1.hlines(1/15.46*1000, -0.5, 3.5, linestyle='dashed', color='black', label='frame rate')
ax1.legend()
ax1.set_xticks([])
ax1.set_xlabel('init batch size (frames)')
ax1.set_ylabel('time (ms)')
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(['500', '1000', '1500', '3000' ])
#plt.savefig(savef+'Fig7h_tpf_v3.7.pdf')