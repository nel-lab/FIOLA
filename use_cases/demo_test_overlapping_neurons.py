#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:15:34 2021

@author: @caichangjia
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import time, sleep
from threading import Thread

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass
#%%
from viola.metrics import metric
from viola.nmf_support import normalize
from viola.violaparams import violaparams
from viola.viola import VIOLA
import scipy.io
from viola.match_spikes import match_spikes_greedy, compute_F1
#sys.path.append('/home/nel/Code/NEL_LAB/VIOLA/use_cases')
#sys.path.append(os.path.abspath('/Users/agiovann/SOFTWARE/VIOLA'))
from use_cases.test_run_viola import run_viola # must be in use_cases folder

#%%
ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/overlapping_neurons'
names = sorted(os.listdir(ROOT_FOLDER))

one_neuron_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron'
one_neuron_names = ['454597_Cell_0_40x_patch1', '456462_Cell_3_40x_1xtube_10A2',
         '456462_Cell_3_40x_1xtube_10A3', '456462_Cell_5_40x_1xtube_10A5',
         '456462_Cell_5_40x_1xtube_10A6', '456462_Cell_5_40x_1xtube_10A7', 
         '462149_Cell_1_40x_1xtube_10A1', '462149_Cell_1_40x_1xtube_10A2',
         '456462_Cell_4_40x_1xtube_10A4', '456462_Cell_6_40x_1xtube_10A10',
         '456462_Cell_5_40x_1xtube_10A8', '456462_Cell_5_40x_1xtube_10A9', 
         '462149_Cell_3_40x_1xtube_10A3', '466769_Cell_2_40x_1xtube_10A_6',
         '466769_Cell_2_40x_1xtube_10A_4', '466769_Cell_3_40x_1xtube_10A_8', 
         '09282017Fish1-1', '10052017Fish2-2', 'Mouse_Session_1']

#%%
select = np.array(range(len(names)))

for idx, name in enumerate(np.array(names)[select]):
    num_frames_init = 20000
    border_to_0 = 0
    flip = True
    thresh_range= [3, 4]
    erosion=0 
    use_rank_one_nmf=False
    hals_movie='hp_thresh'
    semi_nmf=False
    update_bg = False
    use_spikes= False
    use_batch=True
    batch_size=100
    center_dims=None
    initialize_with_gpu=False
    do_scale = False
    adaptive_threshold=True
    filt_window=15
    freq = 15
    do_plot = True
    step = 2500
    
    options = {
        'border_to_0': border_to_0,
        'flip': flip,
        'num_frames_init': num_frames_init, 
        'thresh_range': thresh_range,
        'erosion':erosion, 
        'use_rank_one_nmf': use_rank_one_nmf,
        'hals_movie': hals_movie,
        'semi_nmf':semi_nmf,  
        'update_bg': update_bg,
        'use_spikes':use_spikes, 
        'use_batch':use_batch,
        'batch_size':batch_size,
        'initialize_with_gpu':initialize_with_gpu,
        'do_scale': do_scale,
        'adaptive_threshold': adaptive_threshold,
        'filt_window': filt_window, 
        'freq':freq,
        'do_plot':do_plot,
        'step': step}
    
    fr = 400.8
    fnames = os.path.join(ROOT_FOLDER, name, name+'.tif')  # files are motion corrected before
    path_ROIs = os.path.join(ROOT_FOLDER, name, name+'_ROIs.hdf5')
    run_viola(fnames, path_ROIs, fr=fr, online_gpu=True, options=options)

#%%
select = np.array(range(9))[:]
init_frames = 20000
result_all = {}

for idx, name in enumerate(np.array(names)[select]):
    neuron_idxs = [int(name[6]), int(name[8])]
    f1_scores = []                
    prec = []
    rec = []
    thr = []
    result = {}

    for idxx, neuron_idx in enumerate(neuron_idxs):       
        gt_path = os.path.join(one_neuron_folder, one_neuron_names[neuron_idx], 
                               one_neuron_names[neuron_idx]+'_output.npz')
        dict1 = np.load(gt_path, allow_pickle=True)
        length = dict1['v_sg'].shape[0]    
        
        vi_folder = os.path.join(ROOT_FOLDER, name, 'viola')
        vi_files = sorted([file for file in os.listdir(vi_folder) if 'v2.1' in file and 'hp_thresh' in file])# and '24000' in file])
        print(f'files number: {len(vi_files)}')
        if len(vi_files) > 1:
            vi_files = [file for file in vi_files if '15000' in file]
        vi_file = vi_files[0]
        vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
        
        vi_spatial = vi.H.copy()
        vi_temporal = vi.t_s.copy()
        vi_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vi.index])[np.argsort(vi.seq)][idxx]
        thr.append(vi.thresh_factor[0])
        
        n_cells = 1
        vi_result = {'F1':[], 'precision':[], 'recall':[]}        
        rr = {'F1':[], 'precision':[], 'recall':[]}        
        vi_spikes = np.delete(vi_spikes, np.where(vi_spikes >= dict1['v_t'].shape[0])[0])        
        dict1_v_sp_ = dict1['v_t'][vi_spikes]
         
        if 'Cell' in name:
            for i in range(len(dict1['sweep_time']) - 1):
                dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
        
        precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned\
                            = metric(one_neuron_names[neuron_idx], dict1['sweep_time'], dict1['e_sg'], 
                                  dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                  dict1['v_sg'], dict1_v_sp_ , 
                                  dict1['v_t'], dict1['v_sub'],init_frames=init_frames, save=False)
            
        p = len(e_match)/len(v_spike_aligned)
        r = len(e_match)/len(e_spike_aligned)
        f = (2 / (1 / p + 1 / r))

        f1_scores.append(f)                
        prec.append(p)
        rec.append(r)
        result[neuron_idx] = {'f1':f, 'precision':p, 'recall':r}
    result_all[name] = result
    
    
#%%
SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neuron'
np.save(os.path.join(SAVE_FOLDER, f'viola_F1_v2.1'), result_all)    



#%%
import shutil
ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/overlapping_neurons'
files = sorted(os.listdir(ROOT_FOLDER))
files = [file[:-4] for file in files if 'ROI' not in file and '.tif' in file]

for file in files:
    folder = os.path.join(ROOT_FOLDER, file)
    try:
        os.makedirs(folder)
        print('make folder')
    except:
        print('already exist')
        
for file in files:
    try:
        shutil.move(os.path.join(ROOT_FOLDER, file+'.tif'), os.path.join(ROOT_FOLDER, file))
    except:
        pass
    try:
        shutil.move(os.path.join(ROOT_FOLDER, file+'_ROIs.hdf5'), os.path.join(ROOT_FOLDER, file))
    except:
        pass
    
#%%
names = sorted(os.listdir(ROOT_FOLDER))
for name in names:
    folder = os.path.join(ROOT_FOLDER, name, 'viola')
    try:
        os.makedirs(folder)
        print('make folder')
    except:
        print('already exist')
            