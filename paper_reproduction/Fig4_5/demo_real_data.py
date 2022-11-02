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
from fiola.utilities import metric, normalize, match_spikes_greedy, compute_F1, load, signal_filter
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
import scipy.io
#sys.path.append('/home/nel/Code/NEL_LAB/fiola/use_cases')
#sys.path.append(os.path.abspath('/Users/agiovann/SOFTWARE/fiola'))
from paper_reproduction.Fig4_5.test_run_fiola import run_fiola # must be in use_cases folder

#%%
ROOT_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/overlapping_neurons'
names = sorted(os.listdir(ROOT_FOLDER))

one_neuron_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron'
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
select = np.array(range(len(names)))[np.array([0])]

for num_layers in [1, 3, 5, 10]:
    for idx, name in enumerate(np.array(names)[select]):
        mode = 'voltage'
        num_frames_init = 20000
        num_frames_total = 100000
        border_to_0 = 2
        flip = True
        use_rank_one_nmf=False
        hals_movie='hp_thresh'
        semi_nmf=False
        update_bg = False
        use_spikes= False
        batch_size=1
        center_dims=None
        initialize_with_gpu=False
        do_scale = False
        adaptive_threshold=True
        filt_window=15
        minimal_thresh=3
        step=2500
        template_window=2
        trace_with_neg=False
        nb=0
        ms=[0, 0]
        #num_layers = num_layers
            
        options = {
            'mode': mode, 
            'border_to_0': border_to_0,
            'flip': flip,
            'num_frames_init': num_frames_init, 
            'num_frames_total': num_frames_total, 
            'use_rank_one_nmf': use_rank_one_nmf,
            'hals_movie': hals_movie,
            'semi_nmf':semi_nmf,  
            'update_bg': update_bg,
            'use_spikes':use_spikes, 
            'batch_size':batch_size,
            'initialize_with_gpu':initialize_with_gpu,
            'do_scale': do_scale,
            'adaptive_threshold': adaptive_threshold,
            'filt_window': filt_window,
            'minimal_thresh': minimal_thresh,
            'step': step, 
            'template_window':template_window, 
            'num_layers': num_layers, 
            'nb': nb, 
            'trace_with_neg':trace_with_neg, 
            'ms': ms}
        
        fr = 400.8
        fnames = os.path.join(ROOT_FOLDER, name, name+'.tif')  # files are motion corrected before
        path_ROIs = os.path.join(ROOT_FOLDER, name, name+'_ROIs.hdf5')
        run_fiola(fnames, path_ROIs, fr=fr, options=options)

#%%
select = np.array(range(0))[:]
init_frames = 20000
result_all = {}
method = ['fiola', 'meanroi'][0]

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
        
        if method == 'fiola': 
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
            
        elif method == 'meanroi':
            mov = load(os.path.join(ROOT_FOLDER, name, name+'.tif'))
            spatial = load(os.path.join(ROOT_FOLDER, name, name+'_ROIs.hdf5'))#.squeeze()
            mov = mov.reshape([mov.shape[0], -1], order='F')
            spatial_F = [np.where(sp.reshape(-1, order='F')>0) for sp in spatial]
            t_temporal = np.array([-mov[:, sp].mean((1,2)) for sp in spatial_F])
            t_spatial = spatial
            
            # t_temporal_p = signal_filter(t_temporal, freq=15, fr=400.8)
            # t_temporal_p[:, :30] = 0
            # t_temporal_p[:, -30:] = 0  
            # t_temporal_p = t_temporal_p[idxx:idxx+1]
            # #v_temporal = t_temporal_p.squeeze()             
            # #thresh = 3
            # t_spikes = np.array(extract_spikes(t_temporal_p, threshold=thresh)).squeeze()
            # t_spikes = np.delete(t_spikes, np.where(t_spikes >= dict1['v_t'].shape[0])[0])
            
            
            from fiola.signal_analysis_online import SignalAnalysisOnlineZ
            saoz = SignalAnalysisOnlineZ(mode='voltage', window=10000, step=5000, detrend=True, flip=False,
                                         do_scale=False, template_window=2, robust_std=False, adaptive_threshold=True, fr=400, freq=15, 
                                         minimal_thresh=3.0, online_filter_method = 'median_filter', filt_window = 15, do_plot=False)               
            saoz.fit(t_temporal[:, :init_frames], num_frames=t_temporal.shape[1])
            for n in range(init_frames, t_temporal.shape[1]):
                saoz.fit_next(t_temporal[:, n: n+1], n)
            t_spikes = [np.unique(saoz.index[idxx]) for idxx in range(len(saoz.index))]
            t_spikes = t_spikes[idxx][1:] # remove the first one
            t_spikes = np.delete(t_spikes, np.where(t_spikes >= dict1['v_t'].shape[0])[0])        
            dict1_v_sp_ = dict1['v_t'][t_spikes]
         
        if 'Cell' in name:
            for i in range(len(dict1['sweep_time']) - 1):
                dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
        
        precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned, spnr\
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
init_frames = 20000
result_all = {}
method = ['fiola', 'meanroi'][0]

for num_layers in [1, 3, 5, 10, 30]:
    print(num_layers)
    
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
            
            if method == 'fiola': 
                vi_folder = os.path.join(ROOT_FOLDER, name, 'viola')
                #vi_files = sorted([file for file in os.listdir(vi_folder) if 'v2.1' in file and 'hp_thresh' in file])# and '24000' in file])
                vi_files = [file for file in os.listdir(vi_folder) if f'num_layers_{num_layers}_' in file]
                print(f'files number: {len(vi_files)}')
                if len(vi_files) > 1:
                    print(vi_files)
                    vi_files = [file for file in vi_files if '15000' in file]
                vi_file = vi_files[0]
                vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
                
                #vi_spatial = vi.H.copy()
                vi_temporal = vi.t_s.copy()
                vi_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vi.index])[np.argsort(vi.seq)][idxx]
                thr.append(vi.thresh_factor[0])
                
                n_cells = 1
                vi_result = {'F1':[], 'precision':[], 'recall':[]}        
                rr = {'F1':[], 'precision':[], 'recall':[]}        
                vi_spikes = np.delete(vi_spikes, np.where(vi_spikes >= dict1['v_t'].shape[0])[0])        
                dict1_v_sp_ = dict1['v_t'][vi_spikes]
                
            elif method == 'meanroi':
                mov = load(os.path.join(ROOT_FOLDER, name, name+'.tif'))
                spatial = load(os.path.join(ROOT_FOLDER, name, name+'_ROIs.hdf5'))#.squeeze()
                mov = mov.reshape([mov.shape[0], -1], order='F')
                spatial_F = [np.where(sp.reshape(-1, order='F')>0) for sp in spatial]
                t_temporal = np.array([-mov[:, sp].mean((1,2)) for sp in spatial_F])
                t_spatial = spatial
                
                # t_temporal_p = signal_filter(t_temporal, freq=15, fr=400.8)
                # t_temporal_p[:, :30] = 0
                # t_temporal_p[:, -30:] = 0  
                # t_temporal_p = t_temporal_p[idxx:idxx+1]
                # #v_temporal = t_temporal_p.squeeze()             
                # #thresh = 3
                # t_spikes = np.array(extract_spikes(t_temporal_p, threshold=thresh)).squeeze()
                # t_spikes = np.delete(t_spikes, np.where(t_spikes >= dict1['v_t'].shape[0])[0])
                
                
                from fiola.signal_analysis_online import SignalAnalysisOnlineZ
                saoz = SignalAnalysisOnlineZ(mode='voltage', window=10000, step=5000, detrend=True, flip=False,
                                             do_scale=False, template_window=2, robust_std=False, adaptive_threshold=True, fr=400, freq=15, 
                                             minimal_thresh=3.0, online_filter_method = 'median_filter', filt_window = 15, do_plot=False)               
                saoz.fit(t_temporal[:, :init_frames], num_frames=t_temporal.shape[1])
                for n in range(init_frames, t_temporal.shape[1]):
                    saoz.fit_next(t_temporal[:, n: n+1], n)
                t_spikes = [np.unique(saoz.index[idxx]) for idxx in range(len(saoz.index))]
                t_spikes = t_spikes[idxx][1:] # remove the first one
                t_spikes = np.delete(t_spikes, np.where(t_spikes >= dict1['v_t'].shape[0])[0])        
                dict1_v_sp_ = dict1['v_t'][t_spikes]
             
            if 'Cell' in name:
                for i in range(len(dict1['sweep_time']) - 1):
                    dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
                dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
            
            precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned, spnr\
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
    result_all[num_layers] = result    
    
#%%
SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neuron'
np.save(os.path.join(SAVE_FOLDER, f'fiola_F1_num_layers_v3.0'), result_all)    

#%%
ff = np.zeros((2, 5))
for idx, num_layers in enumerate([1, 3, 5, 10, 30]):
    ff[0, idx] = result_all[num_layers][0]['f1']
    ff[1, idx] = result_all[num_layers][1]['f1']
    

#%%
fig = plt.figure()
ax1 = plt.subplot()
colors=['blue', 'orange']
methods = ['FIOLA', 'meanroi online']
for idx in range(2):
    ax1.bar(x+width*(idx-0.5), [ff.mean(1), ff1.mean(1)][idx], width, color=colors[idx], label=methods[idx])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
#ax1.spines['bottom'].set_visible(False)
ax1.set_ylabel('F1 Score')
ax1.set_xticks(x)
ax1.set_xticklabels(label)
ax1.xaxis.set_ticks_position('none') 
ax1.yaxis.set_tick_params(length=8)
ax1.set_ylim([0.5,1])
ax1.legend(frameon=False)
fig.tight_layout()


#%%
width=0.1
x=np.array([0, 1])
for idx in range(5):
    plt.bar(x + width*(idx-1.5), ff[:, idx], width, label=['1', '3', '5', '10', '30'][idx])
plt.legend()
plt.xticks(x, ['cell1', 'cell2'])
plt.title('real_data_num_iterations')

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
            