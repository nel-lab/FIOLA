#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:48:17 2020
Pipeline for online analysis of voltage imaging data
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
@author: @agiovann, @caichangjia, @cynthia
"""
import h5py
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
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
#
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import normalize, normalize_piecewise, match_spikes_greedy, compute_F1, load, signal_filter, extract_spikes
#sys.path.append('/media/nel/storage/Code/NEL_LAB/fiola/use_cases')
#sys.path.append(os.path.abspath('/Users/agiovann/SOFTWARE/fiola'))
#from use_cases.test_run_fiola import run_fiola # must be in use_cases folder
from paper_reproduction.Fig4_5.test_run_fiola import run_fiola


#%%
mode = ['overlapping', 'non_overlapping', 'positron'][1]
dropbox_folder = '/media/nel/storage/NEL-LAB Dropbox/'
#dropbox_folder = '/Users/agiovann/Dropbox/'

if mode == 'overlapping':
    ROOT_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/data/voltage_data/simulation/overlapping'
    SAVE_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/result/test_simulations/overlapping'
    names = [f'viola_sim3_{i}' for i in range(1, 19)]
    #names = [f'viola_sim3_{i}' for i in range(4, 7)]
    #names = [f'viola_sim6_{i}' for i in range(2, 20, 3)]

elif mode == 'non_overlapping':
    ROOT_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/data/voltage_data/simulation/non_overlapping'
    SAVE_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/result/test_simulations/non_overlapping'
    #names = [f'viola_sim5_{i}' for i in range(1, 8)]
    names = [f'viola_sim5_{i}' for i in range(2, 8, 2)]
    #names = [f'viola_sim7_{i}' for i in range(2, 9)]
elif mode == 'positron':
    ROOT_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/data/voltage_data/simulation/test/sim4_positron'
    SAVE_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/result/test_simulations/test/sim4_positron'
    names = [f'viola_sim4_{i}' for i in range(1, 13)]
   
#%%
for num_layers in [10, 30]:
    mode = 'voltage'
    num_frames_init = 10000
    num_frames_total = 75000
    border_to_0 = 2
    flip = True
    use_rank_one_nmf=False
    hals_movie='hp_thresh'
    semi_nmf=False
    update_bg = True
    use_spikes= False
    batch_size=1
    center_dims=None
    initialize_with_gpu=True
    do_scale = False
    adaptive_threshold=True
    filt_window=15
    minimal_thresh=3
    step=2500
    template_window=2
    trace_with_neg=False
    #num_layers = 3
        
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
        'trace_with_neg':trace_with_neg}
    
    #%%
    for name in names:
        fnames = os.path.join(ROOT_FOLDER, name, name+'.hdf5')
        print(f'NOW PROCESSING: {fnames}')
        path_ROIs = os.path.join(ROOT_FOLDER, name, 'viola', 'ROIs_gt.hdf5')
        run_fiola(fnames, path_ROIs, fr=400, options=options)
    
#%%  
# for num_layers in [1, 3, 5, 10, 30]:
#     for trace_with_neg in [True, False]:
#distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
#names = [f'viola_sim3_{i}' for i in range(1, 19)]
#names = [f'viola_sim5_{i}' for i in range(1, 8)]
#names = [f'viola_sim6_{i}' for i in range(2, 20, 3)]
#names = [f'viola_sim7_{i}' for i in range(2, 9)]
test = []
t_range = [10000, 75000]
for idx, dist in enumerate(distance):
    #if idx == 1:
    vi_result_all = []
    spnr_all = []
    if mode == 'overlapping':
        #select = np.arange(idx * 3, (idx + 1) * 3)
        select = np.array([idx])
    else:
        select = np.array(range(len(names)))
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
        gt_file = gt_files[0]
        gt = scipy.io.loadmat(os.path.join(folder, gt_file))
        gt = gt['simOutput'][0][0]['gt']
        spikes = gt['ST'][0][0][0]
        
        vi_folder = os.path.join(folder, 'viola')
        vi_files = sorted([file for file in os.listdir(vi_folder) if 'v3.1' in file and '30_' in file])
        #vi_files = sorted([file for file in os.listdir(vi_folder) if 'v3.0' in file and 'layers_1' in file])
        #vi_files = sorted([file for file in os.listdir(vi_folder) if 'v3.0' in file and f'layers_{num_layers}_' in file and f'trace_with_neg_{trace_with_neg}' in file])
        if len(vi_files) > 1:
            raise Exception('number of files greater than one')
        vi_file = vi_files[0]
        vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
        
        #vi_spatial = vi.H.copy()
        vi_temporal = vi.t_s.copy()
        vi_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vi.index])[np.argsort(vi.seq)]
        
        n_cells = spikes.shape[0]
        vi_result = {'F1':[], 'precision':[], 'recall':[]}        
        rr = {'F1':[], 'precision':[], 'recall':[]}
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s1 = s1 - 1     # matlab first element is 1
            s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
            s2 = vi_spikes[idx]
            s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        vi_result_all.append(rr)
        
        spnr = []
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s1 = s1 - 1
            s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
            s2 = vi_spikes[idx]
            s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
            t2 = vi_temporal[idx]
            t2 = normalize_piecewise(t2, 5000)
            ss = np.mean(t2[s2])
            spnr.append(ss)        
        spnr_all.append(spnr)

    folder_name = vi_file[:-4]
    try:
        os.makedirs(os.path.join(SAVE_FOLDER, folder_name))
        print('make folder')
    except:
        print('already exist')
    np.save(os.path.join(SAVE_FOLDER, folder_name, f'viola_result_{t_range[0]}_{t_range[1]}'), vi_result_all)
    np.save(os.path.join(SAVE_FOLDER, folder_name, f'viola_result_spnr_{t_range[0]}_{t_range[1]}_v3.0'), spnr_all)
    print(len(vi_result_all))
    np.save(os.path.join(SAVE_FOLDER, folder_name,  f'viola_result_{t_range[0]}_{t_range[1]}_{dist}'), vi_result_all)
    ##np.save(os.path.join(SAVE_FOLDER, folder_name, f'viola_result_spnr_{t_range[0]}_{t_range[1]}_v2.0_without_shrinking'), spnr_all)
    test.append(np.mean(vi_result_all[0]['F1']))

#%%
hh = vi.H.reshape((100,100,9), order='F')   
plt.imshow(hh[:,:,3]) 

plt.plot(vi_temporal[9])   

#%%

m = load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/non_overlapping/viola_sim5_7/viola_sim5_7.hdf5')
cm.movie(m).save('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/non_overlapping/viola_sim5_7/viola_sim5_7.tiff')


#%%
#'t_' means tested algorithm
a_idx = 2 # algorithm idx
algo = ['volpy', 'suite2p', 'meanroi', 'meanroi_online'][a_idx]
thresh_list = np.arange(2.0, 4.1, 0.1)
best_thresh = [False, True, True, False][a_idx]

if mode == 'overlapping':
    t_range = [10000, 20000]
    names = [f'viola_sim3_{i}' for i in range(1, 19)]
    distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
elif mode == 'non_overlapping':
    t_range = [10000, 75000]
    names = [f'viola_sim5_{i}' for i in range(1, 8)]
    distance = [0]
    
save_folder = os.path.join(SAVE_FOLDER, f'{algo}_result_v3.1')
try:
    os.makedirs(save_folder)
    print('make folder')
except:
    print('already exist')

for ii, dist in enumerate(distance):  
    t_result_all = []
    spnr_all = []
    best_thresh_all = []

    if mode == 'overlapping':
        select = np.arange(ii * 3, (ii + 1) * 3)
    else:
        select = np.array(range(len(names)))
    
    for name in np.array(names)[select]:
        print(f'now processing {name}')
        folder = os.path.join(ROOT_FOLDER, name)
        gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
        gt_file = gt_files[0]
        gt = scipy.io.loadmat(os.path.join(folder, gt_file))
        gt = gt['simOutput'][0][0]['gt']
        spikes = gt['ST'][0][0][0]
        temporal = gt['trace'][0][0][:,:,0]
        path_ROIs = os.path.join(folder, 'viola', 'ROIs_gt.hdf5')
        with h5py.File(path_ROIs,'r') as h5:
            spatial = np.array(h5['mov'])
        plt.imshow(spatial.sum(0))
                
        #summary_file = os.listdir(os.path.join(folder, 'volpy'))
        #summary_file = [file for file in summary_file if 'summary' in file][0]
        #summary = cm.load(os.path.join(folder, 'volpy', summary_file))
        
        if algo == 'meanroi' or 'meanroi_online':
            movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
            mov = load(os.path.join(folder, movie_file))
            mov = mov.reshape([mov.shape[0], -1], order='F')
            spatial_F = [np.where(sp.reshape(-1, order='F')>0) for sp in spatial]
            t_temporal = np.array([-mov[:, sp].mean((1,2)) for sp in spatial_F])
            t_spatial = spatial

            """
            idx = 2
            plt.plot(m_temporal[idx,:1000])
            spk = spikes[idx][spikes[idx]<1000]
            plt.vlines(spk, ymin=m_temporal[idx,:1000].max(), ymax=m_temporal[idx,:1000].max()+10 )
            plt.savefig(os.path.join(SAVE_FOLDER, 'eg_trace_amplitude_0.15.pdf'))
            """
     
        elif algo == 'volpy':
            t_folder = os.path.join(folder, 'volpy')
            t_files = sorted([file for file in os.listdir(t_folder) if 'adaptive_threshold' in file])
            t_file = t_files[0]
            t = np.load(os.path.join(t_folder, t_file), allow_pickle=True).item()
            t_spatial = t['weights'].copy()
            t_temporal = t['ts'].copy()
            t_ROIs = t['ROIs'].copy()
            t_ROIs = t_ROIs * 1.0
            t_templates = t['templates'].copy()
            t_spikes = t['spikes'].copy()
        
        if not best_thresh:
            print('not finding best thresh')
            n_cells = t_spatial.shape[0]
            t_result = {'F1':[], 'precision':[], 'recall':[]}        
            rr = {'F1':[], 'precision':[], 'recall':[]}
            
            if algo == 'meanroi_online':
                from fiola.signal_analysis_online import SignalAnalysisOnlineZ
                saoz = SignalAnalysisOnlineZ(mode='voltage', window=10000, step=5000, detrend=True, flip=False,
                                             do_scale=False, template_window=2, robust_std=False, adaptive_threshold=True, fr=400, freq=15, 
                                             minimal_thresh=3.0, online_filter_method = 'median_filter', filt_window = 15, do_plot=False)               
                saoz.fit(t_temporal[:, :t_range[0]], num_frames=t_temporal.shape[1])
                for n in range(t_range[0], t_temporal.shape[1]):
                    saoz.fit_next(t_temporal[:, n: n+1], n)
                t_spikes = [np.unique(saoz.index[idxx]) for idxx in range(len(saoz.index))]

            for idx in range(n_cells):
                s1 = spikes[idx].flatten()
                s1 = s1 - 1
                s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
                s2 = t_spikes[idx]
                s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
                idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
                F1, precision, recall= compute_F1(s1, s2, idx1_greedy, idx2_greedy)
                rr['F1'].append(F1)
                rr['precision'].append(precision)
                rr['recall'].append(recall)
            t_result_all.append(rr)
        else:
            print('find best thresh')
            from scipy.signal import medfilt
            n_cells = t_spatial.shape[0]
            t_result = {'F1':[], 'precision':[], 'recall':[]}        
            t_result_0 = {'F1':[], 'precision':[], 'recall':[]}        
            rr = {'F1':[], 'precision':[], 'recall':[]}
            for thresh in thresh_list:
                print(thresh)
                rr = {'F1':[], 'precision':[], 'recall':[]}
                t_temporal_p = np.array([tt - medfilt(tt, kernel_size=15) for tt in t_temporal])
                t_spikes = extract_spikes(t_temporal_p, threshold=thresh)
                
                for idx in range(n_cells):
                    s1 = spikes[idx].flatten()    
                    s1 = s1 - 1
                    s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
                    s2 = t_spikes[idx]
                    s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
                    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
                    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
                    rr['F1'].append(F1)
                    rr['precision'].append(precision)
                    rr['recall'].append(recall)
                if not t_result['F1']:
                    t_result = rr
                    best_thresh = thresh
                    new_F1 = np.array(rr['F1']).sum()/n_cells
                    print(f'update: thresh:{thresh}, best_F1:{new_F1}, new_F1:{new_F1}')
                else:
                    best_F1 = np.array(t_result['F1']).sum()/n_cells
                    new_F1 = np.array(rr['F1']).sum()/n_cells
                    #print(new_F1)
                    if new_F1 > best_F1:
                        print(f'update: thresh:{thresh}, best_F1:{best_F1}, new_F1:{new_F1}')
                        t_result = rr
                        best_thresh = thresh
                        
                # for idx in range(n_cells):
                #     s1 = spikes[idx].flatten().copy()    
                #     s1 = s1 - 1
                #     s1 = s1[s1<=t_range[0]]
                #     s2 = t_spikes[idx].copy()
                #     s2 = s2[s2<=t_range[0]]
                #     idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
                #     F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
                #     rr0['F1'].append(F1)
                #     rr0['precision'].append(precision)
                #     rr0['recall'].append(recall)
                    
                #     s1 = spikes[idx].flatten().copy()    
                #     s1 = s1 - 1
                #     s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
                #     s2 = t_spikes[idx].copy()
                #     s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
                #     idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
                #     F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
                #     rr['F1'].append(F1)
                #     rr['precision'].append(precision)
                #     rr['recall'].append(recall)

                # if not t_result['F1']:
                #     t_result = rr
                #     t_result_0 = rr0
                #     best_thresh = thresh
                #     new_F1 = np.array(rr0['F1']).sum()/n_cells
                #     print(f'update: thresh:{thresh}, best_F1:{new_F1}, new_F1:{new_F1}')
                # else:
                #     best_F1 = np.array(t_result_0['F1']).sum()/n_cells
                #     new_F1 = np.array(rr0['F1']).sum()/n_cells
                #     print(f'update: thresh:{thresh}, best_F1:{best_F1}, new_F1:{new_F1}')
                #     #print(new_F1)
                #     if new_F1 > best_F1:   
                #         print('***')
                #         print(f'current F1: {np.array(t_result["F1"]).sum()/n_cells}')
                #         t_result = rr
                #         t_result_0 = rr0
                #         best_thresh = thresh
                #t_result.append(np.array(rr['F1']).sum()/n_cells)
            t_result_all.append(t_result)
        #best_thresh_all.append(best_thresh)
        
        if mode == 'non_overlapping':
            spnr = []
            for idx in range(n_cells):
                s1 = spikes[idx].flatten()
                s1 = s1 - 1
                s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
                t2 = t_temporal_p[idx]
                t2 = normalize_piecewise(t2, 5000)
                ss = np.mean(t2[s1])
                spnr.append(ss)        
            spnr_all.append(spnr)
    
    # t_result_all = np.array(t_result_all)
    # t_result_all = t_result_all[:, np.argmax(t_result_all.mean(0))]
    
    
    if mode == 'non_overlapping':
        np.save(os.path.join(save_folder, f'{algo}_F1_{t_range[0]}_{t_range[1]}_v3.1'), t_result_all)
        np.save(os.path.join(save_folder, f'{algo}_spnr_{t_range[0]}_{t_range[1]}_v3.1'), spnr_all)    
    elif mode == 'overlapping':
        np.save(os.path.join(save_folder, f'{algo}_F1_{t_range[0]}_{t_range[1]}_{dist}'), t_result_all)
    
#%%
plt.plot(t_temporal_p[0])
plt.vlines(t_spikes[0], 500, 1000, color='black')


##################################################################################################################
##################################################################################################################
#%% Fig 4a
from caiman.utils.visualization import plot_contours

mm = cm.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/overlapping/viola_sim3_6/viola_sim3_6.hdf5')
ROIs = cm.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/overlapping/viola_sim3_6/viola/ROIs_gt.hdf5')
plt.imshow(mm.mean(0), cmap='gray')
plt.scatter(centers[:, 1], centers[:, 0], color='r', s=1)
from caiman.base.rois import com
r = ROIs.reshape([ROIs.shape[0], -1], order='F').T
centers = com(r, d1=100, d2=100)
plot_contours(r, mm.mean(0), cmap='gray', display_numbers=False)
plt.axis('off')
plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/overlapping_eg.pdf')





#%% Fig 4c, F1 score
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
VIOLA_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' not in file]
VOLPY_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
volpy_files = os.listdir(VOLPY_FOLDER)
volpy_files = [os.path.join(VOLPY_FOLDER, file) for file in volpy_files if '.npy' in file and 'spnr' not in file and 'v2.0' in file]
VIOLA_FOLDER1 = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_[8, 4]_minimal_thresh_3_template_window_2_v2.1'
v1 = os.listdir(VIOLA_FOLDER1)
v1 = [os.path.join(VIOLA_FOLDER1, file) for file in v1 if 'spnr' not in file]
VIOLA_FOLDER2 = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_[8, 4]_minimal_thresh_3_template_window_0_v2.1'
v2 = os.listdir(VIOLA_FOLDER2)
v2 = [os.path.join(VIOLA_FOLDER2, file) for file in v2 if 'spnr' not in file]
folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/meanroi_result_v3.1'
m = os.listdir(folder)
m = [os.path.join(folder, file) for file in m if '.npy' in file and 'meanroi' in file and 'spnr' not in file and '3.1' in file]
#m1 = os.listdir(folder)
#m1 = [os.path.join(folder, file) for file in m1 if '.npy' in file and 'meanroi' in file and 'spnr' not in file and 'best_thresh_False' in file]

files = viola_files+volpy_files + v1 + v2 + m 
result_all = [np.load(file, allow_pickle=True) for file in files]

for idx, results in enumerate(result_all):
    try:
        #if idx == 0:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15', linewidth = 3)
        #else:
        #    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15')
    except:
        if idx == 1:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='^', markersize=9, linestyle=':')    
        elif idx == 6:
            plt.plot(x, results, marker='^', markersize=9, linestyle=':')   
        elif idx == 4:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='^', markersize=9, linestyle=':')    
        else:    
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='.', markersize=15)    

    plt.legend(['FIOLA_25ms', 'VolPy', 'FIOLA_17.5ms', 'FIOLA_12.5ms', 'MeanROI', 'MeanROIOnline'], frameon=False)
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score for minimally overlapping neurons')
    plt.ylim([0, 1])
    plt.xticks([0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=8)
    ax.yaxis.set_tick_params(length=8)

ff = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig4'
plt.savefig(os.path.join(ff, f'Fig4c.pdf'))

#%% Fig 4d, SPNR
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
VIOLA_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' in file]
VOLPY_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
volpy_files = os.listdir(VOLPY_FOLDER)
volpy_files = [os.path.join(VOLPY_FOLDER, file) for file in volpy_files if '.npy' in file and 'spnr' in file and 'v2.0' in file]
VIOLA_FOLDER1 = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_[8, 4]_minimal_thresh_3_template_window_2_v2.1'
v1 = os.listdir(VIOLA_FOLDER1)
v1 = [os.path.join(VIOLA_FOLDER1, file) for file in v1 if 'spnr' in file]
VIOLA_FOLDER2 = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_[8, 4]_minimal_thresh_3_template_window_0_v2.1'
v2 = os.listdir(VIOLA_FOLDER2)
v2 = [os.path.join(VIOLA_FOLDER2, file) for file in v2 if 'spnr' in file]
folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/meanroi_result_v3.1'
m = os.listdir(folder)
m = [os.path.join(folder, file) for file in m if 'meanroi' in file and 'spnr' in file]
files = viola_files+volpy_files+v1+v2+m
result_all = [np.load(file, allow_pickle=True) for file in files]

for idx, results in enumerate(result_all):
    if idx == 0 or idx == 2 or idx == 3:
        plt.plot(x, [np.array(result).sum()/len(result) for result in results],  marker='.', markersize=15)
    else:
        plt.plot(x, [np.array(result).sum()/len(result) for result in results],  marker='^', markersize=9, linestyle=':')
                 
    plt.legend(['FIOLA_25ms', 'VolPy', 'FIOLA_17.5ms', 'FIOLA_12.5ms', 'MeanROI'], frameon=False)
    plt.xlabel('spike amplitude')
    plt.ylabel('SPNR')
    plt.title('SPNR for minimally overlapping neurons')
    plt.xticks([0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=8)
    ax.yaxis.set_tick_params(length=8)

ff = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig4'
plt.savefig(os.path.join(ff, f'Fig4d.pdf'))

#%% Fig4e
fig = plt.figure()
ax1 = plt.subplot(111)
xx = np.arange(1, 6)
vi = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping_sim7/fiola_result_num_layers_30_trace_with_neg_False_v3.0/viola_result_10000_75000.npy', allow_pickle=True)
me = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping_sim7/meanroi_result_v3.0/meanroi_F1_10000_75000_v3.1.npy', allow_pickle=True)
#vi = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping_sim7/fiola_result_num_layers_30_trace_with_neg_False_v3.0/viola_result_spnr_10000_75000_v3.0.npy', allow_pickle=True)
#me = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping_sim7/meanroi_result_v3.0/meanroi_spnr_10000_75000_v3.1.npy', allow_pickle=True)
select = np.array([1, 2, 3, 4, 6])
rr = [np.array([np.mean(v['F1']) for v in vi])[select], np.array([np.mean(v['F1']) for v in me])[select]]
#rr = [[np.mean(v) for v in vi], [np.mean(v) for v in me]]
ax1.bar(xx-0.1, rr[0], width=0.2, label='FIOLA')
ax1.bar(xx+0.1, rr[1], width=0.2, color='C4', label='MeanROI')
ax1.set_xlabel('number of neurons in the FOV')
ax1.set_ylabel('F1 score')
ax1.set_ylim([0.5,1])
ax1.set_xticks(xx)
lab = np.array(['3', '5', '10', '20', '30', '40', '50'])[select]
ax1.set_xticklabels(lab)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.xaxis.set_tick_params(length=8)
ax1.yaxis.set_tick_params(length=8)
ax1.legend(frameon=False)

ff = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig4'
plt.savefig(os.path.join(ff, f'Fig4e.pdf'))

#%% Fig 4f. F1 score vs Number of iterations
fig = plt.figure()
ax2 = plt.subplot(111)
trace_with_neg = False
methods = ['layers_3', 'layers_5', 'layers_10', 'layers_30']
v3 = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/fiola_result_num_layers_3_trace_with_neg_False_v3.1/viola_result_10000_75000.npy', allow_pickle=True)
v5 = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/fiola_result_num_layers_5_trace_with_neg_False_v3.1/viola_result_10000_75000.npy', allow_pickle=True)
v10 = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/fiola_result_num_layers_10_trace_with_neg_False_v3.1/viola_result_10000_75000.npy', allow_pickle=True)
v30 = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/fiola_result_num_layers_30_trace_with_neg_False_v3.1/viola_result_10000_75000.npy', allow_pickle=True)
v = [v3, v5, v10, v30]

xx = np.array([0, 1, 2])
#methods = ['viola', 'layer_1', 'layer_3']
colors = ['b', 'orange', 'g', 'r', 'black']
for idx, method in enumerate(methods):

    ax2.bar(xx + 0.1 * (idx-2), [np.mean(r['F1']) for r in v[idx]], width=0.1, #yerr=[np.std(r['F1']) for r in results], 
            color=colors[idx])
     
ax2.set_xlabel('spike amplitude')
ax2.set_ylabel('F1 score')
ax2.set_ylim([0.2,1])
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['0.075', '0.125', '0.175'])
ax2.legend(methods, frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.xaxis.set_tick_params(length=8)
ax2.yaxis.set_tick_params(length=8)
plt.tight_layout()
ff = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig4'
plt.savefig(os.path.join(ff, f'Fig4f_50neurons.pdf'))

#%%
trace_with_neg = False
methods = ['layers_3', 'layers_5', 'layers_10', 'layers_30']
result_all = {}
for m in methods:
    result_all[m] = {} 
distance = [f'dist_{i}' for i in [3]]
x = [round(0.075 + 0.05 * i, 3) for i in range(3)] 
for method in methods:
    #f = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/'
    f = 
    load_folder = [ff for ff in os.listdir(f) if method+'_' in ff and 'v3.0' in ff and f'trace_with_neg_{trace_with_neg}' in ff]
    print(len(load_folder))
    load_folder = f+load_folder[0]

    files = np.array(sorted(os.listdir(load_folder)))#[0]#[np.array([5, 0, 1, 2, 3, 4, 6])]
    #files = [file for file in files if method in file and dist+'.npy' in file]
    #files = [file for file in files if dist+'.npy' in file]
    print(len(files))
    for file in files:
        if file == 'viola_result_10000_20000_dist_3.npy':
            result_all[method][file] = np.load(os.path.join(load_folder, file), allow_pickle=True)

#
#fig = plt.figure()
#ax = plt.subplot()
#xx = np.array(range(len(methods)))
xx = np.array([0, 1, 2])
#methods = ['viola', 'layer_1', 'layer_3']
colors = ['b', 'orange', 'g', 'r', 'black']
for idx, method in enumerate(methods):
    results = result_all[method]
    key = list(results.keys())[0]
    results = results[key]
    #print(results)
    ax2.bar(xx + 0.1 * (idx-2), [np.mean(r['F1']) for r in results], width=0.1, #yerr=[np.std(r['F1']) for r in results], 
            color=colors[idx])
     
ax2.set_xlabel('spike amplitude')
ax2.set_ylabel('F1 score')
ax2.set_ylim([0.5,1])
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['0.075', '0.125', '0.175'])
ax2.legend(methods, frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.xaxis.set_tick_params(length=8)
ax2.yaxis.set_tick_params(length=8)
plt.tight_layout()
#plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/overlapping_v3.1.pdf')

#%% Fig4h. Overlapping neurons for volpy with different overlapping areas
area = []
for idx, dist in enumerate(distance):  
    select = np.arange(idx * 3, (idx + 1) * 3)
    v_result_all = []
    #names = [f'sim3_{i}' for i in range(10, 17)]
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
        gt_file = gt_files[0]
        gt = scipy.io.loadmat(os.path.join(folder, gt_file))
        gt = gt['simOutput'][0][0]['gt']
        spikes = gt['ST'][0][0][0]
        spatial = gt['IM2'][0][0]
        temporal = gt['trace'][0][0][:,:,0]
        spatial[spatial <= np.percentile(spatial, 99.5)] = 0 # just empirical
        spatial[spatial > 0] = 1
        spatial = spatial.transpose([2, 0, 1])
        percent = (np.where(spatial.sum(axis=0) > 1)[0].shape[0])/(np.where(spatial.sum(axis=0) > 0)[0].shape[0])
        area.append(percent)   
        
area = np.round(np.unique(area), 2)[::-1]
area = np.append(area, 0)

#%%
#fig = plt.figure(figsize=(12, 8))
result_all = {'volpy':{}, 'viola':{}, 'meanroi':{}, 'meanroi_online':{}, 'layer_3':{}, 'layer_1':{}}
distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
x = [round(0.075 + 0.05 * i, 3) for i in range(3)] 
for method in ['viola', 'volpy', 'meanroi', 'meanroi_online', 'layer_3', 'layer_1']:
    if method == 'viola':
        SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/fiola_result_num_layers_30_trace_with_neg_False_v3.0'
    elif method == 'volpy':
        SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/volpy_result'
    elif method == 'meanroi':
        SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/meanroi_result_v3.1'
    elif method == 'meanroi_online':
        SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/meanroi_online_result'
    elif method == 'layer_3':
        SAVE_FOLDER ='/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/fiola_result_num_layers_3_v3.0'
    elif method == 'layer_1':
        SAVE_FOLDER ='/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/fiola_result_num_layers_1_v3.0'
    
    for dist in distance:   
        files = np.array(sorted(os.listdir(SAVE_FOLDER)))#[0]#[np.array([5, 0, 1, 2, 3, 4, 6])]
        #files = [file for file in files if method in file and dist+'.npy' in file]
        files = [file for file in files if dist+'.npy' in file]
        print(len(files))
        for file in files:
            result_all[method][file] = np.load(os.path.join(SAVE_FOLDER, file), allow_pickle=True)

xx = np.array([0, 1, 2])
methods = ['viola', 'volpy', 'meanroi']
colors = ['blue', 'orange', 'green', 'purple', 'red', 'black']

#%%
fig = plt.figure()
#ax = plt.subplot(131)
ax1 = plt.subplot(111)
#ax2 = plt.subplot(133)
xx = np.array([3, 2, 1, 0, 0])
methods = ['viola', 'volpy', 'meanroi']
for method in methods:
    for idx, key in enumerate(result_all[method].keys()):
        if idx in [0, 1, 2, 4]:
            results = result_all[method][key]
            #print(results)
            if method in key:
                if method == 'viola':
                    ax1.bar(xx[idx]-0.2 , [np.mean(r['F1']) for r in results][1], width=0.2,
                           #yerr=[np.std(r['F1']) for r in results],  
                             color='C0')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                elif method == 'volpy':
                    ax1.bar(xx[idx], [np.mean(r['F1']) for r in results][1], width=0.2,
                           #yerr=[np.std(r['F1']) for r in results], 
                             color='C1')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                # elif method == 'meanroi':
                #     # ax1.plot(x, [np.array(r['F1']).sum()/len(r['F1']) for r in results], 
                #     #           marker='v',markersize=6, linestyle='-.', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                #     ax1.plot(x, results, 
                #               marker='v',markersize=6, linestyle='-.', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                elif method == 'meanroi':
                    ax1.bar(xx[idx]+0.2, [np.mean(r['F1']) for r in results][1], width=0.2,
                           #yerr=[np.std(r['F1']) for r in results], 
                             color='C4')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                
ax1.set_xlabel('overlapping area')
ax1.set_ylabel('F1 score')
ax1.set_ylim([0.5,1])
ax1.set_xticks([0, 1, 2, 3])
lab = ['0%', '16%', '36%', '63%']
ax1.set_xticklabels(lab)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.xaxis.set_tick_params(length=8)
ax1.yaxis.set_tick_params(length=8)

ff = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig4'
plt.savefig(os.path.join(ff, f'Fig4h.pdf'))

#%%
fig, axs = plt.subplots(1, 3)
sp = [0.075, 0.125, 0.175]

for i in range(3):
    xx = np.array([3, 2, 1, 0, 0])
    methods = ['viola', 'volpy', 'meanroi']
    for method in methods:
        for idx, key in enumerate(result_all[method].keys()):
            if idx in [0, 1, 2, 4]:
                results = result_all[method][key]
                #print(results)
                if method in key:
                    if method == 'viola':
                        axs[i].bar(xx[idx]-0.2 , [np.mean(r['F1']) for r in results][i], width=0.2,
                               #yerr=[np.std(r['F1']) for r in results],  
                                 color='C0', label='FIOLA')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                    elif method == 'volpy':
                        axs[i].bar(xx[idx], [np.mean(r['F1']) for r in results][i], width=0.2,
                               #yerr=[np.std(r['F1']) for r in results], 
                                 color='C1', label='VolPy')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                    # elif method == 'meanroi':
                    #     # axs[i].plot(x, [np.array(r['F1']).sum()/len(r['F1']) for r in results], 
                    #     #           marker='v',markersize=6, linestyle='-.', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                    #     axs[i].plot(x, results, 
                    #               marker='v',markersize=6, linestyle='-.', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                    elif method == 'meanroi':
                        axs[i].bar(xx[idx]+0.2, [np.mean(r['F1']) for r in results][i], width=0.2,
                               #yerr=[np.std(r['F1']) for r in results], 
                                 color='C4', label='MeanROI')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                    
    
    axs[i].set_ylim([0.3,1])
    axs[i].set_xticks([0, 1, 2, 3])
    lab = ['0%', '16%', '36%', '63%']
    axs[i].set_xticklabels(lab)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].xaxis.set_tick_params(length=8)
    axs[i].yaxis.set_tick_params(length=8)
    axs[i].set_title(f'spike amplitude {sp[i]}')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys(), frameon=False)
    #axs[i].xaxis.set_visible(False)
    
    if i == 0:
        axs[i].set_ylabel('F1 score')
        axs[i].set_xlabel('overlapping area')
        axs[i].set_title(f'{sp[i]}')
        
    else:
        axs[i].yaxis.set_ticklabels([])
        axs[i].set_title(f'{sp[i]}')
ff = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig4'
plt.savefig(os.path.join(ff, f'supp.pdf'))


#%%
fig = plt.figure()
ax = plt.subplot()
xx = np.array([0, 1, 2])
methods = ['viola', 'layer_1', 'layer_3']

for method in methods:
    for idx, key in enumerate(result_all[method].keys()):
        if idx in [1]:
            results = result_all[method][key]
            #print(results)
            if method == 'viola':
                ax.bar(xx-0.2 , [np.mean(r['F1']) for r in results], width=0.2,
                       yerr=[np.std(r['F1']) for r in results],  
                         color='blue')
            elif method == 'layer_3':
                ax.bar(xx, [np.mean(r['F1']) for r in results], width=0.2,
                        yerr=[np.std(r['F1']) for r in results], 
                          color='orange')
            elif method == 'layer_1':
                ax.bar(xx+0.2, [np.mean(r['F1']) for r in results], width=0.2,
                        yerr=[np.std(r['F1']) for r in results], 
                          color='green')
                
ax.set_xlabel('spike amplitude')
ax.set_ylabel('F1 score')
ax.set_ylim([0.35,1])
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['0.075', '0.125', '0.175'])
ax.legend(methods, frameon=False)
#ax.set_title(f'Fiola vs VolPy F1 score with different overlapping areas')
#ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(length=8)
ax.yaxis.set_tick_params(length=8)

           
#SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'

#SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
#plt.savefig(os.path.join(SAVE_FOLDER, f'F1_overlapping_fiola&volpy.pdf'))
#%%
fig = plt.figure()
#ax = plt.subplot(131)
ax1 = plt.subplot(111)
#ax2 = plt.subplot(133)
xx = np.array([3, 2, 1, 0, 0])
methods = ['viola', 'volpy', 'meanroi', 'viola_long']
for method in methods:
    if 'viola_long' not in method:
        for idx, key in enumerate(result_all[method].keys()):
            if idx in [0, 1, 2, 4]:
                results = result_all[method][key]
                #print(results)
                if method in key:
                    if method == 'viola':
                        ax1.bar(xx[idx]-0.2 , [np.mean(r['F1']) for r in results][1], width=0.2,
                               #yerr=[np.std(r['F1']) for r in results],  
                                 color='blue')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                    elif method == 'volpy':
                        ax1.bar(xx[idx]+0.2, [np.mean(r['F1']) for r in results][1], width=0.2,
                               #yerr=[np.std(r['F1']) for r in results], 
                                 color='orange')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                    # elif method == 'meanroi':
                    #     # ax1.plot(x, [np.array(r['F1']).sum()/len(r['F1']) for r in results], 
                    #     #           marker='v',markersize=6, linestyle='-.', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                    #     ax1.plot(x, results, 
                    #               marker='v',markersize=6, linestyle='-.', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                    elif method == 'meanroi':
                        ax1.bar(xx[idx]+0.4, [np.mean(r['F1']) for r in results][1], width=0.2,
                               #yerr=[np.std(r['F1']) for r in results], 
                                 color='green')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
    else:
        for idx, key in enumerate(result_all['viola'].keys()):
            if idx in [0, 1, 2, 4]:
                ax1.bar(xx[idx], test[idx], width=0.2,
                       #yerr=[np.std(r['F1']) for r in results], 
                         color='purple')#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                
                
ax1.set_xlabel('overlapping area')
ax1.set_ylabel('F1 score')
ax1.set_ylim([0.5,1])
ax1.set_xticks([0, 1, 2, 3])
lab = ['0%', '16%', '36%', '63%']
ax1.set_xticklabels(lab)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.xaxis.set_tick_params(length=8)
ax1.yaxis.set_tick_params(length=8)



#%% Supplementary figure
mean = []
std = []
f = []
v = []
f1 = []
f2 = []
for idx, results in enumerate(result_all):
    mean.append([np.array(result['F1']).sum()/len(result['F1']) for result in results])
    std.append([np.array(result['F1']).std() for result in results])
    if idx == 0:
        f.append([np.array(result['F1']) for result in results])
    if idx == 1:
        v.append([np.array(result['F1']) for result in results])
    if idx == 2:
        f1.append([np.array(result['F1']) for result in results])
    if idx == 3:
        f2.append([np.array(result['F1']) for result in results])
        
rr = [f,v,f1,f2]

#%%
sig = []
for amp in range(7):
    temp = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if i < j:
                #temp[i, j] = stats.ttest_ind(rr[i][0][amp], rr[j][0][amp], equal_var=False).pvalue
                temp[i, j] = stats.wilcoxon(rr[i][0][amp], rr[j][0][amp]).pvalue
    sig.append(temp)
#%%   
mean = np.array(mean)
std = np.array(std)

labels = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()
width = 0.35  # the width of the bars
rects1 = ax.bar(x - width/2, mean[0], width, yerr=[[0]*7, std[0]], capsize=5, label=f'FIOLA')
rects2 = ax.bar(x + width/2, mean[1], width, yerr=[[0]*7, std[1]], capsize=5, label=f'VolPy')
#rects3 = ax.bar(x  , mean[2], width/4, yerr=[[0]*7, std[2]], capsize=5, label=f'FIOLA_17.5ms')
#rects4 = ax.bar(x + width/4, mean[3], width/4, yerr=[[0]*7, std[3]], capsize=5, label=f'FIOLA_12.5ms')
for i in range (7):
    temp = sig[i][0,1]
    if temp > 0.05:
        string = 'ns'
    elif temp <=0.001:
        string = '***'
    elif temp <= 0.01:
        string = '**'
    elif temp <= 0.05:
        string = '*'
    barplot_annotate_brackets(0, 1, string, [i-width/2, i+width/2], 
                              [mean[0,i], mean[1,i]], yerr=[std[0,i], std[1,i]])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Spike amplitude')
ax.set_ylabel('F1 score')
ax.legend(ncol=2, frameon=False, loc=0)
#ax.xaxis.set_ticks_position('none') 
#ax.yaxis.set_ticks_position('none') 
#ax.set_yticks([])
#ax.set_ylim([0,1])
fig.tight_layout()
plt.show()
plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1/supp/suppl_simulation_f1_significance_wilcoxon.png')

#%%
def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)




#%% 

               
  
                
                
                
    
    
#%%
from scipy import stats
rvs1 = stats.norm.rvs(loc=7,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)
stats.ttest_ind(rvs1, rvs2)
plt.boxplot(rvs1)
plt.boxplot(rvs2)

#stats.ttest_ind(volpy,viola2,  equal_var = False)



#%%
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
VIOLA_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if '.npy' in file and 'spnr' not in file]
VOLPY_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
volpy_files = os.listdir(VOLPY_FOLDER)
volpy_files = [os.path.join(VOLPY_FOLDER, file) for file in volpy_files if '.npy' in file and 'spnr' not in file and 'v2.0' in file]
files = viola_files+volpy_files
result_all = [np.load(file, allow_pickle=True) for file in files]

for idx, results in enumerate(result_all):
    try:
        #if idx == 0:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15', linewidth = 3)
        #else:
        #    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15')
    except:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='.', markersize='15')    

    plt.legend(['Viola','Viola_without_decreasing_threshold', 'VolPy'])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score for non-overlapping neurons')
    plt.ylim([0, 1])
    plt.xticks([0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=8)
    ax.yaxis.set_tick_params(length=8)

#plt.savefig(os.path.join(SAVE_FOLDER, f'F1_Viola_vs_VolPy_{t_range[0]}_{t_range[1]}_new.pdf'))




#%% SPNR
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
VIOLA_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' in file]
VOLPY_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
volpy_files = os.listdir(VOLPY_FOLDER)
volpy_files = [os.path.join(VOLPY_FOLDER, file) for file in volpy_files if '.npy' in file and 'spnr' in file and 'v2.0' in file]
files = viola_files+volpy_files
result_all = [np.load(file, allow_pickle=True) for file in files]

for idx, results in enumerate(result_all):
    if idx == 0:
        plt.plot(x, [np.array(result).sum()/len(result) for result in results],  marker='.', markersize=15)
    else:
        plt.plot(x, [np.array(result).sum()/len(result) for result in results],  marker='^', markersize=9, linestyle=':')
                 
    plt.legend(['Fiola', 'VolPy'])
    plt.xlabel('spike amplitude')
    plt.ylabel('SPNR')
    plt.title('SPNR for non-overlapping neurons')
    plt.xticks([0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=8)
    ax.yaxis.set_tick_params(length=8)

SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(SAVE_FOLDER, f'SPNR_Fiola_vs_VolPy_{t_range[0]}_{t_range[1]}.pdf'))


#%%
ROOT_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/non_overlapping/'
os.listdir(ROOT_FOLDER)
names = [f'viola_sim5_{i}' for i in range(2, 7, 2)]

name = names[2]
fnames = os.path.join(ROOT_FOLDER, name, name+'.hdf5')
with h5py.File(fnames,'r') as h5:
    mov = np.array(h5['mov'])

mm = np.mean(mov, axis=0)
plt.figure(); plt.imshow(mm, cmap='gray'); 
ax = plt.gca(); ax.get_yaxis().set_visible(False); ax.get_xaxis().set_visible(False)
SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(SAVE_FOLDER, f'Mean_image.pdf'))

#%%
t = []
sp = []
for name in names:
    vi_folder = os.path.join(ROOT_FOLDER, name, 'viola')
    vi_file = [os.path.join(vi_folder, file) for file in os.listdir(vi_folder) if 'v2.0' in file][0]
    vi = np.load(vi_file, allow_pickle=True).item()
    
    folder = os.path.join(ROOT_FOLDER, name)
    gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
    gt_file = gt_files[0]
    gt = scipy.io.loadmat(os.path.join(folder, gt_file))
    gt = gt['simOutput'][0][0]['gt']
    spikes = gt['ST'][0][0][0]
    spatial = gt['IM2'][0][0]
    temporal = gt['trace'][0][0][:,:,0]
    spatial[spatial <= np.percentile(spatial, 99.6)] = 0
    spatial[spatial > 0] = 1
    spatial = spatial.transpose([2, 0, 1])
    #idx = 0
    #vi_temporal = vi.t_s[idx]
    t.append(vi.t0)
    sp.append(spikes)
    #plt.plot(normalize(vi_temporal))
    #plt.vlines(spikes[idx], 6, 8)
    #plt.xlim([6000, 6500])    
#%%
amp = [0.075, 0.125, 0.175]
i = [0, 1, 2]
xlims = [[200, 1000], [200, 1000], [200, 1000]]
fig, axs = plt.subplots(3,1)
    
for idx, tt in enumerate(t):
    axs[idx].plot(normalize(tt[i[idx]]), c='orange')
    axs[idx].vlines(spikes[i[idx]]-1, normalize(tt[i[idx]]).max()+0.5, normalize(tt[i[idx]]).max()+1.5)
    axs[idx].set_xlim(xlims[idx]) 
    axs[idx].spines['top'].set_visible(False)
    axs[idx].spines['right'].set_visible(False)
    axs[idx].spines['left'].set_visible(False)
    axs[idx].spines['bottom'].set_visible(False)
    axs[idx].get_yaxis().set_visible(False)
    axs[idx].get_xaxis().set_visible(False)
    axs[idx].text(120,0, amp[idx])
    if idx == 0:
        axs[idx].text(120,10, 'spike amplitude')
    if idx == 2:
        axs[idx].hlines( -5, 200, 300)
        axs[idx].text(200,-10, '0.25s')
SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(SAVE_FOLDER, f'example_traces.pdf'))
    


    

#%%
import scipy.io 
folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/test'
gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
gt_file = gt_files[0]
gt = scipy.io.loadmat(os.path.join(folder, gt_file))
gt = gt['simOutput'][0][0]['gt']
spikes = gt['ST'][0][0][0]
spatial = gt['IM2'][0][0]
temporal = gt['trace'][0][0][:,:,0]
spatial[spatial <= np.percentile(spatial, 99.6)] = 0
spatial[spatial > 0] = 1
spatial = spatial.transpose([2, 0, 1])
#np.save(os.path.join(folder, 'ROIs.npy'), spatial)



#%% resutl is stored in vio.pipeline.saoz object
#plt.plot(vio.pipeline.saoz.t_s[1][:scope[1]])
for i in range(vio.H.shape[1]):
    if i == 3:
        plt.figure()
        plt.imshow(mov[0], cmap='gray')
        plt.imshow(vio.H.reshape((100, 100, vio.H.shape[1]), order='F')[:,:,i], alpha=0.3)
        plt.figure()
        plt.plot(vio.pipeline.saoz.t_s[i][:scope[1]])

#%%
np.savez(os.path.join(folder, 'viola_result'), vio)        
#%%
from matplotlib.widgets import Slider
def view_components(estimates, img, idx, frame_times=None, gt_times=None):
    """ View spatial and temporal components interactively
    Args:
        estimates: dict
            estimates dictionary contain results of VolPy
            
        img: 2-D array
            summary images for detection
            
        idx: list
            index of selected neurons
    """
    n = len(idx) 
    fig = plt.figure(figsize=(10, 10))

    axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])
    ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
    ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
    ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])    
    s_comp = Slider(axcomp, 'Component', 0, n, valinit=0)
    vmax = np.percentile(img, 98)
    if frame_times is not None:
        pass
    else:
        frame_times = np.array(range(len(estimates.t0[0])))
    
    def arrow_key_image_control(event):

        if event.key == 'left':
            new_val = np.round(s_comp.val - 1)
            if new_val < 0:
                new_val = 0
            s_comp.set_val(new_val)

        elif event.key == 'right':
            new_val = np.round(s_comp.val + 1)
            if new_val > n :
                new_val = n  
            s_comp.set_val(new_val)
        
    def update(val):
        i = np.int(np.round(s_comp.val))
        print(f'Component:{i}')

        if i < n:
            
            ax1.cla()
            imgtmp = estimates.weights[idx][i]
            ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
            ax1.set_title(f'Spatial component {i+1}')
            ax1.axis('off')
            
            ax2.cla()
            ax2.plot(frame_times, estimates.t0[idx][i], alpha=0.8)
            ax2.plot(frame_times, estimates.t_sub[idx][i])            
            ax2.plot(frame_times, estimates.t_rec[idx][i], alpha = 0.4, color='red')
            ax2.plot(frame_times[estimates.spikes[idx][i]],
                     1.05 * np.max(estimates.t0[idx][i]) * np.ones(estimates.spikes[idx][i].shape),
                     color='r', marker='.', fillstyle='none', linestyle='none')
            if gt_times is not None:
                ax2.plot(gt_times,
                     1.15 * np.max(estimates['t'][idx][i]) * np.ones(gt_times.shape),
                     color='blue', marker='.', fillstyle='none', linestyle='none')
                ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes', 'gt_spikes'])
            else:
                ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes'])
            ax2.set_title(f'Signal and spike times {i+1}')
            #ax2.text(0.1, 0.1, f'snr:{round(estimates["snr"][idx][i],2)}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
            #ax2.text(0.1, 0.07, f'num_spikes: {len(estimates["spikes"][idx][i])}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            #ax2.text(0.1, 0.04, f'locality_test: {estimates["locality"][idx][i]}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            
            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None',
                       alpha=0.5, cmap=plt.cm.hot)
            ax3.axis('off')
            
    s_comp.on_changed(update)
    s_comp.set_val(0)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    plt.show()     

vio.pipeline.saoz.reconstruct_signal()
estimates = vio.pipeline.saoz
estimates.spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vio.pipeline.saoz.index])
weights = vio.H.reshape((100, 100, vio.H.shape[1]), order='F')
weights = weights.transpose([2, 0, 1])
estimates.weights = weights
#idx = list(range(len(estimates.t_s)))
idx = np.argsort(vio.seq)


view_components(estimates, img_mean, idx = list(range(51)))
   
        
        
#%%
from match_spikes import match_spikes_greedy, compute_F1
rr = {'F1':[], 'precision':[], 'recall':[]}
for idx in range(n_cells):
    s1 = spikes[idx].flatten()
    s2 = estimates.spikes[np.argsort(vio.seq)][idx]
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
    rr['F1'].append(F1)
    rr['precision'].append(precision)
    rr['recall'].append(recall)  
    
plt.boxplot(rr['F1']); plt.title('viola')
plt.boxplot(rr['precision'])
plt.boxplot(rr['recall'])
        
        
#%%
vpy = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/test/volpy_viola_sim1_1_adaptive_threshold.npy', allow_pickle=True).item()        
rr = {'F1':[], 'precision':[], 'recall':[]}
for idx in range(n_cells):
    s1 = spikes[idx].flatten()
    s2 = vpy['spikes'][idx]
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
    rr['F1'].append(F1)
    rr['precision'].append(precision)
    rr['recall'].append(recall)  

plt.boxplot(rr['F1']); plt.title('volpy')
plt.boxplot(rr['precision'])
plt.boxplot(rr['recall'])
        
#%%
from spikepursuit import signal_filter     
for idx in [1]:
    plt.plot(normalize(vpy['t'][idx]))
    sg = estimates.trace[np.argsort(vio.seq)][idx]
    #sgg  = signal_filter(sg, 1/3, 400, order=3, mode='high')   
    plt.plot(normalize(sg))
    
#%% adaptive threshold on viola traces
from spikepursuit import denoise_spikes
viola_trace = estimates.t0[np.argsort(vio.seq)].copy()   
rr = {'F1':[], 'precision':[], 'recall':[]}

for idx in range(n_cells):
    s1 = spikes[idx].flatten()
    s2 = denoise_spikes(viola_trace[idx], window_length=8, threshold_method='adaptive_threshold')[1]   
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
    rr['F1'].append(F1)
    rr['precision'].append(precision)
    rr['recall'].append(recall)  
    
plt.boxplot(rr['F1']); plt.title('adaptive threshold on viola trace')
   
        
#%% saoz on volpy traces
from signal_analysis_online import SignalAnalysisOnlineZ
saoz = SignalAnalysisOnlineZ(window = 10000, step = 5000, detrend=False, flip=False, 
                 do_scale=False, robust_std=False, frate=400, freq=15, 
                 thresh_range=[3.2, 4], mfp=0.2, online_filter_method = 'median_filter',
                 filt_window = 9, do_plot=False)

saoz.fit(vpy['t'][:,:5000], 20000)
for n in range(5000, 20000):
    saoz.fit_next(vpy['t'][:, n:n+1], n)

viola_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz.index])

rr = {'F1':[], 'precision':[], 'recall':[]}
for idx in range(n_cells):
    s1 = spikes[idx].flatten()
    s2 = viola_spikes[idx]
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
    rr['F1'].append(F1)
    rr['precision'].append(precision)
    rr['recall'].append(recall)  

plt.boxplot(rr['F1']); plt.title('viola spikes on volpy trace')

        
#%%


#%%
algo = ['volpy', 'suite2p', 'meanroi'][2]
names = [f'viola_sim5_{i}' for i in range(1, 8)]
t_range = [10000, 75000]
for idx, dist in enumerate(distance):  
    t_result_all = []
    spnr_all = []
    if mode == 'overlapping':
        select = np.arange(idx * 3, (idx + 1) * 3)
    else:
        select = np.array(range(len(names)))
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
        gt_file = gt_files[0]
        gt = scipy.io.loadmat(os.path.join(folder, gt_file))
        gt = gt['simOutput'][0][0]['gt']
        spikes = gt['ST'][0][0][0]
        
        #summary_file = os.listdir(os.path.join(folder, 'volpy'))
        #summary_file = [file for file in summary_file if 'summary' in file][0]
        #summary = cm.load(os.path.join(folder, 'volpy', summary_file))
        
        v_folder = os.path.join(folder, 'volpy')
        v_files = sorted([file for file in os.listdir(v_folder) if 'adaptive_threshold' in file])
        v_file = v_files[0]
        v = np.load(os.path.join(v_folder, v_file), allow_pickle=True).item()
        
        v_spatial = v['weights'].copy()
        v_temporal = v['ts'].copy()
        v_ROIs = v['ROIs'].copy()
        v_ROIs = v_ROIs * 1.0
        v_templates = v['templates'].copy()
        v_spikes = v['spikes'].copy()
        
        n_cells = v_spatial.shape[0]
        v_result = {'F1':[], 'precision':[], 'recall':[]}        
        rr = {'F1':[], 'precision':[], 'recall':[]}
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s1 = s1 - 1
            s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
            s2 = v_spikes[idx]
            s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall= compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        v_result_all.append(rr)
        
        spnr = []
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s1 = s1 - 1
            s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
            t2 = v_temporal[idx]
            t2 = normalize_piecewise(t2, 5000)
            ss = np.mean(t2[s1])
            spnr.append(ss)        
        spnr_all.append(spnr)
   # np.save(os.path.join(SAVE_FOLDER, f'volpy_adaptive_threshold_{t_range[0]}_{t_range[1]}_v2.0'), v_result_all)
    np.save(os.path.join(SAVE_FOLDER, f'volpy_spnr_adaptive_threshold_{t_range[0]}_{t_range[1]}_v2.0'), spnr_all)    
    #np.save(os.path.join(ROOT_FOLDER, 'result_overlap', f'{dist}', f'volpy_{dist}_thresh_adaptive'), v_save_result)

    #plt.plot(v_temporal[0]);
    #plt.vlines(spikes[0], 0, 150)        
        
        
#%% F1 score
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
VIOLA_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' not in file]
VOLPY_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
volpy_files = os.listdir(VOLPY_FOLDER)
volpy_files = [os.path.join(VOLPY_FOLDER, file) for file in volpy_files if '.npy' in file and 'spnr' not in file and 'v2.0' in file]
VIOLA_FOLDER1 = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_9_minimal_thresh_3_template_window_0_v2.1'
v1 = os.listdir(VIOLA_FOLDER1)
v1 = [os.path.join(VIOLA_FOLDER1, file) for file in v1 if 'spnr' not in file]
files = viola_files+volpy_files + v1
result_all = [np.load(file, allow_pickle=True) for file in files]

for idx, results in enumerate(result_all):
    try:
        #if idx == 0:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15', linewidth = 3)
        #else:
        #    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15')
    except:
        if idx == 0 or idx == 2:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='.', markersize=15)    
        else: 
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='^', markersize=9, linestyle=':')    

    plt.legend(['Fiola', 'VolPy'])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score for non-overlapping neurons')
    plt.ylim([0, 1])
    plt.xticks([0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=8)
    ax.yaxis.set_tick_params(length=8)

SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(SAVE_FOLDER, f'F1_Viola_vs_VolPy_{t_range[0]}_{t_range[1]}.pdf'))        
        
        
        
        #%%
        mode = ['viola', 'volpy'][0]
        result_all = {}
        distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
        SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
        x = [round(0.075 + 0.05 * i, 3) for i in range(3)] 
        for dist in distance:   
            files = np.array(sorted(os.listdir(SAVE_FOLDER)))#[0]#[np.array([5, 0, 1, 2, 3, 4, 6])]
            files = [file for file in files if mode in file and dist+'.npy' in file]
            print(len(files))
            for file in files:
                result_all[file] = np.load(os.path.join(SAVE_FOLDER, file), allow_pickle=True)
            
        for idx, key in enumerate(result_all.keys()):
            results = result_all[key]
            #print(results)
            if mode in key:
                plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
                         marker='.',markersize=15, label=f'{mode}_{area[idx]:.0%}_{distance[idx]}')
                
            print([np.array(result['F1']).sum()/len(result['F1']) for result in results])
            #elif 'volpy' in key:
            #    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
            #             marker='^',markersize=15, label=f'volpy_{area[idx]:.0%}')
            plt.legend()
            plt.xlabel('spike amplitude')
            plt.ylabel('F1 score')
            plt.title(f'{mode} F1 score with different overlapping areas')
            
        #plt.savefig(os.path.join(SAVE_FOLDER, f'F1_overlapping_{mode}.pdf'))
        
        
 elif algo == 'suite2p':
            import numpy as np
            from scipy import stats
            from suite2p.extraction.extract import compute_masks_and_extract_traces
            
            # make stat array
            stat = []
            for u in np.unique(masks)[1:]:
                ypix,xpix = np.nonzero(masks==u)
                npix = len(ypix)
                stat.append({'ypix': ypix, 'xpix': xpix, 'npix': npix, 'lam': np.ones(npix, np.float32)})
            stat = np.array(stat)
            
            # run extraction
            F, Fneu,_,_ = compute_masks_and_extract_traces(ops, stat)
            
            # subtract neuropil
            dF = F - ops['neucoeff'] * Fneu
            
            # compute activity statistics for classifier
            sk = stats.skew(dF, axis=1)
            sd = np.std(dF, axis=1)
            for k in range(F.shape[0]):
                stat[k]['skew'] = sk[k]
                stat[k]['std']  = sd[k]
            
            fpath = ops['save_path']
            # save ops
            np.save(ops['ops_path'], ops)
            # save results
            np.save(os.path.join(fpath,'F.npy'), F)
            np.save(os.path.join(fpath,'Fneu.npy'), Fneu)
            np.save(os.path.join(fpath,'stat.npy'), stat)       
 for method in ['viola', 'volpy', 'meanroi', 'meanroi_online']:
     for idx, key in enumerate(result_all[method].keys()):
         if idx in [0,1,2,4]:
             results = result_all[method][key]
             #print(results)
             if method in key:
                 if method == 'viola':
                     plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
                              marker='.',markersize=10, color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                 elif method == 'volpy':
                     plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
                              marker='^',markersize=6, linestyle=':', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                 elif method == 'meanroi':
                     # plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
                     #           marker='v',markersize=6, linestyle='-.', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                     plt.plot(x, results, 
                               marker='v',markersize=6, linestyle='-.', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                 elif method == 'meanroi_online':
                     plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
                              marker='<',markersize=6, linestyle='--', color=colors[idx])#label=f'{method}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                 
 plt.xlabel('spike amplitude')
 plt.ylabel('F1 score')
 plt.ylim([0.35,1])
 plt.xticks([0.075, 0.125, 0.175])
 plt.title(f'Fiola vs VolPy F1 score with different overlapping areas')
 ax = plt.gca()
 ax.spines['top'].set_visible(False)
 ax.spines['right'].set_visible(False)
 ax.xaxis.set_tick_params(length=8)
 ax.yaxis.set_tick_params(length=8)

 colors = ['blue', 'orange', 'green', 'red']
 for cc, col in enumerate(colors):
     ax.plot(np.NaN, np.NaN, c=colors[cc], label=["63%", "36%", "16%", "0%"][cc])

 ax2 = ax.twinx()
 markers = ['.', '^', 'v', '<']
 linestyles = ['-', ':', '-.', '--']
 markersizes = [10, 6, 6, 6]
 for ss, sty in enumerate(markers):
     ax2.plot(np.NaN, np.NaN, marker=markers[ss],linestyle=linestyles[ss], 
              label=["Fiola", "VolPy", "MeanROI", "MeanROi_online"][ss], c='black', markersize=markersizes[ss])
 ax2.get_yaxis().set_visible(False)
 ax2.spines['top'].set_visible(False)
 ax2.spines['right'].set_visible(False)


 ax.legend(loc=4,  frameon=False)
 ax2.legend(loc=(0.6, 0.03),  frameon=False)
            
 #SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'

 #SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
 #plt.savefig(os.path.join(SAVE_FOLDER, f'F1_overlapping_fiola&volpy.pdf'))       
 
 
 
#%%
for name in names:
   fnames_mat = f'/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/{name}/{name}.mat'
   m = scipy.io.loadmat(fnames_mat)
   m = cm.movie(m['dataAll'].transpose([2, 0, 1]))
   fnames = fnames_mat[:-4] + '.hdf5'
   m.save(fnames)
   
#%% Non overlapping
for idx, dist in enumerate(distance):
#    for ff in range(3):
    for name in np.array(names)[select]:    
        vi_result_all = []
        folder = os.path.join(ROOT_FOLDER, name)
        gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
        gt_file = gt_files[0]
        gt = scipy.io.loadmat(os.path.join(folder, gt_file))
        gt = gt['simOutput'][0][0]['gt']
        spikes = gt['ST'][0][0][0]
        
        vi_folder = os.path.join(folder, 'viola')
        vi_files = sorted([file for file in os.listdir(vi_folder) if 'template_window_2' in file and 'filt_window_9' in file])
        ff = 0
        vi_file = vi_files[ff]
        vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
        
        vi_spatial = vi.H.copy()
        vi_temporal = vi.t_s.copy()
        vi_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vi.index])[np.argsort(vi.seq)]
        
        n_cells = spikes.shape[0]
        vi_result = {'F1':[], 'precision':[], 'recall':[]}        
        rr = {'F1':[], 'precision':[], 'recall':[]}
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s1 = s1[np.logical_and(s1>=t_range[0], s1<t_range[1])]
            s2 = vi_spikes[idx]
            s2 = s2[np.logical_and(s2>=t_range[0], s2<t_range[1])]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        vi_result_all.append(rr)
        print(np.array(vi_result_all[0]['F1']).mean())
        
#%%
step = 2500
plt.figure()
plt.plot(vi.t_s[1])
#plt.title(idx)
for idx, tt in enumerate(vi.thresh[1]):
    if idx == 0:
        plt.hlines(tt, 0, 30000)
    else:
        plt.hlines(tt, 30000 + (idx -1) * step, 30000 + idx * step)

#%%        
k = []
idx = 3
step = 5000
tt = vi_temporal[idx].copy()
for i in np.arange(0, 75000, step):
    ss = spikes[idx][np.logical_and(spikes[idx] > i, spikes[idx] < i+step)]
    k.append(np.mean(tt[ss-1]))
    #k.append(np.percentile(tt[i:i+1000], 95))
plt.plot(np.arange(0, 75000, step), k)

#methods = ['viola', 'volpy', 'layer_1']

# for method in methods:
#     for idx, key in enumerate(result_all[method].keys()):
#         if idx in [1]:
#             results = result_all[method][key]
#             #print(results)
#             if method == 'viola':
#                 ax.bar(xx-0.2 , [np.mean(r['F1']) for r in results], width=0.2,
#                        #yerr=[np.std(r['F1']) for r in results],  
#                          color='blue')
#             elif method == 'volpy':
#                 ax.bar(xx, [np.mean(r['F1']) for r in results], width=0.2,
#                        #yerr=[np.std(r['F1']) for r in results], 
#                          color='orange')
#             elif method == 'meanroi':
#                 ax.bar(xx+0.2, [np.mean(r['F1']) for r in results], width=0.2,
#                        #yerr=[np.std(r['F1']) for r in results], 
#                          color='green')

# ax.set_xlabel('spike amplitude')
# ax.set_ylabel('F1 score')
# ax.set_ylim([0.5,1])
# ax.set_xticks([0, 1, 2])
# ax.set_xticklabels(['0.075', '0.125', '0.175'])
# ax.legend(methods, frameon=False)
# #ax.set_title(f'Fiola vs VolPy F1 score with different overlapping areas')
# #ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.xaxis.set_tick_params(length=8)
# ax.yaxis.set_tick_params(length=8)


# #
# #fig = plt.figure()
# #ax = plt.subplot()
# #xx = np.array([0, 1, 2, 3, 3])
# xx = np.array([3, 2, 1, 0, 0])