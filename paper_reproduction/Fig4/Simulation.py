#!/usr/bin/env python
"""
Further ananlysis on simulation data and Code for reproducing fig4.
"""
import h5py
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False})
import matplotlib.pyplot as plt
import numpy as np
import pyximport
pyximport.install()
import os
import scipy.io
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
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
from paper_reproduction.Fig7.Fig7_utilities import barplot_annotate_brackets, barplot_pvalue
from fiola.utilities import normalize, normalize_piecewise, match_spikes_greedy, compute_F1, load, signal_filter, extract_spikes
#sys.path.append('/media/nel/storage/Code/NEL_LAB/fiola/use_cases')
#sys.path.append(os.path.abspath('/Users/agiovann/SOFTWARE/fiola'))
#from use_cases.test_run_fiola import run_fiola # must be in use_cases folder
from paper_reproduction.Fig4_5.test_run_fiola import run_fiola
from paper_reproduction.utilities import multiple_dfs

#%%
mode = ['overlapping', 'non_overlapping', 'positron'][0]
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
# for num_layers in [1, 3, 5, 10, 30]:
#     for trace_with_neg in [True, False]:
distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
#names = [f'viola_sim3_{i}' for i in range(1, 19)]
#names = [f'viola_sim5_{i}' for i in range(2, 8, 2)]
#names = [f'viola_sim6_{i}' for i in range(2, 20, 3)]
#names = [f'viola_sim7_{i}' for i in range(2, 9)]
test = []
t_range = [10000, 75000]
for idx, dist in enumerate(distance):
    #if idx == 1:
    vi_result_all = []
    time_all = []
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
        vi_files = sorted([file for file in os.listdir(vi_folder) if 'v3.3' in file and 'layers_3_' in file ])
        #vi_files = sorted([file for file in os.listdir(vi_folder) if 'v3.0' in file and 'layers_1' in file])
        #vi_files = sorted([file for file in os.listdir(vi_folder) if 'v3.0' in file and f'layers_{num_layers}_' in file and f'trace_with_neg_{trace_with_neg}' in file])
        if len(vi_files) > 1:
            raise Exception('number of files greater than one')
        vi_file = vi_files[0]
        vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
        
        time_all.append(np.mean(np.diff(vi.timing_online)))
        
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
            ss = np.mean(t2[s1])
            spnr.append(ss)        
        spnr_all.append(spnr)

    folder_name = vi_file[:-4]
    try:
        os.makedirs(os.path.join(SAVE_FOLDER, folder_name))
        print('make folder')
    except:
        print('already exist')
    #np.save(os.path.join(SAVE_FOLDER, folder_name, f'viola_result_{t_range[0]}_{t_range[1]}'), vi_result_all)
    np.save(os.path.join(SAVE_FOLDER, folder_name, f'viola_result_{t_range[0]}_{t_range[1]}_timing'), time_all)
    
    # np.save(os.path.join(SAVE_FOLDER, folder_name, f'viola_result_spnr_{t_range[0]}_{t_range[1]}_v3.0'), spnr_all)
    # print(len(vi_result_all))
    #np.save(os.path.join(SAVE_FOLDER, folder_name,  f'viola_result_{t_range[0]}_{t_range[1]}_{dist}'), vi_result_all)
    ##np.save(os.path.join(SAVE_FOLDER, folder_name, f'viola_result_spnr_{t_range[0]}_{t_range[1]}_v2.0_without_shrinking'), spnr_all)
    test.append(np.mean(vi_result_all[0]['F1']))

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
                
            t_result_all.append(t_result)
        
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
    
    
    
    # if mode == 'non_overlapping':
    #     np.save(os.path.join(save_folder, f'{algo}_F1_{t_range[0]}_{t_range[1]}_v3.1'), t_result_all)
    #     np.save(os.path.join(save_folder, f'{algo}_spnr_{t_range[0]}_{t_range[1]}_v3.1'), spnr_all)    
    # elif mode == 'overlapping':
    #     np.save(os.path.join(save_folder, f'{algo}_F1_{t_range[0]}_{t_range[1]}_{dist}'), t_result_all)

##################################################################################################################
##################################################################################################################

