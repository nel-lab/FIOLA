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
#%%
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import normalize, normalize_piecewise, match_spikes_greedy, compute_F1
#sys.path.append('/home/nel/Code/NEL_LAB/fiola/use_cases')
#sys.path.append(os.path.abspath('/Users/agiovann/SOFTWARE/fiola'))
from use_cases.test_run_fiola import run_fiola # must be in use_cases folder

#%%
mode = ['overlapping', 'non_overlapping', 'positron'][1]
dropbox_folder = '/home/nel/NEL-LAB Dropbox/'
#dropbox_folder = '/Users/agiovann/Dropbox/'

if mode == 'overlapping':
    ROOT_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/data/voltage_data/simulation/overlapping'
    SAVE_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/result/test_simulations/overlapping'
    names = [f'viola_sim3_{i}' for i in range(1, 2)]
elif mode == 'non_overlapping':
    ROOT_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/data/voltage_data/simulation/non_overlapping'
    SAVE_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/result/test_simulations/non_overlapping'
    names = [f'viola_sim5_{i}' for i in range(1, 8)]
elif mode == 'positron':
    ROOT_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/data/voltage_data/simulation/test/sim4_positron'
    SAVE_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/result/test_simulations/test/sim4_positron'
    names = [f'viola_sim4_{i}' for i in range(1, 13)]
   
#%%
t_range = [10000, 75000]

num_frames_init = 10000
border_to_0 = 2
flip = True
thresh_range= [2.8, 5.0]
erosion=0 
use_rank_one_nmf=False
hals_movie='hp_thresh'
semi_nmf=False
update_bg = True
use_spikes= False
use_batch=True
batch_size=100
center_dims=None
initialize_with_gpu=True
do_scale = False
adaptive_threshold=True
filt_window=15
minimal_thresh=3
step=2500
template_window=2
    
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
    'minimal_thresh': minimal_thresh,
    'step': step, 
    'template_window':template_window}

#%%
for name in names:
    fnames = os.path.join(ROOT_FOLDER, name, name+'.hdf5')
    print(f'NOW PROCESSING: {fnames}')
    path_ROIs = os.path.join(ROOT_FOLDER, name, 'viola', 'ROIs_gt.hdf5')
    
    run_fiola(fnames, path_ROIs, fr=400, online_gpu=True, options=options)
    
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

#%%  
distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
names = [f'viola_sim3_{i}' for i in range(1, 19)]
#names = [f'viola_sim5_{i}' for i in range(1, 8)]
#t_range = [10000, 20000]
for idx, dist in enumerate(distance):
    vi_result_all = []
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
        
        vi_folder = os.path.join(folder, 'viola')
        vi_files = sorted([file for file in os.listdir(vi_folder) if 'filt_window_[8, 4]' in file and 'template_window_0' in file])
        if len(vi_files) > 1:
            raise Exception('number of files greater than one')
        vi_file = vi_files[0]
        vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
        
        vi_spatial = vi.H.copy()
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
    np.save(os.path.join(SAVE_FOLDER, folder_name, f'viola_result_spnr_{t_range[0]}_{t_range[1]}'), spnr_all)
    #np.save(os.path.join(ROOT_FOLDER, 'result_overlap', f'{dist}', f'viola_result'), vi_save_result)
    #np.save(os.path.join(SAVE_FOLDER, folder_name, f'viola_result_spnr_{t_range[0]}_{t_range[1]}_v2.0_without_shrinking'), spnr_all)
    

#%%
hh = vi.H.reshape((100,100,9), order='F')   
plt.imshow(hh[:,:,3]) 

plt.plot(vi_temporal[9])   



#%%
names = [f'viola_sim5_{i}' for i in range(1, 8)]
t_range = [10000, 75000]
for idx, dist in enumerate(distance):  
    v_result_all = []
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
    
##################################################################################################################
##################################################################################################################
#%% F1 score
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
VIOLA_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' not in file]
VOLPY_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
volpy_files = os.listdir(VOLPY_FOLDER)
volpy_files = [os.path.join(VOLPY_FOLDER, file) for file in volpy_files if '.npy' in file and 'spnr' not in file and 'v2.0' in file]
VIOLA_FOLDER1 = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_9_minimal_thresh_3_template_window_0_v2.1'
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

SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(SAVE_FOLDER, f'F1_Viola_vs_VolPy_{t_range[0]}_{t_range[1]}.pdf'))

#%% Fig 4c, F1 score 
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
VIOLA_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' not in file]
VOLPY_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
volpy_files = os.listdir(VOLPY_FOLDER)
volpy_files = [os.path.join(VOLPY_FOLDER, file) for file in volpy_files if '.npy' in file and 'spnr' not in file and 'v2.0' in file]
VIOLA_FOLDER1 = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_[8, 4]_minimal_thresh_3_template_window_2_v2.1'
v1 = os.listdir(VIOLA_FOLDER1)
v1 = [os.path.join(VIOLA_FOLDER1, file) for file in v1 if 'spnr' not in file]
VIOLA_FOLDER2 = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_[8, 4]_minimal_thresh_3_template_window_0_v2.1'
v2 = os.listdir(VIOLA_FOLDER2)
v2 = [os.path.join(VIOLA_FOLDER2, file) for file in v2 if 'spnr' not in file]
files = viola_files+volpy_files + v1 + v2
result_all = [np.load(file, allow_pickle=True) for file in files]

for idx, results in enumerate(result_all):
    try:
        #if idx == 0:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15', linewidth = 3)
        #else:
        #    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15')
    except:
        if idx != 1:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='.', markersize=15)    
        else: 
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='^', markersize=9, linestyle=':')    

    plt.legend(['FIOLA_25ms', 'VolPy', 'FIOLA_17.5ms', 'FIOLA_12.5ms'], frameon=False)
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

#SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
#plt.savefig(os.path.join(SAVE_FOLDER, f'F1_Fiola&VolPy_non_symm_median_{t_range[0]}_{t_range[1]}_1.pdf'))

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
rects1 = ax.bar(x - width/2, mean[0], width, yerr=[[0]*7, std[0]], capsize=5, label=f'FIOLA_25ms')
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
#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1/supp/suppl_simulation_f1_significance.png')

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
VIOLA_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if '.npy' in file and 'spnr' not in file]
VOLPY_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
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
VIOLA_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' in file]
VOLPY_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
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

SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(SAVE_FOLDER, f'SPNR_Fiola_vs_VolPy_{t_range[0]}_{t_range[1]}.pdf'))

#%%
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
VIOLA_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' in file]
VOLPY_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/volpy_result'
volpy_files = os.listdir(VOLPY_FOLDER)
volpy_files = [os.path.join(VOLPY_FOLDER, file) for file in volpy_files if '.npy' in file and 'spnr' in file and 'v2.0' in file]
VIOLA_FOLDER1 = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_[8, 4]_minimal_thresh_3_template_window_2_v2.1'
v1 = os.listdir(VIOLA_FOLDER1)
v1 = [os.path.join(VIOLA_FOLDER1, file) for file in v1 if 'spnr' in file]
VIOLA_FOLDER2 = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_filt_window_[8, 4]_minimal_thresh_3_template_window_0_v2.1'
v2 = os.listdir(VIOLA_FOLDER2)
v2 = [os.path.join(VIOLA_FOLDER2, file) for file in v2 if 'spnr' in file]
files = viola_files+volpy_files+v1+v2
result_all = [np.load(file, allow_pickle=True) for file in files]

for idx, results in enumerate(result_all):
    if idx == 0 or idx == 2 or idx == 3:
        plt.plot(x, [np.array(result).sum()/len(result) for result in results],  marker='.', markersize=15)
    else:
        plt.plot(x, [np.array(result).sum()/len(result) for result in results],  marker='^', markersize=9, linestyle=':')
                 
    plt.legend(['FIOLA_25ms', 'VolPy', 'FIOLA_17.5ms', 'FIOLA_12.5ms'], frameon=False)
    plt.xlabel('spike amplitude')
    plt.ylabel('SPNR')
    plt.title('SPNR for non-overlapping neurons')
    plt.xticks([0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=8)
    ax.yaxis.set_tick_params(length=8)

SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(SAVE_FOLDER, f'SPNR_Fiola_vs_VolPy_{t_range[0]}_{t_range[1]}.pdf'))


#%% Overlapping neurons for volpy with different overlapping areas
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
mode = ['viola', 'volpy'][0]
result_all = {}
distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
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
    
plt.savefig(os.path.join(SAVE_FOLDER, f'F1_overlapping_{mode}.pdf'))


#%%
result_all = {'volpy':{}, 'viola':{}}
distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
x = [round(0.075 + 0.05 * i, 3) for i in range(3)] 
for mode in ['viola', 'volpy']:
    if mode == 'viola':
        SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
    else:
        SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/volpy_result'
        
    for dist in distance:   
        files = np.array(sorted(os.listdir(SAVE_FOLDER)))#[0]#[np.array([5, 0, 1, 2, 3, 4, 6])]
        files = [file for file in files if mode in file and dist+'.npy' in file]
        print(len(files))
        for file in files:
            result_all[mode][file] = np.load(os.path.join(SAVE_FOLDER, file), allow_pickle=True)
    
colors = ['blue', 'orange', 'green', 'purple', 'red', 'black']
for mode in ['viola', 'volpy']:
    for idx, key in enumerate(result_all[mode].keys()):
        if idx in [0,1,2,4]:
            results = result_all[mode][key]
            #print(results)
            if mode in key:
                if mode == 'viola':
                    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
                             marker='.',markersize=10, color=colors[idx])#label=f'{mode}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
                else:
                    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
                             marker='^',markersize=6, linestyle=':', color=colors[idx])#label=f'{mode}_{area[idx]:.0%}_{distance[idx]}', color=colors[idx])
        
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
markers = ['.', '^']
linestyles = ['-', ':']
markersizes = [10, 6]
for ss, sty in enumerate(markers):
    ax2.plot(np.NaN, np.NaN, marker=markers[ss],linestyle=linestyles[ss], 
             label=["Fiola", "VolPy"][ss], c='black', markersize=markersizes[ss])
ax2.get_yaxis().set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)


ax.legend(loc=4,  frameon=False)
ax2.legend(loc=(0.6, 0.03),  frameon=False)
           
#SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'

SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(SAVE_FOLDER, f'F1_overlapping_fiola&volpy.pdf'))
    

#%%
ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/non_overlapping/'
os.listdir(ROOT_FOLDER)
names = [f'viola_sim5_{i}' for i in range(2, 7, 2)]

name = names[2]
fnames = os.path.join(ROOT_FOLDER, name, name+'.hdf5')
with h5py.File(fnames,'r') as h5:
    mov = np.array(h5['mov'])

mm = np.mean(mov, axis=0)
plt.figure(); plt.imshow(mm, cmap='gray'); 
ax = plt.gca(); ax.get_yaxis().set_visible(False); ax.get_xaxis().set_visible(False)
SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
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
SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(SAVE_FOLDER, f'example_traces.pdf'))
    


    

#%%
import scipy.io 
folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/test'
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
vpy = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/test/volpy_viola_sim1_1_adaptive_threshold.npy', allow_pickle=True).item()        
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
        
        
        
        
        
        
        
        
        
        
        
        