#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:48:17 2020
Pipeline for online analysis of voltage imaging data
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
@author: @agiovann, @caichangjia, @cynthia
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

from nmf_support import normalize
from violaparams import violaparams
from viola import VIOLA
import scipy.io
from match_spikes import match_spikes_greedy, compute_F1
sys.path.append('/home/nel/Code/NEL_LAB/VIOLA/use_cases')
from test_simulation_run_viola import run_viola

#%%
mode = ['overlapping', 'non_overlapping', 'positron'][2]
if mode == 'overlapping':
    ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/overlapping'
    SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping'
    names = [f'viola_sim3_{i}' for i in range(1, 7)]
elif mode == 'non_overlapping':
    ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/non_overlapping'
    SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping'
    names = [f'viola_sim2_{i}' for i in range(1, 8)]
elif mode == 'positron':
    ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/test/sim4_positron'
    SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/test/sim4_positron'
    names = [f'viola_sim4_{i}' for i in range(1, 13)]
    
#%%
distance = [f'dist_{i}' for i in [1, 3]]#, 5, 7, 10, 15]]
t_range = [10000, 20000]
border_to_0 = 2
flip = True
num_frames_init = 10000
num_frames_total=20000
thresh_range= [3, 4]
erosion=0 
update_bg = True
use_spikes=True 
initialize_with_gpu=False
adaptive_threshold=True
filt_window=15
    
options = {
    'border_to_0': border_to_0,
    'flip': flip,
    'num_frames_total': num_frames_total, 
    'thresh_range': thresh_range,
    'erosion':erosion, 
    'update_bg': update_bg,
    'use_spikes':use_spikes, 
    'initialize_with_gpu':initialize_with_gpu,
    'adaptive_threshold': adaptive_threshold,
    'filt_window': filt_window}

#%%
for name in names:
    fnames = os.path.join(ROOT_FOLDER, name, name+'.hdf5')
    path_ROIs = os.path.join(ROOT_FOLDER, name, 'viola', 'ROIs_gt.hdf5')
    run_viola(fnames, path_ROIs, fr=400, options=options)
    
#%%
for idx, dist in enumerate(distance[0]):
    for ff in range(3):
        vi_result_all = []
        folder = os.path.join(ROOT_FOLDER, name)
        gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
        gt_file = gt_files[0]
        gt = scipy.io.loadmat(os.path.join(folder, gt_file))
        gt = gt['simOutput'][0][0]['gt']
        spikes = gt['ST'][0][0][0]
        
        vi_folder = os.path.join(folder, 'viola')
        vi_files = sorted([file for file in os.listdir(vi_folder) if 'viola' in file and 'use_spikes_False' in file])
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
            s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
            s2 = vi_spikes[idx]
            s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        vi_result_all.append(rr)
        print(np.array(vi_result_all[0]['F1']).mean())

#%%  
for idx, dist in enumerate(distance):
    vi_result_all = []
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
        vi_files = sorted([file for file in os.listdir(vi_folder) if 'viola' in file and 'use_spikes_False_small_mask' in file])
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
            s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
            s2 = vi_spikes[idx]
            s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        vi_result_all.append(rr)
    np.save(os.path.join(SAVE_FOLDER, 'mask_erosion', f'viola_result_10000_20000_{dist}'), vi_result_all)
    #np.save(os.path.join(ROOT_FOLDER, 'result_overlap', f'{dist}', f'volpy_{dist}_thresh_adaptive'), vi_save_result)

#%%
hh = vi.H.reshape((100,100,9), order='F')   
plt.imshow(hh[:,:,3]) 

plt.plot(vi_temporal[9])   



#%%
for idx, dist in enumerate(distance):  
    v_result_all = []
    if overlap == True:
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
            s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
            s2 = v_spikes[idx]
            s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall= compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        v_result_all.append(rr)

    np.save(os.path.join(SAVE_FOLDER, f'volpy_adaptive_threshold_10000_20000_{dist}'), v_result_all)

    
    #np.save(os.path.join(ROOT_FOLDER, 'result_overlap', f'{dist}', f'volpy_{dist}_thresh_adaptive'), v_save_result)

#%%
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping'
files = np.array(sorted(os.listdir(result_folder)))
files = np.array(['viola_result_10000_20000.npy', 'volpy_adaptive_threshold_10000_20000.npy'])
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
for idx, results in enumerate(result_all):
    try:
        if idx == 0:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15', linewidth = 3)
        else:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15')
    except:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='.', markersize='15')    

    plt.legend(['Viola', 'VolPy'])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score for non-overlapping neurons')

plt.savefig(os.path.join(SAVE_FOLDER, 'F1_Viola_vs_VolPy_10000_20000.pdf'))

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
    #elif 'volpy' in key:
    #    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
    #             marker='^',markersize=15, label=f'volpy_{area[idx]:.0%}')


    plt.legend()
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title(f'{mode} F1 score with different overlapping areas')
    
plt.savefig(os.path.join(SAVE_FOLDER, f'F1_overlapping_{mode}.pdf'))


#%%
mode = ['viola', 'volpy'][0]
result_all = {}
#distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/overlapping/mask_erosion'
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
    #elif 'volpy' in key:
    #    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
    #             marker='^',markersize=15, label=f'volpy_{area[idx]:.0%}')


    plt.legend()
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title(f'{mode} F1 score with different overlapping areas and masks erosion')
    
plt.savefig(os.path.join(SAVE_FOLDER, f'F1_overlapping_{mode}.pdf'))




#%%
import scipy.io 
folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/test'
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
vpy = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/test/volpy_viola_sim1_1_adaptive_threshold.npy', allow_pickle=True).item()        
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
        
        
        
        
        
        
        
        
        
        
        
        