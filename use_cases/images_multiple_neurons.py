#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:01:48 2020
Running tests for viola vs caiman and volpy
@author: caichangjia
"""
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import nnls    
from signal_analysis_online import SignalAnalysisOnlineZ
from skimage import measure
from sklearn.decomposition import NMF

from caiman_functions import signal_filter, to_3D, to_2D, bin_median, play
from F1_score_computation import compute_distances, match_spikes_linear_sum, match_spikes_greedy, compute_F1
from metrics import metric
from nmf_support import hals, select_masks, normalize, nmf_sequential
from skimage.io import imread
from running_statistics import OnlineFilter
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%% files for processing
n_neurons = ['1', '2', 'many'][2]

if n_neurons in ['1', '2']:
    movie_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
                   '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron',
                   '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/'][1]
    
    movie_lists = ['454597_Cell_0_40x_patch1_mc.tif', '456462_Cell_3_40x_1xtube_10A2_mc.tif',
                 '456462_Cell_3_40x_1xtube_10A3_mc.tif', '456462_Cell_5_40x_1xtube_10A5_mc.tif',
                 '456462_Cell_5_40x_1xtube_10A6_mc.tif', '456462_Cell_5_40x_1xtube_10A7_mc.tif', 
                 '462149_Cell_1_40x_1xtube_10A1_mc.tif', '462149_Cell_1_40x_1xtube_10A2_mc.tif',
                 '456462_Cell_4_40x_1xtube_10A4_mc.tif', '456462_Cell_6_40x_1xtube_10A10_mc.tif',
                 '456462_Cell_5_40x_1xtube_10A8_mc.tif', '456462_Cell_5_40x_1xtube_10A9_mc.tif', 
                 '462149_Cell_3_40x_1xtube_10A3_mc.tif', '466769_Cell_2_40x_1xtube_10A_6_mc.tif',
                 '466769_Cell_2_40x_1xtube_10A_4_mc.tif', '466769_Cell_3_40x_1xtube_10A_8_mc.tif', 
                 '09282017Fish1-1_mc.tif', '10052017Fish2-2_mc.tif', 'Mouse_Session_1_mc.tif']
    
    frate_all = np.array([400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 995.02, 400.8 ,
       400.8 , 400.8 , 400.8 , 400.8 , 995.02, 995.02, 995.02, 995.02,
       300.  , 300.  , 400.  ])
    
    fnames = [os.path.join(movie_folder, file) for file in movie_lists]
    
    combined_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons',
                    '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons'][0]
    
    combined_lists = ['neuron0&1_x[1, -1]_y[1, -1].tif', 
                   'neuron0&1_x[2, -2]_y[2, -2].tif', 
                   'neuron1&2_x[4, -2]_y[4, -2].tif', 
                   'neuron1&2_x[6, -2]_y[8, -2].tif']
elif n_neurons == 'many':
    movie_folder = ['/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons'][1]
   
    movie_lists = ['demo_voltage_imaging_mc.hdf5', 
                   'FOV4_50um_mc.hdf5',
                   '06152017Fish1-2_portion.hdf5', 
                   'FOV4_50um.hdf5',
                   'IVQ32_S2_FOV1_processed_mc.hdf5']
    
#%% Choosing datasets
if n_neurons == '1':
    file_set = [1]
    name = movie_lists[file_set[0]]
    belong_Marton = True
    if ('Fish' in name) or ('Mouse' in name):
        belong_Marton = False
    frate = frate_all[file_set[0]]
    mov = imread(os.path.join(movie_folder, name))
    with h5py.File(os.path.join(movie_folder, name[:-7]+'_ROI.hdf5'),'r') as h5:
        mask = np.array(h5['mov'])
    if mask.shape[0] != 1:
        mask = mask[np.newaxis,:]

elif n_neurons == '2':
    file_set = [0, 1]
    name = combined_lists[0]
    frate = 400
    mov = imread(os.path.join(combined_folder, name))
    with h5py.File(os.path.join(combined_folder, name[:-4]+'_ROIs.hdf5'),'r') as h5:
       mask = np.array(h5['mov'])

elif n_neurons == 'many':
    name = movie_lists[4]
    frate = 300
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
       mov = np.array(h5['mov'])
    with h5py.File(os.path.join(movie_folder, name[:-18]+'_ROIs.hdf5'),'r') as h5:
       mask = np.array(h5['mov'])


#%% Load VolPy result
name_estimates = ['demo_voltage_imaging_estimates.npy', 
                  'FOV4_50um_estimates.npz', 
                  '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_06152017Fish1-2.npy',
                  '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_IVQ32_S2_FOV1.npy', 
                  '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_FOV4_35um.npy' ]
#name_estimates = [os.path.join(movie_folder, name) for name in name_estimates]
estimates = np.load(name_estimates[3], allow_pickle=True).item()

#%% Load CaImAn result
name_estimates = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/caiman_FOV4_50um.npy', 
                  '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/caiman_IVQ32_S2_FOV1.npy', 
                  '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/caiman_06152017Fish1-2.npy']

caiman_estimates = np.load(name_estimates[1], allow_pickle=True).item()

#%% Load VioLa result
name_estimates = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/viola_06152017Fish1-2.npy',
                  '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/viola_FOV4_35um.npy',
                  '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/viola_IVQ32_S2_FOV1.npy']
saoz = np.load(name_estimates[2], allow_pickle=True).item()

#%% caiman sspikes
from scipy import signal
sg = signal_filter(caiman_estimates.C, freq=15, fr=1000)
caiman_spikes = []
for idx, data in enumerate(sg):
    data = data - np.median(data)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    thresh = 5 * std
    locs = signal.find_peaks(data, height=thresh)[0]
    caiman_spikes.append(locs)

#%% Traces
seq = saoz.seq
idx_list = np.where((saoz.index_track> 30))[0]
idx_list=np.array([2])
idx_volpy  = seq[idx_list]
#idx_volpy = np.where(np.array([len(estimates['spikes'][k]) for k in range(len(estimates['spikes']))])>30)[0]
#idx_list = np.array([np.where(seq==idx_volpy[k])[0][0] for k in range(idx_volpy.size)])
length = idx_list.size
fig, ax = plt.subplots(idx_list.size,1)
colorsets = plt.cm.tab10(np.linspace(0,1,10))
colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]
scope=[0, 17000]

score_viola = []
score_caiman = []

for n, idx in enumerate(idx_list):
    idx_volpy = seq[idx]
    #idx_caiman = caiman_estimates.idx[tp_comp][n]
    #idx_caiman = [12, 13, 9][n]
    idx_caiman = 0
    ax.plot(np.arange(scope[1]), normalize(estimates['ts'][idx_volpy]), 'c', linewidth=0.5, color='black', label='volpy')
    ax.plot(np.arange(scope[1]), normalize(saoz.t_s[idx, :scope[1]])-0.5, 'c', linewidth=0.5, color='red', label='viola')
    ax.plot(np.arange(scope[1]), normalize(signal_filter(caiman_estimates.C, freq=15, fr=400)[idx_caiman].flatten())-1, 'c', linewidth=0.5, color='orange', label='caiman')
    #ax.plot(np.arange(20000), normalize(saoz.t_s[idx, :20000]), 'c', linewidth=0.5, color='red', label='viola')
    #ax.plot(normalize(saoz.t_s[idx]), color='blue', label=f'viola_t_s')
    add = 2
    spikes_online = list(set(saoz.index[idx]) - set([0]))
    ax.vlines(estimates['spikes'][idx_volpy], add+1.9, add+2.2, color='black')
    ax.vlines(spikes_online, add+1.55, add+1.85, color='red')
    ax.vlines(caiman_spikes[idx_caiman], add+1.2, add+1.5, color='orange')

    if n<length-1:
        ax.get_xaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 
        ax.spines['bottom'].set_visible(False) 
        ax.spines['left'].set_visible(False) 
        ax.set_yticks([])
    
    if n==length-1:
        ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  
        ax.spines['left'].set_visible(True) 
        ax.set_xlabel('Frames')
    ax.set_ylabel('o')
    ax.get_yaxis().set_visible(True)
    ax.yaxis.label.set_color(colorsets[np.mod(n,9)])
    ax.set_xlim([7500, 12500])
    
    
    s1 = estimates['spikes'][idx_volpy]
    s2 = sorted(np.array(spikes_online))
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=10)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
    print(f'F1:{round(F1, 3)}, precision:{round(precision,3)}, recall:{round(recall, 3)}')
    score_viola.append([F1, precision, recall])
    
    s1 = estimates['spikes'][idx_volpy]
    s2 = caiman_spikes[idx_caiman]
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=10)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
    print(f'F1:{round(F1, 3)}, precision:{round(precision,3)}, recall:{round(recall, 3)}')
    score_caiman.append([F1, precision, recall])
        
plt.tight_layout()
#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/FOV4_50um.pdf')
#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/06152017Fish1-2.pdf')
plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/IVQ32_S2_FOV1.pdf')
 
#%%
seq = saoz.seq
idx_list = np.where((saoz.index_track> 50))[0]
#idx_list=np.array([2])
idx_volpy  = seq[idx_list]
#idx_volpy = np.where(np.array([len(estimates['spikes'][k]) for k in range(len(estimates['spikes']))])>30)[0]
#idx_list = np.array([np.where(seq==idx_volpy[k])[0][0] for k in range(idx_volpy.size)])
length = idx_list.size
fig, ax = plt.subplots(idx_list.size,1)
colorsets = plt.cm.tab10(np.linspace(0,1,10))
colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]
scope=[0, 20000]

score_viola = []
score_caiman = []

for n, idx in enumerate(idx_list):
    idx_volpy = seq[idx]
    idx_caiman = caiman_estimates.idx[tp_comp][n]
    #idx_caiman = [12, 13, 9][n]
    #idx_caiman = np.array([0])
    ax[n].plot(np.arange(scope[1]), normalize(estimates['ts'][idx_volpy]), 'c', linewidth=0.5, color='black', label='volpy')
    ax[n].plot(np.arange(scope[1]), normalize(saoz.t_s[idx, :scope[1]])-0.5, 'c', linewidth=0.5, color='red', label='viola')
    ax[n].plot(np.arange(scope[1]), normalize(signal_filter(caiman_estimates.C, freq=15, fr=400)[idx_caiman])-1, 'c', linewidth=0.5, color='orange', label='caiman')
    #ax[n].plot(np.arange(20000), normalize(saoz.t_s[idx, :20000]), 'c', linewidth=0.5, color='red', label='viola')
    #ax[n].plot(normalize(saoz.t_s[idx]), color='blue', label=f'viola_t_s')

    spikes_online = list(set(saoz.index[idx]) - set([0]))
    ax[n].vlines(estimates['spikes'][idx_volpy], 1.9, 2.2, color='black')
    ax[n].vlines(spikes_online, 1.55, 1.85, color='red')
    ax[n].vlines(caiman_spikes[idx_caiman], 1.2, 1.5, color='orange')

    if n<length-1:
        ax[n].get_xaxis().set_visible(False)
        ax[n].spines['right'].set_visible(False)
        ax[n].spines['top'].set_visible(False) 
        ax[n].spines['bottom'].set_visible(False) 
        ax[n].spines['left'].set_visible(False) 
        ax[n].set_yticks([])
    
    if n==length-1:
        ax[n].legend()
        ax[n].spines['right'].set_visible(False)
        ax[n].spines['top'].set_visible(False)  
        ax[n].spines['left'].set_visible(True) 
        ax[n].set_xlabel('Frames')
    ax[n].set_ylabel('o')
    ax[n].get_yaxis().set_visible(True)
    ax[n].yaxis.label.set_color(colorsets[np.mod(n,9)])
    ax[n].set_xlim(scope)
    
    
    s1 = estimates['spikes'][idx_volpy]
    s2 = sorted(np.array(spikes_online))
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=3)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
    print(f'F1:{round(F1, 3)}, precision:{round(precision,3)}, recall:{round(recall, 3)}')
    score_viola.append([F1, precision, recall])
    
    s1 = estimates['spikes'][idx_volpy]
    s2 = caiman_spikes[idx_caiman]
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=3)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
    print(f'F1:{round(F1, 3)}, precision:{round(precision,3)}, recall:{round(recall, 3)}')
    score_caiman.append([F1, precision, recall])
        
plt.tight_layout()
#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/FOV4_50um.pdf')
#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/06152017Fish1-2.pdf')
 
#%%
print(f'viola: {np.array(score_viola).mean(0).round(3)}')
print(f'caiman: {np.array(score_caiman).mean(0).round(3)}')




#%% Spatial contours
Cn = mov[0]
vmax = np.percentile(Cn, 99)
vmin = np.percentile(Cn, 5)
plt.figure()
plt.imshow(Cn, interpolation='None', vmax=vmax, vmin=vmin, cmap=plt.cm.gray)
plt.title('Neurons location')
d1, d2 = Cn.shape
#cm1 = com(mask.copy().reshape((N,-1), order='F').transpose(), d1, d2)
colors='yellow'
for n, idx in enumerate(idx_list):
    contours = measure.find_contours(mask[seq[idx]], 0.5)[0]
    plt.plot(contours[:, 1], contours[:, 0], linewidth=1, color=colorsets[np.mod(n,9)])
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/FOV4_50um_footprints.pdf')
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/06152017Fish1-2_footprints.pdf')   
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/IVQ32_S2_FOV1_footprints.pdf')   


#%%
H_new = saoz.H
#idx = 0
#idx_volpy = seq[idx]
#idx = np.where(seq==idx_volpy)[0][0]
idx = idx_list[0]
spikes_online = list(set(saoz.index[idx]) - set([0]))
plt.figure();plt.imshow(mov[0], cmap='gray');
plt.imshow(H_new[:,idx].reshape(mov.shape[1:], order='F'), alpha=0.5, cmap='gray');plt.colorbar()

#%%
#plt.figure();plt.imshow(H_new[:,idx].reshape(mov.shape[1:], order='F'))
plt.figure()
plt.plot(saoz.t0[idx], color='blue', label=f'online_{len(spikes_online)}')
plt.plot(normalize(saoz.t[idx]), color='red', label=f'online_t_s')
plt.plot(normalize(estimates['t'][idx_volpy]), color='orange', label=f'volpy_{estimates["spikes"][idx_volpy].shape[0]}')
plt.plot(normalize(saoz.t_rec[idx]), color='red', label=f'online_t_rec')
plt.plot(saoz.t_sub[idx], color='red', label=f'online_t_sub')
plt.vlines(spikes_online, -1.6, -1.2, color='blue', label='online_spikes')
plt.vlines(estimates['spikes'][idx_volpy], -1.8, -1.4, color='orange', label='volpy_spikes')
print(len(list(set(saoz.index[idx]))))
plt.legend()       
    

#%%
for idx in range(0, 1):
    #idx = 9
    plt.figure();plt.plot(signal_filter(caiman_estimates.C, freq=15, fr=300)[idx])
    plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape((-1, 96), order='F'))
    plt.title(str(idx))
    
#%% match caiman spatial footprints and volpy spatial footprint   
caiman_estimates.idx
plt.figure();plt.plot(signal_filter(caiman_estimates.C, freq=15, fr=400)[idx])
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape((512, 128), order='F'))
    
mask0 = mask[seq[idx_list]]
mask1 = caiman_estimates.A.toarray()[:, caiman_estimates.idx].reshape((512, 128, -1), order='F').transpose([2, 0, 1])
mask1[mask1>0.02] = 1
plt.figure();plt.imshow(mask0.sum(0));plt.colorbar();plt.show()
        
from caiman.base.rois import nf_match_neurons_in_binary_masks
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        mask0, mask1, thresh_cost=1, min_dist=10, print_assignment=True,
        plot_results=True, Cn=mov[0], labels=['viola', 'cm'])    



#%%
caiman_estimates.idx
plt.figure();plt.plot(signal_filter(caiman_estimates.C, freq=15, fr=400)[idx])
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape((512, 128), order='F'))
    
#%%
#mask0 = mask[seq[idx_list]]
mask0 = vpy['weights'][idx_list].copy()
mask0[mask0<0] = 0
#mask0[mask0>0] = 1
mask1 = caiman_estimates.A.toarray().copy()[:, :].reshape((512, 128, -1), order='F').transpose([2, 0, 1])
mask1[mask1>0.02] = 1
plt.figure();plt.imshow(mask0.sum(0));plt.colorbar();plt.show()
plt.figure();plt.imshow(mask1.sum(0));plt.colorbar();plt.show()
        
from caiman.base.rois import nf_match_neurons_in_binary_masks
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        mask0, mask1, thresh_cost=1, min_dist=10, print_assignment=True,
        plot_results=True, Cn=caiman_estimates.Cn, labels=['volpy', 'caiman'])    
