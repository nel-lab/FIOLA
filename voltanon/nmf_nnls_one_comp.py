#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:30:11 2020
Use hals algorithm to refine spatial components extracted by rank-1 nmf. 
Use nnls with gpu for signal extraction 
Use sao for spike extraction
@author: @agiovann, @caichangjia, @cynthia
"""
#%%
from caiman_functions import signal_filter, to_3D, to_2D
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import nnls    
from signal_analysis_online import SignalAnalysisOnline
from sklearn.decomposition import NMF
from metrics import metric
from nmf_support import hals, select_masks, normalize
from skimage.io import imread
import h5py
from running_statistics import OnlineFilter
#%% files for processing
n_neurons = ['1', '2', 'many'][0]

if n_neurons in ['1', '2']:
    movie_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
                   '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
                   '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/'][1]
    
    movie_lists = ['454597_Cell_0_40x_patch1_mc.tif', '456462_Cell_3_40x_1xtube_10A2_mc.tif',
                 '456462_Cell_3_40x_1xtube_10A3_mc.tif', '456462_Cell_5_40x_1xtube_10A5_mc.tif',
                 '456462_Cell_5_40x_1xtube_10A6_mc.tif', '456462_Cell_5_40x_1xtube_10A7_mc.tif', 
                 '462149_Cell_1_40x_1xtube_10A1_mc.tif', '462149_Cell_1_40x_1xtube_10A2_mc.tif',
                 '456462_Cell_4_40x_1xtube_10A4_mc.tif', '456462_Cell_6_40x_1xtube_10A10_mc.tif',
                 '456462_Cell_5_40x_1xtube_10A8_mc.tif', '456462_Cell_5_40x_1xtube_10A9_mc.tif', 
                 '462149_Cell_3_40x_1xtube_10A3_mc.tif', '466769_Cell_2_40x_1xtube_10A_6_mc.tif',
                 '466769_Cell_2_40x_1xtube_10A_4_mc.tif', '466769_Cell_3_40x_1xtube_10A_8_mc.tif']
    
    freq_400 = [True, True, True, True, True, True, False, True, True, True, True, True, False, False, False, False]

    fnames = [os.path.join(movie_folder, file) for file in movie_lists[:16]]

    combined_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons',
                    '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons'][0]
    
    combined_lists = ['neuron0&1_x[1, -1]_y[1, -1].tif', 
                   'neuron0&1_x[2, -2]_y[2, -2].tif', 
                   'neuron1&2_x[4, -2]_y[4, -2].tif', 
                   'neuron1&2_x[6, -2]_y[8, -2].tif']
elif n_neurons == 'many':
    movie_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data'][0]
    
    movie_lists = ['demo_voltage_imaging_mc.hdf5', 
                   'FOV4_50um.hdf5']
#%%    
v_sg = []
all_f1_scores = []
all_prec = []
all_rec = []
all_snr = []

#%% Choosing datasets
for kk in range(8, 16):
    print(f'now processing neurons number {kk}')
    file_set = [kk]
    name = movie_lists[file_set[0]]
    if freq_400[file_set[0]] == True:
        frate = 400
    else:
        frate = 1000
    mov = imread(os.path.join(movie_folder, name))
    with h5py.File(os.path.join(movie_folder, name[:-7]+'_ROI.hdf5'),'r') as h5:
        mask = np.array(h5['mov'])

    #%%
    print(f'loading the movie')
    # original movie
    y = to_2D(-mov).copy()     
    y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
    y_filt = y_filt 
    
    do_plot = True
    if do_plot:
        plt.figure()
        plt.imshow(mov[0])
        plt.figure()
        if n_neurons == 'many':
            plt.imshow(mask.sum(0))    
        else:            
            for i in range(mask.shape[0]):
                plt.imshow(mask[i], alpha=0.5)
    
    #%% Use nmf sequentially to extract all neurons in the region
    print(f'One component nmf')
    num_frames_init = 10000
    y_seq = y_filt[:num_frames_init,:].copy()
    W_tot = []
    H_tot = []
    mask_2D = to_2D(mask)
    std = [np.std(y_filt[:, np.where(mask_2D[i]>0)[0]].mean(1)) for i in range(len(mask_2D))]
    seq = np.argsort(std)[::-1]
    print(f'sequence of rank1-nmf: {seq}')
    
    for i in seq:
        model = NMF(n_components=1, init='nndsvd', max_iter=100, verbose=False)
        y_temp, _ = select_masks(y_seq, mov[:num_frames_init].shape, mask=mask[i])
        W = model.fit_transform(np.maximum(y_temp,0))
        H = model.components_
        y_seq = y_seq - W@H
        W_tot.append(W)
        H_tot.append(H)
        if do_plot:
            plt.figure();plt.plot(W);
            plt.figure();plt.imshow(H.reshape(mov.shape[1:], order='F'));plt.colorbar()
    H = np.vstack(H_tot)
    W = np.hstack(W_tot)
    #%% Use hals to optimize masks
    print(f'HALS')
    update_bg = False
    y_input = np.maximum(y_filt[:num_frames_init], 0)
    y_input =to_3D(y_input, shape=(num_frames_init,mov.shape[1],mov.shape[2]), order='F').transpose([1,2,0])
    
    H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y.shape[1],1)) / y.shape[1],
                                 np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, 
                                 update_bg=update_bg, use_spikes=True)
    if do_plot:
        for i in range(mask.shape[0]):
            plt.figure();plt.imshow(H_new[:,i].reshape(mov.shape[1:], order='F'));plt.colorbar()
    
    from running_statistics import OnlineFilter       
    from time import time
    fe = slice(0,None)
    if update_bg:
        trace_nnls = np.array([nnls(np.hstack((H_new, b)),yy)[0] for yy in (-y)[fe]])
    else:
        trace_nnls = np.array([nnls(H.T,yy)[0] for yy in (-y)[fe]])
    filter_method = ['offline', 'butter', 'dc_blocker'][2]
    if filter_method == 'offline':
        trace_nnls = signal_filter(trace_nnls.T,freq = 1/3, fr=frate).T
        trace_nnls -= np.median(trace_nnls, 0)[np.newaxis, :]
        trace_nnls = -trace_nnls.T
        trace_all = trace_nnls 
    elif filter_method == 'butter': # filter online
        freq = 1/3
        trace_all = trace_nnls.T
        onFilt = OnlineFilter(freq=freq, fr=frate, mode='high')
        trace_filt = np.zeros_like(trace_all)
        trace_filt[:,:20000] = onFilt.fit(trace_all[:,:20000])    
        time0 = time()
        for i in range(20000, trace_all.shape[-1]):
            trace_filt[:,i] = onFilt.fit_next(trace_all[:,i])
        print(time()-time0)
        trace_all -= np.median(trace_all.T, 0)[np.newaxis, :].T
        trace_all = -trace_filt
    elif filter_method == 'dc_blocker':
        trace = trace_nnls.T
        trace_filt = np.zeros(trace.shape)        
        for tp in range(trace.shape[1]):
            if tp > 0:
                trace_filt[:, tp] = trace[:, tp] - trace[:, tp - 1] + 0.9 * trace_filt[:, tp - 1]
        trace_all = -trace_filt

   #plt.plot(trace_all/trace_all.max(), label='mean');plt.plot(trace1.flatten()/trace1.flatten().max(), label='nmf');plt.legend()
#%% Extract spikes and compute F1 score
    print(f'Spikes detection and F1 score')
    for idx, k in enumerate(list(file_set)):
        trace = trace_all[seq[idx]:seq[idx]+1, :].copy()
        #trace = dc_blocked[np.newaxis,:].copy()
        sao = SignalAnalysisOnline(thresh_STD=4, percentile_thr_sub=99)
        #sao.fit(trr_postf[:20000], len())
        #trace=dict1['v_sg'][np.newaxis, :]
        sao.fit(trace[:, :20000], num_frames=100000, frate=frate)
        for n in range(20000, trace.shape[1]):
            sao.fit_next(trace[:, n: n+1], n)
        sao.compute_SNR()
        print(f'SNR: {sao.SNR}')
        indexes = np.array((list(set(sao.index[0]) - set([0]))))  
        name_traces = '/'.join(fnames[k].split('/')[:-2] + ['data_new', 
                                   fnames[k].split('/')[-1][:-7]+'_output.npz'])
        # F1 score
        dict1 = np.load(name_traces, allow_pickle=True)
        indexes = np.delete(indexes, np.where(indexes >= dict1['v_t'].shape[0])[0])
        
        dict1_v_sp_ = dict1['v_t'][indexes]
        v_sg.append(dict1['v_sg'])
            
        for i in range(len(dict1['sweep_time']) - 1):
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
        dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
        
        precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned\
                            = metric(dict1['sweep_time'], dict1['e_sg'], 
                                  dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                  dict1['v_sg'], dict1_v_sp_ , 
                                  dict1['v_t'], dict1['v_sub'],save=False)
        
        print(np.array(F1).round(2).mean())
        all_f1_scores.append(np.array(F1).round(2))
        all_prec.append(np.array(precision).round(2))
        all_rec.append(np.array(recall).round(2))
        all_snr.append(sao.SNR[0].round(3))
  
#%%   
print(f'average_F1:{np.mean([np.nanmean(fsc) for fsc in all_f1_scores])}')
print(f'F1:{np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)}')
print(f'prec:{np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)}'); 
print(f'rec:{np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)}')

#%%
dictionary = {'all_f1_scores': all_f1_scores, 'all_prec': all_prec, 'all_rec': all_rec, 'all_snr': all_snr}
np.save(os.path.join(movie_folder, 'viola_test_result.npy'), dictionary)


