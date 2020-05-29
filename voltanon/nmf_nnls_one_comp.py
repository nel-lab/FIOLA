#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:43:15 2020
One component nmf
@author: @agiovann and @caichangjia
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import nnls    
from signal_analysis_online import SignalAnalysisOnline
from sklearn.decomposition import NMF

import caiman as cm
from caiman.base.rois import nf_read_roi_zip
from caiman.source_extraction.volpy.spikepursuit import signal_filter
from caiman.base.movies import to_3D
from metrics import metric
from nmf_support import hals, combine_datasets, select_masks, normalize
#%% files for processing
base_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
               '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
               '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/'][1]
#lists = ['454597_Cell_0_40x_patch1.tif', '456462_Cell_3_40x_1xtube_10A2.tif',
#            '456462_Cell_3_40x_1xtube_10A3.tif', '456462_Cell_5_40x_1xtube_10A5.tif',
#             '456462_Cell_5_40x_1xtube_10A6.tif', '456462_Cell_5_40x_1xtube_10A7.tif', 
#             '462149_Cell_1_40x_1xtube_10A1.tif', '462149_Cell_1_40x_1xtube_10A2.tif', ]
lists = ['454597_Cell_0_40x_patch1_mc.tif', '456462_Cell_3_40x_1xtube_10A2_mc.tif',
             '456462_Cell_3_40x_1xtube_10A3_mc.tif', '456462_Cell_5_40x_1xtube_10A5_mc.tif',
             '456462_Cell_5_40x_1xtube_10A6_mc.tif', '456462_Cell_5_40x_1xtube_10A7_mc.tif', 
             '462149_Cell_1_40x_1xtube_10A1_mc.tif', '462149_Cell_1_40x_1xtube_10A2_mc.tif', ]
fnames = [os.path.join(base_folder, file) for file in lists]

#%% test pipeline on one neuron
all_f1_scores = []
all_prec = []
all_rec = []
all_snr = []
frate = 400
for k in list(range(0, 8)):
    mov = cm.load(fnames[k])#.resize(.5,.5,1)
    y = (cm.movie(-mov)).to_2D().copy()     
    y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
    name_traces = '/'.join(fnames[k].split('/')[:-2] + ['data_new', 
                               fnames[k].split('/')[-1][:-7]+'_output.npz'])
    dims = mov.shape[1:]
    mask = nf_read_roi_zip((fnames[k][:-7] + '_ROI.zip'), dims=dims)
        
    mode = ['direct_multiplication', 'nmf_nnls'][1]
    detrend_before = True
    
    if mode == 'direct_multiplication':
        trace = np.mean(y_filt[:, cm.movie(mask).to_2D()[0]>0], axis=1)[np.newaxis, :]
        
    elif mode == 'nmf_nnls':
        # rank-1 NMF
        num_frames_init = 20000        
        y_seq = y_filt[:num_frames_init,:].copy()
        model = NMF(n_components=1, init='nndsvd', max_iter=100, verbose=True)
        y_temp, _ = select_masks(y_seq, mov[:num_frames_init].shape, mask=mask[0])
        W = model.fit_transform(np.maximum(y_temp,0))
        H = model.components_
        plt.figure();plt.plot(W);
        plt.figure();plt.imshow(H.reshape(mov.shape[1:], order='F'));plt.colorbar()
            
        # hals optimization
        y_input = np.maximum(y_filt[:num_frames_init], 0)
        y_input = cm.movie(to_3D(y_input, shape=(num_frames_init, dims[0], dims[1]), order='F')).transpose([1,2,0])
        
        H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y.shape[1],1)) / y.shape[1],
                                     np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, 
                                     update_bg=False, use_spikes=True)
        
        plt.figure();plt.imshow(H_new.reshape(mov.shape[1:], order='F'));plt.colorbar()
        
        # nnls
        fe = slice(0,None)
        trace_all = np.array([nnls(H_new,yy)[0] for yy in (y - (y).min())[fe]]) 
        trace_all = signal_filter(trace_all.T,freq = 1/3, fr=frate).T
        trace_all = trace_all - np.median(trace_all, 0)[np.newaxis, :]
        plt.plot(trace_all)
        trace_all = trace_all.T
        trace = trace_all

    # SAO for spike extraction
    sao = SignalAnalysisOnline()
    sao.fit(trace[:, :20000], num_frames=100000)
    for n in range(20000, trace.shape[1]):
        sao.fit_next(trace[:, n: n+1], n)
    sao.compute_SNR()
    print(f'SNR: {sao.SNR}')
    indexes = np.array((list(set(sao.index[0]) - set([0]))))  
    plt.figure(); plt.plot(sao.trace_rm.flatten())
    
    # F1 score
    dict1 = np.load(name_traces, allow_pickle=True)
    dict1_v_sp_ = dict1['v_t'][indexes]
            
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
    #print(f'average_sub:{np.nanmean(all_corr_subthr,axis=0)}')
    print(f'F1:{np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)}')
    print(f'prec:{np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)}'); 
    print(f'rec:{np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)}')
    print(f'snr:{np.array(all_snr).round(3)}')
