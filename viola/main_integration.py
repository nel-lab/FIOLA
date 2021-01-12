#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:30:11 2020
Use hals algorithm to refine spatial components extracted by rank-1 nmf. 
Use nnls with gpu for signal extraction 
Use online template matching (saoz) for spike extraction
@author: @agiovann, @caichangjia, @cynthia
"""
#%%
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
from scipy.optimize import nnls    
from viola.signal_analysis_online import SignalAnalysisOnlineZ
from skimage import measure
from sklearn.decomposition import NMF

from viola.caiman_functions import signal_filter, to_3D, to_2D, bin_median, play
from viola.metrics import metric
from viola.nmf_support import hals, select_masks, normalize, nmf_sequential
from skimage.io import imread
from viola.running_statistics import OnlineFilter
from viola.match_spikes import match_spikes_greedy, compute_F1

from time import time
# from viola.pipeline_gpu import Pipeline, get_model
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass
#%% files for processing
n_neurons = ['1', '2', 'many', 'test'][0]

if n_neurons in ['1', '2']:
    movie_folder = ['/Users/agiovan/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron/',
                   '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron',
                   '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/'][0]
    
    movie_lists = ['454597_Cell_0_40x_patch1', '456462_Cell_3_40x_1xtube_10A2',
                 '456462_Cell_3_40x_1xtube_10A3', '456462_Cell_5_40x_1xtube_10A5',
                 '456462_Cell_5_40x_1xtube_10A6', '456462_Cell_5_40x_1xtube_10A7', 
                 '462149_Cell_1_40x_1xtube_10A1', '462149_Cell_1_40x_1xtube_10A2',
                 '456462_Cell_4_40x_1xtube_10A4', '456462_Cell_6_40x_1xtube_10A10',
                 '456462_Cell_5_40x_1xtube_10A8', '456462_Cell_5_40x_1xtube_10A9', 
                 '462149_Cell_3_40x_1xtube_10A3', '466769_Cell_2_40x_1xtube_10A_6',
                 '466769_Cell_2_40x_1xtube_10A_4', '466769_Cell_3_40x_1xtube_10A_8', 
                 '09282017Fish1-1', '10052017Fish2-2', 'Mouse_Session_1']
    
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
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons', 
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/test'][2]
   
    movie_lists = ['demo_voltage_imaging_mc.hdf5', 
                   'FOV4_50um_mc.hdf5',
                   '06152017Fish1-2_portion.hdf5', 
                   'FOV4_50um.hdf5', 
                   'viola_sim1_1.hdf5']
    
elif n_neurons == 'test':
    movie_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/overlapping/viola_sim3_18',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/overlapping/viola_sim3_16',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/overlapping/viola_sim3_4',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/overlapping/viola_sim3_2',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/overlapping/viola_sim3_5',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/non_overlapping/viola_sim2_7'][4]
    movie_lists = ['viola_sim3_18.hdf5',   # overlapping 8 neurons
                   'viola_sim3_16.hdf5',
                   'viola_sim3_4.hdf5',
                   'viola_sim3_2.hdf5',
                   'viola_sim3_5.hdf5',
                   'viola_sim2_7.hdf5']    # non-overlapping 50 neurons
    
#%% Choosing datasets
if n_neurons == '1':
    file_set = [-2]
    name = movie_lists[file_set[0]]
    belong_Marton = True
    if ('Fish' in name) or ('Mouse' in name):
        belong_Marton = False
    frate = frate_all[file_set[0]]
    mov = imread(os.path.join(movie_folder, name, name+'_mc.tif'))
    with h5py.File(os.path.join(movie_folder, name, name+'_ROI.hdf5'),'r') as h5:
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
    name = movie_lists[0]
    frate = 400
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
       mov = np.array(h5['mov'])
    with h5py.File(os.path.join(movie_folder, name[:-5]+'_ROIs.hdf5'),'r') as h5:
       mask = np.array(h5['mov'])
       
elif n_neurons == 'test':
    name = movie_lists[4]
    frate = 400
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
       mov = np.array(h5['mov'])
    with h5py.File(os.path.join(movie_folder, 'viola', 'ROIs_gt.hdf5'),'r') as h5:
       mask = np.array(h5['mov'])
    

#%% Preliminary processing
# Remove border pixel of the motion corrected movie
border_pixel = 2
mov[:, :border_pixel, :] = mov[:, border_pixel:border_pixel + 1, :]
mov[:, -border_pixel:, :] = mov[:, -border_pixel-1:-border_pixel, :]
mov[:, :, :border_pixel] = mov[:, :, border_pixel:border_pixel + 1]
mov[:, :, -border_pixel:] = mov[:, :, -border_pixel-1:-border_pixel]
      
# original movie !!!!
flip = True
if flip == True:
    y = to_2D(-mov).copy()
else:
    y = to_2D(mov).copy()
use_signal_filter = True   
if use_signal_filter:  # consume lots of memory
    y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
else:   # maybe not a good idea
    y_filt = np.zeros(y.shape)        
    for tp in range(y.shape[1]):
        if tp > 0:
            y_filt[:, tp] = y[:, tp] - y[:, tp - 1] + 0.995 * y_filt[:, tp - 1]
 
do_plot = True
if do_plot:
    plt.figure()
    plt.imshow(mov[0])
    plt.figure()
    if n_neurons == 'many' or 'test':
        plt.imshow(mask.sum(0))    
    else:            
        for i in range(mask.shape[0]):
            plt.imshow(mask[i], alpha=0.5)

#%% Use nmf sequentially to extract all neurons in the region
num_frames_init = 10000
y_seq = y_filt[:num_frames_init,:].copy()

mask_2D = to_2D(mask)
std = [np.std(y_filt[:, np.where(mask_2D[i]>0)[0]].mean(1)) for i in range(len(mask_2D))]
seq = np.argsort(std)[::-1]
print(f'sequence of rank1-nmf: {seq}')
W, H = nmf_sequential(y_seq, mask=mask, seq=seq, small_mask=True)
nA = np.linalg.norm(H)
H = H/nA
W = W*nA

#%%
update_bg = True
use_spikes = True
hals_positive = False

if hals_positive:
    y_input = np.maximum(y_filt[:num_frames_init], 0)
    H_new,W_new,b,f = hals(y_input[:num_frames_init].T, H.T, W.T, np.ones((y_filt.shape[1],1)) / y_filt.shape[1],
                                 np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, 
                                 update_bg=update_bg, use_spikes=use_spikes, frate=frate)

else:
    H_new,W_new,b,f = hals(y_filt[:num_frames_init].T, H.T, W.T, np.ones((y_filt.shape[1],1)) / y_filt.shape[1],
                                 np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, 
                                 update_bg=update_bg, use_spikes=use_spikes, frate=frate)
if do_plot:
    plt.figure();plt.imshow(H_new.sum(axis=1).reshape(mov.shape[1:], order='F'));plt.colorbar()
    plt.figure();plt.imshow(b.reshape(mov.shape[1:], order='F'));plt.colorbar()

if update_bg:
     H_new = np.hstack((H_new, b))

# normalization will enable gpu-nnls extracting bg signal 
H_new = H_new / norm(H_new, axis=0)
    
#%% Motion correct and use NNLS to extract signals
# You can skip rank 1-nmf, hals step if H_new is saved
#np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data//multiple_neurons/FOV4_50um_H_new.npy', H_new)
#H_new = np.load(os.path.join(movie_folder, name[:-8]+'_H_new.npy'))
use_GPU = False
use_batch = False
if use_GPU:
    mov_in = mov 
    Ab = H_new.astype(np.float32)
    template = bin_median(mov_in, exclude_nans=False)
    center_dims =(template.shape[0], template.shape[1])
    if not use_batch:
        b = mov[0].reshape(-1, order='F')
        x0 = nnls(Ab,b)[0][:,None]
        AtA = Ab.T@Ab
        Atb = Ab.T@b
        n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
        theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
        mc0 = mov_in[0:1,:,:, None]

        
        model = get_model(template, center_dims, Ab, 30)
        model.compile(optimizer='rmsprop', loss='mse')
        spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, mov_in[:,:,:20000])
        traces_viola = spike_extractor.get_traces(20000)

    else:
    #FOR BATCHES:
        batch_size = 20
        num_frames = 20000
        num_components = Ab.shape[-1]

        template = bin_median(mov_in, exclude_nans=False)
        b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')
        x0=[]
        for i in range(batch_size):
            x0.append(nnls(Ab,b[:,i])[0])
        x0 = np.array(x0).T
        AtA = Ab.T@Ab
        Atb = Ab.T@b
        n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
        theta_2 = (Atb/n_AtA).astype(np.float32)

        from viola.batch_gpu import Pipeline, get_model
        model_batch = get_model(template, center_dims, Ab, num_components, batch_size)
        model_batch.compile(optimizer = 'rmsprop',loss='mse')
        mc0 = mov_in[0:batch_size, :, :, None][None, :]
        x_old, y_old = np.array(x0[None,:]), np.array(x0[None,:])
        spike_extractor = Pipeline(model_batch, x_old, y_old, mc0, theta_2, mov_in, num_components, batch_size)
        spikes_gpu = spike_extractor.get_spikes(1000)
        traces_viola = []
        for spike in spikes_gpu:
            for i in range(batch_size):
                traces_viola.append([spike[:,:,i]])

    traces_viola = np.array(traces_viola).squeeze().T
    trace_all = traces_viola.copy()

else:
    #Use nnls to extract signal for neurons or not and filter
    
    fe = slice(0,None)
    trace_nnls = np.array([nnls(H_new,yy)[0] for yy in (-y)[fe]])
    trace_all = trace_nnls.T.copy() 
    

#%% Viola spike extraction, result is in the estimates object
if True:
    trace = trace_all[:].copy()
    saoz = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                  detrend=True, flip=True, 
                                  frate=frate, thresh_range=[2.8, 5.0], 
                                  filt_window=15, mfp=0.1)
    saoz.fit(trace[:, :10000], num_frames=trace.shape[1])
    for n in range(10000, trace.shape[1]):
        saoz.fit_next(trace[:, n: n+1], n)
    saoz.compute_SNR()
    saoz.reconstruct_signal()
    print(f'thresh:{saoz.thresh}')
    print(f'SNR: {saoz.SNR}')
    print(f'Mean_SNR: {np.array(saoz.SNR).mean()}')
    print(f'Spikes based on mask sequence: {(saoz.index>0).sum(1)[np.argsort(seq)]}')
    estimates = saoz
    estimates.spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz.index])
    weights = H_new.reshape((mov.shape[1], mov.shape[2], H_new.shape[1]), order='F')
    weights = weights.transpose([2, 0, 1])
    estimates.weights = weights
    
#%% Visualization
    idx = 0
    plt.imshow(estimates.weights[idx])   # weight
    plt.plot(trace[idx])                # original trace
    plt.plot(estimates.t_s[idx])        # after template matching

    
#%%
    #SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/ephys_voltage/10052017Fish2-2'
    #save_path = os.path.join(SAVE_FOLDER, 'viola', f'viola_update_bg_{update_bg}_use_spikes_{use_spikes}')
    #np.save(save_path, estimates)    
    
#%% Load simulation groundtruth
    import scipy.io
    vi_result_all = []
    gt_files = [file for file in os.listdir(movie_folder) if 'SimResults' in file]
    gt_file = gt_files[0]
    gt = scipy.io.loadmat(os.path.join(movie_folder, gt_file))
    gt = gt['simOutput'][0][0]['gt']
    spikes = gt['ST'][0][0][0]


    
#%% Compute F1 score for spike extraction
    n_cells = spikes.shape[0]
    rr = {'F1':[], 'precision':[], 'recall':[]}
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        s2 = estimates.spikes[np.argsort(seq)][idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
        F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
        rr['F1'].append(F1)
        rr['precision'].append(precision)
        rr['recall'].append(recall)  
        
    plt.boxplot(rr['F1']); plt.title('viola')
    
#%%    
##############################################################################################################
    
    #%%
    vpy = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/test/volpy_viola_sim1_1_adaptive_threshold.npy', allow_pickle=True).item()        
    idx = 0
    plt.plot(normalize(vpy['t'][idx]))
    plt.plot(normalize(saoz_5000.t0[np.argsort(seq)][idx]))
    
    #%%
    trace = trace_all[:].copy()
    saoz_5000 = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                      detrend=True, flip=True, 
                                      frate=frate, thresh_range=[2.8, 5.0], 
                                      filt_window=15, mfp=0.1)
    saoz_5000.fit(trace[:, :5000], num_frames=trace.shape[1])
    for n in range(5000, trace.shape[1]):
        saoz_5000.fit_next(trace[:, n: n+1], n)
    saoz_5000.compute_SNR()
    saoz_5000.reconstruct_signal()
#%%
    trace = trace_all[:].copy()
    saoz_20000 = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                      detrend=True, flip=True, 
                                      frate=frate, thresh_range=[2.8, 5.0], 
                                      mfp=0.2)
    saoz_20000.fit(trace[:, :20000], num_frames=trace.shape[1])
    saoz_20000.compute_SNR()
    saoz_20000.reconstruct_signal()
    
    
    #%% I want to see distribution of thresh 
    x = saoz_5000.thresh_factor.flatten()[:50]
    y = saoz_20000.thresh_factor.flatten()[:50]
    #plt.scatter(x+np.random.rand(50)/50 , y+np.random.rand(55)/50) 
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)
    x_pred = np.arange(2.5, 4, 0.1)[:, np.newaxis]    
    y_pred = lr.predict(x_pred)
    plt.scatter(x+np.random.rand(50)/50 , y+np.random.rand(50)/50) 
    plt.plot(x_pred, y_pred)    
    
    plt.figure(); plt.hist(x); plt.hist(y, color='r'); plt.legend(['5000', '20000']);plt.title('distribution of threshold')
    # Conclusion: running offline with 20000 frames yield more consistent thresh distribution result
    
    #%% I want to see trace produced by 5000 and 20000
    idx = 1
    x1 = saoz_5000.t_s[idx]
    plt.plot(x1)
    x2 = saoz_20000.t_s[idx]
    plt.plot(x2)
    # trace does not look bad to me
    
    #%%
    for idx in range(1,4):
        plt.figure(); plt.plot(saoz_5000.t[idx]); plt.plot(saoz_5000.t_sub[idx]); #plt.plot(saoz_5000.t_d[idx])  
    #%%
    s1 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_5000.index])
    s2 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_20000.index])

    n_cells = 50
    from match_spikes import match_spikes_greedy, compute_F1
    rr = {'F1':[], 'precision':[], 'recall':[], 'TP':[], 'FP':[], 'FN':[]}
    for idx in range(n_cells):
        ss1 = spikes[seq][idx].flatten()
        ss2 = s1[idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(ss1, ss2, max_dist=4)
        F1, precision, recall, TP, FP, FN = compute_F1(ss1, ss2, idx1_greedy, idx2_greedy)   
        for keys, values in rr.items():
            rr[keys].append(eval(keys))
    
    fig, ax = plt.subplots(1, 3)
    plt.suptitle('viola initialized 5000 frames')
    ax[0].boxplot(rr['F1']); ax[0].set_title('F1')
    ax[1].boxplot(rr['precision']); ax[1].set_title('precision')
    ax[2].boxplot(rr['recall']); ax[2].set_title('recall')
    plt.tight_layout()
    
    print(f'Mean: {np.array(rr["F1"]).mean()}, Median: {np.median(np.array(rr["F1"]))}')
    
    rr1 = rr.copy()
    
    #%%
    n_cells = 50
    from match_spikes import match_spikes_greedy, compute_F1
    rr = {'F1':[], 'precision':[], 'recall':[], 'TP':[], 'FP':[], 'FN':[]}
    for idx in range(n_cells):
        ss1 = spikes[seq][idx].flatten()
        ss2 = s2[idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(ss1, ss2, max_dist=4)
        F1, precision, recall, TP, FP, FN = compute_F1(ss1, ss2, idx1_greedy, idx2_greedy)   
        for keys, values in rr.items():
            rr[keys].append(eval(keys))
    
    fig, ax = plt.subplots(1, 3)
    plt.suptitle('viola initialized 5000 frames')
    ax[0].boxplot(rr['F1']); ax[0].set_title('F1')
    ax[1].boxplot(rr['precision']); ax[1].set_title('precision')
    ax[2].boxplot(rr['recall']); ax[2].set_title('recall')
    plt.tight_layout()
    
    print(f'Mean: {np.array(rr["F1"]).mean()}, Median: {np.median(np.array(rr["F1"]))}')
    
    rr2 = rr.copy()
    
    #%% see spikes
    idx = 1
    x1 = saoz_5000.t0[idx]
    x1_sub = saoz_5000.t_sub[idx]
    plt.plot(x1)
    plt.plot(x1_sub)
    x2 = saoz_20000.t0[idx]
    x2_sub = saoz_20000.t_sub[idx]
    plt.plot(x2)
    plt.plot(x2_sub)
    plt.legend(['5000', '5000_sub', '20000', '20000_sub'])
    h = np.max([np.max(x1), np.max(x2)])
    plt.scatter(s1[idx], np.ones((len(s1[idx])))*h, color='blue')
    plt.scatter(s2[idx], np.ones((len(s2[idx])))*h+0.5, color='green')
    plt.vlines(spikes[seq][idx], 0, h+0.5)
    
    print(f'{[{keys:rr1[keys][idx]} for keys in rr1]}')    
    print(f'{[{keys:rr2[keys][idx]} for keys in rr2]}')    
    
    
    
    
    #%%
    # threshold is not that important
    # online filter suspicious to me
    # online filter is not good, it produces more FP
    
    #%%
    from match_spikes import match_spikes_greedy, compute_F1
    s2 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_20000.index])
        
    filt_windows = np.arange(9, 25, 2)
    mfps = np.arange(0.005,0.05, 0.005)
    rr_all = {}
    for filt_window in filt_windows:
        for mfp in mfps:
            trace = trace_all[:].copy()
            saoz_5000 = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                              detrend=True, flip=True, 
                                              frate=frate, thresh_range=[2.8, 5.0], 
                                              filt_window=filt_window, mfp=mfp)
            saoz_5000.fit(trace[:, :5000], num_frames=trace.shape[1])
            for n in range(5000, trace.shape[1]):
                saoz_5000.fit_next(trace[:, n: n+1], n)
            #saoz_5000.compute_SNR()
            #saoz_5000.reconstruct_signal()
            s1 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_5000.index])
            n_cells = 50
            rr = {'F1':[], 'precision':[], 'recall':[], 'TP':[], 'FP':[], 'FN':[]}
            for idx in range(n_cells):
                ss1 = spikes[seq][idx].flatten()
                ss2 = s1[idx]
                idx1_greedy, idx2_greedy = match_spikes_greedy(ss1, ss2, max_dist=4)
                F1, precision, recall, TP, FP, FN = compute_F1(ss1, ss2, idx1_greedy, idx2_greedy)   
                for keys, values in rr.items():
                    rr[keys].append(eval(keys))
            
            """
            fig, ax = plt.subplots(1, 3)
            plt.suptitle('viola initialized 5000 frames')
            ax[0].boxplot(rr['F1']); ax[0].set_title('F1')
            ax[1].boxplot(rr['precision']); ax[1].set_title('precision')
            ax[2].boxplot(rr['recall']); ax[2].set_title('recall')
            plt.tight_layout()
            """
            rr_all[(filt_window, mfp)]= rr.copy()
    
    #%%
    F1_mean = [np.median(np.array(rr['F1'])) for rr in rr_all.values()]
    plt.plot(filt_windows, F1_mean)    
    
    #%%
    mat = np.zeros((len(filt_windows), len(mfps)))
    for idx_x, filt_window in enumerate(filt_windows):
        for idx_y, mfp in enumerate(mfps):
            rr = rr_all[(filt_window, mfp)]
            mat[idx_x, idx_y] = np.median(np.array(rr['F1']))
    
    plt.title('Heatmap for F1 score'); plt.imshow(mat); plt.colorbar(); plt.yticks(list(range(len(filt_windows))), filt_windows)
    plt.xticks(list(range(len(mfps))), mfps); plt.xlabel('mfps'); plt.ylabel('filt_windows')
    


    #%% use adaptive threshold  
    from match_spikes import match_spikes_greedy, compute_F1
    s2 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_20000.index])
        
    filt_windows = np.array([15])
    rr_all = {}
    for filt_window in filt_windows:
        trace = trace_all[:].copy()
        saoz_5000 = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                          detrend=True, flip=True, 
                                          frate=frate, thresh_range=[2.8, 5.0], 
                                          filt_window=filt_window, mfp=0.2, do_plot=False)
        saoz_5000.fit(trace[:, :5000], num_frames=trace.shape[1])
        for n in range(5000, trace.shape[1]):
            saoz_5000.fit_next(trace[:, n: n+1], n)
        #saoz_5000.compute_SNR()
        #saoz_5000.reconstruct_signal()
        s1 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_5000.index])
        n_cells = 50
        rr = {'F1':[], 'precision':[], 'recall':[], 'TP':[], 'FP':[], 'FN':[]}
        for idx in range(n_cells):
            ss1 = spikes[seq][idx].flatten()
            ss2 = s1[idx]
            idx1_greedy, idx2_greedy = match_spikes_greedy(ss1, ss2, max_dist=4)
            F1, precision, recall, TP, FP, FN = compute_F1(ss1, ss2, idx1_greedy, idx2_greedy)   
            for keys, values in rr.items():
                rr[keys].append(eval(keys))
        
        fig, ax = plt.subplots(1, 3)
        plt.suptitle('viola adaptive threshold initialized 5000 frames')
        ax[0].boxplot(rr['F1']); ax[0].set_title('F1')
        ax[1].boxplot(rr['precision']); ax[1].set_title('precision')
        ax[2].boxplot(rr['recall']); ax[2].set_title('recall')
        plt.tight_layout()
        
        rr_all[(filt_window, mfp)]= rr.copy()      

#%%
    mat = np.zeros((len(filt_windows), len([1])))
    for idx_x, filt_window in enumerate(filt_windows):
        rr = rr_all[(filt_window, mfp)]
        mat[idx_x, 0] = np.median(np.array(rr['F1']))
    
    plt.title('Heatmap for F1 score adaptive threshold'); plt.imshow(mat); plt.colorbar(); plt.yticks(list(range(len(filt_windows))), filt_windows)
    plt.xticks(list(range(len(mfps))), mfps); plt.xlabel('mfps'); plt.ylabel('filt_windows')
        
    # two problems, online filtering problem and spike extraction problem
    # filtering problem: small filt_window (9 frames as default, that means 4 frames lag) works much worse than (15 frames)
    # threshold problem: adaptive threshold works well; mfp the smaller the better (0.01 best, 0.2 default), not suitable in this case
    # after changing filt_window size to 15, changing to adaptive threshold method, F1 score 0.935 
    #%% Load VolPy result
    name_estimates = ['demo_voltage_imaging_estimates.npy', 'FOV4_50um_estimates.npz', 
                      '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_06152017Fish1-2.npy',
                      '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_IVQ32_S2_FOV1.npy', 
                      '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_FOV4_35um.npy' ]
    #name_estimates = [os.path.join(movie_folder, name) for name in name_estimates]
    estimates = np.load(name_estimates[4], allow_pickle=True).item()
    
    #%% Check neurons
    idx = 5
    idx_volpy = seq[idx]
    idx = np.where(seq==idx_volpy)[0][0]
    spikes_online = list(set(saoz.index[idx]) - set([0]))
    #plt.figure();plt.imshow(H_new[:,idx].reshape(mov.shape[1:], order='F'));plt.colorbar()
    plt.figure();plt.imshow(H_new[:,idx].reshape(mov.shape[1:], order='F'))
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

    #%% Traces
    idx_list = np.where((saoz.index_track> 50))[0]
    idx_volpy  = seq[idx_list]
    #idx_volpy = np.where(np.array([len(estimates['spikes'][k]) for k in range(len(estimates['spikes']))])>30)[0]
    #idx_list = np.array([np.where(seq==idx_volpy[k])[0][0] for k in range(idx_volpy.size)])
    length = idx_list.size
    fig, ax = plt.subplots(idx_list.size,1)
    colorsets = plt.cm.tab10(np.linspace(0,1,10))
    colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]
    scope=[0, 20000]
    
    for n, idx in enumerate(idx_list):
        idx_volpy = seq[idx]
        ax[n].plot(np.arange(scope[1]), normalize(estimates['t'][idx_volpy]), 'c', linewidth=0.5, color='orange', label='volpy')
        ax[n].plot(np.arange(scope[1]), normalize(saoz.t_s[idx, :scope[1]]), 'c', linewidth=0.5, color='blue', label='viola')
        #ax[n].plot(np.arange(20000), normalize(saoz.t_s[idx, :20000]), 'c', linewidth=0.5, color='red', label='viola')
        #ax[n].plot(normalize(saoz.t_s[idx]), color='blue', label=f'viola_t_s')
    
        spikes_online = list(set(saoz.index[idx]) - set([0]))
        ax[n].vlines(spikes_online, 1.2, 1.5, color='red', label='viola_spikes')
        ax[n].vlines(estimates['spikes'][idx_volpy], 1.5, 1.8, color='black', label='volpy_spikes')
    
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
            
    plt.tight_layout()
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/whole_FOV/traces.pdf')
 
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
        #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/whole_FOV/spatial_masks.pdf')
    
    #%%
    saoz.seq = seq
    saoz.H = H_new
    np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/viola_FOV4_35um', saoz)
#%% Extract spikes and compute F1 score
if n_neurons in ['1', '2']:
    for idx, k in enumerate(list(file_set)):
        name_traces = '/'.join(fnames[k].split('/')[:-2] + ['one_neuron_result', 
                                   fnames[k].split('/')[-1][:-7]+'_output.npz'])
        # F1 score
        dict1 = np.load(name_traces, allow_pickle=True)
        length = dict1['v_sg'].shape[0]
        
        trace = trace_all[seq[idx]:seq[idx]+1, :].copy()
        #trace = dc_blocked[np.newaxis,:].copy()
        saoz = SignalAnalysisOnlineZ(do_scale=True, thresh_range=[2.8, 5], frate=frate, robust_std=False, detrend=True, flip=True)
        #saoz.fit(trr_postf[:20000], len())
        #trace=dict1['v_sg'][np.newaxis, :]
        saoz.fit(trace_all[:20000], trace_all.shape[1])
        for n in range(20000, trace_all.shape[1]):
            saoz.fit_next(trace_all[:, n:n+1], n)
        saoz.compute_SNR()
        indexes = np.array(list(set(saoz.index[0]) - set([0])))
        thresh = saoz.thresh_factor[0, 0]
        snr = saoz.SNR
        indexes = np.delete(indexes, np.where(indexes >= dict1['v_t'].shape[0])[0])
        
        dict1_v_sp_ = dict1['v_t'][indexes]
        v_sg.append(dict1['v_sg'])
            
        if belong_Marton:
            for i in range(len(dict1['sweep_time']) - 1):
                dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
        
        precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned\
                            = metric(dict1['sweep_time'], dict1['e_sg'], 
                                  dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                  dict1['v_sg'], dict1_v_sp_ , 
                                  dict1['v_t'], dict1['v_sub'],save=False, belong_Marton=belong_Marton)
        
        p = len(e_match)/len(v_spike_aligned)
        r = len(e_match)/len(e_spike_aligned)
        f = (2 / (1 / p + 1 / r))
print(np.array(F1).round(2).mean())
all_f1_scores.append(np.array(F1).round(2))
all_prec.append(np.array(precision).round(2))
all_rec.append(np.array(recall).round(2))
all_thresh.append(thresh)
all_snr.append(snr)
compound_f1_scores.append(f)                
compound_prec.append(p)
compound_rec.append(r)
print(f'average_F1:{np.mean([np.nanmean(fsc) for fsc in all_f1_scores])}')
#print(f'average_sub:{np.nanmean(all_corr_subthr,axis=0)}')
print(f'F1:{np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)}')
print(f'prec:{np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)}'); 
print(f'rec:{np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)}')
print(f'average_compound_f1:{np.mean(np.array(compound_f1_scores)).round(3)}')
print(f'compound_f1:{np.array(compound_f1_scores).round(2)}')
print(f'compound_prec:{np.array(compound_prec).round(2)}')
print(f'compound_rec:{np.array(compound_rec).round(2)}')
print(f'snr:{np.array(all_snr).round(2)}')
dict2 = {}
dict2['trace'] = saoz.trace
dict2['indexes'] = sorted(indexes)
dict2['t_s'] = saoz.t_s
dict2['snr'] = saoz.SNR
dict2['sub'] = saoz.sub
dict2['template'] = saoz.PTA
dict2['thresh'] = saoz.thresh
dict2['thresh_factor'] = saoz.thresh_factor
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result'
np.save(os.path.join(save_folder, 'spike_detection_saoz_'+ name[:-7]  +'_output'), dict2)
        
#%%
dict1 = {}
dict1['average_F1'] = np.mean([np.nanmean(fsc) for fsc in all_f1_scores])
dict1['F1'] = np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)
dict1['prec'] = np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)
dict1['rec'] = np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)
dict1['average_compound_f1'] = np.mean(np.array(compound_f1_scores)).round(3)
dict1['compound_f1'] = np.array(compound_f1_scores).round(2)
dict1['compound_prec'] = np.array(compound_prec).round(2)
dict1['compound_rec'] =  np.array(compound_rec).round(2)
dict1['snr'] = np.array(all_snr).round(2).T

np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/saoz_training.npy', dict1)
   
#%%
if n_neurons == '1':
#plt.plot(dict1['v_t'], saoz.trace.flatten())
plt.plot(dict1['v_t'], normalize(saoz.trace.flatten()))
plt.plot(dict1['e_t'], normalize(dict1['e_sg'])/10)
plt.vlines(dict1_v_sp_, -2, -1)    


elif n_neurons == '2':
show_frames = 80000 
#%matplotlib auto
t1 = normalize(trace_all[0])
t2 = normalize(trace_all[1])
t3 = normalize(v_sg[0])
t4 = normalize(v_sg[1])
plt.plot(dict1['v_t'][:show_frames], t1[:show_frames] + 0.5, label='neuron1')
plt.plot(dict1['v_t'][:show_frames], t2[:show_frames], label='neuron2')
plt.plot(dict1['v_t'][:show_frames], t3[:show_frames] + 0.5, label='gt1')
plt.plot(dict1['v_t'][:show_frames], t4[:show_frames], label='gt2')
#plt.plot(dict1['e_t'], t4-3, label='ele')
#plt.vlines(dict1['e_sp'], -3, -2.5, color='black')
plt.legend()
  
#%%   
print(f'average_F1:{np.mean([np.nanmean(fsc) for fsc in all_f1_scores])}')
print(f'F1:{np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)}')
print(f'prec:{np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)}'); 
print(f'rec:{np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)}')

#%% corr
corr = []
#file_set = np.arange(0,8)
names = []
file_set = np.array([7])
for idx, k in enumerate(list(file_set)):
    name_traces = '/'.join(fnames[k].split('/')[:-2] + ['one_neuron_result', 
                               fnames[k].split('/')[-1][:-7]+'_output.npz'])
    # F1 score
    dict1 = np.load(name_traces, allow_pickle=True)
    corr.append(np.corrcoef(dict1['e_sub'][:20000], dict1['v_sub'][:20000])[0, 1])
    names.append(fnames[k].split('/')[-1][:13])
    
#%%
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

plt.bar(file_set, corr)    
plt.xticks(file_set, names, rotation='vertical', fontsize=4)  
plt.ylabel('corrcoef')  
np.array(corr).mean()
plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Backup/2020-06-29-Jannelia-meeting/bar_plot.pdf')
#%%
plt.figure()
plt.plot(normalize(dict1['e_sub'][:20000]), label='ephys sub')
plt.plot(normalize(dict1['v_sub'][:20000]), label='volpy sub')
#np.corrcoef(dict1['e_sub'][:20000], dict1['v_sub'][:20000])
plt.legend()
plt.xlim([0,3000])
plt.ylim([-1,1])
plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Backup/2020-06-29-Jannelia-meeting/dataset_7.pdf')

#%% reconstruction
shape = mask.shape[1:]
scope = [0, 1000]
saoz.reconstruct_movie(H_new, shape, scope)

#%%
mov_new = saoz.mov_rec.copy()
play(mov_new, fr=400, q_min=0.01)        
#mv_bl = mv.computeDFF(secsWindow=0.1)[0]

#%%
y = to_2D(-mov[scope[0]:scope[1]]).copy()
y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
mov_detrend = to_3D(y_filt, shape=mov[scope[0]:scope[1]].shape)
play(mov_detrend, q_min=80, q_max=98)

