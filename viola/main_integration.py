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
import os
from scipy.optimize import nnls    
from signal_analysis_online import SignalAnalysisOnlineZ
from skimage import measure
from sklearn.decomposition import NMF

from caiman_functions import signal_filter, to_3D, to_2D, bin_median
from metrics import metric
from nmf_support import hals, select_masks, normalize, nmf_sequential
from skimage.io import imread
from running_statistics import OnlineFilter
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
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data'][1]
    
    movie_lists = ['demo_voltage_imaging_mc.hdf5', 
                   'FOV4_50um_mc.hdf5']
    
#%% Choosing datasets
if n_neurons == '1':
    file_set = [i]
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
    name = movie_lists[0]
    frate = 400
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
       mov = np.array(h5['mov'])
    with h5py.File(os.path.join(movie_folder, name[:-8]+'_ROIs.hdf5'),'r') as h5:
       mask = np.array(h5['mov'])

#%% Preliminary processing
# Remove border pixel of the motion corrected movie
border_pixel = 2
mov[:, :border_pixel, :] = mov[:, border_pixel:border_pixel + 1, :]
mov[:, -border_pixel:, :] = mov[:, -border_pixel-1:-border_pixel, :]
mov[:, :, :border_pixel] = mov[:, :, border_pixel:border_pixel + 1]
mov[:, :, -border_pixel:] = mov[:, :, -border_pixel-1:-border_pixel]
      
# original movie
y = to_2D(-mov).copy()
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
    if n_neurons == 'many':
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

#%% Use hals to optimize masks
update_bg = False
y_input = np.maximum(y_filt[:num_frames_init], 0)
y_input = to_3D(y_input, shape=(num_frames_init,mov.shape[1],mov.shape[2]), order='F').transpose([1,2,0])

H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y_filt.shape[1],1)) / y_filt.shape[1],
                             np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, 
                             update_bg=update_bg, use_spikes=True)
plt.close('all')
if do_plot:
    for i in range(mask.shape[0]):
        plt.figure();plt.imshow(H_new[:,i].reshape(mov.shape[1:], order='F'));plt.colorbar()
    plt.figure();plt.imshow(H_new.sum(axis=1).reshape(mov.shape[1:], order='F'));plt.colorbar()
    
#%% Motion correct and use NNLS to extract signals
# You can skip rank 1-nmf, hals step if H_new is saved
#np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/FOV4_50um_H_new.npy', H_new)
H_new = np.load(os.path.join(movie_folder, name[:-8]+'_H_new.npy'))
use_GPU = True
use_batch = False
if use_GPU:
    mov_in = mov 
    Ab = H_new.astype(np.float32)
    #template = np.median(mov_in, axis=0)
    template = bin_median(mov_in, exclude_nans=False)
    center_dims = (template.shape[0], template.shape[1])
    if not use_batch:
        #b = to_2D(mov).T[:, 0]
        b = mov[0].reshape(-1, order='F')
        x0 = nnls(Ab,b)[0][:,None]
        AtA = Ab.T@Ab
        Atb = Ab.T@b
        n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
        theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
        mc0 = mov_in[0:1,:,:, None]

        from pipeline_gpu import Pipeline, get_model
        model = get_model(template, center_dims, Ab, 30)
        model.compile(optimizer='rmsprop', loss='mse')
        spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, mov_in[:,:,:100000])
        traces_viola = spike_extractor.get_spikes(3000)

    else:
    #FOR BATCHES:
        batch_size = 20
        num_frames = 20000
        num_components = Ab.shape[-1]

        #template = np.median(mov_in, axis=0)
        template = bin_median(mov_in, exclude_nans=False)
        #b = to_2D(mov).T[:, 0:batch_size]
        b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')
        x0=[]
        for i in range(batch_size):
            x0.append(nnls(Ab,b[:,i])[0])
        x0 = np.array(x0).T
        AtA = Ab.T@Ab
        Atb = Ab.T@b
        n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
        theta_2 = (Atb/n_AtA).astype(np.float32)

        from batch_gpu import Pipeline, get_model
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
    traces_viola = signal_filter(traces_viola,freq = 1/3, fr=frate).T
    traces_viola -= np.median(traces_viola, 0)[np.newaxis, :]
    traces_viola = -traces_viola.T
    trace_all = np.hstack([traces_viola,np.zeros((traces_viola.shape[0],1))])

else:
    #Use nnls to extract signal for neurons or not and filter
    from running_statistics import OnlineFilter       
    from time import time
    fe = slice(0,None)
    if update_bg:
        trace_nnls = np.array([nnls(np.hstack((H_new, b)),yy)[0] for yy in (-y)[fe]])
    else:
        trace_nnls = np.array([nnls(H_new,yy)[0] for yy in (-y)[fe]])
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
                trace_filt[:, tp] = trace[:, tp] - trace[:, tp - 1] + 0.995 * trace_filt[:, tp - 1]
        trace_all = -trace_filt
        trace_all = trace_all - np.median(trace_all)
    
#%% Viola spike extraction
if n_neurons == 'many':
    trace = trace_all[:].copy()
    saoz = SignalAnalysisOnlineZ(do_scale=True, freq=15, detrend=True, flip=True)
    saoz.fit(trace[:, :10000], num_frames=trace.shape[1])
    for n in range(10000, trace.shape[1]):
        saoz.fit_next(trace[:, n: n+1], n)
    saoz.compute_SNR()
    print(f'thresh:{saoz.thresh}')
    print(f'SNR: {saoz.SNR}')
    print(f'Mean_SNR: {np.array(saoz.SNR).mean()}')
    #print(f'sequence of rank1-nmf: {seq}')
    #print(f'Spikes:{(saoz.index>0).sum(1)}')
    print(f'Spikes based on mask sequence: {(saoz.index>0).sum(1)[np.argsort(seq)]}')

    #%% Load VolPy result
    name_estimates = ['demo_voltage_imaging_estimates.npy', 'FOV4_50um_estimates.npz']
    name_estimates = [os.path.join(movie_folder, name) for name in name_estimates]
    estimates = np.load(name_estimates[0], allow_pickle=True).item()
    
    #%% Check neurons
    idx = 1
    idx_volpy = seq[idx]
    #idx = np.where(seq==idx_volpy)[0][0]
    spikes_online = list(set(saoz.index[idx]) - set([0]))
    plt.figure();plt.imshow(H_new[:,idx].reshape(mov.shape[1:], order='F'));plt.colorbar()
    plt.figure()
    plt.plot(normalize(saoz.t_d[idx]), color='blue', label=f'online_{len(spikes_online)}')
    plt.plot(normalize(saoz.t_s[idx]), color='red', label=f'online_t_s')
    plt.plot(normalize(estimates['t'][idx_volpy]), color='orange', label=f'volpy_{estimates["spikes"][idx_volpy].shape[0]}')
    plt.vlines(spikes_online, -1.6, -1.2, color='blue', label='online_spikes')
    plt.vlines(estimates['spikes'][idx_volpy], -1.8, -1.4, color='orange', label='online_spikes')
    print(len(list(set(saoz.index[idx]))))
    plt.legend()   

    #%% Traces
    #idx_list = np.where((saoz.index_track>30))[0]
    idx_volpy = np.where(np.array([len(estimates['spikes'][k]) for k in range(len(estimates['spikes']))])>50)[0]
    idx_list = np.array([np.where(seq==idx_volpy[k])[0][0] for k in range(idx_volpy.size)])
    length = idx_list.size
    fig, ax = plt.subplots(idx_list.size,1)
    colorsets = plt.cm.tab10(np.linspace(0,1,10))
    colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]
    scope=[11000, 14000]
    
    for n, idx in enumerate(idx_list):
        idx_volpy = seq[idx]
        ax[n].plot(np.arange(20000), normalize(estimates['t'][idx_volpy]), 'c', linewidth=0.5, color='orange', label='volpy')
        ax[n].plot(np.arange(20000), normalize(saoz.trace[idx, :20000]), 'c', linewidth=0.5, color='blue', label='viola')
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
        saoz = SignalAnalysisOnlineZ()
        #saoz.fit(trr_postf[:20000], len())
        #trace=dict1['v_sg'][np.newaxis, :]
        saoz = SignalAnalysisOnlineZ(frate=frate, robust_std=False)
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
