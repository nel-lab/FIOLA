#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:30:11 2020

@author: andrea
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:25:59 2020
Use hals algorithm to refine spatial components extracted by rank-1 nmf. 
Use nnls and sao for further processing.
@author: @agiovann and @caichangjia
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

#%% files for processing
base_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
               '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
               '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/'][-1]
lists = ['454597_Cell_0_40x_patch1_mc.tif', '456462_Cell_3_40x_1xtube_10A2_mc.tif',
             '456462_Cell_3_40x_1xtube_10A3_mc.tif', '456462_Cell_5_40x_1xtube_10A5_mc.tif',
             '456462_Cell_5_40x_1xtube_10A6_mc.tif', '456462_Cell_5_40x_1xtube_10A7_mc.tif', 
             '462149_Cell_1_40x_1xtube_10A1_mc.tif', '462149_Cell_1_40x_1xtube_10A2_mc.tif',
             '456462_Cell_4_40x_1xtube_10A4_mc.tif', '456462_Cell_6_40x_1xtube_10A10_mc.tif',
             '456462_Cell_5_40x_1xtube_10A8_mc.tif', '456462_Cell_5_40x_1xtube_10A9_mc.tif', 
             '462149_Cell_3_40x_1xtube_10A3_mc.tif', '466769_Cell_2_40x_1xtube_10A_6_mc.tif',
             '466769_Cell_2_40x_1xtube_10A_4_mc.tif', '466769_Cell_3_40x_1xtube_10A_8_mc.tif']
fnames = [os.path.join(base_folder, file) for file in lists]
freq_400 = [True, True, True, True, True, True, False, True, True, True, True, True, False, False, False, False]
movie_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons',
                '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons'][-1]

#%% Combine datasets
file_set = [0, 1]
name_set = fnames[file_set[0]: file_set[1] + 1]

name = 'neuron0&1_x[1, -1]_y[1, -1].tif'
frate = 400
mov = imread(os.path.join(movie_folder, name))
with h5py.File(os.path.join(movie_folder, name[:-4]+'_ROIs.hdf5'),'r') as h5:
   mask = np.array(h5['mov'])

# original movie
y = to_2D(-mov).copy()     
y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
y_filt = y_filt 

do_plot = False
if do_plot:
    plt.figure()
    plt.imshow(mov[0])
    plt.figure()
    plt.imshow(mask[0], alpha=0.5)
    plt.imshow(mask[1], alpha=0.5)

#%% Use nmf sequentially to extract all neurons in the region
num_frames_init = 20000
y_seq = y_filt[:num_frames_init,:].copy()
W_tot = []
H_tot = []
seq = [1,0]
for i in seq:
    model = NMF(n_components=1, init='nndsvd', max_iter=100, verbose=True)
    y_temp, _ = select_masks(y_seq, mov[:num_frames_init].shape, mask=mask[i])
    W = model.fit_transform(np.maximum(y_temp,0))
    H = model.components_
    y_seq = y_seq - W@H
    W_tot.append(W)
    H_tot.append(H)
    plt.figure();plt.plot(W);
    plt.figure();plt.imshow(H.reshape(mov.shape[1:], order='F'));plt.colorbar()
H = np.vstack(H_tot)
W = np.hstack(W_tot)
#%% Use hals to optimize masks
update_bg = False
y_input = np.maximum(y_filt[:num_frames_init], 0)
y_input =to_3D(y_input, shape=(num_frames_init,30,30), order='F').transpose([1,2,0])

H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y.shape[1],1)) / y.shape[1],
                             np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, 
                             update_bg=update_bg, use_spikes=True)
plt.close('all')
if do_plot:
    for i in range(2):
        plt.figure();plt.imshow(H_new[:,i].reshape(mov.shape[1:], order='F'));plt.colorbar()
    

#%%
use_GPU = False
if use_GPU: 
    mov_in = mov.transpose([1,2,0])
    Ab = H_new.astype(np.float32)
    template = np.median(mov_in, axis=-1)
    mc0 = mov_in[:,:,0:1][None, :]
    b =  to_2D(mov).T[:, 0]
    x0 = nnls(Ab,b)[0][:,None]
    AtA = Ab.T@Ab
    Atb = Ab.T@b
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
    #%%
    from pipeline_gpu import Pipeline, get_model
    model = get_model(template, Ab, 30)
    model.compile(optimizer='rmsprop', loss='mse')
    spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, mov_in[:,:,:100000])
    traces_viola = spike_extractor.get_spikes(100000)
    #%%
    traces_viola = np.array(traces_viola).squeeze().T
    traces_viola = signal_filter(traces_viola,freq = 1/3, fr=frate).T
    traces_viola -= np.median(traces_viola, 0)[np.newaxis, :]
    traces_viola = -traces_viola.T
    trace_all = np.hstack([traces_viola,np.zeros((traces_viola.shape[0],1))])

else:
    #%% Use nnls to extract signal for neurons
    fe = slice(0,None)
    if update_bg:
        trace_nnls = np.array([nnls(np.hstack((H_new, b)),yy)[0] for yy in (-y)[fe]])
    else:
        trace_nnls = np.array([nnls(H_new,yy)[0] for yy in (-y)[fe]])
    
    trace_nnls = signal_filter(trace_nnls.T,freq = 1/3, fr=frate).T
    trace_nnls -= np.median(trace_nnls, 0)[np.newaxis, :]
    trace_nnls = -trace_nnls.T
    if do_plot:
        plt.figure()
        plt.plot(trace_nnls.T) 
    trace_all = trace_nnls
#%%
# idxv = 0
# plt.plot(np.hstack([traces_viola[idxv],0])-trace_nnls[idxv])
# plt.plot(np.hstack([traces_viola[idxv],0])-trace_nnls[idxv])
# #%%
# plt.plot(trace_nnls[idxv])
# plt.plot(traces_viola[idxv])
#%%
#%% Extract spikes and compute F1 score
v_sg = []
all_f1_scores = []
all_prec = []
all_rec = []
all_snr = []
for idx, k in enumerate(list(file_set)):
    trace = trace_all[seq[idx]:seq[idx]+1, :]
    sao = SignalAnalysisOnline(thresh_STD=4)
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
show_frames = 50000 
#%matplotlib auto
t1 = normalize(trace_all[0])
t2 = normalize(trace_all[1])
t3 = normalize(v_sg[0])
t4 = normalize(v_sg[1])
plt.plot(dict1['v_t'][:show_frames], t1[:show_frames] + 0.5, label='neuron1')
plt.plot(dict1['v_t'][:show_frames], t2[:show_frames], label='neuron2')
plt.plot(dict1['v_t'][:show_frames], t3[:show_frames], label='gt1')
plt.plot(dict1['v_t'][:show_frames], t4[:show_frames]+0.5, label='gt2')
#plt.plot(dict1['e_t'], t4-3, label='ele')
#plt.vlines(dict1['e_sp'], -3, -2.5, color='black')
plt.legend()
  
#%%   
print(f'average_F1:{np.mean([np.nanmean(fsc) for fsc in all_f1_scores])}')
print(f'F1:{np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)}')
print(f'prec:{np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)}'); 
print(f'rec:{np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)}')
plt.pause(15)