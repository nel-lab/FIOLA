#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:25:59 2020
nmf nnls on multiple neurons
@author: @agiovann and @caichangjia
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import nnls    
from signal_analysis_online import SignalAnalysisOnline
from sklearn.decomposition import NMF

import caiman as cm
from caiman.source_extraction.volpy.spikepursuit import signal_filter
from caiman.base.movies import to_3D
from metrics import metric
from nmf_support import hals, select_masks, normalize
#%% files for processing
base_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
               '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
               '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
               '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data',
               '/home/nel/caiman_data/example_movies/volpy'][4]
fname = ['FOV4_50um.hdf5', 'demo_voltage_imaging.hdf5'][1]
mov = cm.load(os.path.join(base_folder, fname))
mask = cm.load(os.path.join(base_folder, fname[:-5]+'_ROIs.hdf5'))

#%%
# original movie
y = (cm.movie(-mov)).to_2D().copy()
frate = 400    
shape = mov.shape
y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
mask_2D = mask.to_2D()
n_neurons = mask.shape[0]

plt.figure();plt.imshow(mov[0])
plt.figure();plt.imshow(mask.sum(0))

#%% Use nmf sequentially to extract all neurons in the region
std = [np.std(y_filt[:, np.where(mask_2D[i]>0)[0]].mean(1)) for i in range(len(mask_2D))]
seq = np.argsort(std)[::-1]
plt.imshow(mask[seq[0:3]].sum(0))
#array([23,  6,  7,  8, 10, 20, 15, 17,  2,  9, 19, 18, 21, 22,  3, 14, 30,
#       25,  4,  5, 24, 13, 12, 29, 27,  1, 28, 26,  0, 31, 16, 11])
#%%
num_frames_init = 10000
y_seq = y_filt[:num_frames_init,:].copy()
W_tot = []
H_tot = []
from time import time
t_init = time()
for i in seq:
    print(f'now processing neuron {i}')
    model = NMF(n_components=1, init='nndsvd', max_iter=100, verbose=False)
    y_temp, _ = select_masks(y_seq, mov[:num_frames_init].shape, mask=mask[i])
    W = model.fit_transform(np.maximum(y_temp,0))
    H = model.components_
    y_seq = y_seq - W@H
    W_tot.append(W)
    H_tot.append(H)
    #plt.figure();plt.plot(W);
    #plt.figure();plt.imshow(H.reshape(mov.shape[1:], order='F'));plt.colorbar()
H = np.vstack(H_tot)
W = np.hstack(W_tot)

print(f'time lapse {time() - t_init}')


#%% Use hals to optimize masks
update_bg = True
y_input = np.maximum(y_filt[:num_frames_init], 0)
y_input = cm.movie(to_3D(y_input, shape=(num_frames_init,shape[1],shape[2]), order='F')).transpose([1,2,0])

H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y.shape[1],1)) / y.shape[1],
                             np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, 
                             update_bg=update_bg, use_spikes=True)

plt.figure();plt.imshow(H_new.reshape((mov.shape[1], mov.shape[2], n_neurons), order='F').sum(2));plt.colorbar()


#%% Use nnls to extract signal for neurons
fe = slice(0,None)
if update_bg:
    trace_all = np.array([nnls(np.hstack((H_new, b)),yy)[0] for yy in (-y)[fe]]) 
else:
    trace_all = np.array([nnls(H_new,yy)[0] for yy in (-y)[fe]]) 

trace_all = signal_filter(trace_all.T,freq = 1/3, fr=frate).T
trace_all = trace_all - np.median(trace_all, 0)[np.newaxis, :]
trace_all = -trace_all.T
plt.plot(trace_all[:16].T)

#%% Extract spikes and compute F1 score
trace = trace_all[:].copy()
sao = SignalAnalysisOnline(thresh_STD=None)
sao.fit(trace[:, :10000], num_frames=trace.shape[1])
for n in range(10000, trace.shape[1]):
    sao.fit_next(trace[:, n: n+1], n)
sao.compute_SNR()
print(f'thresh:{sao.thresh}')
print(f'SNR: {sao.SNR}')
print(f'Mean_SNR: {np.array(sao.SNR).mean()}')
print(f'Spikes:{(sao.index>0).sum(1)}')


#%%
idx = 1
plt.plot(sao.trace_rm[idx, :])
indexes = np.array((list(set(sao.index[idx]) - set([0]))))  
plt.vlines(indexes, -2, -1)

    
    
