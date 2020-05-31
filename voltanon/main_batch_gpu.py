#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:37:29 2020

@author: nellab
"""
#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
import tensorflow_addons as tfa
from queue import Queue
from threading import Thread
from past.utils import old_div
from skimage import io
import numpy as np
import pylab as plt
# import caiman as cm
import multiprocessing as mp
from tensorflow.python.keras import backend as K
#%%
base_folder = '/home/nellab/SOFTWARE/SANDBOX/src/'
#base_folder = '/home/andrea/software/SANDBOX/src/'
with np.load(base_folder+'regression_n.01.01_less_neurons.npz', allow_pickle=True) as ld:
    Y_tot = ld['Y']
import h5py
import scipy
with h5py.File('/home/nellab/caiman_data/example_movies/memmap__d1_512_d2_512_d3_1_order_C_frames_1825_.hdf5','r') as f:
        
    data = np.array(f['estimates']['A']['data'])
    indices = np.array(f['estimates']['A']['indices'])
    indptr = np.array(f['estimates']['A']['indptr'])
    shape = np.array(f['estimates']['A']['shape'])
    idx_components = f['estimates']['idx_components']
    A_sp_full = scipy.sparse.csc_matrix((data[:], indices[:], indptr[:]), shape[:])
    YrA_full = np.array(f['estimates']['YrA'])
    C_full = np.array(f['estimates']['C']) 
    b_full = np.array(f['estimates']['b']) 
    f_full = np.array(f['estimates']['f'])
    A_sp_full = A_sp_full[:,idx_components ]
    C_full = C_full[idx_components]
    YrA_full = YrA_full[idx_components]
#%%
a2 = np.transpose(Y_tot.reshape(512, 512, 1825))
#%%
from batch_gpu import Pipeline, get_model
#%%
batch_size=1
template = np.median(a2, axis=0)
f, Y =  f_full[:, 0][:, None], Y_tot[:, 0][:, None]
YrA = YrA_full[:, 0][:, None]
C = C_full[:, 0][:, None]

Ab = np.concatenate([A_sp_full.toarray()[:], b_full], axis=1).astype(np.float32)
b = Y_tot[:, 0]
AtA = Ab.T@Ab
Atb = Ab.T@b
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)

Cf = np.concatenate([C_full+YrA,f_full], axis=0)
x0 = Cf[:,0:batch_size].copy()
#%%
model = get_model(template, Ab, batch_size)
model.compile(optimizer='rmsprop', loss='mse')
#%%
num_frames = 1800
mc0 = a2[0:batch_size, :, :, None][None, :]
x_old, y_old = np.array(x0[None,:]), np.array(x0[None,:])
spike_extractor = Pipeline(model, x_old, y_old, mc0, theta_2, a2, batch_size)
spikes_gpu = spike_extractor.get_spikes(num_frames)
#%%
temp = []
for spike in spikes_gpu:
    for i in range(batch_size):
        temp.append(spike[i])
spikes = np.array(temp).squeeze().T
# plt.plot(spikes)
for vol, ca in zip(spikes, (C_full + YrA_full)[:,:num_frames]):
#    print(tf.reduce_sum(vol), tf.reduce_sum(ca))
    plt.cla()
    plt.plot((ca), label='caiman') 
    plt.plot((vol), label='volpy')
    plt.pause(1)