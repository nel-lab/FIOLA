#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:18:19 2020

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
import cv2
import timeit
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
#a = np.load(base_folder+'movie.npy')
#a = np.array(a[:, :, :])
from caiman.base.movies import to_3D
a2 = to_3D(Y_tot, (512, 512, 1825))
#%%
from pipeline_gpu import Pipeline, get_model
#%%
template = np.median(a2, axis=2)
model = get_model(template, A_sp_full, b_full)
#%%
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

Cf = np.concatenate([C+YrA,f], axis=0)
x0 = Cf[:,0].copy()[:,None]
#%%
#cfnn1 = []
#x_old, y_old = x0[None, :], x0[None, :]
#template = np.median(a, axis=0)
##template = template[138:-138, 138:-138, None, None]
#
#mc = MotionCorrect(template)
##ct2 = compute_theta2(Ab, n_AtA)
##reg = NNLS(theta_1)
#
#for i in range(100):
##    plt.imshow(a[0, :, :])
#    mc_out = mc(a[i, :, :, None][None, :])
##    mc_out = Y_tot[:, i:i+1].T
##    ct2_out = ct2(mc_out)
##    nnls_out = reg([y_old, x_old, tf.convert_to_tensor(np.zeros_like(1), dtype=tf.int8), ct2_out])
##    for k in range(1, 10):
##        nnls_out = reg(nnls_out)
##
##    y_old, x_old = nnls_out[0], nnls_out[1]
##    tf.print(tf.reduce_sum(y_old), tf.reduce_sum(x_old), tf.reduce_sum(nnls_out[3]), "y, x, weight")
#    cfnn1.append(mc_out)    
#
#cfnn1 = np.squeeze(np.array(cfnn1)).T
#for vol, ca in zip(cfnn1, (C_full + YrA_full)[:,:100]):
##    print(tf.reduce_sum(vol), tf.reduce_sum(ca))
#    plt.cla()
#    plt.plot((vol), label='volpy')
#    plt.plot((ca), label='caiman')    
#    plt.pause(1)
#%%
model.compile(optimizer='rmsprop', loss='mse')
#%%

#%%
#spikes2 = []
#x_old, y_old = x0[None, :], x0[None, :]
#for idx in range(10):
#    fr = a2[:, :, idx:idx+1][None, :]
##    fr = mov_corr[idx:idx+1, :]
#    nnls_out = model([fr, x_old, y_old, tf.convert_to_tensor([[0]], dtype=tf.int8)])[0]
#    y_old, x_old = nnls_out[0], nnls_out[1]
#    spikes2.append(x_old)
#%%
mc0 = a2[:, :, 0:1][None, :]
spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, a2)
spikes_gpu = spike_extractor.get_spikes(1800)
#%%
spikes = np.array(spikes_gpu).squeeze().T
# plt.plot(spikes)
for vol, ca in zip(spikes, (C_full + YrA_full)[:,:100]):
#    print(tf.reduce_sum(vol), tf.reduce_sum(ca))
    plt.cla()
    plt.plot((vol), label='volpy')
    plt.plot((ca), label='caiman')    
    plt.pause(1)


