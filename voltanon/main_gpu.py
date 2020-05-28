#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:18:19 2020

@author: nellab
"""
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
import caiman as cm
import multiprocessing as mp
from tensorflow.python.keras import backend as K
#%%
with np.load('/home/nellab/SOFTWARE/SANDBOX/src/regression_n.01.01_less_neurons.npz', allow_pickle=True) as ld:
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
a = cm.load('/home/nellab/caiman_data/example_movies/n.01.01._rig__d1_512_d2_512_d3_1_order_F_frames_1825_.mmap', in_memory=True)
a = np.array(a[:, :, :])
#%%
from nnls_gpu import compute_theta2, NNLS
from pipeline_gpu import Pipeline, get_model
from motion_correction_gpu import MotionCorrect
#%%
template = np.median(a, axis=0)
model = get_model(template, A_sp_full, b_full)
f, Y =  f_full[:, 0][:, None], Y_tot[:, 0][:, None]
YrA = YrA_full[:, 0][:, None]
C = C_full[:, 0][:, None]

Ab = np.concatenate([A_sp_full.toarray()[:], b_full], axis=1).astype(np.float32)
b = Y_tot[:, 0]
AtA = Ab.T@Ab
Atb = Ab.T@b
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)

Cf = np.concatenate([C+YrA,f], axis=0)
Cf_bc = Cf.copy()
x0 = Cf[:,0].copy()[:,None]
#%%
model.compile(optimizer='rmsprop', loss='mse')
mc0 = a[0, :, :].reshape((1, 512, 512, 1))
spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, a)
spikes = spike_extractor.get_spikes(1000)
