#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:18:19 2020

@author: nellab
"""
#%%
import os
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
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
from caiman_functions import to_3D, to_2D
#%%
base_folder = '/home/nellab/SOFTWARE/SANDBOX/src/'
# base_folder = '/home/andrea/software/SANDBOX/src/'
# with np.load('/home/nellab/NEL-LAB Dropbox/NEL/SIMONS_HOME_FOLDER/CaImAnOld/JG10982_171121_field3_stim_00002_00001.tif', allow_pickle=True) as ld:
#     Y_tot = ld['Y']
import h5py
import scipy
with h5py.File('/home/nellab/NEL-LAB Dropbox/NEL/SIMONS_HOME_FOLDER/CaImAnOld/JG10982_171121_field3_stim_00002_00001_results.hdf5','r') as f:
# with h5py.File('/home/andrea/software/SANDBOX/src/memmap__d1_512_d2_512_d3_1_order_C_frames_1825_.hdf5','r') as f:
# with h5py.File('/home/nellab/NEL-LAB Dropbox/NEL/SIMONS_HOME_FOLDER/CaImAnOld/example_movies/k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034_results.hdf5','r') as f:
    
    data = np.array(f['estimates']['A']['data'])
    indices = np.array(f['estimates']['A']['indices'])
    indptr = np.array(f['estimates']['A']['indptr'])
    shape = np.array(f['estimates']['A']['shape'])
    # idx_components = np.array(f['estimates']['idx_components'])
    idx_components = np.arange(shape[-1])
    A_sp_full = scipy.sparse.csc_matrix((data[:], indices[:], indptr[:]), shape[:])
    YrA_full = np.array(f['estimates']['YrA'])
    C_full = np.array(f['estimates']['C']) 
    b_full = np.array(f['estimates']['b'])
    b_full = b_full
    f_full = np.array(f['estimates']['f'])
    A_sp_full = A_sp_full[:,idx_components ]
    C_full = C_full[idx_components]
    YrA_full = YrA_full[idx_components]
#%%

a2 = io.imread("/home/nellab/NEL-LAB Dropbox/NEL/SIMONS_HOME_FOLDER/CaImAnOld/JG10982_171121_field3_stim_00002_00001.tif")
# a2 = io.imread("/home/nellab/NEL-LAB Dropbox/NEL/SIMONS_HOME_FOLDER/CaImAnOld/example_movies/k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034.tif")
#a = np.load(base_folder+'movie.npy')
#a = np.array(a[:, :, :])
# a2 = to_3D(a)
Y_tot = to_2D(a2).T

#%%
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


Cff = np.concatenate([C_full+YrA_full,f_full], axis=0)
Cf = np.concatenate([C+YrA,f], axis=0)
x0 = Cf[:,0].copy()[:,None]
#%%
# q = Queue()
from pipeline_gpu import Pipeline, get_model
model = get_model(template, (256, 256), Ab.astype(np.float32), 30)
#%%
# from motion_correction_gpu import MotionCorrect
# cfnn1 = []
# shifts = []
# # x_old, y_old = x0[None, :], x0[None, :]
# template = np.median(a2, axis=0)
# #template = template[138:-138, 138:-138, None, None]
# shp = template.shape[0]//1
# mc = MotionCorrect(template, (256, 256))
# #ct2 = compute_theta2(Ab, n_AtA)
# #reg = NNLS(theta_1)

# for i in range(1825):
# #    plt.imshow(a[0, :, :])
#     mc_out = mc(a2[i, :, :, None][None, :].astype(np.float32))
# #    mc_out = Y_tot[:, i:i+1].T
# #    ct2_out = ct2(mc_out)
# #    nnls_out = reg([y_old, x_old, tf.convert_to_tensor(np.zeros_like(1), dtype=tf.int8), ct2_out])
# #    for k in range(1, 10):
# #        nnls_out = reg(nnls_out)
# #
# #    y_old, x_old = nnls_out[0], nnls_out[1]
# #    tf.print(tf.reduce_sum(y_old), tf.reduce_sum(x_old), tf.reduce_sum(nnls_out[3]), "y, x, weight")
#     cfnn1.append(mc_out[0]) 
#     shifts.append(mc_out[1])
#%%
# tempx1, tempy1 = [], []
# for i in range(1000):
#     tempx1.append(shifts[i][0].numpy())
#     tempy1.append(shifts[i][1].numpy())
# tempx1 = np.array(tempx1).squeeze()
# tempy1 = np.array(tempy1).squeeze()
#%%
#jg
# a0x, a0y, a1x, a1y, a2x, a2y, a3x, a3y = np.load("jgx0.npy"), np.load("jgy0.npy"), np.load("jgwholex.npy"), np.load("jgwwoley.npy"), np.load("jgx1.npy") , np.load("jgy1.npy"), np.load("jgx2.npy"), np.load("jgy2.npy")
# #k56
# b0x, b0y, b1x, b1y, b2x, b2y, b3x, b3y = np.load("k560x.npy"), np.load("k560y.npy"), np.load("k56wholex.npy"), np.load("k56wholey.npy"), np.load("k56x1.npy") , np.load("k56y1.npy"), np.load("k56x2.npy"), np.load("k56y2.npy")
# #%%
# plt.plot(a0y[:1000]);plt.plot(a1y);plt.plot(a2y);plt.plot(a3y);plt.legend(["CaImAn shifts", "Vol shifts 100%", "Vol shifts 50%", "vol shifts 75%"]);plt.title("Y-shifts for JG file")
# #%%
# a = [a0x, a0y, a1x, a1y, a2x, a2y, a3x, a3y]
# b = [b0x[:1000], b0y[:1000], b1x, b1y, b2x, b2y, b3x, b3y]
# a_res = []
# b_res = []
# from scipy import stats
# for i in range(2, len(a), 2):
#     a_res.append(stats.pearsonr(a0x, a[i]))
#     a_res.append(stats.pearsonr(a0y, a[i+1]))
# for j in range(2, len(b), 2):
#     b_res.append(stats.pearsonr(b0x[:1000], b[j]))
#     b_res.append(stats.pearsonr(b0y, b[j+1]))
#%%
# cfnn1 = np.squeeze(np.array(cfnn1)).T
# for vol, ca in zip(cfnn1, (C_full + YrA_full)[:,:100]):
# #    print(tf.reduce_sum(vol), tf.reduce_sum(ca))
#     plt.cla()
#     plt.plot((vol), label='volpy')
#     plt.plot((ca), label='caiman')    
#     plt.pause(1)
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
from motion_correction_gpu import MotionCorrect
mc0 = np.expand_dims(a2[0:1, :, :], axis=3)
spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, a2)
out = spike_extractor.get_spikes(1825)
#%%
spikes_gpu = out[0]
spikes = np.array(spikes_gpu).squeeze().T
#%%
print(np.linalg.norm(Y_tot-Ab@spikes)/np.linalg.norm(Y_tot))
#%%
# plt.plot(spikes)
for vol, ca in zip(spikes, Cff[:,:300]):
#    print(tf.reduce_sum(vol), tf.reduce_sum(ca))
    plt.cla()
    plt.plot((vol), label='volpy')
    plt.plot((ca), label='caiman')    
    plt.xlim([0,300])
    plt.pause(1)



