#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:12:08 2020

@author: nellab
"""

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
from pipeline_gpu import Pipeline, get_model
from scipy.optimize import nnls
import h5py

from caiman_functions import signal_filter, to_3D, to_2D
import matplotlib.pyplot as plt
from scipy.optimize import nnls    
from signal_analysis_online import SignalAnalysisOnline
from sklearn.decomposition import NMF
from metrics import metric
from nmf_support import hals, select_masks, normalize
from skimage.io import imread
from running_statistics import OnlineFilter
#%% HALS
from math import sqrt
def HALS4activity(Yr, A, noisyC, AtA=None, iters=5, tol=1e-3, groups=None,
                  order=None):
    """Solves C = argmin_C ||Yr-AC|| using block-coordinate decent. Can use
    groups to update non-overlapping components in parallel or a specified
    order.

    Args:
        Yr : np.array (possibly memory mapped, (x,y,[,z]) x t)
            Imaging data reshaped in matrix format

        A : scipy.sparse.csc_matrix (or np.array) (x,y,[,z]) x # of components)
            Spatial components and background

        noisyC : np.array  (# of components x t)
            Temporal traces (including residuals plus background)

        AtA : np.array, optional (# of components x # of components)
            A.T.dot(A) Overlap matrix of shapes A.

        iters : int, optional
            Maximum number of iterations.

        tol : float, optional
            Change tolerance level

        groups : list of sets
            grouped components to be updated simultaneously

        order : list
            Update components in that order (used if nonempty and groups=None)

    Returns:
        C : np.array (# of components x t)
            solution of HALS

        noisyC : np.array (# of components x t)
            solution of HALS + residuals, i.e, (C + YrA)
    """

    AtY = A.T.dot(Yr)
    num_iters = 0
    C_old = np.zeros_like(noisyC)
    C = noisyC.copy()
    if AtA is None:
        AtA = A.T.dot(A)
    AtAd = AtA.diagonal() + np.finfo(np.float32).eps

    # faster than np.linalg.norm
    def norm(c): return sqrt(c.ravel().dot(c.ravel()))
    while (norm(C_old - C) >= tol * norm(C_old)) and (num_iters < iters):
        C_old[:] = C
        if groups is None:
            if order is None:
                order = list(range(AtY.shape[0]))
            for m in order:
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C)) / AtAd[m]
                C[m] = np.maximum(noisyC[m], 0)
        else:
            for m in groups:
                noisyC[m] = C[m] + ((AtY[m] - AtA[m].dot(C)).T/AtAd[m]).T
                C[m] = np.maximum(noisyC[m], 0)
        num_iters += 1
    return C, noisyC
#%%
base_folder = "/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/nnls"
mov_ind = 0
filename = ["/k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034_results.hdf5",
             "/JG10982_171121_field3_stim_00002_00001_results.hdf5",
             "/memmap__d1_512_d2_512_d3_1_order_C_frames_1825_.hdf5",
             "/FOV1_35um_ROIs.hdf5"][mov_ind]
tifname = ["/k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034.tif",
           "/JG10982_171121_field3_stim_00002_00001.tif",
           "/regression_n.01.01_less_neurons.npz",
           '/FOV1_35um.hdf5'][mov_ind]

path = base_folder + filename
path1 = base_folder + tifname

namespace = ["k56", "jg", "neu", "vol"][mov_ind]
#%%
voltage = False
if not voltage:
    import h5py
    import scipy
    with h5py.File(path,'r') as f:
        print(f.keys())
        img_min =  np.array(f['img_min'])[()]
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
        
    a2 = io.imread(path1).astype(np.float32)
    a2 -= img_min
else:
    frate = 400
    with h5py.File(path1,'r') as f:
        print(f.keys())
        a2 = np.array(f["mov"]).astype(np.float32)
    a2 -= a2.min()
    with h5py.File(path, "r") as f:
        mask = np.array(f["mov"]).astype(np.float32)
    Y_tot = to_2D(a2).T
    mov = a2[:5000]
#%% FOR VOLTAGE
if voltage:
    y = to_2D(-mov).copy()     
    y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
    y_filt = y_filt
    Y = Y_tot[:,:300]

    do_plot = False
    num_frames_init = 5000
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

    update_bg = False
    y_input = np.maximum(y_filt[:num_frames_init], 0)
    y_input =to_3D(y_input, shape=(num_frames_init,mov.shape[1],mov.shape[2]), order='F').transpose([1,2,0])
        
    H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y.shape[1],1)) / y.shape[1],
                                  np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, 
                                  update_bg=update_bg, use_spikes=True)
    

    mov_in = mov 
    Ab = H_new.astype(np.float32)
    template = np.median(mov_in, axis=0)
    center_dims = (template.shape[0], template.shape[1])
    
    b = to_2D(mov).T[:, 0]
    x0 = nnls(Ab,b)[0][:,None]
    AtA = Ab.T@Ab
    Atb = Ab.T@b
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
    mc0 = mov_in[0:1,:,:, None]
    #%% HALS
    count = 0
    hals = []
    for frame in Y.T[1:]:
        count += 1
        cc = (HALS4activity(frame, Ab, noisyC = W[-1], AtA=AtA, iters=5, groups=None)[0])
        hals.append(cc)
    hals = np.array(hals)
    #%%
    
    from pipeline_gpu import Pipeline, get_model
    model = get_model(template, center_dims, Ab, 30)
    model.compile(optimizer='rmsprop', loss='mse')
    spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, mov_in[:,:,:100000])
    traces_viola = spike_extractor.get_spikes(5000)
    #%%
    
    #%%
    traces = np.array(traces_viola).squeeze().T
    # gt = np.load(base_folder+"/FOV4_50um_estimates.npz", allow_pickle=True)[()]["t"][:,:5000]
    plt.plot(traces[0]);plt.plot(hals.T[0])
#%% CALCIUM
if voltage == False:
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
    mc0 = np.expand_dims(a2[0:1, :, :], axis=3)
#%%
    tf.keras.backend.clear_session()
    layer_depths = [30]
    for d in layer_depths:
        filepath = namespace + str(d)
        model = get_model(template, (256, 256), Ab.astype(np.float32), d)
        model.compile(optimizer='rmsprop', loss='mse')
        spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, a2)
        out = spike_extractor.get_spikes(30)
        spikes_gpu = out[0]
        spikes = np.array(spikes_gpu).squeeze().T
        np.save(filepath+"traces", spikes)
        np.save(filepath+"times", out[1])
#%%
Cf_bc = [Cf[:,0].copy()]
count = 0
times = []
for frame in Y.T[1:]:
    count += 1
    cc = (HALS4activity(frame, A, noisyC = Cf_bc[-1], AtA=AtA, iters=5, groups=None)[0])
#    Cf_bc.append(HALS4activity(frame, A, noisyC = Cf_bc[-1], AtA=AtA, iters=5, groups=None)[0])
    times.append(time()-t_0)
print(time()-t_0) 
Cf_bc = np.squeeze(np.array(Cf_bc).T)
print(np.linalg.norm(Y-Ab@Cf_bc)/np.linalg.norm(Y))  
#%%
gt = np.load("k56_groundtruth.npy")
a = np.load("k565.npy")
for i in range(100,150):
    plt.plot(gt[i]);plt.plot(a[i])
    plt.pause(0.5)
    plt.cla()
#%%
np.savez('FOV1_35_vol_data.npz', nnlsgt = np.load("FOV_1_35_nnls.npy"), nnls5 = np.load("FOV1_35_5.npy"),nnls10 = np.load("FOV1_35_10.npy"), nnls20=np.load("FOV1_35_20.npy"), nnls30=np.load("FOV1_35_30.npy"), Y_tot=y, Ab=Ab)
#%%
from skimage.transform import resize
from motion_correction_gpu import MotionCorrectTest
# frame_sizes = [256, 512, 768]
frame_sizes = [256]
for size in frame_sizes:
    mov = resize(a2, (a2.shape[0], size, size))
    template = np.median(mov, axis=0)
    mot_corr = MotionCorrectTest(template, (size, size))
    shifts = []
    for i in range(50):
        out = mot_corr(a2[i, :,:,None][None, :])
        shifts.append(out[1])