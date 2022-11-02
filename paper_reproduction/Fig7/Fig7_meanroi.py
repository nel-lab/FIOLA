#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:08:51 2022

@author: nel
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pyximport
pyximport.install()
from tensorflow.python.client import device_lib
from time import time, sleep
import scipy
import sys
sys.path.append('/home/nel/CODE/VIOLA')

import caiman as cm
from caiman.base.rois import com
from caiman.source_extraction.cnmf.utilities import get_file_size
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.temporal import constrained_foopsi
from fiola.demo_initialize_calcium import run_caiman_init
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, play, bin_median, to_2D, local_correlations, movie_iterator, compute_residuals
from Fig7.Fig7_utilities import *

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)
    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1

#%%
def run_meanroi(mov, num_frames_init, num_frames_total, b_masks):
    batch_size = 100
    t_all = []

    # reshape binary masks
    b_masks = b_masks.copy().transpose([1, 2, 0])
    b_masks = b_masks.reshape([-1, b_masks.shape[-1]], order='F') 

    # load batch of frames for processing
    for idx in range(num_frames_init, num_frames_total, batch_size):
        if (idx) % 1000 == 0:
            logging.info(f'{idx} frames processed online') 

        mm = mov[idx:idx+batch_size]
        mm = mm.transpose([1, 2, 0])
        mm = mm.reshape([-1, mm.shape[-1]], order='F')
        
        t_batch = []
        for idx in range(len(b_masks.T)):
            pixels = np.where(b_masks[:, idx] >0)[0]
            t = mm[pixels].mean(0)
            t_batch.append(t)
        t_batch = np.array(t_batch).T
        t_all.append(t_batch)    
    t_all = np.array(t_all)  
    t_all = t_all.transpose([2, 0, 1]).reshape((-1, t_all.shape[0] * t_all.shape[1]), order='C')      
    return t_all



#%%
num_frames_init = 0
num_frames_total = 31900
fnames = '/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000._rig__d1_796_d2_512_d3_1_order_F_frames_31933_.mmap'
mov = cm.load(fnames)

cnm = load_CNMF('/media/nel/storage/fiola/R2_20190219/3000/memmap__d1_796_d2_512_d3_1_order_C_frames_3000__v3.7.hdf5')
A = cnm.estimates.A.toarray()
dims = [796, 512]
#A = A[:, select]     
A = A.reshape([dims[0], dims[1], -1], order='F')
A = A.transpose([2, 0, 1])
masks = A
b_masks = masks_to_binary(masks.copy())
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(masks.sum(0))
# ax[0].imshow(binary_masks.sum(0), vmax=1, alpha=0.3)

t_m = run_meanroi(mov, num_frames_init, num_frames_total, b_masks)
savef = '/media/nel/storage/fiola/R2_20190219/meanroi/'
np.save(savef + 'traces_v3.13.npy', t_m)

#%%      
for ii in range(500, 600, 5):
    t1 = (cnm.estimates.C+cnm.estimates.YrA)[ii, :3000]
    t2 = t_m[ii, :3000]
    t1 = (t1 - t1.mean()) / t1.std()
    t2 = (t2 - t2.mean()) / t2.std()
    plt.figure()    
    plt.plot(t1)
    plt.plot(t2-5)
    
#%%
t_m = np.load('/media/nel/storage/fiola/R2_20190219/meanroi/traces_v3.13.npy')
t_n = StandardScaler().fit_transform(t_m.T).T
t_dec = []
for idx, tt in enumerate(t_n):
    if idx % 100 == 0:
        print(f'processing trace {idx}')
    c_full, bl, c1, g, sn, s_full, lam = constrained_foopsi(tt, p=1, s_min=0.1)
    t_dec.append(s_full)
t_dec = np.array(t_dec)
savef = '/media/nel/storage/fiola/R2_20190219/meanroi/'
np.save(savef + 'traces_dec_v3.13.npy', t_dec)

#%%
fio = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_None_with_detrending_v3.12.npy', allow_pickle=True).item()
f_t = fio.trace
f_d = fio.trace_deconvolved

#%%
for i in np.arange(10, 100, 10):
    plt.figure()
    #plt.title(f'fr:30 Hz sampling from 90 Hz, tau: {tau}s, nonlinearity_model: {nonlinearity_model}')
    #plt.plot(t_m[i])
    plt.plot(t_n[i])
    plt.plot(t_dec[i])
    plt.title(f'{i}')
    plt.figure()
    plt.plot(f_t[i])
    plt.plot(f_d[i])
    plt.xlabel('frames')
#plt.legend(['fluorescence trace', 'calcium trace', 'trueSpikes'])
#plt.show(block=False)
