#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:02:58 2021

@author: nel
"""
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.io import imread
from time import time, sleep
from threading import Thread

from fiola.utilities import normalize
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
#%%
mov = imread('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/demo_K53/k53.tif')
movie_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data'][0]
name = 'demo_voltage_imaging.hdf5'
if '.hdf5' in name:
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
        mov = np.array(h5['mov'])
elif '.tif' in name:
    mov = imread(name)
    
#%%
from fiola.utilities import bin_median, play
from fiola.gpu_mc_nnls import get_mc_model, get_nnls_model, run_gpu_motion_correction, get_model, Pipeline
import tensorflow as tf
import timeit
from scipy.optimize import nnls  

template = bin_median(mov, exclude_nans=False)
template = template.astype(np.float32)
mov = mov.astype(np.float32)
batch_size = 20
center_dims = None


#%%
mc_mov, shifts, times = run_gpu_motion_correction(mov, template, batch_size, ms_h=10, ms_w=10, 
                                       use_fft=True, normalize_cc=True, center_dims=(256,256), return_shifts=True)
plt.plot(np.diff(times))
plt.plot(np.array(x_sh).flatten()); plt.plot(np.array(y_sh).flatten())
mc_mov = np.vstack(mc_mov)
a = mc_mov.reshape((-1, template.shape[0], template.shape[1]), order='F')
play(a, gain=3, q_min=5, q_max=99.99)

#%%
Ab = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/demo/Ab.npy')
mov = fio.mov.copy()
#mov = -mov
trace = run_gpu_nnls(mov, Ab, batch_size=20)
plt.plot(trace[8])
#%%
from fiola.utilities import to_2D
fe = slice(0,None)
trace_nnls = np.array([nnls(Ab,yy)[0] for yy in (to_2D(mov).copy())[fe]])

#%%
tt = run_gpu_motion_correction_nnls(mov, template, batch_size=1, Ab=Ab, ms_h=10, ms_w=10, 
                          use_fft=True, normalize_cc=True, center_dims=None, return_shifts=False, num_layers=10)

#%%
spike_extractor = Pipeline(mov, template, batch_size, Ab, ms_h=10, ms_w=10, 
                          use_fft=True, normalize_cc=True, center_dims=(80, 80), return_shifts=False, num_layers=10)

spikes_gpu = spike_extractor.get_traces(len(mov))
traces_fiola = []
for spike in spikes_gpu:
    for i in range(batch_size):
        traces_fiola.append([spike[:,:,i]])
traces_fiola = np.array(traces_fiola).squeeze().T
trace = traces_fiola.copy()

plt.plot(trace[8])


