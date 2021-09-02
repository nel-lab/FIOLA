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
from fiola.gpu_mc_nnls import get_mc_model, get_nnls_model, run_gpu_motion_correction, get_model
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
def run_gpu_motion_correction_nnls(mov, template, batch_size, Ab, **kwargs):
    """
    Run GPU NNLS for source extraction

    Parameters
    ----------
    mov: ndarray
        motion corrected movie
    template : ndarray
        the template used for motion correction
    batch_size: int
        number of frames used for motion correction each time. The default is 1.
    Ab: ndarray (number of pixels * number of spatial footprints)
        spatial footprints for neurons and background        
    
    Returns
    -------
    trace: ndarray
        extracted temporal traces 
    """
    
    def generator():
        if len(mov) % batch_size != 0 :
            raise ValueError('batch_size needs to be a factor of frames of the movie')
        for idx in range(len(mov) // batch_size):
            yield {"m":mov[None, idx*batch_size:(idx+1)*batch_size,...,None], 
                   "y":y, "x":x, "k":[[0.0]]}
            
    def get_frs():
        dataset = tf.data.Dataset.from_generator(generator, 
                                                 output_types={"m": tf.float32,
                                                               "y": tf.float32,
                                                               "x": tf.float32,
                                                               "k": tf.float32}, 
                                                 output_shapes={"m":(1, batch_size, dims[0], dims[1], 1),
                                                                "y":(1, num_components, batch_size),
                                                                "x":(1, num_components, batch_size),
                                                                "k":(1, 1)})
        return dataset
    
    if kwargs['return_shifts'] == True:
        raise ValueError('return shifts should be False in the full model')
    
    times = []
    out = []
    flag = 500
    index = 0
    dims = mov.shape[1:]
    
    b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')         
    x0 = np.array([nnls(Ab,b[:,i])[0] for i in range(batch_size)]).T
    x, y = np.array(x0[None,:]), np.array(x0[None,:]) 
    num_components = Ab.shape[-1]
    
    model = get_model(template, Ab=Ab, batch_size=batch_size, **kwargs)
    model.compile(optimizer='rmsprop', loss='mse')   
    estimator = tf.keras.estimator.model_to_estimator(model)
    
    print('now start motion correction and source extraction')
    start = timeit.default_timer()
    for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
        out.append(i)
        times.append(timeit.default_timer()-start)
        index += 1    
        if index * batch_size >= flag:
            print(f'processed {flag} frames')
            flag += 500            
    
    print('finish motion correction and source extraction')
    print(f'total timing:{times[-1]}')
    print(f'average timing per frame:{times[-1] / len(mov)}')
    
    trace = []; 
    for ou in out:
        keys = list(ou.keys())
        trace.append(ou[keys[0]][0])        
    trace = np.hstack(trace)
    
    return trace





