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

#%%
from fiola.utilities import bin_median, play
from fiola.gpu_mc_nnls import get_mc_model
import tensorflow as tf
import timeit

template = bin_median(mov, exclude_nans=False)
batch_size = 100
center_dims = (256, 256)

def run_gpu_motion_correction(mov, template, batch_size=1, ms_h=10, ms_w=10, center_dims=None):

    def generator():
        if len(mov) % batch_size != 0 :
            raise ValueError('batch_size needs to be a factor of frames of the movie')
        for idx in range(len(mov) // batch_size):
            yield{"m":mov[None, idx*batch_size:(idx+1)*batch_size,...,None]}
                 
    def get_frs():
        dataset = tf.data.Dataset.from_generator(generator, output_types={'m':tf.float32}, 
                                                 output_shapes={"m":(1, batch_size, dims[0], dims[1], 1)})
        return dataset

    times = []
    out = []
    dims = mov.shape[1:]
    mc_model = get_mc_model(template=template, center_dims=center_dims, batch_size=batch_size, ms_h=ms_h, ms_w=ms_w)
    mc_model.compile(optimizer='rmsprop', loss='mse')   
    estimator = tf.keras.estimator.model_to_estimator(keras_model = mc_model)
    start = timeit.default_timer()
    for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
        out.append(i)
        times.append(timeit.default_timer()-start)

    return out, times

#%%
out, times = run_gpu_motion_correction(mov, template, batch_size=20, ms_h=10, ms_w=10, center_dims=(256, 256))
mc_mov = []
x_sh = []
y_sh = []
for ou in out:
    keys = list(ou.keys())
    mc_mov.append(ou[keys[0]])
    x_sh.append(ou[keys[1]])
    y_sh.append(ou[keys[2]])
plt.plot(np.diff(times))
plt.plot(np.array(x_sh).flatten()); plt.plot(np.array(y_sh).flatten())
mc_mov = np.vstack(mc_mov)
a = mc_mov.reshape((-1, template.shape[0], template.shape[1]), order='F')
play(mov, gain=3, q_min=5, q_max=99.99)