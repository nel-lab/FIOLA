#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:52:19 2021

@author: nellab
"""
#%% imports!
import numpy as np
import pylab as plt
import os
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
import tensorflow_addons as tfa
from queue import Queue
from threading import Thread
from past.utils import old_div
from skimage import io
from skimage.transform import resize
import cv2
import timeit
import multiprocessing as mp
from tensorflow.python.keras import backend as K
from viola.caiman_functions import to_3D, to_2D
import scipy
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%% get files
import glob
many =  True
if many:
    names = glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*.tif')
else:
    # names.append('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/mesoscope.hdf5')
    # names+=glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/viola_movies/*.hdf5')
    names = glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron/*/*.tif') #One Neuron Tests

j=4
movie = names[j]
mov = io.imread(movie)
full = True
print(movie)
#%% optional rotation
mov = np.transpose(mov, (0, 2, 1))
plt.imshow(mov[0])
full=True
#%% motion correct layer setup
from viola.motion_correction_gpu import MotionCorrectTest
template = np.load(movie[:-4]+"_template.npy")
template = np.transpose(template)
#template = temp
# template = np.median(mov[:2000], axis=0)
if full:
    mc_layer = MotionCorrectTest(template, mov[0].shape, ms_h=10, ms_w=10)
else:
    mc_layer = MotionCorrectTest(template, (mov[0].shape[0]//2, mov[0].shape[1]//2))
    
#%% run mc
shifts = []
new_mov = []
for i in range(len(mov//4)):
    fr = mc_layer(mov[i, :, :, None][None, :].astype(np.float32))
    shifts.append(fr[1]) 
    new_mov.append(fr[0]) #movie
new_mov = np.array(new_mov).squeeze() 
#%% reshape shifts
x_shift, y_shift = [], []
for i in range(len(shifts)):
    x_shift.append(shifts[i][0].numpy())
    y_shift.append(shifts[i][1].numpy())
x_shift = np.array(x_shift).squeeze()
y_shift = np.array(y_shift).squeeze() 
#%%
np.save(movie[:73]+"transp_test/"+movie[73:-4]+"_viola_shifts", shifts)
#%% plot
v_big = np.load(movie[:-4]+"_viola_shifts.npy")
plt.plot(x_shift)
plt.plot(v_big[1])
plt.plot(y_shift)
plt.plot(v_big[0])
#%%
err_x = []
err_y = []
for i,val in enumerate(x_shift):
    err_x.append(val-v_big[1][i])
    err_y.append(y_shift[i]-v_big[0][i])

#%% "play" movie
for fr in new_mov:
    plt.imshow(fr);
    plt.pause(0.2)
    plt.cla()
 
 