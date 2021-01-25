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
many =  False
if many:
    names = glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*.tif')
else:
    # names.append('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/mesoscope.hdf5')
    # names+=glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/viola_movies/*.hdf5')
    names = glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron/*/*.tif') #One Neuron Tests

j=14
movie = names[j]
mov = io.imread(movie)
full = True
print(movie)

#%% motion correct layer setup
from viola.motion_correction_gpu import MotionCorrectTest
# template = np.load(movie[:-4]+"_template.npy")
#template = temp
template = np.median(mov[:2000], axis=0)
if full:
    mc_layer = MotionCorrectTest(template, mov[0].shape, ms_h=1, ms_w=1)
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

x_shift, y_shift = [], []
for i in range(len(shifts)):
    x_shift.append(shifts[i][0].numpy())
    y_shift.append(shifts[i][1].numpy())
x_shift = np.array(x_shift).squeeze()
y_shift = np.array(y_shift).squeeze() 
#%% plot
plt.plot(x_shift[:100])
plt.plot(y_shift[:100])
#%% "play" movie
for fr in new_mov:
    plt.imshow(fr);
    plt.pause(0.2)
    plt.cla()
    
#%% need to get H_new from main_integration
from viola.nnls_gpu import NNLS, compute_theta2
from scipy.optimize import nnls

Ab = H_new.astype(np.float32)
b = mov[0].reshape(-1, order='F')
x0 = nnls(Ab,b)[0][:,None].astype(np.float32)
x_old, y_old = x0, x0
AtA = Ab.T@Ab
Atb = Ab.T@b
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
mc0 = mov[0:1,:,:, None]
        
c_th2 = compute_theta2(Ab, n_AtA)
n = NNLS(theta_1)
num_layers = 10
#%%
nnls_out = []
k = np.zeros_like(1)
for i in range(20000):
    mc = np.reshape(np.transpose(new_mov[i]), [-1])[None, :]
    th2 = c_th2(mc)
    x_kk = n([y_old, x_old, k, th2])

    
    for j in range(1, num_layers):
        x_kk = n(x_kk)
        
    y_old, x_old = x_kk[0], x_kk[1]
    nnls_out.append(y_old)
 
 