#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:26:14 2021

@author: andreagiovannucci
"""


#%%
from numpy.fft import ifftshift
import math
import numpy as np
import tensorflow as tf
from tensorflow_addons.utils import types
from typing import Optional
import tensorflow_probability as tfp
import pylab as plt
#%%
shape = np.array([500,300,100])
shifts = np.array([0.25,2.65,-4.85], dtype=np.float32)
a = np.zeros(shape, dtype=np.float32);
a[shape[0]//2,shape[1]//2,shape[2]//2] = 1
# a = plt.imread('/Users/andreagiovannucci/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/N.00.00/images_N.00.00/image02860.tif')
# a = np.repeat(a,[6], axis=-1)
# shape = a.shape
mn,mx = np.min(a), np.max(a)

at = tf.convert_to_tensor(a[None,:,:,:,None], dtype=tf.float32)
image = at
#%%
# def apply_shifts_dft(img, shifts, diffphase=np.array([0],dtype=np.complex64)):
#     src_freq = np.fft.fftn(img)
#     shifts = np.array(list(shifts[:-1][::-1]) + [shifts[-1]])
#     nc, nr, nd = np.array(np.shape(src_freq), dtype=float)
#     Nr = ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.)))
#     Nc = ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.)))
#     Nd = ifftshift(np.arange(-np.fix(nd / 2.), np.ceil(nd / 2.)))

#     Nr, Nc, Nd = np.meshgrid(Nr, Nc, Nd)
#     sh_0 = -shifts[0] * Nr / nr
#     sh_1 = -shifts[1] * Nc / nc
#     sh_2 = -shifts[2] * Nd / nd
#     sh_tot = (sh_0 +  sh_1 + sh_2)
#     Greg = src_freq * np.exp(-1j * 2 * np.pi * sh_tot)
#     Greg = Greg*(np.exp(1j * diffphase))
#     new_img = np.real(np.fft.ifftn(Greg))
    
#     return new_img[None,:,:,:,None]

@tf.function
def apply_shifts_dft_tf(img, shifts, diffphase=tf.cast([0],dtype=tf.complex64)):
    img = tf.cast(img, dtype=tf.complex64)
    shifts =  (shifts[1], shifts[0], shifts[2]) #p.array(list(shifts[:-1][::-1]) + [shifts[-1]])
    src_freq = tf.signal.fft3d(img)
    nshape = tf.cast(tf.shape(src_freq), dtype=tf.float32)
    nc = nshape[0]
    nr = nshape[1]
    nd = nshape[2]
    Nr = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(nr / 2.), tf.math.ceil(nr / 2.)))
    Nc = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(nc / 2.), tf.math.ceil(nc / 2.)))
    Nd = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(nd / 2.), tf.math.ceil(nd / 2.)))

    Nr, Nc, Nd = tf.meshgrid(Nr, Nc, Nd)
    sh_0 = -shifts[0] * Nr / nr
    sh_1 = -shifts[1] * Nc / nc
    sh_2 = -shifts[2] * Nd / nd
    sh_tot = (sh_0 +  sh_1 + sh_2)
    Greg = src_freq * tf.math.exp(-1j * 2 * math.pi * tf.cast(sh_tot, dtype=tf.complex64))
    
    

    #todo: check difphase and eventually dot product?
    Greg = Greg * tf.math.exp(1j * diffphase)
    new_img = tf.math.real(tf.signal.ifft3d(Greg))
    

    return new_img[None,:,:,:,None]
#%%
at_t_3D= apply_shifts_dft(a, -shifts)
img = image[0,:,:,:,0]
print(np.unravel_index(np.argmax(a), shape=shape))
print(np.unravel_index(np.argmax(at_t_3D), shape=shape))
i,j,k = np.unravel_index(np.argmax(at_t_3D), shape=shape)
print(np.max(at))
print(np.max(at_t_3D))
plt.figure()
plt.subplot(3,1,1)
plt.imshow(at_t_3D[0,i,:,:,0])    
plt.subplot(3,1,2)
plt.imshow(at_t_3D[0,:,j,:,0])
plt.subplot(3,1,3)
plt.imshow(at_t_3D[0,:,:,k,0],vmin=mn, vmax = mx)
#%%"
at_t_3D = apply_shifts_dft_tf(tf.cast(a,dtype=tf.float32), -shifts)
print(np.unravel_index(np.argmax(a), shape=shape))
print(np.unravel_index(np.argmax(at_t_3D), shape=shape))
i,j,k = np.unravel_index(np.argmax(at_t_3D), shape=shape)
print(np.max(at))
print(np.max(at_t_3D))
plt.figure()
plt.subplot(3,1,1)
plt.imshow(at_t_3D[0,i,:,:,0])    
plt.subplot(3,1,2)
plt.imshow(at_t_3D[0,:,j,:,0])
plt.subplot(3,1,3)
plt.imshow(at_t_3D[0,:,:,k,0],vmin=mn, vmax = mx)