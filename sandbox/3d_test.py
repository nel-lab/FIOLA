#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:59:07 2021

@author: nellab
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#%% Create the nxnxn random matrix and add "neurons"
simulation = np.random.rand(100,100,100)
simulation = simulation/10

max_list = np.random.randint(0, 100, 75)

for i in range(0, len(max_list), 3):
    simulation[max_list[i], max_list[i+1], max_list[i+2]] = 1
    
#%% Simulate movie
mov = []
shifts_x = np.random.randint(0, 10, 500)
shifts_y = np.random.randint(0, 10, 500)
shifts_z = np.random.randint(0, 10, 500)

for i in range(len(shifts_x)):
    mov.append(np.roll(np.roll(np.roll(simulation, shifts_x[i]), shifts_y[i], axis=1), shifts_x[i], axis=2))

mov = np.asarray(mov).astype(np.float32)
template = np.median(mov, axis=0).astype(np.float32)
# template = template[10:-10, 10:-10, 10:-10]
#%% Test 3d conv
padding='VALID'
strides = [1,1,1,1,1]
data = np.expand_dims(mov[0], axis=3)[None, :]
filt = np.reshape(template, (80, 80, 80, 1, 1))
out = tf.nn.conv3d(data, filt, strides, padding)
#%% Test  motion correction
from motion_correction_3d import MotionCorrect3D

mc_layer = MotionCorrect3D(template, mov[0].shape)

for i in range(len(data)):
    fr = mc_layer(data[i].astype(np.float32))