#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:36:44 2021

@author: nel
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
from skimage.transform import resize,  rescale
import cv2
import timeit
import multiprocessing as mp
from tensorflow.python.keras import backend as K
from viola.caiman_functions import to_3D, to_2D
import scipy
import glob
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_XLA_FLAGS"]="--tf_xla_enable_xla_devices"

#%% get files
#%%
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
fls  = sorted(glob.glob("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/IterTiming/*_500*"))
#%% 
names = []
data = []

for fl in fls:
    names.append(fl[-13:-6])
    data.append(np.load(fl))
    
fig, ax = plt.subplots()
ax.errorbar(names[:3], [np.mean(val) for val in data[:3]], yerr=[np.std(val) for val in data[:3]])
ax.errorbar(names[:3], [np.mean(val) for val in data[3:]], yerr=[np.std(val) for val in data[3:]])  
# plt.yscale("log", nonposy="clip")  
    
