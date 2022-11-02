#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:02:21 2021

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
from skimage.transform import resize
import cv2
import timeit
import multiprocessing as mp
from tensorflow.python.keras import backend as K
from viola.caiman_functions import to_3D, to_2D
import scipy
import glob

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx("float32")
#%% get files
names = glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*.tif')
names += glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*.hdf5')
names+=glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/*um/*m.hdf5')
names+=glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/*min/*n.hdf5')
#%% organize data
from collections import defaultdict
err_x = defaultdict(list)
err_y = defaultdict(list)
nshort = []
for n in range(len(names)):
    nshort.append(names[n][73:-1]) #90
    cm = np.load(names[n][:-4]+"_cm_on_shifts.npy").squeeze()
    tempx1, tempy1 = [], []
    for i in range(len(cm)):
        tempx1.append(cm[i][0])
        tempy1.append(cm[i][1])

    v_big = np.load(names[n][:73]+names[n][73:-4]+"_viola_full_shifts.npy").squeeze()
    tempx,tempy=[],[]
    for i in range(len(cm)):
        tempx.append(v_big[i][0])
        tempy.append(v_big[i][1])
    v_big = [tempx,tempy]
    v_small = np.load(names[n][:73]+names[n][73:-4]+"_viola_small_shifts.npy").squeeze()
    tempx,tempy=[],[]
    for i in range(len(cm)):
        tempx.append(v_small[i][0])
        tempy.append(v_small[i][1])
    v_small = [tempx,tempy]
    len_half = len(cm)//2
    
    outx, outy  = [],  []
    for i in range(len_half,len(tempx1)):
        # if tempx1[i] != 0:
        outx.append(tempx1[i]-v_big[0][i])
        # if tempy1[i] != 0:
        outy.append(tempy1[i]-v_big[1][i])
    mnx, mny = np.nanmean(outx), np.nanmean(outy)
    
    for x,y in zip(outx, outy):
        err_x[nshort[-1]].append(x-mnx)
        err_y[nshort[-1]].append(y-mny)
        
    fill_len = 10000-len(err_x[nshort[-1]])
    err_x[nshort[-1]] += [np.nan]*fill_len 
    err_y[nshort[-1]] += [np.nan]*fill_len
    #np.pad(err_x[nshort[-1]], (20000,), mode="constant", constant_values=(-5,))
    #np.pad(err_y[nshort[-1]], (20000,), mode="constant", constant_values=(-5,))

    
    outx, outy  = [],  []
    for i in range(len_half,len(tempx1)):
        # if tempx1[i] != 0:
        outx.append(tempx1[i]-v_small[0][i])
        # if tempy1[i] != 0:
        outy.append(tempy1[i]-v_small[1][i])
    mnx, mny = np.nanmean(outx), np.nanmean(outy)
    
    for x,y in zip(outx, outy):
        err_x[nshort[-1]].append(x-mnx)
        err_y[nshort[-1]].append(y-mny)
        
    fill_len = 10000-len(err_x[nshort[-1]])
    err_x[nshort[-1]] += [np.nan]*fill_len 
    err_y[nshort[-1]] += [np.nan]*fill_len
    # np.pad(err_x[nshort[-1]], (40000,), mode="constant", constant_values=(np.nan,))
    # np.pad(err_y[nshort[-1]], (40000,), mode="constant", constant_values=(np.nan,))
nshort += ["FOV"]
err_x["FOV"] = ["full"]*10000+["crop"]*10000
err_y["FOV"] = ["full"]*10000+["crop"]*10000
#%% violin setup
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")
# tot_err_x_full = [val for val in err_x.values()]
# tot_err_y_full = [val for val in err_y.values()]
# x_mean = [np.mean(val) for val in tot_err_x_full]
# y_mean = [np.mean(val) for val in tot_err_y_full]
dfx = pd.DataFrame()
dfy = pd.DataFrame()
i=0
for x in err_x:
    dfx.loc[:,i]=pd.Series(err_x[x])
    i+=1
i=0
for y in err_y:
    dfy.loc[:,i] = pd.Series(err_y[y])
    i+=1
#dfx.loc[:,i] = pd.Series(["full"]*20000+["crop"]*20000)
#dfy.loc[:,i] = pd.Series(["full"]*20000+["crop"]*20000)
#df.drop(columns=[8], inplace=True)
#df.iloc[9,0]=10

dfx.columns = nshort
dfy.columns = nshort
dfx = pd.melt(dfx, id_vars=["FOV"], var_name="File Name", value_name="Error")
dfy = pd.melt(dfy, id_vars=["FOV"], var_name="File Name", value_name="Error")
#%%  plot
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
fig, axs = plt.subplots(2)
#a1 = sns.violinplot(data=dfx, bw=0.2, split=True, ax = axs[0], width=1.2, hue=dfx["FOV"])
#a2 = sns.violinplot(data=dfy, bw=0.2, split=True, ax = axs[1], width=1.2)
sns.violinplot(x=dfx['File Name'],y=dfx['Error'],hue=dfx['FOV'],split=True, ax=axs[0], palette="viridis")
sns.violinplot(x=dfy['File Name'],y=dfy['Error'],hue=dfy['FOV'],split=True, ax=axs[1], palette="viridis")
axs[0].set_xticklabels(nshort[:-1])
axs[1].set_xticklabels(nshort[:-1])
axs[0].set_title("Error in X-shifts")
axs[1].set_title("Error in Y-shifts")
