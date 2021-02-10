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
    names += glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/viola_movies/*.hdf5')
    names += glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*.hdf5')
else:
    # names.append('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/mesoscope.hdf5')
    # names+=glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/viola_movies/*.hdf5')
    names = glob.glob('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron/*/*.tif') #One Neuron Tests
#%%
j = 6 # 0,2,3,4,6,9,9,-1
movie = names[j]
mov = io.imread(movie)
full = True
print(movie)
#%%
j=8
movie = names[j]
import h5py
with h5py.File(movie, "r") as f:
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    mov = np.array(f['mov'])
full = True
#%% optional rotation
# mov = np.transpose(mov, (0, 2, 1))
# plt.imshow(mov[0])
# full=True
#%% motion correct layer setup
from viola.motion_correction_gpu import MotionCorrectTest
template = np.load(movie[:-4]+"_template_on.npy")
# template = np.transpose(template)
#template = temp
# template = np.median(mov[:2000], axis=0)
#%%
if full:
    mc_layer = MotionCorrectTest(template, mov[0].shape, ms_h=10, ms_w=10)
else:
    mc_layer = MotionCorrectTest(template, (mov[0].shape[0]//2, mov[0].shape[1]//2), ms_h=10, ms_w=10)
    
#%% run mc
shifts = []
# new_mov = []
for i in range(len(mov)):
    fr = mc_layer(mov[i, :, :, None][None, :].astype(np.float32))
    shifts.append(fr[1]) 
    # new_mov.append(fr[0]) #movie
# new_mov = np.array(new_mov).squeeze() 
# reshape shifts
x_shift, y_shift = [], []
for i in range(len(shifts)):
    x_shift.append(shifts[i][0].numpy())
    y_shift.append(shifts[i][1].numpy())
x_shift = np.array(x_shift).squeeze()
y_shift = np.array(y_shift).squeeze() 
#%% save
if full:
    # np.save(movie[:73]+"new_cm/"+movie[73:-4]+"_viola_full_shifts", shifts)
    np.save(movie[:-4]+"_viola_full_shifts", shifts)
else:
    # np.save(movie[:73]+"new_cm/"+movie[73:-4]+"_viola_small_shifts", shifts)
    np.save(movie[:-4]+"_viola_small_shifts", shifts)
 
#%% plot
# v_big = np.load(movie[:-4]+"_viola_shifts.npy")
# plt.plot(x_shift)
# plt.plot(v_big[1])
# plt.plot(y_shift)
# plt.plot(v_big[0])
#%%  plotting traces
j=9
movie = names[j]
cm = np.load(movie[:-4]+"_cm_on_shifts.npy")
cmx, cmy = [], []
for i in range(len(cm)):
    cmx.append(cm[i][0])
    cmy.append(cm[i][1])
cmx = np.array(cmx).squeeze()
cmy = np.array(cmy).squeeze()

v_big = np.load(names[j][:73]+names[j][73:-4]+"_viola_full_shifts.npy").squeeze()
tempx,tempy=[],[]
for i in range(len(cm)):
    tempx.append(v_big[i][0])
    tempy.append(v_big[i][1])
v_big = [tempx,tempy]

v_small = np.load(names[j][:73]+names[j][73:-4]+"_viola_small_shifts.npy").squeeze()
tempx,tempy=[],[]
for i in range(len(cm)):
    tempx.append(v_small[i][0])
    tempy.append(v_small[i][1])
v_small = [tempx,tempy]
len_half = len(cm)//2

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42

fig, axs = plt.subplots(2)
fig.suptitle(names[j][73:-4])
axs[0].plot(cmx[len_half:], label="caiman");
axs[0].plot(v_small[0][len_half:], label="cropped viola");
axs[0].plot(v_big[0][len_half:], label="full viola");
axs[1].plot(cmy[len_half:], label="caiman")
axs[1].plot(v_small[1][len_half:], label="cropped viola");
axs[1].plot(v_big[1][len_half:], label="full viola");
plt.legend()
# fig.savefig(movie[:-4]+"_tracefig.pdf", bbox_inches="tight", format="pdf")
#%% error for motcorr
from collections import defaultdict
err_x = defaultdict(list)
err_y = defaultdict(list)
nshort = []
for n in [2,3,4,6,8,9,-3,-1]:
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
    
    for i in range(len_half,len(tempx1)):
        # if tempx1[i] != 0:
        err_x[nshort[-1]].append(tempx1[i]-v_big[0][i])
        # if tempy1[i] != 0:
        err_y[nshort[-1]].append(tempy1[i]-v_big[1][i])
    fill_len = 10000-len(err_x[nshort[-1]])
    err_x[nshort[-1]] += [np.nan]*fill_len 
    err_y[nshort[-1]] += [np.nan]*fill_len
    #np.pad(err_x[nshort[-1]], (20000,), mode="constant", constant_values=(-5,))
    #np.pad(err_y[nshort[-1]], (20000,), mode="constant", constant_values=(-5,))

    
    for i in range(len_half,len(tempx1)):
        # if tempx1[i] != 0:
        err_x[nshort[-1]].append(tempx1[i]-v_small[0][i])
        # if tempy1[i] != 0:
        err_y[nshort[-1]].append(tempy1[i]-v_small[1][i])
    fill_len = 10000-len(err_x[nshort[-1]])
    err_x[nshort[-1]] += [np.nan]*fill_len 
    err_y[nshort[-1]] += [np.nan]*fill_len
    # np.pad(err_x[nshort[-1]], (40000,), mode="constant", constant_values=(np.nan,))
    # np.pad(err_y[nshort[-1]], (40000,), mode="constant", constant_values=(np.nan,))
nshort += ["FOV"]
err_x["FOV"] = ["full"]*10000+["crop"]*10000
err_y["FOV"] = ["full"]*10000+["crop"]*10000

#%% violin plotting
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
#%%
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
#%%
fig, axs = plt.subplots(2)
axs[0].violinplot(tot_err_x_full) 
#axs[0].scatter(range(1,6), x_mean)
axs[0].title.set_text('X shift squared error')
#axs[0].set_xticklabels(nshort)
axs[1].violinplot(tot_err_y_full)
#axs[1].scatter(range(1,6), y_mean)
axs[1].title.set_text('Y shift squared error')

#%% "play" movie
for fr in new_mov:
    plt.imshow(fr);
    plt.pause(0.2)
    plt.cla()
 
 