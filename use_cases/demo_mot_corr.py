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
from skimage.transform import resize,  rescale
import cv2
import timeit
import multiprocessing as mp
from tensorflow.python.keras import backend as K
from fiola.utilities import to_3D, to_2D
import scipy
import glob
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_XLA_FLAGS"]="--tf_xla_enable_xla_devices"

#%% get files
many =  True
names = []
if many:
    names = glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*.tif')
    names += glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*.tif')
    names += glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/viola_movies/*.hdf5')
    names += glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*.hdf5')
    j = 0 # 0,2,3,4,6,9,9,-1
    movie = names[j]
    # movie = "/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/k53_256.tif"
    mov = io.imread(movie)
    full = True
    print(movie)
else:
    # names.append('/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/mesoscope.hdf5')
    names+=glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/*/*.hdf5')
    names += glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/viola_movies/*.hdf5')
    cm_names = glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/viola_movies/*cm_on_shifts.npy') #One Neuron Tests
    vi_names = glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/viola_movies/*full_shifts.npy') #One Neuron Tests
    j=-5
    movie = names[j]
    # movie="/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/nnls/FOV4_50um_ROIs.hdf5"
    import h5py
    with h5py.File(movie, "r") as f:
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        mov = np.array(f['mov'])
    full = True
# optional rotation
# mov = np.transpose(mov, (0, 2, 1))
# plt.imshow(mov[0])
# full=True
# motion correct layer setup
# from viola.motion_correction_gpu import MotionCorrectTest
# template = np.load(names[j][:-4]+"_template_on.npy")
template = np.load("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00001_template_on.npy")
# template = np.load("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/1MP_SIMPLE_Stephan__001_001_template_on.npy")
# template =  np.load("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/FOV1_35um/FOV1_35um._template_on.npy")
# template = np.transpose(template)
#template = temp
# template = np.median(mov[:2000], axis=0)
#%%
shifts = sorted(glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*_viola_full_shifts.npy'))[1:]
shifts += sorted(glob.glob("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/viola_movies/*_viola_full_shifts.npy"))
rshifts = sorted(glob.glob("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*_viola_small_shifts.npy"))[1:]
rshifts += sorted(glob.glob("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/viola_movies/*_viola_small_shifts.npy"))
cshifts = sorted(glob.glob("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*_cm_on_shifts.npy"))
cshifts += sorted(glob.glob("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/viola_movies/*cm_on_shifts.npy"))
outx = np.array([])
outy = np.array([])
cmx = np.array([])
cmy = np.array([])

for c,s,r in zip(cshifts, shifts, rshifts):
    t = np.load(r)
    x_shift = []
    y_shift = []
    for i in range(len(t)):
        x_shift.append(t[i][0])
        y_shift.append(t[i][1])
    x_shift = np.array(x_shift).squeeze()
    y_shift = np.array(y_shift).squeeze()
    outx=np.append(outx, x_shift)
    outy=np.append(outy,  y_shift)
    
    tc = np.load(r)
    x_shiftc = []
    y_shiftc = []
    for i in range(len(tc)):
        x_shiftc.append(tc[i][0])
        y_shiftc.append(tc[i][1])
    x_shiftc = np.array(x_shiftc).squeeze()
    y_shiftc = np.array(y_shiftc).squeeze()
    
    cm = np.load(c).squeeze()
    tempx1, tempy1 = [], []
    for i in range(len(cm)):
        tempx1.append(cm[i][0])
        tempy1.append(cm[i][1])
        
    cmx=np.append(cmx, tempx1)
    cmy=np.append(cmy, tempy1)
    

    
    print(s)
    print(c)
    print("stdx", np.std(x_shift-tempx1))
    print("stdy", np.std(y_shift-tempy1))
    # out[s] = [np.std(x_shift-tempx1), np.std(y_shift-tempy1)]
    # out[r] = [np.std(x_shiftc-tempx1), np.std(y_shiftc-tempy1)] 
    print()
#%%
print(s)
print("stdx", np.std(abs(outx-cmx)))
print("stdy", np.std(abs(outy-cmy)))
print("meanx", np.mean(abs(outx-cmx)))
print("meany", np.mean(abs(outy-cmy)))
#%%FOR ESTIMATOR TIMING
out = [0]*(mov.shape[0])
def generator():
    for frame in mov:
        # print(np.mean(frame))
        yield{"m":frame[None,:,:,None]}
             
def get_frs():
    dataset = tf.data.Dataset.from_generator(generator, output_types={'m':tf.float32}, output_shapes={"m":(1,mov.shape[1],mov.shape[2],1)})
    return dataset
#%%  
from FFT_MOTION import get_mc_model, MotionCorrect
import timeit
import time
import tensorflow.keras as keras
model = get_mc_model(template[:,:,None,None], (256, 256))
model.compile(optimizer='rmsprop', loss='mse')
estimator = tf.keras.estimator.model_to_estimator(model)
times = [0]*(mov.shape[0])
curr=0
start = timeit.default_timer()
for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
    out[curr]=i
    times[curr]=time.time()-start
    curr += 1
    break
print(timeit.default_timer()-start)
plt.plot(np.diff(times))
#%%
x=[]
y=[]
frs = []
kys  = list(out[0].keys())
for val in  out:
    frs.append(val[kys[0]])
    x.append(np.squeeze(val[kys[1]]))
    y.append(np.squeeze(val[kys[2]]))
plt.plot(x);plt.plot(cmx)
#%%
# f = movie[70:73]
f = input("names: ")
save_loc =  "/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/MC/"
np.save(save_loc+f+"_times_2.0", np.diff(times[1:-1]))
#%%
if full:
    mc_layer = MotionCorrectTest(template, mov[0].shape, ms_h=10, ms_w=10)
else:
    mc_layer = MotionCorrectTest(template, (mov[0].shape[0]//2, mov[0].shape[1]//2), ms_h=10, ms_w=10)
    
#%% run mc
shifts = []
# new_mov = []
for i in range(len(mov)//4):
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

#%%  plotting traces
movie = names[4]
cm = np.load(movie[:-4]+"_cm_on_shifts.npy")
# cm = np.load(cm_names[0])
cmx, cmy = [], []
for i in range(len(cm)):
    cmx.append(cm[i][0])
    cmy.append(cm[i][1])
cmx = np.array(cmx).squeeze()
cmy = np.array(cmy).squeeze()

v_big = np.load(names[j][:73]+names[j][73:-4]+"_viola_full_shifts.npy").squeeze()
# v_big = np.load(vi_names[1])
tempx,tempy=[],[]
for i in range(len(cm)):
    tempx.append(v_big[i][0].squeeze())
    tempy.append(v_big[i][1].squeeze())
v_big = [tempx,tempy]
plt.plot(cmx);plt.plot(v_big[0]);plt.plot(x)
plt.plot(cmy);plt.plot(v_big[1]);plt.plot(y)
#%%
np.save(save_loc + f + "_vf_shifts", [x,y])
#%%
v_small = np.load(names[j][:73]+names[j][73:-4]+"_viola_small_shifts.npy").squeeze()
tempx,tempy=[],[]
for i in range(len(cm)):
    tempx.append(v_small[i][0])
    tempy.append(v_small[i][1])
v_small = [tempx,tempy]
#%%
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
 
 