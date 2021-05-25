#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:35:16 2021

@author: cxd00
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
import h5py
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import glob

#%% set folders
calcium = False

if calcium:
    base_folder = "/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE"
    dataset = ["/N.00.00", "/N.01.01", "/N.02.00", "/N.03.00.t", "/N.04.00.t", "/YST"][2]
    slurm_data = base_folder + dataset + "/results_analysis_online_sensitive_SLURM_01.npz"
else:
    base_folder = "/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data"
    dataset = ["/FOV1", "/FOV1_35um", "/FOV2_80um", "/FOV4_50um", "/403106_3min"][2]
    H_new = np.load(base_folder + dataset + dataset + "_H_new.npy")
    with h5py.File(base_folder + dataset + dataset + ".hdf5",'r') as h5:
       a2 = np.array(h5['mov'])       

#%% get ground truth data
with np.load(slurm_data, allow_pickle=True) as ld:
    print(list(ld.keys()))
    locals().update(ld)
    Ab_gt = Ab[()].toarray()
    num_bckg = f.shape[0]
    b_gt, A_gt = Ab_gt[:,:num_bckg], Ab_gt[:,num_bckg:]
    num_comps = Ab_gt.shape[-1]
    f_gt, C_gt = Cf[:num_bckg], Cf[num_bckg:num_comps]
    noisyC = noisyC[num_bckg:num_comps]
#%% start comps : save the indices where each spatial footprint first appears
start_comps = []
for i in range(num_comps):
    start_comps.append(np.where(np.diff(Cf[i])>0)[0][0])
plt.plot(start_comps,np.arange(num_comps),'.')
#%% initialize with i frames, calculate GT
for i in [len(noisyC[0])//4]: #50%
    included_comps = np.where(np.array(start_comps)<i)[0]
    A_start = A_gt[:,included_comps]
    C_start = C_gt[included_comps]
    noisyC_start = noisyC[included_comps] # ground truth
    b_gt = b_gt
    f_gt = f_gt  
#%% pick up images as a movie file
dirname = base_folder + dataset + "/images_" + dataset[1:]
a2 = []
#for fname in os.listdir(dirname):r
for i in range(noisyC.shape[1]//2):
    fname = "image" + str(i).zfill(5) + ".tif"
    im = io.imread(os.path.join(dirname, fname))
    # a2.append(resize(im, (256, 256)))
    a2.append(im)

    #a2.append(im[0:125, 125:256])    
#%% image normalization for movie
img_norm = np.std(a2, axis=0)
img_norm += np.median(img_norm)
a2 = a2/img_norm[None, :, :]
#a2 = to_2D(np.asarray(Y_tot)).T
#%% one-time calculations and template
# template = np.median(a2[:len(a2)//2], axis=0) # template created w/ first half
#f, Y =  f_full[:, 0][:, None], Y_tot[:, 0][:, None]
#YrA = YrA_full[:, 0][:, None]
# YrA = 
#C = C_full[:, 0][:, None]

#Ab = np.concatenate([A_sp_full.toarray()[:], b_full], axis=1).astype(np.float32) # A_gt
#b = Y_tot[:, 0]
Ab_gt_start = np.concatenate([A_start, b_gt], axis=1).astype(np.float32)
AtA = Ab_gt_start.T@Ab_gt_start
Atb = Ab_gt_start.T@b_gt
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_1 = (np.eye(Ab_gt_start.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)


#Cff = np.concatenate([C_full+YrA_full,f_full], axis=0)
Cf = np.concatenate([noisyC_start,f_gt], axis=0)
x0 = Cf[:,0].copy()[:,None]
#%% fiola nnls
from viola.nnls_gpu import NNLS, compute_theta2
from scipy.optimize import nnls


# Ab = H_new.astype(np.float32)
Ab  = Ab_gt_start
# b = a2[0].reshape(-1, order='F')
b = b[:,0]

x0 = nnls(Ab,b)[0][:,None].astype(np.float32)
x_old, y_old = x0, x0
AtA = Ab.T@Ab
Atb = Ab.T@b
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
#%%
mc0 = a2[0:1,:,:, None]
c_th2 = compute_theta2(Ab_gt_start, n_AtA)
n = NNLS(theta_1)
#%%
# from temp_new_pipeline import get_nnls_model
model = get_nnls_model(template, Ab_gt_start)
#%%
num_layers = 30

nnls_out = []
k = np.zeros_like(1)
shifts = [0.0, 0.0]
for i in range(500):
    mc = np.reshape(np.transpose(a2[i]), [-1])[None, :]
    (th2, shifts) = c_th2(mc, shifts)
    x_kk = n([y_old, x_old, k, th2, shifts])

    
    for j in range(1, num_layers):
        x_kk = n(x_kk)
        
    y_old, x_old = x_kk[0], x_kk[1]
    nnls_out.append(y_old)
nnls_out = np.array(nnls_out).squeeze().T
# np.save(base_folder + dataset + "/v_nnls_"+str(num_layers), nnls_out)
#%% nnls
y = to_2D(a2[:30])
from scipy.optimize import nnls
# H_new = np.load(base_folder + dataset + dataset + "_H_new.npy")
H_new = Ab_gt_start
fe = slice(0,None)
trace_nnls = np.array([nnls(H_new,fr)[0] for fr in (y)[fe]])
trace_nnls = trace_nnls.T 
#%% model creation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
shapes = a2[0].shape
from viola.pipeline_gpu import Pipeline, get_model
model = get_model(template, shapes, Ab_gt_start.astype(np.float32), 30)
#model = get_model(template, shapes, newAb, 30)
model.compile(optimizer='rmsprop', loss='mse')
#%% extract traces
a2 = np.asarray(a2)
mc0 = np.expand_dims(a2[0:1, :, :], axis=3)
trace_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, a2)
#%%
out = trace_extractor.get_traces(500)

test_traces = out
test_traces = np.array(test_traces).T.squeeze()

#%% fig gen
from scipy.stats import pearsonr
names = []
base_folder = "../../NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE"
datasets = ["/N.00.00", "/N.01.01", "/N.02.00", "/N.03.00.t", "/N.04.00.t", "/YST"]
bgs = [2,1,2,3,3,3,0,0,0]
for val in datasets:
    names.append(base_folder + val)
base_folder = "/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data"
# datasets += ["/FOV1", "/FOV2_80um", "/FOV4_50um"]
datasets += ["/FOV1"]
for val in datasets[6:]:
    names.append(base_folder + val)

from collections import defaultdict        
err = defaultdict(list)
for i,n in enumerate(names):
    gt = np.load(n + "/nnls.npy")
    bg = bgs[i]
    nnls30= np.load(n +"/v_nnls_30.npy")
    nnls10= np.load(n +"/v_nnls_10.npy")
    nnls5= np.load(n+ "/v_nnls_5.npy")
    
    num_comps = nnls30.shape[0]
    l = nnls30.shape[1]//2
    gt = np.append(gt[bg:num_comps], gt[0:bg], axis=0)
    fill_len = 263-num_comps
        
    if i == 2:
        err[n] = [pearsonr(gt[j], nnls30[j, l:])[0] for j in range(nnls30.shape[0])]
        err[n]+=[np.nan]*fill_len
        err[n]+=[pearsonr(gt[j], nnls10[j, l:])[0] for j in range(nnls10.shape[0])]
        err[n]+=[np.nan]*fill_len
        err[n]+=[pearsonr(gt[j], nnls5[j, l:])[0] for j in range(nnls5.shape[0])]
        err[n]+=[np.nan]*fill_len
        # err[n] = [np.nan if err[n][i] < 0.7 else err[n][i] for i in range(len(err[n]))]
        # plt.plot(gt[-1]);plt.plot(nnls30[-1,l:])
        # err[n+"10"] = [pearsonr(gt[j], nnls10[j, l:])[0] for j in range(nnls10.shape[0]-bg)]
        # err[n+"5"] = [pearsonr(gt[j], nnls5[j, l:])[0] for j in range(nnls5.shape[0]-bg)]        
    else:
        err[n] = [pearsonr(gt[j, l:], nnls30[j, l:])[0] for j in range(nnls30.shape[0])]
        err[n]+=[np.nan]*fill_len
        err[n]+=[pearsonr(gt[j, l:], nnls10[j, l:])[0] for j in range(nnls10.shape[0])]
        err[n]+=[np.nan]*fill_len
        err[n]+=[pearsonr(gt[j, l:], nnls5[j, l:])[0] for j in range(nnls5.shape[0])]
        err[n]+=[np.nan]*fill_len
        # err[n] = [np.nan if err[n][i] < 0.7 else err[n][i] for i in range(len(err[n]))]
        # err[n+"10"] = [pearsonr(gt[j, l:], nnls10[j, l:])[0] for j in range(nnls10.shape[0]-bg)]
        # err[n+"5"] = [pearsonr(gt[j, l:], nnls5[j, l:])[0] for j in range(nnls5.shape[0]-bg)]
        # plt.plot(gt[-3]);plt.plot(nnls30[-3])

#%% to look at the "bad" traces:
    #NOTE: for N.02.00 I only  saved teh back half of the scipy optimize NNLS so  you'll have to only plot the back 10,000 frames
k=1   
n=names[k]
bg = bgs[k]
bad = []
for i,val in enumerate(err[n]):
    if val < 0.8:
        bad.append(i)

nnls30 = np.load(n +"/v_nnls_30.npy")
num_comps = nnls30.shape[0]
gt = np.load(n + "/nnls.npy")
gt = np.append(gt[bg:num_comps], gt[0:bg], axis=0)

for idx in bad:
    if idx < num_comps:
        plt.plot(gt[idx])
        plt.plot(nnls30[idx])
        plt.pause(5)
        plt.cla()

#%% plotting
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")
df = pd.DataFrame()
i=0
for key in err:
    df.loc[:,i]=pd.Series(err[key])
    i+=1
df.columns = datasets
df["layers"] = [30]*263 + [10]*263 + [5] * 263
 
df = pd.melt(df, id_vars=["layers"], var_name="file", value_name="Error")
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
fig,ax = plt.subplots(1)

sns.boxplot(x=df["file"], y=df["Error"], hue=df["layers"], palette="viridis")
ax.set_xticklabels(datasets)

