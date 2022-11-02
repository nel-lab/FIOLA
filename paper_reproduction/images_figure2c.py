#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:30:24 2021

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
from fiola.utilities import to_3D, to_2D
import scipy
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx("float32")
#%% load caiman outputs
datasets = ["N00",  "N01", "N02", "N03", "N04", "YST", "k53"]
dataset = datasets[0]
base_folder = "/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/"
A = np.load(base_folder + dataset + "_half_A.npy", allow_pickle=True)[()].toarray()
b = np.load(base_folder + dataset + "_half_b.npy",allow_pickle=True)[()]
C = np.load(base_folder + dataset + "_C.npy", allow_pickle=True)[()]
f = np.load(base_folder + dataset + "_f.npy", allow_pickle=True)[()]
# noisyC = np.load(base_folder + dataset +  "_half_noisyC.npy", allow_pickle=True)[()]
#%% variable creation
num_bckg = f.shape[0]
num_comps = A.shape[-1]
lful  = C.shape[-1]
lhaf = C.shape[-1]//2+1

Cf = np.concatenate((C, f), axis=0)
Ab = np.concatenate((A, b), axis=1)

# start_comps = []
# for i in range(num_comps):
#     if max(Cf[i]) > 0:
#         start_comps.append(np.where(np.diff(Cf[i])>0)[0][0])
    
for i in [lhaf-1]: #50%
    # included_comps = np.where(np.array(start_comps)<i)[0]
    included_comps = np.load(base_folder + dataset +"_incl.npy")
    A_start = A[:, included_comps]
    C_start = C[included_comps]
    # noisyC_start = noisyC[included_comps+num_bckg] # ground truth
    Cf_start = np.concatenate((C_start, f), axis=0)

print(len(included_comps))   
#%% load movie
# a2 = io.imread("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00001.tif")
a2 = io.imread("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/N.00.00/mov_YST.tif")
a2 -= np.min(a2[:lhaf])
# img_norm = np.std(a2[:200], axis=0)
# img_norm = np.load(base_folder+"k53_img_norm.npy")
# img_norm += np.median(img_norm)
# a2 = a2/img_norm[None, :, :]
template = np.median(a2, axis=0)
# template = np.load("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00001_template_on.npy")
#%% set up calculations
from scipy.optimize import nnls
Ab = np.concatenate((A_start,b), axis=1)
plt.imshow(Ab[:,:].sum(-1).reshape(template.shape, order='F'))
#x0 = nnls(Ab,b[:,0])[0][:,None].astype(np.float32)
x0 = Cf_start[:,0].copy()[:,None].astype(np.float32)
x_old, y_old = x0, x0
AtA = Ab.T@Ab
Atb = Ab.T@b
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
mc0 = a2[0:1,:,:, None]
#%% model creation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# shapes = a2[0].shape
from viola.pipeline_gpu import Pipeline, get_model

model = get_model(template, template.shape, Ab.astype(np.float32), 30)
#%% run model
model.compile(optimizer='rmsprop', loss='mse')
a2 = np.asarray(a2)
# mc0 = np.expand_dims(a2[0:1, :, :], axis=3)
trace_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, a2)

import time
start = time.time()
out = trace_extractor.get_traces(a2.shape[0])
print(time.time()-start)

test_traces = np.array(out[0]).squeeze().T
shifts = out[1]
times = out[2]
plt.plot(np.diff(times[1:-1])) 
#%% plot
from  scipy import stats
full_C = np.load(base_folder + dataset+"_C.npy", allow_pickle=True)[()][included_comps]
full_f = np.load(base_folder + dataset+"_f.npy", allow_pickle=True)[()]
full_Cf = np.concatenate((full_C, full_f))
incl_comps_alt = np.concatenate((included_comps+num_bckg, range(num_bckg)))
full_noisyC = np.load(base_folder + dataset+"_noisyC.npy", allow_pickle=True)[()][incl_comps_alt]

rs = []
norms = []
snrs = []
# from caiman.source_extraction.cnmf.temporal import constrained_foopsi

for fi, ca in zip(test_traces[:-num_bckg,lhaf-1:], full_noisyC[:, lhaf-1:]):
    # normscore = np.linalg.norm(fi-ca)/np.linalg.norm(ca)
    # norms.append(normscore)
    # c, bl, c1, g, sn, s, lam  = constrained_foopsi(vol, p=1)
    #  use c
    #  first 750 vs last 750
    
    rscore =  stats.pearsonr(fi, ca*(ca>0))[0]
    rs.append(rscore)
    
    # snr = max(ca) / stats.iqr(ca)
    # snrs.append(snr)
plt.plot(rs)
#np.save(base_folder+  dataset + "_rscore",  rs)    
#     if (snr < 0) and (snr > 5):
#         print(snr, rscore)
#         plt.cla()
#         plt.plot(fi);plt.plot(ca)
#         plt.pause(1)
    
# plt.scatter(x=range(len(rs)),  y=rs, color="red")
# plt.scatter(x=range(len(norms)), y=norms,  color="blue")
# plt.pause(1)
# plt.cla()
# plt.scatter(x=range(len(norms)), y=norms)
#%%
snrs  = np.load("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/"+dataset+"_SNR.npy")
snrs = snrs[included_comps+num_bckg]
rs = np.array(rs)
plt.scatter(-snrs,rs,alpha=0.2)
# plt.scatter(-snrs[snrs<-20],rs[snrs<-20], alpha=0.2);
# plt.yscale("log", nonposy="clip")
# plt.xlim([0,1000])
#%%
# plt.figure()
metrics = {}
for d in datasets[:-1]:
    print(d)
    snrs  = np.load("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/"+d+"_SNR.npy")
    print(np.mean(snrs))
    included_comps = np.load(base_folder  + d +"_incl.npy")
    snrs = snrs[included_comps+num_bckg]
    rs = np.load(base_folder +  d + "_rscore.npy")
    # plt.scatter(-snrs, rs, alpha=0.2)
    # plt.yscale("log", nonposy="clip")
    metrics[d]  = [snrs, rs]

metrics['N04'][1].shape
#%%
import seaborn
import  pandas as pd
rss  = []
names = []
fitnesses  = []
for i in range(6):
    rss += list(metrics[datasets[i]][1])
    rl = len(metrics[datasets[i]][1])
    names += [datasets[i]]*rl
    fitnesses += list(metrics[datasets[i]][0])
    
fitnesses = np.array(fitnesses)
# norm = plt.Normalize(fitnesses.min(),  fitnesses.max())
# n = fitnesses.size()
# sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
# sm.set_array([])

# cmap = seaborn.cubehelix_palette(as_cmap=True)    
df = pd.DataFrame({"names":names, "R":rss, "fitness":fitnesses})
quants = [np.quantile(fitnesses, p) for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
fit = pd.cut(df["fitness"], bins=quants, labels=["10", '20', '30', '40', '50', '60', '70', '80', '90', '100'])
df["classes"]  = fit
# cmap = seaborn.cubehelix_palette(rot = -0.2, as_cmap=True)
#%%
seaborn.stripplot(x="names", y=-np.log10(1-df["R"]), hue="classes", data=df, jitter=True, alpha=1,  s=10, palette=seaborn.color_palette("rocket", 10))
# plt.hlines(-np.log10(0.1), 0, 5)
ylab = -np.log10(1-np.array([0.5, 0.9,  0.95, 0.99, 0.999]))
ylabe = [0.5, 0.9,  0.95, 0.99, 0.999]
plt.yticks(ylab, ylabe)
# plt.ytick_labels(ylab)
# exp = lambda x:  10**(x)
# plt.set_yscale('function', functions=(exp))
# plt.yscale("log")  
#%%
for  i,s in enumerate(snrs):
    if s>-1500 and s<-1000:
        if rs[i] < 0.97:
            print(i, rs[i], s)
            plt.figure()
            ca = full_noisyC[i, 1500:]
            plt.plot((ca*(ca>0)))
            plt.plot(test_traces[i, 1500:])
            plt.title(str(i)+" "+str(rs[i])+" "+str(s))
    if s>-100 and s<-20:
        if rs[i] > 0.979:
            print(i, rs[i], s)
            plt.figure()
            ca = full_noisyC[i, 1500:]
            plt.plot((ca*(ca>0)))
            plt.plot(test_traces[i, 1500:])
            plt.title(str(i)+" "+str(rs[i])+" "+str(s))
#%%
fig,ax = plt.subplots(2)
ca = full_noisyC[106, 1500:]
ca = (ca*(ca>0))
ax[0].plot(ca)
ax[0].plot(test_traces[106, 1500:])
# ax[0].set_xlabel("106 "+str(rs[106])+" "+str(snrs[106]))

ca = full_noisyC[136, 1500:]
ca = (ca*(ca>0))
ax[1].plot(ca)
ax[1].plot(test_traces[136, 1500:])
# ax[1].set_xlabel("136 "+str(rs[136])+" "+str(snrs[136]))
    #%%
Y_tot = to_2D(a2)[1500:].T
print(np.linalg.norm(Y_tot-Ab@test_traces[:,1500:])/np.linalg.norm(Y_tot))
print(np.linalg.norm(Y_tot-Ab@full_noisyC[:,1500:])/np.linalg.norm(Y_tot))

plt.scatter(x=range(len(rs)),  y=rs)
plt.pause(1)
plt.scatter(x=range(len(norms)),  y=norms)
#%%   compare visually
if True:
    for vol, ca in zip(test_traces[:-num_bckg,:], full_noisyC[:, :]):
#    print(tf.reduce_sum(vol), tf.reduce_sum(ca))
        rscore =  stats.pearsonr(vol[lhaf:], (ca*(ca>0))[lhaf:])[0]
        if (rscore < 0.8):
            plt.cla()
            plt.plot((ca*(ca>0))[lhaf:], label='caiman')
            plt.plot((vol[lhaf:]), label='fiola', color="red")
            
            # plt.xlim([200,400])
        # plt.ylim([0,10])
        # break
            plt.pause(1) 
# pre-rectification caiman ratio max / IQR ==> x-axis, yaxis ==> RScore w/ rectification
#%%
plt.plot(full_noisyC[23, 1500:]);plt.plot(test_traces[23, 1500:]);
                                                    