#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:03:58 2022

@author: nel
"""

#%% Import all
import tensorflow as tf
from fiola.gpu_mc_nnls import get_nnls_model
import matplotlib.cbook as cbook
import seaborn as sns
import suite2p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from skimage import io
import glob
from tifffile import imread
import h5py

#%% load datasets for 6A
fiola_all = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/*/*_times.npy")
fiola_crop = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/Crop/time*.npy")
cm_times = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/26feb_finalsub/cm_*.npy"))
fiola_batch = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/Batch/Neurs/times_*.npy")
def filter_helper( path):
    dims = ["256","768", "crop"]
    return all(x not in path for x in dims)
fiola_all = sorted(list(filter(lambda x: filter_helper(x), fiola_all)), key=lambda y: (int(y.split("/")[-2]), int(y.split("_")[-2])))
fiola_crop = sorted(list(filter(lambda x: filter_helper( x), fiola_crop)), key=lambda y: (int(y.split("_")[-1][:-4]), int(y.split("_")[-2])))
fiola_batch = sorted(list(filter(lambda x: filter_helper(x), fiola_batch)), key=lambda y: (int(y.split("_")[-2]), int(y.split("_")[-1][:-4])))
cm_times = sorted(list(filter(lambda x: filter_helper( x), cm_times)))

#%% calculate means and stdevs
cou = 24 #24 for full figure
offset = 4 #4 for full figure

batch, grph, crop, cm = [0]*cou,[0]*cou,[0]*cou,[0]*cou
batch_sd, grph_sd, crop_sd, cm_sd = [0]*cou,[0]*cou,[0]*cou,[0]*cou
batch_all,grph_all,crop_all,cm_all = [],[],[],[]
dsets = []
for i in range(6): # 6 for full figure

    cm_data = np.load(cm_times[i], allow_pickle=True)[()][3000:]
    cm[i*offset]  = np.mean(cm_data)
    cm_sd[i*offset] =  np.std(cm_data)
    cm_all = np.concatenate([cm_all, cm_data])
    dsets.append(cm_data)
    grph_data = np.load(fiola_all[i], allow_pickle=True)*1000
    grph[i*offset+1]  = np.mean(grph_data)
    grph_sd[i*offset+1] =  np.std(grph_data)
    grph_all = np.concatenate([grph_all,grph_data])
    dsets.append(grph_data)
    crop_data = np.load(fiola_crop[i], allow_pickle=True)*1000
    crop[i*offset+2]  = np.mean(crop_data)
    crop_sd[i*offset+2] =  np.std(crop_data)
    crop_all = np.concatenate([crop_all,crop_data])
    dsets.append(crop_data)
    batch_data = np.load(fiola_batch[i], allow_pickle=True)*10
    batch[i*offset+3] = np.mean(batch_data)
    batch_sd[i*offset+3] = np.std(batch_data)
    batch_all = np.concatenate([batch_all,batch_data])
    dsets.append(batch_data) # order matters here!

#%% plot for 6A
# labels = ["sep100_1024","grph100_1024","egr100_1024","sep500_1024","grph500_1024","egr500_1024",\
          # "sep100_512","grph100_512","egr100_512","sep500_512","grph500_512","egr500_512"]
# labels = ["b100","g100","c100","b200","g200","c200","b500","g500","c500"]
# labels = ["b100", "f100","c100", "x100","b200","f200","c200", "x200","b500","f500","c500", "x500"]
labels = range(1, 25) # range(24) for full fig

fig, ax = plt.subplots()
ax.bar(labels, cm, yerr=cm_sd, label="caiman",zorder=1)
ax.bar(labels, grph, yerr=grph_sd,  label="full pipeline",zorder=1)
ax.bar(labels, crop, yerr=crop_sd, label="crop",zorder=1)
ax.bar(labels, batch, yerr=batch_sd, label="batch", zorder=1)
# ax.boxplot(dsets, whis=(0.1,99.9))
# ax.violinplot(dsets, widths=0.8, showextrema=False)
"""
Uncomment for scatter plot
for j,dset in enumerate(dsets):

    jitter_labels = [j+4*(i//(len(dset)//3)) +np.random.rand()*0.5 -0.25 for i in range(len(dset))]
    jitter_labels = [j+(i//(len(dset)))+np.random.rand()*0.2 -0.1 for i in range(len(dset))]
    ax.scatter(jitter_labels,dset,zorder=2,facecolors="none", edgecolors="gray")
"""
# ax.scatter(labels, grph_data,zorder=2)
# ax.scatter(labels, crop,zorder=2)
# ax.scatter(labels, cm,zorder=2)
ax.set_yscale('log')
plt.hlines(2.5, 0, len(labels))
plt.legend()

#%% Load datasets for 6C
# load mc/nnls/deconv separate times
mc_files =  sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_*mc.npy"))
nnls_files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_*100_nnls_time.npy"))
nnls_files += sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_*500_nnls_time.npy"))
# nnls_files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_256*_nnls_time.npy"))
nnls_files = sorted(nnls_files)
nnls_old = [
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/NNLS/1024/k53_times_100_1024.npy",
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/NNLS/1024/k53_times_500_1024.npy",
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/NNLS/512/k53_times_100_512.npy",
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/NNLS/512/k53_times_500_512.npy"]
deconv_files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/Nature Methods Resubmission/Timing Johannes/k53/*_deconv.npy"))
# load graph pipeline times
full_graph_files = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_*100.npy")
full_graph_files += glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_*500.npy")
# full_graph_files = sorted(full_graph_files)
# load eager pipeline times
full_eager_files = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_*100.npy")
full_eager_files += glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_*500.npy")
# full_eager_files = sorted(full_eager_files)
#%% hardcoding for full
mc_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_512_mc.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_1024_mc.npy']
nnls_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_512_100_nnls_time.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_512_500_nnls_time.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_1024_100_nnls_time.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_1024_500_nnls_time.npy']
deconv_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/Nature Methods Resubmission/Timing Johannes/k53/512_100_deconv.npy',
                '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/Nature Methods Resubmission/Timing Johannes/k53/512_500_deconv.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/Nature Methods Resubmission/Timing Johannes/k53/1024_100_deconv.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/Nature Methods Resubmission/Timing Johannes/k53/1024_500_deconv.npy']
full_graph_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_512_100.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_512_500.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_1024_100.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_1024_500.npy']
full_eager_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_512_100.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_512_500.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_1024_100.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_1024_500.npy']
#%%
mc_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_256_mc.npy']
nnls_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_256_100_nnls_time.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_256_200_nnls_time.npy',
              '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_256_500_nnls_time.npy']
deconv_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/Nature Methods Resubmission/Timing Johannes/k53/256_100_deconv.npy',
                '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/Nature Methods Resubmission/Timing Johannes/k53/256_200_deconv.npy',
                '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/Nature Methods Resubmission/Timing Johannes/k53/256_500_deconv.npy']
full_graph_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_256_100.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_256_200.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_256_500.npy']
full_eager_files = ['/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_256_100.npy',
                    '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_256_200.npy',
 '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_256_500.npy']
#%% calculate means and stdevs
cou = 12 #12 for full figure
mc, nnls, dcv, grph, egr = [0]*cou,[0]*cou,[0]*cou,[0]*cou,[0]*cou
mc_sd, nnls_sd, dcv_sd, grph_sd, egr_sd = [0]*cou,[0]*cou,[0]*cou,[0]*cou,[0]*cou
offset = 3
dsets = []
jitter_labels = []
for i in range(4): # 4 for full figure

    grph_data = np.load(full_graph_files[i], allow_pickle=True)*1000
    grph[i*offset+1]  = np.mean(grph_data)
    grph_sd[i*offset+1] =  np.std(grph_data)
    # dsets.append(grph_data)
    jitter_labels.append([i*3+1+(l//len(grph_data))+np.random.rand()*0.5 -0.25 for l in range(len(grph_data))])

    mc_data = np.load(mc_files[0], allow_pickle=True)*1000 # i//2 for full, 0 for 256
    mc[i*offset] = np.mean(mc_data)
    mc_sd[i*offset] = np.std(mc_data)
    # dsets.append(mc_data)
    jitter_labels.append([i*3+(l//len(mc_data))+np.random.rand()*0.5 -0.25 for l in range(len(mc_data))])    
    
    nnls_data = np.load(nnls_files[i], allow_pickle=True)*1000
    nnls[i*offset] = np.mean(nnls_data)
    nnls_sd[i*offset] = np.std(nnls_data)
    med=nnls_data + np.mean(mc_data)
    # dsets.append(med)
    jitter_labels.append([i*3+(l//len(med))+np.random.rand()*0.5 -0.25 for l in range(len(med))])

    dcv_data = np.load(deconv_files[i], allow_pickle=True)*1000
    dcv[i*offset] = np.mean(dcv_data)
    dcv_sd[i*offset] = np.std(dcv_data)
    fin=dcv_data + np.mean(mc_data)+ np.mean(nnls_data)
    # dsets.append(fin)
    jitter_labels.append([i*3+(l//len(fin))+np.random.rand()*0.5 -0.25 for l in range(len(fin))])
                 
    egr_data = np.load(full_eager_files[i], allow_pickle=True)*1000
    egr[i*offset+2]  = np.mean(egr_data)
    egr_sd[i*offset+2] =  np.std(egr_data)
    # dsets.append(egr_data)
    jitter_labels.append([i*3+2+(l//len(egr_data))+np.random.rand()*0.5 -0.25 for l in range(len(egr_data))])

    dsets.append(grph_data)
    print(len(dsets[-1]), 1)
    dsets.append(mc_data)
    print(len(dsets[-1]))
    dsets.append(med)
    print(len(dsets[-1]))
    dsets.append(fin)
    print(len(dsets[-1]))
    dsets.append(egr_data)
    print(len(dsets[-1]))

#%% plot 6c
labels = ["sep100_1024","grph100_1024","egr100_1024","sep500_1024","grph500_1024","egr500_1024",\
            "sep100_512","grph100_512","egr100_512","sep500_512","grph500_512","egr500_512"]
# labels = ["sep100_256","grph100_256","egr100_256","sep200_256","grph200_256","egr200_256","sep500_256","grph500_256","egr500_256"]
# labels = range(1,21)
fig, ax = plt.subplots()
ax.bar(labels, mc, yerr=mc_sd, label="mc")
ax.bar(labels, nnls, yerr=nnls_sd, bottom=mc, label="nnls") 
ax.bar(labels, dcv, yerr=dcv_sd, bottom = np.sum([mc,nnls],axis=0), label="deconv")
ax.bar(labels, grph, yerr=grph_sd,  label="full pipeline")
ax.bar(labels, egr, yerr=egr_sd, label="eager") 
ax.boxplot(dsets, whis=(0.1,99.9))
ax.set_yscale("log")
# for j,dset in enumerate(dsets):
#     ax.scatter(jitter_labels[j], dset,zorder=2,s=0.1)  
   
    
