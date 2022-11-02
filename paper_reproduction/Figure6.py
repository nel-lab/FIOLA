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
cm_times = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/k53_*_tna.npy"))
fiola_batch = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/Batch/Neurs/times_*.npy")
def filter_helper(dims, path):
    return all(x not in path for x in dims)
fiola_all = sorted(list(filter(lambda x: filter_helper(["256","1024","crop"], x), fiola_all)))
fiola_crop = sorted(list(filter(lambda x: filter_helper(["256","1024","768"], x), fiola_crop)))
fiola_batch = sorted(list(filter(lambda x: filter_helper(["256","1024","768"], x), fiola_batch)))

#%% plot roughly
#%% calculate means and stdevs
cou = 9 #9 #12 for full figure
batch, grph, crop = [0]*cou,[0]*cou,[0]*cou
batch_sd, grph_sd, crop_sd = [0]*cou,[0]*cou,[0]*cou
for i in range(3): #3): # 4 for full figure

    batch_data = np.load(fiola_batch[i], allow_pickle=True)/100
    batch[i*3] = np.mean(batch_data)
    batch_sd[i*3] = np.std(batch_data)
    grph_data = np.load(fiola_all[i], allow_pickle=True)
    grph[i*3+1]  = np.mean(grph_data)
    grph_sd[i*3+1] =  np.std(grph_data)
    crop_data = np.load(fiola_crop[i], allow_pickle=True)
    crop[i*3+2]  = np.mean(crop_data)
    crop_sd[i*3+2] =  np.std(crop_data)

#%% plot
# labels = ["sep100_1024","grph100_1024","egr100_1024","sep500_1024","grph500_1024","egr500_1024",\
#           "sep100_512","grph100_512","egr100_512","sep500_512","grph500_512","egr500_512"]
# labels = ["b100","g100","c100","b200","g200","c200","b500","g500","c500"]
labels = ["b100", "f100","c100","b200","f200","c200","b500","f500","c500"]
fig, ax = plt.subplots()
ax.bar(labels, batch, yerr=batch_sd, label="batch")
ax.bar(labels, grph, yerr=grph_sd,  label="full pipeline")
ax.bar(labels, crop, yerr=crop_sd, label="crop") 
ax.set_yscale('log')
plt.hlines(0.0025, 0, 10)

#%% Load datasets for 6C
# load mc/nnls/deconv separate times
mc_files =  sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_256*mc.npy"))
# nnls_files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_*100_nnls_time.npy"))
# nnls_files += sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_*500_nnls_time.npy"))
nnls_files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_fi_256*_nnls_time.npy"))
nnls_files = sorted(nnls_files)
nnls_old = [
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/NNLS/1024/k53_times_100_1024.npy",
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/NNLS/1024/k53_times_500_1024.npy",
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/NNLS/512/k53_times_100_512.npy",
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/NNLS/512/k53_times_500_512.npy"]
deconv_files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/Nature Methods Resubmission/Timing Johannes/k53/256*_deconv.npy"))
# load graph pipeline times
full_graph_files = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_256*.npy")
# full_graph_files += glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_graph/k53_*500.npy")
full_graph_files = sorted(full_graph_files)
# load eager pipeline times
full_eager_files = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_256*.npy")
# full_eager_files += glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/k53_*_500.npy")
full_eager_files = sorted(full_eager_files)

#%% calculate means and stdevs
cou = 9 #12 for full figure
mc, nnls, dcv, grph, egr = [0]*cou,[0]*cou,[0]*cou,[0]*cou,[0]*cou
mc_sd, nnls_sd, dcv_sd, grph_sd, egr_sd = [0]*cou,[0]*cou,[0]*cou,[0]*cou,[0]*cou
for i in range(3): # 4 for full figure
    mc_data = np.load(mc_files[0], allow_pickle=True) # i//2 for full
    mc[i*3] = np.mean(mc_data)
    mc_sd[i*3] = np.std(mc_data)
    nnls_data = np.load(nnls_files[i], allow_pickle=True)
    nnls[i*3] = np.mean(nnls_data)
    nnls_sd[i*3] = np.std(nnls_data)
    dcv_data = np.load(deconv_files[i], allow_pickle=True)
    dcv[i*3] = np.mean(dcv_data)
    dcv_sd[i*3] = np.std(dcv_data)
    grph_data = np.load(full_graph_files[i], allow_pickle=True)
    grph[i*3+1]  = np.mean(grph_data)
    grph_sd[i*3+1] =  np.std(grph_data)
    egr_data = np.load(full_eager_files[i], allow_pickle=True)
    egr[i*3+2]  = np.mean(egr_data)
    egr_sd[i*3+2] =  np.std(egr_data)

#%% plot
# labels = ["sep100_1024","grph100_1024","egr100_1024","sep500_1024","grph500_1024","egr500_1024",\
#           "sep100_512","grph100_512","egr100_512","sep500_512","grph500_512","egr500_512"]
labels = ["sep100_256","grph100_256","egr100_256","sep200_256","grph200_256","egr200_256","sep500_256","grph500_256","egr500_256"]
fig, ax = plt.subplots()
ax.bar(labels, mc, yerr=mc_sd, label="mc")
ax.bar(labels, nnls, yerr=nnls_sd, bottom=mc, label="nnls") 
ax.bar(labels, dcv, yerr=dcv_sd, bottom = np.sum([mc,nnls],axis=0), label="deconv")
ax.bar(labels, grph, yerr=grph_sd,  label="full pipeline")
ax.bar(labels, egr, yerr=egr_sd, label="eager")   
    
    
