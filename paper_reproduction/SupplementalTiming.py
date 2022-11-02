#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 17:26:28 2022

@author: nel
"""
#%% Imports
from fiola.gpu_mc_nnls import get_mc_model, get_nnls_model, get_model, Pipeline
import tensorflow as tf
import tensorflow.keras as keras
from multiprocessing import Queue
import glob
import caiman as cm
import numpy as np
import matplotlib.pyplot as plt
from time import time

#%% Eager MC/NNLS Figure File Load
base_file_mc = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/MC/k53_*.npy"
base_file_nnls = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/fiola_eager/NNLS/k53_*.npy"
#%% Eager MC/NNLS Figure calculations
mc_avg = []
mc_std = []
nnls_avg = []
nnls_std = []
for dims in [512, 1024]:
    mc_file = str(dims).join(base_file_mc.split("*"))
    mc = np.diff(np.load(mc_file))*1000
    for neurs in [100, 500]:
        nnls_file = (str(dims) + "_"+ str(neurs)).join(base_file_nnls.split("*"))
        nnls = np.diff(np.load(nnls_file))*1000
        
        mc_avg.append(np.mean(mc))
        mc_std.append(np.std(mc))
        nnls_avg.append(np.mean(nnls))
        nnls_std.append(np.std(nnls))

#%% Eager MC/NNLS Figure plot
labels = ["512_100", "512_500", "1024_100", "1024_500"]
width = 0.25
fig, ax = plt.subplots()
ax.bar(labels, mc_avg, width, yerr=mc_std)
ax.bar(labels, nnls_avg, width, yerr=nnls_std, bottom=mc_avg)
ax.set_ylabel("ms")
ax.legend()
plt.show()
#%% Supplementary Figure 7
#%% SF7 file collection
fiola_all = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/*/*_times.npy")
def filter_helper(dims, path):
    return all(x not in path for x in dims)
fiola_256 = sorted(list(filter(lambda x: filter_helper(["1024", "512", "768","crop"], x), fiola_all)))
fiola_512 = sorted(list(filter(lambda x: filter_helper(["1024", "256", "768","crop"], x), fiola_all)))
fiola_1024 = sorted(list(filter(lambda x: filter_helper(["512", "256", "768","crop"], x), fiola_all)))
print(len(fiola_256), len(fiola_512), len(fiola_1024))

#%% SF7 plotting
fig, ax = plt.subplots()
fiolas = [fiola_256, fiola_512, fiola_1024]
for j in range(2,3):
    for i in range(3):    
        data = np.load(fiolas[i][j], allow_pickle=True)
        ax.scatter(range(1,len(data)+ 1), data)
        print(fiolas[i][j][-20:], np.mean(data)*1000)
plt.hlines([0.0025, 0.005, 0.01225], 0, 3000)
plt.legend(["256_100","256_200","256_500","512_100","512_200","512_500","1024_100","1024_200","1024_500",])