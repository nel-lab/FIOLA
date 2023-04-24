#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 23:17:45 2023

@author: nel
"""

import tensorflow as tf
from fiola.gpu_mc_nnls import get_nnls_model
import matplotlib.cbook as cbook
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import glob
import h5py

#%% set up files for Fig 2
files_cm = sorted(glob.glob(
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*cm_on_shifts.npy"))
files_fiola = sorted(glob.glob(
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*viola_full_shifts.npy"))
files_ops = sorted(glob.glob(
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*s2p.npy"))
files_voltage = sorted(glob.glob(
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/*/*cm_on_shifts.npy"))
files = files_cm + files_voltage

#%% Fig 2b error calculation using caiman as gt
dataset_x = {"alg": [], "dset": [], "err": []}
dataset_y = {"alg": [], "dset": [], "err": []}
for i, f in enumerate(files):
    base_file = f[:-16]
    names = ["1MP", "YST", "K37", "K53", "Meso", "403106", "FOV1_35", "FOV_3_35"]
    cm = np.load(f)
    ff = np.squeeze(np.load(base_file + "viola_full_shifts.npy"))
    fc = np.squeeze(np.load(base_file + "viola_small_shifts.npy"))

    leng = len(cm[:, 0])
    start = leng//2  # leng//2

    cmx, cmy = cm[start:, 0]-np.mean(cm[:, 0]), cm[start:, 1]-np.mean(cm[:, 1])
    ffx, ffy = ff[start:, 0]-np.mean(ff[:, 0]), ff[start:, 1]-np.mean(ff[:, 1])
    fcx, fcy = fc[start:, 0]-np.mean(fc[:, 0]), fc[start:, 1]-np.mean(fc[:, 1])
    try:
        s2p_ops = np.load(files_ops[i],  allow_pickle=True)[()]
        s2px, s2py = (s2p_ops["yoff1"][start:, 0] + s2p_ops["yoff"][start:]
                      ), (s2p_ops["xoff1"][start:, 0] + s2p_ops["xoff"][start:])
    except: # if rigid, no subpixel shits => using caiman values to make the error null
        s2px, s2py = -cmx, -cmy
    
    print(base_file.split("/")[-1], cm.shape, ff.shape, fc.shape, s2p_ops["yoff"].shape)
    errffx, errffy = (np.subtract(ffx, cmx)), (np.subtract(ffy, cmy))
    errfcx, errfcy = (np.subtract(fcx, cmx)), (np.subtract(fcy, cmy))
    errs2x, errs2y = (np.subtract(-s2px, cmx)), (np.subtract(-s2py, cmy))

    dataset_x["err"] += list(errffx - np.median(errffx)) \
        + list(errfcx - np.median(errfcx)) \
        + list(errs2x - np.median(errs2x))
    dataset_y["err"] += list(errffy - np.median(errffy)) \
        + list(errfcy - np.median(errfcy)) \
        + list(errs2y - np.median(errs2y))
    dataset_x["alg"] += ["full_fiola"]*start + \
        ["crop_fiola"]*start + ["suite2p"]*start
    dataset_x["dset"] += [names[i]]*3*start
    dataset_y["alg"] += ["full_fiola"]*start + \
        ["crop_fiola"]*start + ["suite2p"]*start
    dataset_y["dset"] += [names[i]]*3*start

# %% box plot generation for 2b (motion correction)
df = pd.DataFrame(dataset_x)
ax = sns.boxplot(x=df["dset"],
            y=df["err"],
            hue=df["alg"],
            whis=[1,99])
ax.set_ylim([-2,2])

#%% Generate timings for Fig 2c
dataset = {"alg": [], "typ": [], "time": []}
dataset_fr = {"alg": [], "time": []}
timed = list()
dfc = pd.DataFrame()
for i, f in enumerate(["fc", "fi", "cm", "s2", "s2r"]):
    base_file = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_"
    dat512 = np.load(base_file + "512_" + f+".npy", allow_pickle=True)[()]
    dat1024 = np.load(base_file + "1024_" + f+".npy", allow_pickle=True)[()]
    max_len = 2998
    if f[0] == "f" or f == "cm":
        init512 = dat512["init"]
        init1024 = dat1024["init"]
        print(f, dat512["mc"].shape, dat1024["mc"].shape)
        tot512 = np.sum(dat512["mc"][1:])
        tot1024 = np.sum(dat1024["mc"][1:])
        dataset["alg"] += [f+"512", f+"512", f+"1024", f+"1024"]
        dataset["typ"] += ["init", "frames", "init", "frames"]
        dataset["time"] += [init512, tot512, init1024, tot1024]

        dataset_fr["time"] += list(dat512["mc"][1:])
        dataset_fr["alg"] += [f+"512"]*(-1+len(dat512["mc"]))
        dataset_fr["time"] += list(dat1024["mc"][1:])
        dataset_fr["alg"] += [f+"1024"]*(-1+len(dat1024["mc"]))
        
        filler = [None]* (max_len-len(dat512["mc"][1:]))
        dfc[f+"_512"] = np.concatenate((list(dat512["mc"][1:]), filler))
        dfc[f+"_1024"] = np.concatenate((list(dat1024["mc"][1:]), filler))

    elif f == "s2":
        init512 = dat512["fiola_finishTemplate"] - dat512["fiola_startRegInit"]
        init1024 = dat1024["fiola_finishTemplate"] - \
            dat1024["fiola_startRegInit"]
        print(f, len(dat512["fiola_batchStart"][1:]), len(dat1024["fiola_batchStart"][1:]))
        tot512 = np.sum(dat512["fiola_batchStart"][1:])/2
        tot1024 = np.sum(dat1024["fiola_batchStart"][1:])/2
        dataset["alg"] += [f+"512", f+"512", f+"1024", f+"1024"]
        dataset["typ"] += ["init", "frames", "init", "frames"]
        dataset["time"] += [init512, tot512, init1024, tot1024]

        dataset_fr["time"] += list(np.divide(dat512["fiola_batchStart"][1:], 100))
        dataset_fr["alg"] += [f+"512"]*30
        dataset_fr["time"] += list(np.divide(dat1024["fiola_batchStart"][1:], 100))
        dataset_fr["alg"] += [f+"1024"]*30
        filler  = [None]*(max_len-len(dat512["fiola_batchStart"][1:]))
        dfc[f+"_512"] = np.concatenate((list(dat512["fiola_batchStart"][1:]),  filler))
        dfc[f+"_1024"] =  np.concatenate((list(dat1024["fiola_batchStart"][1:]),  filler))
#%% Create statistics dictionary
stats = {}
data_fr_custom = {}
base_file = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_'
nmes = ["512_cm", "512_fi", "512_fc", "512_s2", "512_s2r",
        "1024_cm", "1024_fi", "1024_fc", "1024_s2", "1024_s2r"]
for n in nmes:
    if "s" in n:
        temp="test_19-09-22/"
        data_fr_custom[n] = 1000*np.array(np.load(base_file + temp + n +
                                            ".npy", allow_pickle=True))[0:]
    else:
        if "cm" in n:
            multiplier = 1
            offset=1
        else:
            multiplier = 1000
            offset=1500
        data_fr_custom[n] = multiplier * \
            np.array(np.load(base_file + n + ".npy",
                     allow_pickle=True)[()]["mc"][1:])
    print(n, data_fr_custom[n].shape)
#%%
#%% set up statistics for Fig 2c
count = 0
for key in data_fr_custom.keys():
    print(data_fr_custom[key].shape)
    stats[key] = cbook.boxplot_stats(data_fr_custom[key], labels=str(count))[0]
    stats[key]["q1"], stats[key]["q3"] = np.percentile(
        data_fr_custom[key], [5, 95])
    stats[key]["whislo"], stats[key]["whishi"] = np.percentile(
        data_fr_custom[key], [0.1, 99.9])

    outliers = []
    for val in stats[key]["fliers"]:
        if val >= stats[key]['whishi'] or val <= stats[key]["whislo"]:
            outliers.append(val)
    stats[key]["fliers"] = outliers
    count += 1

#%% Plot Fig 2c for timings
colors = ["coral", "blue", "orange", "green", "green", "coral", "blue", "orange", "green", "green"]
fig, ax = plt.subplots(1, 1)
bplot = ax.bxp(stats.values(),  positions=range(10),  patch_artist=True)
ax.set_yscale("log")
ax.set_xticklabels(nmes)
ax.set_ylim([0, 10000])
for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)       

