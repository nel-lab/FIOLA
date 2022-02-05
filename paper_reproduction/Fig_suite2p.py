#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 00:55:01 2021

@author: nel
"""

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
# %% set up folders
shift_fig = True
if shift_fig:
    k53_size = ["_256", "_512", "_1024"][1]
    filepath = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p' + k53_size
    movpath = glob.glob(filepath+"/*.tif")
    data = imread(movpath[0])
    n_time, Ly, Lx = data.shape
    file_folder = filepath
else:
    filepath = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts'
    movies = sorted(glob.glob(filepath + "/*/*.tif") +
                    glob.glob(filepath + "/*/*.hdf5"))
    idx = 2
    try:
        data = imread(movies[idx])
    except:
        f = h5py.File(movies[idx], 'r')
        data = f['mov'][()]
        f.close()
    short_names = ["1mp", "k37", "k53", "mso", "yst"]
    file_folder = filepath + "/s2p_" + short_names[idx]
    n_time, Ly, Lx = data.shape
    print(movies[idx], n_time, Lx, Ly)

# %% set suite2 pparameters
ops = suite2p.default_ops()
ops['batch_size'] = 100
ops['report_timing'] = True
# basedon the suite2p source  code, only nonrigid will generate subpixel shifts
ops['nonrigid'] = True
ops['block_size'] = [Lx, Ly]
ops['maxregshift'] = 0.1
ops["nimg_init"] = n_time//2
ops["subpixel"] = 1000
# ops['maxregshiftNR'] = 15
# ops['snr_thresh']= 1.0
# if idx == 3:
#     ops["h5py"] = file_folder
#     ops["h5py_key"] = "mov"
#     idx = -1
print(ops)

ops['data_path'] = file_folder
db = {"data_path": [file_folder]}
# NOTE: apparently they don't support GPU because  they  didn't see ani mprovement i n speed.
# They  do have GPU  + MATLAB, butnot for their python  code.

# %% Run suite2p with timing
output_ops = suite2p.run_s2p(ops, db)
# %% show i mage
plt.imshow(output_ops["refImg"])

# %% ANALYSIS  &  FIGURE GENERATION
# %% file p ath  boondogglery
files_cm = sorted(glob.glob(
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*cm_on_shifts.npy"))
files_fiola = sorted(glob.glob(
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*viola_full_shifts.npy"))

# %% suite2p output shifts (movementper frame fromregistration)
cm_shiftpath = files_cm[idx]
#np.save(cm_shiftpath[:-16] + "s2p.npy",output_ops, allow_pickle=True)
files_ops = sorted(glob.glob(
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*s2p.npy"))

fiola_shiftPath = files_fiola[idx]
fiola_full_shifts = np.load(fiola_shiftPath)
cm_full_shifts = np.load(cm_shiftpath)
output_ops = np.load(files_ops[idx], allow_pickle=True)[()]
# %%  plot for  quality control
x = 0
x2 = "xoff" if x else "yoff"
show_nr = 1
# procedure:x-shifts  in cm/fiola align  with y-shifts (yoff) in s2p and vice-versa.
start = 0  # n_time//2
fiola_mean = np.mean(fiola_full_shifts[start:, x, 0, 0])
cm_mean = np.mean(cm_full_shifts[start:, x])
fiola_shifts = fiola_full_shifts[start:, x, 0, 0]-fiola_mean
cm_shifts = cm_full_shifts[start:, x]-cm_mean
s2p_shifts = (show_nr*output_ops[x2 + "1"]
              [start:][:, 0] + output_ops[x2][start:])
plt.plot(fiola_shifts)
plt.plot(cm_shifts)
plt.plot(-s2p_shifts+np.mean(s2p_shifts))

# plt.plot(output_ops["yoff1"][1500:]*2+1)
# plt.plot(fiola_full_shifts[1500:,1])
# %%  error  calculation  using cm online  as the  reference  point
dataset_x = {"alg": [], "dset": [], "err": []}
dataset_y = {"alg": [], "dset": [], "err": []}
for i, f in enumerate(files_cm):
    base_file = f[:-16]
    names = ["1MP", "YST", "K37", "K53", "Meso"]
    cm = np.load(f)
    ff = np.squeeze(np.load(base_file + "viola_full_shifts.npy"))
    fc = np.squeeze(np.load(base_file + "viola_small_shifts.npy"))
    # if i==2:
    #     ff= np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/k37_20160109_AM_150um_65mW_zoom2p2_00001_00001_crop_viola_shifts_exp_f.npy")
    s2p_ops = np.load(files_ops[i],  allow_pickle=True)[()]
    leng = len(cm[:, 0])
    start = leng//2  # leng//2

    cmx, cmy = cm[start:, 0]-np.mean(cm[:, 0]), cm[start:, 1]-np.mean(cm[:, 1])
    # if i==2:
    #     ffx,ffy = ff[0, start:]-np.mean(ff[0,start:]), ff[1, start:]-np.mean(ff[1,start:])
    # #         ffx,ffy = ff[0,start:]
    # else:
    ffx, ffy = ff[start:, 0]-np.mean(ff[:, 0]), ff[start:, 1]-np.mean(ff[:, 1])
    fcx, fcy = fc[start:, 0]-np.mean(fc[:, 0]), fc[start:, 1]-np.mean(fc[:, 1])
    s2px, s2py = (s2p_ops["yoff1"][start:, 0] + s2p_ops["yoff"][start:]
                  ), (s2p_ops["xoff1"][start:, 0] + s2p_ops["xoff"][start:])

    dataset_x["err"] += list(np.subtract(ffx, cmx)) \
        + list(np.subtract(fcx, cmx)) \
        + list(np.subtract(-s2px, cmx))
    dataset_y["err"] += list(np.subtract(ffy, cmy)) \
        + list(np.subtract(fcy, cmy)) \
        + list(np.subtract(-s2py, cmy))
    dataset_x["alg"] += ["full_fiola"]*start + \
        ["crop_fiola"]*start + ["suite2p"]*start
    dataset_x["dset"] += [names[i]]*3*start
    dataset_y["alg"] += ["full_fiola"]*start + \
        ["crop_fiola"]*start + ["suite2p"]*start
    dataset_y["dset"] += [names[i]]*3*start
    # if i==2:
    #     break
# %% box plot generation
df = pd.DataFrame(dataset_x)
sns.boxplot(x=df["dset"],
            y=df["err"],
            hue=df["alg"])

# %% timing  calculations (k53 only)
fr = 1024
file_folder = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_" + \
    str(fr)
ops = suite2p.default_ops()
ops['batch_size'] = 100
ops['report_timing'] = True
# basedon the suite2p source  code, only nonrigid will generate subpixel shifts
ops['nonrigid'] = True
ops['block_size'] = [Lx, Ly]
ops['maxregshift'] = 0.1
ops["nimg_init"] = n_time//2
ops["subpixel"] = 1000
# ops['maxregshiftNR'] = 15
# ops['snr_thresh']= 1.0
ops['data_path'] = file_folder
db = {"data_path": [file_folder]}
# %%  RUN s2P => do NOT  RUN > 1x Without clearing the folder
output_ops = suite2p.run_s2p(ops, db)
# %% math fortimes
t0 = output_ops["fiola_start"]
t1 = output_ops["fiola_startRegInit"]
t2 = output_ops["fiola_finishTemplate"]
batchTimes = output_ops["fiola_batchStart"]
t3 = batchTimes[0]
# %%
# %%  save
save = False
if save:
    np.save("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_1024_ff.npy",
            output_ops, allow_pickle=True)
# %% generate timings dataset
dataset = {"alg": [], "typ": [], "time": []}
dataset_fr = {"alg": [], "time": []}
timed = list()

for i, f in enumerate(["fc", "fi", "cm", "s2"]):
    base_file = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_"
    dat512 = np.load(base_file + "512_" + f+".npy", allow_pickle=True)[()]
    dat1024 = np.load(base_file + "1024_" + f+".npy", allow_pickle=True)[()]
    if f[0] == "f" or f == "cm":
        init512 = dat512["init"]
        init1024 = dat1024["init"]
        # if f=="cm":
        #     init512 = t_init_512
        #     init1024 = t_init_1024
        tot512 = np.sum(dat512["mc"])
        tot1024 = np.sum(dat1024["mc"])
        dataset["alg"] += [f+"512", f+"512", f+"1024", f+"1024"]
        dataset["typ"] += ["init", "frames", "init", "frames"]
        dataset["time"] += [init512, tot512, init1024, tot1024]

        dataset_fr["time"] += list(dat512["mc"][1:])
        dataset_fr["alg"] += [f+"512"]*(-1+len(dat512["mc"]))
        dataset_fr["time"] += list(dat1024["mc"][1:])
        dataset_fr["alg"] += [f+"1024"]*(-1+len(dat1024["mc"]))
    elif f == "s2":
        init512 = dat512["fiola_finishTemplate"] - dat512["fiola_startRegInit"]
        init1024 = dat1024["fiola_finishTemplate"] - \
            dat1024["fiola_startRegInit"]
        tot512 = np.sum(dat512["fiola_batchStart"][1:])/2
        tot1024 = np.sum(dat1024["fiola_batchStart"][1:])/2
        dataset["alg"] += [f+"512", f+"512", f+"1024", f+"1024"]
        dataset["typ"] += ["init", "frames", "init", "frames"]
        dataset["time"] += [init512, tot512, init1024, tot1024]

        dataset_fr["time"] += list(np.divide(dat512["fiola_batchStart"][1:], 100))
        dataset_fr["alg"] += [f+"512"]*30
        dataset_fr["time"] += list(np.divide(dat1024["fiola_batchStart"][1:], 100))
        dataset_fr["alg"] += [f+"1024"]*30

# %% seaborn plot
df = pd.DataFrame(dataset)
df.groupby(["typ", "alg"])["time"].sum().unstack("typ")[
    ["init", "frames"]].plot(kind="bar", stacked=True)
# %% seaborn plotfor frame-wise timing NOTINCLUDING INITIALIZATION
df2 = df.sort_values(["typ"]).groupby(["typ", "alg"])[
    "time"].sum().unstack("typ")[["init", "frames"]]/1500
df2.plot(kind="bar", stacked=True)
# %% more
df3 = pd.DataFrame(dataset_fr)
p = sns.boxplot(x=df3["alg"],
                y=df3["time"]*1000,
                order=["fi512", "fc512", "cm512", "s2512",
                       "fi1024", "fc1024", "cm1024", "s21024"],
                whis=[0.1, 99.9])
p.set_yscale("log")
# %%
stats = {}
data_fr_custom = {}
nmes = ["512_fi", "512_fc", "512_cm", "512_s2",
        "1024_fi", "1024_fc", "1024_cm", "1024_s2"]
for n in nmes:
    if "s" in n:
        print(n)
        data_fr_custom[n] = 10*np.array(np.load(base_file + n +
                                        ".npy", allow_pickle=True)[()]["fiola_batchStart"][1:])
    else:
        data_fr_custom[n] = 1000 * \
            np.array(np.load(base_file + n + ".npy",
                     allow_pickle=True)[()]["mc"][1:])
count = 0
for key in data_fr_custom.keys():
    print(key)
    stats[key] = cbook.boxplot_stats(data_fr_custom[key], labels=str(count))[0]
    stats[key]["q1"], stats[key]["q3"] = np.percentile(
        data_fr_custom[key], [5, 95])
    stats[key]["whislo"], stats[key]["whishi"] = np.percentile(
        data_fr_custom[key], [0.1, 99.9])

    # stats[key]["whishi"] = stats[key]["q3"] + 1.5*(stats[key]["q3"]-stats[key]["q1"])
    # stats[key]["whislo"] = stats[key]["q1"] - 1.5*(stats[key]["q3"]-stats[key]["q1"])
    outliers = []
    for val in stats[key]["fliers"]:
        if val >= stats[key]['whishi'] or val <= stats[key]["whislo"]:
            outliers.append(val)
    stats[key]["fliers"] = outliers
    count += 1

colors = ["green", "green", "coral", "blue", "green", "green", "coral", "blue"]
fig, ax = plt.subplots(1, 1)
bplot = ax.bxp(stats.values(),  positions=range(8),  patch_artist=True)
ax.set_yscale("log")
for patch, color in zip(bplot["boxes"], colors):
    print(color)
    patch.set_facecolor(color)

#%% NNLS Timing
stats = {}
data_fr_custom = {}
nmes = ["fi_512", "fb_512","cm_512", "s2p_512",
        "fi_1024", "fb_1024","cm_1024", "s2p_1024"]
for n in nmes:
    if "fb" in n or "s2" in n:
        data_fr_custom[n] = 10 * np.array(np.load(base_file + n +
                                        "_nnls_time.npy"))
    elif "cm" in n:
        print(n)
        data_fr_custom[n] = np.array(np.load(base_file + n + "_nnls_time.npy", allow_pickle= True)[()]["T_track"])
    else:
        data_fr_custom[n] = 1000 * \
            np.array(np.load(base_file + n + "_nnls_time.npy"))
count = 0
for key in data_fr_custom.keys():
    print(key)
    stats[key] = cbook.boxplot_stats(data_fr_custom[key], labels=str(count))[0]
    stats[key]["q1"], stats[key]["q3"] = np.percentile(
        data_fr_custom[key], [5, 95])
    stats[key]["whislo"], stats[key]["whishi"] = np.percentile(
        data_fr_custom[key], [0.1, 99.9])

    # stats[key]["whishi"] = stats[key]["q3"] + 1.5*(stats[key]["q3"]-stats[key]["q1"])
    # stats[key]["whislo"] = stats[key]["q1"] - 1.5*(stats[key]["q3"]-stats[key]["q1"])
    outliers = []
    for val in stats[key]["fliers"]:
        if val >= stats[key]['whishi'] or val <= stats[key]["whislo"]:
            outliers.append(val)
    stats[key]["fliers"] = outliers
    count += 1

colors = ["green", "green", "coral", "blue", "green", "green", "coral", "blue"]
fig, ax = plt.subplots(1, 1)
bplot = ax.bxp(stats.values(),  positions=range(8),  patch_artist=True)
ax.set_yscale("log")
for patch, color in zip(bplot["boxes"], colors):
    print(color)
    patch.set_facecolor(color)

#%%
# dims = (512, 512)
# A = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/k53_A.npy",
#             allow_pickle=True)[()]
# b = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/k53_b.npy",
#             allow_pickle=True)[()]
# Ab = np.concatenate((A.toarray()[:, :], b), axis=1)
# batch_size = 1
# n_split = 1
# nnls_model = get_nnls_model(dims, Ab, batch_size, 30,
#                             n_split, 1)
