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
#%% THE FOLLOWING SECTION IS FOR RUNNING S2P
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
# file_folder = "/media/nel/storage/fiola/R2_20190219/test"
# Lx, Ly = (796, 512)
ops = suite2p.default_ops()
ops['batch_size'] = 1
ops['report_timing'] = True
# basedon the suite2p source  code, only nonrigid will generate subpixel shifts
ops['nonrigid'] = False
ops['block_size'] = [Lx, Ly]
ops['maxregshift'] = 0.1
ops["nimg_init"] = 1500
ops["subpixel"] = 1000
# ops['maxregshiftNR'] = 15
# ops['snr_thresh']= 1.0
# if idx == 3:
# ops["h5py"] = file_folder
# ops["h5py_key"] = "mov"
#     idx = -1
print(ops)

ops['data_path'] = file_folder
ops['save_path'] = file_folder
db = {"data_path": [file_folder]}
# NOTE: apparently they don't support GPU because  they  didn't see ani mprovement i n speed.
# They  do have GPU  + MATLAB, butnot for their python  code.

# %% Run suite2p with timing
output_ops = suite2p.run_s2p(ops, db)
# %% show i mage
np.save(file_folder+ "/full_time_rigid.npy", output_ops, allow_pickle=True)
plt.imshow(output_ops["refImg"])

# %% show dandi spatial footprints
tr = np.load('/media/nel/storage/fiola/R2_20190219/suite2p/plane0/F.npy', allow_pickle=True)
stat = np.load('/media/nel/storage/fiola/R2_20190219/suite2p/plane0/stat.npy', allow_pickle=True)
ops = np.load('/media/nel/storage/fiola/R2_20190219/suite2p/plane0/ops.npy', allow_pickle=True)
import caiman as cm
mm = cm.load('/media/nel/storage/fiola/R2_20190219/test/mov_R2_20190219T210000_init_1000.hdf5')
plt.imshow(mm.mean(0))
mask = np.zeros((4977, 796, 512))

for i in range(len(mask)):
    mask[i, stat[i]['ypix'], stat[i]['xpix']] = 1

# %% ANALYSIS  &  FIGURE GENERATION
#%% FIGURE 2b
# %% file p ath  boondogglery
files_cm = sorted(glob.glob(

    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*cm_on_shifts.npy"))
files_fiola = sorted(glob.glob(
    "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/*viola_full_shifts.npy"))

# %% suite2p output shifts (movementper frame fromregistration)
idx=-3
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
start = 1500  # n_time//2
fiola_mean = np.mean(fiola_full_shifts[start:, x, 0, 0])
cm_mean = np.mean(cm_full_shifts[start:, x])
fiola_shifts = fiola_full_shifts[start:, x, 0, 0]-fiola_mean
cm_shifts = cm_full_shifts[start:, x]-cm_mean
s2p_shifts = (show_nr*output_ops[x2 + "1"]
              [start:][:, 0] + output_ops[x2][start:])
s2p_rigid_shifts_x = output_ops["xoff"]
s2p_rigid_shifts_y = output_ops["yoff"]
plt.plot(fiola_shifts)
plt.plot(cm_shifts)
plt.plot(-s2p_shifts+np.mean(s2p_shifts))
# plt.plot((-s2p_rigid_shifts_x+np.mean(s2p_rigid_shifts_x))[1500:])
plt.plot((-s2p_rigid_shifts_y+np.mean(s2p_rigid_shifts_y))[1500:])

# plt.plot(output_ops["yoff1"][1500:]*2+1)
# plt.plot(fiola_full_shifts[1500:,1])
# %%  error  calculation  using cm online  as the  reference  point for 2b
files_voltage = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/*/*cm_on_shifts.npy"))
files = files_cm + files_voltage
dataset_x = {"alg": [], "dset": [], "err": []}
dataset_y = {"alg": [], "dset": [], "err": []}
for i, f in enumerate(files):
    base_file = f[:-16]
    names = ["1MP", "YST", "K37", "K53", "Meso", "FOV1_35", "FOV_3_35", "403106"]
    cm = np.load(f)
    ff = np.squeeze(np.load(base_file + "viola_full_shifts.npy"))
    fc = np.squeeze(np.load(base_file + "viola_small_shifts.npy"))
    # if i==2:
    #     ff= np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/k37_20160109_AM_150um_65mW_zoom2p2_00001_00001_crop_viola_shifts_exp_f.npy")

    leng = len(cm[:, 0])
    start = leng//2  # leng//2

    cmx, cmy = cm[start:, 0]-np.mean(cm[:, 0]), cm[start:, 1]-np.mean(cm[:, 1])
    # if i==2:
    #     ffx,ffy = ff[0, start:]-np.mean(ff[0,start:]), ff[1, start:]-np.mean(ff[1,start:])
    # #         ffx,ffy = ff[0,start:]
    # else:
    ffx, ffy = ff[start:, 0]-np.mean(ff[:, 0]), ff[start:, 1]-np.mean(ff[:, 1])
    fcx, fcy = fc[start:, 0]-np.mean(fc[:, 0]), fc[start:, 1]-np.mean(fc[:, 1])
    try:
        s2p_ops = np.load(files_ops[i],  allow_pickle=True)[()]
        s2px, s2py = (s2p_ops["yoff1"][start:, 0] + s2p_ops["yoff"][start:]
                      ), (s2p_ops["xoff1"][start:, 0] + s2p_ops["xoff"][start:])
    except:
        s2px, s2py = -cmx, -cmy
    
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
    if i==2:
        break
# %% box plot generation for 2b (motion correction)
df = pd.DataFrame(dataset_y)
ax = sns.boxplot(x=df["dset"],
            y=df["err"],
            hue=df["alg"],
            whis=[1,99])
ax.set_ylim([-2,2])

# %% timing  calculations (k53 only) for 2c
fr = 512
file_folder = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_" + \
    str(fr)
ops = suite2p.default_ops()
ops['batch_size'] = 100
ops['report_timing'] = True
# basedon the suite2p source  code, only nonrigid will generate subpixel shifts
ops['nonrigid'] = False
ops['block_size'] = [Lx, Ly]
ops['maxregshift'] = 0.1
ops["nimg_init"] = n_time//2
ops["subpixel"] = 1000
# ops['maxregshiftNR'] = 15
# ops['snr_thresh']= 1.0
ops['data_path'] = file_folder
db = {"data_path": [file_folder]}
# %%  RUN s2P => do NOT  RUN more than 1x Without clearing the folder
output_ops = suite2p.run_s2p(ops, db)
# %% math for times for 2c
t0 = output_ops["fiola_start"]
t1 = output_ops["fiola_startRegInit"]
t2 = output_ops["fiola_finishTemplate"]
batchTimes = output_ops["fiola_batchStart"][1:]
t3 = batchTimes[0]
print(np.quantile(batchTimes, 0.05),np.quantile(batchTimes, 0.5),np.quantile(batchTimes, 0.95))
# %%  save results
save = False
if save:
    np.save("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_512_s2r2.npy",
            output_ops, allow_pickle=True)
# %% generate timings dataset for 2c
dataset = {"alg": [], "typ": [], "time": []}
dataset_fr = {"alg": [], "time": []}
timed = list()

for i, f in enumerate(["fc", "fi", "cm", "s2", "s2r"]):
    # if i == 1: 
        # break
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
# df = pd.DataFrame(dataset)
# df.groupby(["typ", "alg"])["time"].sum().unstack("typ")[
#     ["init", "frames"]].plot(kind="bar", stacked=True)
#  seaborn plotfor frame-wise timing NOTINCLUDING INITIALIZATION
# df2 = df.sort_values(["typ"]).groupby(["typ", "alg"])[
#     "time"].sum().unstack("typ")[["init", "frames"]]/1500
# df2.plot(kind="bar", stacked=True)
# # %% more
# df3 = pd.DataFrame(dataset_fr)
# p = sns.boxplot(x=df3["alg"],
#                 y=df3["time"]*1000,
#                 order=["fi512", "fc512", "cm512", "s2512",
#                        "fi1024", "fc1024", "cm1024", "s21024"],
#                 whis=[1, 99])
# p.set_yscale("log")

stats = {}
data_fr_custom = {}
nmes = ["512_fi", "512_fc", "512_cm", "512_s2", "512_s2r",
        "1024_fi", "1024_fc", "1024_cm", "1024_s2", "1024_s2r"]
for n in nmes:
    if "s" in n:
        temp=""
        if "r" in n:
            multiplier = 1000
        else:
            multiplier = 10
        if "512" in n:
            temp="test_19-09-22/"
            data_fr_custom[n] = 10*np.array(np.load(base_file + temp + n +
                                            ".npy", allow_pickle=True))
                                    
            print(n)
        else:
            data_fr_custom[n] = multiplier*np.array(np.load(base_file + temp + n +
                                        ".npy", allow_pickle=True)[()]["fiola_batchStart"][1:])
    else:
        if "cm" in n:
            multiplier = 1
        else:
            multiplier = 1000
        data_fr_custom[n] = multiplier * \
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

colors = ["green", "green", "coral", "blue", "blue", "green", "green", "coral", "blue", "blue"]
fig, ax = plt.subplots(1, 1)
bplot = ax.bxp(stats.values(),  positions=range(10),  patch_artist=True)
ax.set_yscale("log")
ax.set_xticklabels(nmes)
for patch, color in zip(bplot["boxes"], colors):
    print(color)
    patch.set_facecolor(color)
#%% Print out the k53 time
print("512_fi times: ", np.percentile(data_fr_custom["512_fi"], [0.1, 1, 25,75,99, 99.9]))
print("512_fc times: ", np.percentile(data_fr_custom["512_fc"], [0.1, 1, 25,75,99, 99.9]))
print("1024_fi times: ", np.percentile(data_fr_custom["1024_fi"], [0.1, 1, 25,75,99, 99.9]))
print("1024_fc times: ", np.percentile(data_fr_custom["1024_fc"], [0.1, 1, 25,75,99, 99.9]))
#%% THE FOLLOWING SECTIONIS CODE FOR FIGURE 3
#%% NNLS Timing for 3e
base_file = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_"
stats = {}
data_fr_custom = {}
# nmes = ["fi_512", "fb_512","cm_512", #"s2p_512",
nmes=["fi_512_100", "fi_512_200", "fi_512_500", "cm_512", "fi_1024_100", "fi_1024_200", "fi_1024_500", "cm_1024"]#, "s2p_1024"]
for n in nmes:
    if "fb" in n or "s2" in n:
        data_fr_custom[n] = 10 * np.array(np.load(base_file + n +
                                        "_nnls_time.npy"))
    elif "cm" in n:
        # print(n)
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

colors = ["green", "green", "green", "coral","green", "green", "green", "coral"]
fig, ax = plt.subplots(1, 1)
bplot = ax.bxp(stats.values(),  positions=range(8),  patch_artist=True)
ax.set_yscale("log")
ax.set_xticklabels(nmes)

for patch, color in zip(bplot["boxes"], colors):
    print(color)
    patch.set_facecolor(color)

#%% JUMP NNLS -- timing for N.... files (3c)
from fiola.gpu_mc_nnls import get_mc_model, get_nnls_model, get_model, Pipeline
import timeit
import time
import tensorflow as tf
import tensorflow.keras as keras
from multiprocessing import Queue
import glob
import caiman as cm
import numpy as np
#%% file load for NNLS timing(Calcium only)
# get tif +  A,b,C files for N... and  YST  (Calcium Comparison)
# movie_name = ["N00", "N01", "N02", "N03", "N04","YST"][5]
# base_file  = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/"
# A_files = sorted(glob.glob(base_file + "*half_A.npy"))
# b_files = sorted(glob.glob(base_file + "*half_b.npy"))
# C_files = sorted(glob.glob(base_file+ "*half_noisyC.npy"))
# movie_files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/N.00.00/images*/*.tif"))
movie_name = "k53"
base_file  = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/"
A_files = sorted(glob.glob(base_file + "*_A.npy"))
b_files = sorted(glob.glob(base_file + "*_b.npy"))
C_files = sorted(glob.glob(base_file+ "*_noisyC.npy"))
movie_files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/N.00.00/images*/*.tif"))
#%% load file
idx = 0
A = np.load(A_files[idx],allow_pickle=True)[()].toarray()#[:,-100:]
b = np.load(b_files[idx],allow_pickle=True)[()]
C = np.load(C_files[idx],allow_pickle=True)[()][-102:]
x0 = C[:,0]
dims = 512
neurs = A.shape[1]+b.shape[1]
Ab = np.concatenate((A,b), axis=1)
mov = cm.load(movie_files)
mov = mov[mov.shape[0]//2:]
fnames = movie_files

#%% generate inputs
from fiola.utilities import HALS4activity
batch_size=1
b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')       
C_init = np.dot(Ab.T, b)
x0 = np.array([HALS4activity(Yr=b[:,i], A=Ab, C=C_init[:, i].copy(), iters=10) for i in range(batch_size)]).T
x0 = x0[:, 0].astype(np.float32)
neurs = 100
dims = 512

#%% create generator
out = []
q = Queue()
q.put((np.concatenate((x0[None],x0[None]), axis=0)[:,:,None]))
times2= []
def generator():
    # print('hi')
    k=[[0.0]]
    for fr in mov:  # CHANGJIA: if you  change this to  mov, then run the cell  at line 451,  you'll see a 2 ms speedup. no idea why
        # print(fr.shape)
        z = q.get()
        # print(z.shape, "debug1")
        # print("unstuck")
        # times2.append(timeit.default_timer())
        yield{"m":fr[None,None,:,:,None], "y":z[1][None], "x":z[0][None], "k":k}
             
def get_frs():
    dataset = tf.data.Dataset.from_generator(generator, output_types={'m':tf.float32, 
                                                                      'y':tf.float32, 
                                                                      'x':tf.float32, 
                                                                      'k':tf.float32}, 
                                             output_shapes={"m":(1,1,512,512,1), 
                                                            "y":(1, neurs, 1),
                                                            "x":(1, neurs, 1), 
                                                            "k":(1, 1)})
    return dataset

#%% set up nnls model
iters = 30
model = get_nnls_model((dims,dims), Ab.astype(np.float32), 1, iters,1,False)
model.compile(optimizer='rmsprop', loss='mse')
estimator  = tf.keras.estimator.model_to_estimator(model)
#%% run model
time_all = []
for iteration in range(5):
    times_nnls_fast = [0]*3000
    traces_nnls = [0]*3000
    count=0
    start = timeit.default_timer()
    for i in estimator.predict(input_fn=get_frs,yield_single_examples=False):
        q.put(np.concatenate((i['nnls'],i['nnls_1'])))
        traces_nnls[count] = i
        times_nnls_fast[count] = timeit.default_timer() - start
        count += 1
    plt.plot(np.diff(times_nnls_fast[1:]))
    time_all.append(times_nnls_fast)
    
ttt = [np.median(np.diff(tt[1:])) for tt in time_all]

#%% run model
times0,times1,traces = [],[],[]
start2 = timeit.default_timer()


tu2 = []
for jj in range(10):

    t_s = timeit.default_timer()
    for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
        # print(i, "PLES")
        # print(i.keys(), "LOOK HERE")
        start1 = timeit.default_timer()
        times0.append(start1-start2)
        q.put(np.concatenate((i['nnls'],i['nnls_1'])))
        traces.append(i)
        start2 = timeit.default_timer()
        times1.append(start2-start1)
    plt.plot((times1[1:-1]))
    t_e = timeit.default_timer()
    print((t_e - t_s)/500)
    
    tu2.append((t_e - t_s)/500)


#%%
into = [mov[0, :, :, None][None, :], x0, x0, [[0.0]]]
start0 = timeit.default_timer()


for i in range(1, 500):
    start = timeit.default_timer()
    out = model(into)[0]
    times[i] = timeit.default_timer()-start
    into = [mov[i+1, :, :, None][None, :], out[0], out[1], out[2]]
    # time.sleep(0.01)

#%% NNLS- pearson's r x number iterations (3e)
base_file  = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/"
from sklearn.metrics import r2_score
# files = glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/*/nnls*.npy")
files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/*/nnls*.npy"))
rscore5, rscore10, rscore30 = [],[],[]
r5err, r10err, r30err = [],[],[]
for file in files:
    nnls_traces = np.load(file);
    v5_traces = np.load(file[:-8] + "v_nnls_5.npy")
    v10_traces = np.load(file[:-8]+ "v_nnls_10.npy")
    v30_traces = np.load(file[:-8] + "v_nnls_30.npy")
    if "02" in  file:
        break
    #rscore5.append(np.corrcoef(nnls_traces, v5_traces))
    
#%% graphing 
save_r  = {}
from scipy.stats import sem, pearsonr
files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/*/nnls*.npy"))
offsets = [2,1,2,3,3,3] # background components
bad = 0
fig, ax = plt.subplots()
for i,f in enumerate(["N00", "N01","N02","N03","N04","YST"]):
    x,y = [],[]
    xerr,yerr = [],[]
    xlow,xhigh,ylow,yhigh = [],[],[],[]
    
    for t in ["5", "10", "30"]:
        ti = np.load(base_file + f+ "_nnls_" + t + "_time.npy")
        ti_mean = np.nanmedian(ti)
        # ti_err = np.nanstd(ti)
        x.append(ti_mean)
        # xerr.append(ti_err)
        xlow.append(x[-1]-np.quantile(ti,0.25))
        xhigh.append(np.quantile(ti,0.75)-x[-1])
        offset = offsets[i]
        nn_sp = np.load(files[i])
        nn_vi = np.load(files[i][:-8]+ "v_nnls_" + t  +  ".npy")[:-offset,-nn_sp.shape[1]:]
        nn_sp = nn_sp[offset:offset+len(nn_vi)]
        r = []
        count = 0
        for (s,v) in zip(nn_sp, nn_vi):
            r_indiv = pearsonr(s,v)[0]
            if r_indiv <= 0:
                r_indiv = 0
                bad += 1
                # plt.figure()
                # plt.plot(s)
                # plt.plot(v)
                # plt.show()
            r.append(r_indiv)
                # yerr.append(sem(r))
            count += 1
        y.append(np.nanmedian(r))
        # yerr.append(np.nanstd(r))
        ylow.append(y[-1]-np.nanquantile(r,0.25))
        yhigh.append(np.nanquantile(r,0.75)-y[-1])
        print(np.nanquantile(r,0.25), np.nanquantile(r,0.75),ylow,yhigh)
    ax.errorbar(x, y, xerr=np.vstack([xlow,xhigh]),yerr=np.vstack([ylow,yhigh]), marker="o", label=f)
    plt.yscale('log')
    plt.legend()
        # break
    save_r[f+t]  = r
        # plt.plot(r2)
exp = lambda x: 10**(x)
log = lambda x: np.log10(x) 
# ax.set_yscale("function", functions=(exp, log)) 
plt.axhline(y=0.999)
ax.set_yticks([0.5,0.9, 0.95, 0.999])  
        
    
#%%
output_ops = np.load('/media/nel/storage/fiola/R2_20190219/full_time.npy', allow_pickle=True)[()]
fiola_keys = [a for a in output_ops.keys() if 'fiola' in a]

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
