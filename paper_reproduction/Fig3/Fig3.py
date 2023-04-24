#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 00:01:32 2023

@author: nel
"""
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import seaborn
import  pandas as pd

#%% Fig 3b
#%% Fig 3c correlations between FIOLA and CaImAn wrt SNR
base_folder = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/"
datasets = ["N00",  "N01", "N02", "N03", "N04", "YST", "k53"]
metrics = {}
for d in datasets[:-1]:
    print(d)
    snrs  = -1* np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/"+d+"_SNR.npy")
    print(np.mean(snrs))
    included_comps = np.load(base_folder  + d +"_incl.npy")
    snrs = snrs[included_comps+2]  # 2 to account  for background
    rs = np.load(base_folder +  d + "_rscore.npy")
    metrics[d]  = [snrs, rs]

#%% Map rscores to snrs
rss  = []
names = []
fitnesses  = []
for i in range(6):
    rss += list(metrics[datasets[i]][1])
    rl = len(metrics[datasets[i]][1])
    names += [datasets[i]]*rl
    fitnesses += list(metrics[datasets[i]][0])
    
fitnesses = np.array(fitnesses)   
df = pd.DataFrame({"names":names, "R":rss, "fitness":fitnesses})
quants = [np.quantile(fitnesses, p) for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
fit = pd.cut(df["fitness"], bins=quants, labels=["10", '20', '30', '40', '50', '60', '70', '80', '90', '100'])
df["classes"]  = fit
#%%  Plot Fig 3c
seaborn.stripplot(x="names", y=-np.log10(1-df["R"]), hue="classes", data=df, jitter=True, alpha=1,  s=3, palette=seaborn.color_palette("rocket", 10))
ylab = -np.log10(1-np.array([0.5, 0.9,  0.95, 0.99, 0.999]))
ylabe = [0.5, 0.9,  0.95, 0.99, 0.999]
plt.yticks(ylab, ylabe)
#%%  Setup for Fig 3d
base_file = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_"
stats = {}
data_fr_custom = {}
nmes=["fi_512_100", "fi_512_200", "fi_512_500", "cm_512", "fi_1024_100", "fi_1024_200", "fi_1024_500", "cm_1024"]#, "s2p_1024"]
for n in nmes:
    if "fb" in n or "s2" in n:
        data_fr_custom[n] = 10 * np.array(np.load(base_file + n +
                                        "_nnls_time.npy"))
    elif "cm" in n:
        data_fr_custom[n] = np.array(np.load(base_file + n + "_nnls_time.npy", allow_pickle= True)[()]["T_track"])[3000:-1]
    else:
        data_fr_custom[n] = 1000 * \
            np.array(np.load(base_file + n + "_nnls_time.npy"))
    print(n, data_fr_custom[n].shape)
count = 0
for key in data_fr_custom.keys():
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
#%% Plot Fig 3d
colors = ["green", "green", "green", "coral","green", "green", "green", "coral"]
fig, ax = plt.subplots(1, 1)
bplot = ax.bxp(stats.values(),  positions=range(8),  patch_artist=True)
ax.set_yscale("log")
ax.set_xticklabels(nmes)

for patch, color in zip(bplot["boxes"], colors):
    print(color)
    patch.set_facecolor(color)
    
#%% NNLS- pearson's r x number iterations For Fig 3e
base_file  = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/"
files = sorted(glob.glob("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/*/nnls*.npy"))
    
#%% Data setup for Fig 3e 
save_r  = {}
from scipy.stats import pearsonr
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
        x.append(ti_mean)
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
            r.append(r_indiv)
            count += 1
        y.append(np.nanmedian(r))
        ylow.append(y[-1]-np.nanquantile(r,0.25))
        yhigh.append(np.nanquantile(r,0.75)-y[-1])
    ax.errorbar(x, y, xerr=np.vstack([xlow,xhigh]),yerr=np.vstack([ylow,yhigh]), marker="o", label=f)
    plt.yscale('log')
    plt.legend()
    save_r[f+t]  = r
    print(nn_vi.shape, f)

exp = lambda x: 10**(x)
log = lambda x: np.log10(x) 
plt.axhline(y=0.999)
ax.set_yticks([0.5,0.9, 0.95, 0.999])  
   