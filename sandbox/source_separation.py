#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:54:44 2020

@author: agiovann
"""

import numpy as np
import pylab as pl
import glob
import scipy
#%%
base_fold = '/Users/agiovann/SOFTWARE/SANDBOX/figures/nnls/'
fnames = glob.glob(base_fold +'*.npy')
fnames.sort()
# print(fnames)
idx = 10
corr_coeffs = []
fnames = glob.glob(base_fold +'vol*.npy')
fnames.sort()
print(fnames)
gts = np.load(fnames[-1]).squeeze()
print(gts.shape)
for fname in fnames:
    traces = np.load(fname).squeeze().T
    print(traces.shape)

    cc = [scipy.stats.pearsonr(tr,np.maximum(gt,0))[0] for (tr, gt) in zip(traces, gts)]
    plt.plot(traces[np.argmax(cc)])
    print (fname+':'+str(np.nanmean(cc)))
    # plt.hist(cc, 100)
# for fname in fnames[:-1]:

    #%%
    
    
    traces.append(np.load(fname).squeeze().T)
    plt.plot(traces[-1][idx].T)
traces = np.array(traces)
legends = [f.split('/')[-1] for f in fnames]
plt.legend(legends)
