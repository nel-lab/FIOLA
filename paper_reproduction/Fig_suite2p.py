#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 00:55:01 2021

@author: nel
"""

import suite2p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from skimage import io
import glob
from tifffile import imread

#%% set up folders
k53_size = ["_256", "_512", "_1024"][2]
filepath = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p' + k53_size
movpath = glob.glob('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*.tif')
data = imread(movpath[1])
n_time, Ly, Lx  = data.shape

#%% set suite2 pparameters
ops = suite2p.default_ops()
ops['batch_size'] = 1
ops['report_timing']  = True
# ops['nonrigid'] = True
# ops['block_size'] = [1024,1024]
ops['maxregshift']  = 0.1
ops["nimg_init"] = n_time//2 
ops["subpixel"] = 1000
# ops['maxregshiftNR'] = 15
# ops['snr_thresh']= 1.0
print(ops)

ops['data_path'] = filepath
db = {"data_path": [filepath]}
## NOTE: apparently they don't support GPU because  they  didn't see ani mprovement i n speed.
## They  do have GPU  + MATLAB, butnot for their python  code.

#%% Run suite2p motion correction with timing
output_ops = suite2p.run_s2p(ops, db)
#%% show i mage
plt.imshow(output_ops["refImg"])
#%% suite2p output shifts (movementper frame fromregistration)
shiftPath= "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/fig1/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00001_cm_on_shifts.npy"
fiola_full_shifts = np.load(shiftPath)
#%%  plot
   plt.plot(fiola_full_shifts[1500:,0])
plt.plot(output_ops["xoff1"][1500:])
# plt.plot(output_ops["yoff1"][1500:]*2+1)
# plt.plot(fiola_full_shifts[1500:,1])

#%% metric testing
ops = suite2p.registration.metrics(output_ops, use_red=False)
#%%
from suite2p.registration import register
refImg = register.pick_initial_reference(ops)
