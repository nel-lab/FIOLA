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
filepath = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p'
movpath = glob.glob('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/*.tif')
data = imread(movpath[0])[:1500]
n_time, Ly, Lx  = data.shape

#%% set suite2 pparameters
ops = suite2p.default_ops()
ops['batch_size'] = 1
ops['report_timing']  = True
ops['nonrigid'] = False
# ops['block_size'] = [548,496]
ops['maxregshift']  = 0.03
ops["nimg_init"]= 1500 
ops["subpixel"]=20
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
plt.plot(output_ops["yoff"])
plt.plot(output_ops["xoff"])
# fiola_full_shifts = np.load("/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/k37_20160109_AM_150um_65mW_zoom2p2_00001_00001_crop_viola_shifts.npy")
# plt.plot(fiola_full_shifts[0])
# plt.plot(fiola_full_shifts[1])
#%% metric testing
ops = suite2p.registration.metrics(ops, use_red=False)
#%%
from suite2p.registration import register
refImg = register.pick_initial_reference(ops)
