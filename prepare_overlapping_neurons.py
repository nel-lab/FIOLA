#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:50:24 2020
This file is used to create overlapping neurons and corresponding masks
@author: @caichangjia
"""
import matplotlib.pyplot as plt
import numpy as np
import os

import caiman as cm
from caiman.base.rois import nf_read_roi_zip
from nmf_support import combine_datasets

#%% files for processing
base_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
               '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron',
               '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/'][1]
lists = ['454597_Cell_0_40x_patch1_mc.tif', '456462_Cell_3_40x_1xtube_10A2_mc.tif',
             '456462_Cell_3_40x_1xtube_10A3_mc.tif', '456462_Cell_5_40x_1xtube_10A5_mc.tif',
             '456462_Cell_5_40x_1xtube_10A6_mc.tif', '456462_Cell_5_40x_1xtube_10A7_mc.tif', 
             '462149_Cell_1_40x_1xtube_10A1_mc.tif', '462149_Cell_1_40x_1xtube_10A2_mc.tif',
             '456462_Cell_4_40x_1xtube_10A4_mc.tif', '456462_Cell_6_40x_1xtube_10A10_mc.tif',
             '456462_Cell_5_40x_1xtube_10A8_mc.tif', '456462_Cell_5_40x_1xtube_10A9_mc.tif', 
             '462149_Cell_3_40x_1xtube_10A3_mc.tif', '466769_Cell_2_40x_1xtube_10A_6_mc.tif',
             '466769_Cell_2_40x_1xtube_10A_4_mc.tif', '466769_Cell_3_40x_1xtube_10A_8_mc.tif']
fnames = [os.path.join(base_folder, file) for file in lists]
freq_400 = [True, True, True, True, True, True, False, True, True, True, True, True, False, False, False, False]

#%% Combine datasets
x_shifts = [6, -2]
y_shifts = [6, -3]
file_set = [0, 2]
name_set = [fnames[file_set[0]], fnames[file_set[1]]]
m1 = cm.load(name_set[0])
m2 = cm.load(name_set[1])
movies = [cm.load(name) for name in name_set]
dims = [mov.shape for mov in movies]
masks = [nf_read_roi_zip((name_set[i][:-7] + '_ROI.zip'), 
                         dims=dims[i][1:]) for i in range(len(name_set))]
num_frames = np.max((dims[0][0], dims[1][0]))
frate = 400

plt.figure();plt.imshow(m1[0]);plt.colorbar()
plt.figure();plt.imshow(m2[0]);plt.colorbar()

mov, mask = combine_datasets(movies, masks, num_frames, x_shifts=x_shifts, 
                             y_shifts=y_shifts, weights=None, shape=(30, 30))

plt.figure();plt.imshow(mov[0]);plt.show()
plt.figure();plt.imshow(mask[0], alpha=0.5);plt.imshow(mask[1], alpha=0.5);plt.show()

print((mask[0]*mask[1]).sum()/(mask[0].sum()+mask[1].sum())*2)
#%%
saving_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/overlapping_neurons'
cm.movie(mov).save(os.path.join(saving_folder, f'neuron{file_set[0]}&{file_set[1]}_x{x_shifts}_y{y_shifts}_0percent.tif'))
cm.movie(np.array(mask)).save(os.path.join(saving_folder, f'neuron{file_set[0]}&{file_set[1]}_x{x_shifts}_y{y_shifts}_0percent_ROIs.hdf5'))




