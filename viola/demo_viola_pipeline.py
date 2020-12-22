#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:48:17 2020
Pipeline for online analysis of voltage imaging data
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
@author: @agiovann, @caichangjia, @cynthia
"""
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.io import imread
from time import time, sleep
from threading import Thread

from nmf_support import normalize
from violaparams import violaparams
from viola import VIOLA

#%% load movie and masks
movie_folder = ['/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons'][1]
    
movie_lists = ['demo_voltage_imaging_mc.hdf5', 
               'FOV4_50um_mc.hdf5', 
               '06152017Fish1-2_portion.hdf5']
name = movie_lists[0]
if '.hdf5' in name:
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
        mov = np.array(h5['mov'])
elif '.tif' in name:
    mov = imread(name)

with h5py.File(os.path.join(movie_folder, name[:-8]+'_ROIs.hdf5'),'r') as h5:
    mask = np.array(h5['mov'])                                               
  
#%% setting params
# dataset dependent parameters
fnames = ''
fr = 400
ROIs = mask

border_to_0 = 2
flip = True
num_frames_init = 10000
num_frames_total = 20000
thresh_range = [3, 4]
erosion = 0 
update_bg = True
use_spikes = False 
initialize_with_gpu = False
adaptive_threshold = True
filt_window = 15
    
opts_dict = {
    'fnames': fnames,
    'fr': fr,
    'ROIs': ROIs,
    'border_to_0': border_to_0,
    'flip': flip,
    'num_frames_total': num_frames_total, 
    'thresh_range': thresh_range,
    'erosion':erosion, 
    'update_bg': update_bg,
    'use_spikes':use_spikes, 
    'initialize_with_gpu':initialize_with_gpu,
    'adaptive_threshold': adaptive_threshold,
    'filt_window': filt_window}

opts = violaparams(params_dict=opts_dict)

#%% process offline for initialization
params = violaparams(params_dict=opts_dict)
vio = VIOLA(params=params)
vio.fit(mov[:num_frames_init])
plt.plot(vio.pipeline.saoz.t_s[0])

#%% process online
scope = [num_frames_init, num_frames_total]
vio.pipeline.load_frame_thread = Thread(target=vio.pipeline.load_frame, daemon=True, args=(mov[scope[0]:scope[1], :, :],))
vio.pipeline.load_frame_thread.start()

start = time()
vio.fit_online()
sleep(0.1) # wait finish
print(f'total time online: {time()-start}')
print(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')

#%% compute the result in vio.estimates object
vio.compute_estimates()

#%% visualize the result
for i in range(vio.H.shape[1]):
    if i == 0:
        plt.figure()
        plt.imshow(mov[0], cmap='gray')
        plt.imshow(vio.H.reshape((mov.shape[1], mov.shape[2], vio.H.shape[1]), order='F')[:,:,i], alpha=0.3)
        plt.figure()
        plt.plot(vio.pipeline.saoz.trace[i][:scope[1]])
        
#%% save the result
save_name = f'viola_result_init_{num_frames_init}_bg_{update_bg}'
np.save(os.path.join(movie_folder, save_name), vio.estimates)

#%%
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)