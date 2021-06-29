#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:48:17 2020
Pipeline for online analysis of fluorescence imaging data
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

from fiola.utilities import normalize
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA

#%% load movie and masks
movie_folder = ['/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data'][3]
    
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

opts = fiolaparams(params_dict=opts_dict)

#%% process offline for initialization
params = fiolaparams(params_dict=opts_dict)
fio = FIOLA(params=params)
fio.fit(mov[:num_frames_init])
plt.plot(fio.pipeline.saoz.t_s[0])

#%% process online
scope = [num_frames_init, num_frames_total]
fio.pipeline.load_frame_thread = Thread(target=fio.pipeline.load_frame, daemon=True, args=(mov[scope[0]:scope[1], :, :],))
fio.pipeline.load_frame_thread.start()

start = time()
fio.fit_online()
sleep(0.1) # wait finish
print(f'total time online: {time()-start}')
print(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')

#%% compute the result in fio.estimates object
fio.compute_estimates()

#%% visualize the result
for i in range(fio.H.shape[1]):
    if i == 1:
        plt.figure()
        plt.imshow(mov[0], cmap='gray')
        plt.imshow(fio.H.reshape((mov.shape[1], mov.shape[2], fio.H.shape[1]), order='F')[:,:,i], alpha=0.3)
        plt.figure()
        plt.plot(fio.pipeline.saoz.trace[i][:scope[1]])
        
#%% save the result
save_name = f'fiola_result_init_{num_frames_init}_bg_{update_bg}'
np.save(os.path.join(movie_folder, save_name), fio.estimates)

#%%
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
