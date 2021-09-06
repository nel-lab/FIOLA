#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:48:17 2020
Pipeline for online analysis of fluorescence imaging data
Voltage dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
Calcium dataset couresy of Sue Ann Koay and David Tank (Princeton University)
@author: @agiovann, @caichangjia, @cynthia
"""
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from skimage.io import imread
from time import time, sleep
from threading import Thread

from fiola.config import load_fiola_config, load_caiman_config
from fiola.utilities import normalize, bin_median, play, to_2D, local_correlations
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA

import logging
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

#%% load movie and masks
folder_idx = 1
movie_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data', 
                '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/demo_K53'][folder_idx]
name = ['demo_voltage_imaging.hdf5', 'k53.tif'][folder_idx]
mode = ['voltage', 'calcium'][folder_idx]
if '.hdf5' in name:
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
        mov = np.array(h5['mov'])
elif '.tif' in name:
    mov = imread(os.path.join(movie_folder, name))

with h5py.File(os.path.join(movie_folder, name.split('.')[0]+'_ROIs.hdf5'),'r') as h5:
    mask = np.array(h5['mov'])                                               
dims = mov.shape[1:]

# init_file_name = os.path.join(movie_folder, 'K53_init.tif')


#%% load configuration; set up FIOLA object
# In the first part, we will first show each part (motion correct, source separation and spike extraction) of 
# FIOLA separately in an offline manner. 
# Then in the second part, we will show the full pipeline and its real time frame-by-frame analysis performance
options = load_fiola_config(mode, mask) 
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)


#%% offline motion correction
fio.dims = mov.shape[1:]
template = bin_median(mov)
mc_mov, shifts, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], mov.min())

display_images = True
if display_images:
    plt.figure(); plt.plot(shifts);plt.legend(['x shifts', 'y shifts']); plt.title('shifts'); plt.show()
    moviehandle = mc_mov.copy().reshape((-1, template.shape[0], template.shape[1]), order='F')
    play(moviehandle, gain=3, q_min=5, q_max=99.99)

#%% optimize masks using hierarchical alternating least squares algorithm
if mode == 'voltage':
    fio.fit_hals(mc_mov, mask)
elif mode == 'calcium':
    # we don't need to optimize masks as we are using spatial footprints from CaImAn
    mask_2D = to_2D(mask, order='C')
    Ab = mask_2D.T
    fio.Ab = Ab / norm(Ab, axis=0)
    
#%% source extraction (nnls)
trace = fio.fit_gpu_nnls(mc_mov, fio.Ab, batch_size=1)

#%% offline spike detection (only available for voltage currently)
fio.saoz = fio.fit_spike_extraction(trace)

#%% put the result in fio.estimates object
fio.compute_estimates()

#%% show results
fio.corr = local_correlations(mc_mov, swap_dim=False)
if display_images:
    fio.view_components(fio.corr)

#%% Now we start the second part. It uses fit method to perform initialization 
# which prepare parameters, spatial footprints etc for real-time analysis
# Then we call fit_online to perform real-time analysis
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
scope = [fio.params.data['num_frames_init'], fio.params.data['num_frames_total']]
fio.fit(mov[:scope[0]])

#%% now fit online
fio.pipeline.load_frame_thread = Thread(target=fio.pipeline.load_frame, 
                                        daemon=True, 
                                        args=(mov[scope[0]:scope[1], :, :],))
fio.pipeline.load_frame_thread.start()

start = time()
fio.fit_online()
sleep(0.1) # wait finish
print(f'total time online: {time()-start}')
print(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')

#%% put the result in fio.estimates object
fio.compute_estimates()

#%% visualize the result, the last component is the background
if display_images:
    fio.view_components(fio.corr)
    
    from caiman.source_extraction.cnmf import cnmf 
    cnm = cnmf.load_CNMF('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/k53/memmap__d1_512_d2_512_d3_1_order_C_frames_3000_.hdf5').estimates
    #np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/demo/Ab.npy', fio.Ab)
    #plt.plot(cnm.C)

    fio.view_components(fio.corr, cnm_estimates=cnm)
        
#%% save the result
save_name = f'{os.path.join(movie_folder, name.split(".")[0])}_fiola_result'
np.save(os.path.join(movie_folder, save_name), fio.estimates)

#%%
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)