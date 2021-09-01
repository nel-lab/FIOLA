#!/usr/bin/env python
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
movie_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data'][0]
name = 'demo_voltage_imaging.hdf5'
if '.hdf5' in name:
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
        mov = np.array(h5['mov'])
elif '.tif' in name:
    mov = imread(name)

with h5py.File(os.path.join(movie_folder, name[:-5]+'_ROIs.hdf5'),'r') as h5:
    mask = np.array(h5['mov'])                                               

#%%
movie_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data', 
                '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/demo_K53'][1]
name = ['demo_voltage_imaging.hdf5', 'k53.tif'][1]
if '.hdf5' in name:
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
        mov = np.array(h5['mov'])
elif '.tif' in name:
    mov = imread(os.path.join(movie_folder, name))

#with h5py.File(os.path.join(movie_folder, name[:-5]+'_ROIs.hdf5'),'r') as h5:
#    mask = np.array(h5['mov'])                                               

mask = np.load(os.path.join(movie_folder, 'masks_caiman.npy'))
mov = mov.astype(np.float32) - mov.min()
dims = mov.shape[1:]

#%% setting params
# dataset dependent parameters
fnames = ''                     # name of the movie, we don't put a name here as movie is already loaded above
fr = 400                        # sample rate of the movie
ROIs = mask                     # a 3D matrix contains all region of interests

num_frames_init =  20000         # number of frames used for initialization
num_frames_total =  30000        # estimated total number of frames for processing, this is used for generating matrix to store data
flip = True                     # whether to flip signal to find spikes   
ms=[5, 5]                      # maximum shift in x and y axis respectively. Will not perform motion correction if None.
thresh_range= [2.8, 5.0]        # range of threshold factor. Real threshold is threshold factor multiply by the estimated noise level
use_rank_one_nmf=False          # whether to use rank-1 nmf, if False the algorithm will use initial masks and average signals as initialization for the HALS
hals_movie='hp_thresh'          # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); original movie (orig)
update_bg = True                # update background components for spatial footprints
use_batch=True                  # whether to process a batch of frames (greater or equal to 1) at the same time. set use batch always as True
batch_size= 40                  # number of frames processing at the same time using gpu 
initialize_with_gpu=True        # whether to use gpu for performing nnls during initialization
do_scale = False                # whether to scale the input trace or not
adaptive_threshold=True         # whether to use adaptive threshold method for deciding threshold level
filt_window=15                  # window size for removing the subthreshold activities 
minimal_thresh=3                # minimal of the threshold 
step=2500                       # step for updating statistics
template_window=2               # half window size of the template; will not perform template matching if window size equals 0
    
options = {
    'fnames': fnames,
    'fr': fr,
    'ROIs': ROIs,
    'flip': flip,
    'ms': ms,
    'num_frames_init': num_frames_init, 
    'num_frames_total':num_frames_total,
    'thresh_range': thresh_range,
    'use_rank_one_nmf': use_rank_one_nmf,
    'hals_movie': hals_movie,
    'update_bg': update_bg,
    'use_batch':use_batch,
    'batch_size':batch_size,
    'initialize_with_gpu':initialize_with_gpu,
    'do_scale': do_scale,
    'adaptive_threshold': adaptive_threshold,
    'filt_window': filt_window,
    'minimal_thresh': minimal_thresh,
    'step': step, 
    'template_window':template_window}

#%% process offline for initialization
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
fio.fit(mov[:num_frames_init])

#%% process online
scope = [num_frames_init, num_frames_total]
fio.pipeline.load_frame_thread = Thread(target=fio.pipeline.load_frame, 
                                        daemon=True, 
                                        args=(mov[scope[0]:scope[1], :, :],))
fio.pipeline.load_frame_thread.start()

start = time()
fio.fit_online()
sleep(0.1) # wait finish
print(f'total time online: {time()-start}')
print(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')

#%% compute the result in fio.estimates object
fio.compute_estimates()

#%% visualize the result, the last component is the background
for i in range(fio.H.shape[1]):
    if i == 8:
        plt.figure()
        plt.imshow(mov[0], cmap='gray')
        plt.imshow(fio.H.reshape((mov.shape[1], mov.shape[2], fio.H.shape[1]), order='F')[:,:,i], alpha=0.3)
        plt.title(f'Spatial footprint of neuron {i}')
        plt.figure()
        plt.plot(fio.pipeline.saoz.trace[i][:scope[1]])
        plt.title(f'Temporal trace of neuron {i} before processing')
        plt.figure()
        plt.plot(normalize(fio.pipeline.saoz.t_s[i][:scope[1]]))
        spikes = np.delete(fio.pipeline.saoz.index[i], fio.pipeline.saoz.index[i]==0)
        h_min = normalize(fio.pipeline.saoz.t_s[i][:scope[1]]).max()
        plt.vlines(spikes, h_min, h_min + 1, color='black')
        plt.legend('trace', 'detected spikes')
        plt.title(f'Temporal trace of neuron {i} after processing')
        
        
#%% save the result
save_name = f'{os.path.join(movie_folder, name)[:-5]}_fiola_result'
np.save(os.path.join(movie_folder, save_name), fio.estimates)

#%%
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)