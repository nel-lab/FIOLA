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

import caiman as cm
from fiola.utilities import normalize, play
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.caiman_init import run_caiman_init

#%% load movie and masks
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
#img_norm = np.std(mov, axis=0)
#img_norm += np.median(img_norm)
#mov = mov/img_norm[None, :, :]
 
#%% setting params
# FIOLA params
fnames = ''                     # name of the movie, we don't put a name here as movie is already loaded above
fr = 30                         # sample rate of the movie
ROIs = mask                     # a 3D matrix contains all region of interests
mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
init_method = 'caiman'          # initialization method 'caiman' or 'masks'. Needs to provide masks or using gui to draw masks if choosing 'masks'
num_frames_init =  1000         # number of frames used for initialization
num_frames_total =  3000        # estimated total number of frames for processing, this is used for generating matrix to store data
flip = False                    # whether to flip signal to find spikes   
ms=[5, 5]                       # maximum shift in x and y axis respectively. Will not perform motion correction if None.
offline_mc_batch_size=100       # number of frames for one batch to perform offline motion correction
thresh_range= [2.8, 5.0]        # range of threshold factor. Real threshold is threshold factor multiply by the estimated noise level
use_rank_one_nmf=False          # whether to use rank-1 nmf, if False the algorithm will use initial masks and average signals as initialization for the HALS
hals_movie=None                 # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
                                # original movie (orig); no HALS needed if the input is from CaImAn (None)
update_bg = True                # update background components for spatial footprints
use_batch=True                  # whether to process a batch of frames (greater or equal to 1) at the same time. set use batch always as True
batch_size= 1                   # number of frames processing at the same time using gpu 
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
    'mode': mode, 
    'init_method':init_method,
    'flip': flip,
    'ms': ms,
    'offline_mc_batch_size': offline_mc_batch_size,
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

# params for caiman init
init_file_name = os.path.join(movie_folder, 'K53_init.tif')
fr = 30             # imaging rate in frames per second
decay_time = 0.4    # length of a typical transient in seconds
dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
# note the lower than usual spatial resolution here
max_shift_um = (12., 12.)       # maximum shift in um
patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

# motion correction parameters
pw_rigid = False       # flag to select rigid vs pw_rigid motion correction
# maximum allowed rigid shift in pixels
max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
# start a new patch for pw-rigid motion correction every x pixels
strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
# overlap between pathes (size of patch in pixels: strides+overlaps)
overlaps = (24, 24)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3

mc_dict = {
    'fnames': init_file_name,
    'fr': fr,
    'decay_time': decay_time,
    'dxy': dxy,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': 'copy'
}

p = 1                    # order of the autoregressive system
gnb = 2                  # number of global background components
merge_thr = 0.85         # merging threshold, max correlation allowed
rf = 15
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 6          # amount of overlap between the patches in pixels
K = 4                    # number of components per patch
gSig = [6, 6]            # expected half size of neurons in pixels
# initialization method (if analyzing dendritic data using 'sparse_nmf')
method_init = 'greedy_roi'
ssub = 2                     # spatial subsampling during initialization
tsub = 2                     # temporal subsampling during intialization
n_processes = None

# parameters for component evaluation
opts_dict = {'fnames': init_file_name,
             'p': p,
             'fr': fr,
             'nb': gnb,
             'rf': rf,
             'K': K,
             'gSig': gSig,
             'stride': stride_cnmf,
             'method_init': method_init,
             'rolling_sum': True,
             'merge_thr': merge_thr,
             'n_processes': n_processes,
             'only_init': True,
             'ssub': ssub,
             'tsub': tsub}

# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier
min_SNR = 5  # signal to noise ratio for accepting a component
rval_thr = 0.85  # space correlation threshold for accepting a component
cnn_thr = 0.99  # threshold for CNN based classifier
cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

quality_dict = {'decay_time': decay_time,
                'min_SNR': min_SNR,
                'rval_thr': rval_thr,
                'use_cnn': True,
                'min_cnn_thr': cnn_thr,
                'cnn_lowest': cnn_lowest}

#%% process offline for initialization
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)

if init_method == 'caiman':
    from caiman.source_extraction.cnmf import cnmf 
    estimates = cnmf.load_CNMF('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/demo_K53/memmap__d1_512_d2_512_d3_1_order_C_frames_1000_.hdf5').estimates
    estimates = run_caiman_init(mov[:num_frames_init], init_file_name, mc_dict, opts_dict, quality_dict)
    estimates.plot_contours(img=estimates.Cn)
    mask = np.hstack((estimates.A.toarray(), estimates.b)).reshape([dims[0], dims[1], -1]).transpose([2, 0, 1])
    trace_init = np.vstack((estimates.C, estimates.f))
    fio.params.data['ROIs'] = mask
    mov_init = cm.load(estimates.fname_new)
    fio.fit(mov_init, trace=trace_init)
else:
    fio.fit(mov[:num_frames_init])

#%% process online
scope = [num_frames_init, num_frames_total]
start = time()

for idx in range(scope[0], scope[1]):
    fio.fit_online_frame(mov[idx:idx+1])    

sleep(3) # wait finish
print(f'total time online: {time()-start}')
print(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')

#%% compute the result in fio.estimates object
#fio.compute_estimates()

#%% visualize the result, the last component is the background
for i in range(10):
    plt.figure()
    #plt.imshow(mov[0], cmap='gray')
    plt.imshow(fio.H.reshape((mov.shape[1], mov.shape[2], fio.H.shape[1]), order='F')[:,:,i])#, alpha=0.7)
    plt.title(f'Spatial footprint of neuron {i}')
    
    if mode == 'voltage':
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
    elif mode == 'calcium':
        plt.figure()
        plt.plot(fio.pipeline.saoz[i][:scope[1]])
        
#%% save the result
save_name = f'{os.path.join(movie_folder, name)[:-5]}_fiola_result'
np.save(os.path.join(movie_folder, save_name), fio.estimates)

#%%
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)