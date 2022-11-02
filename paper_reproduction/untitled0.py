#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:26:49 2022

@author: nel
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib
from time import time
import scipy

from fiola.demo_initialize_calcium import run_caiman_init
import pyximport
pyximport.install()
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from caiman.source_extraction.cnmf.utilities import get_file_size
import caiman as cm
from fiola.utilities import download_demo, load, play, bin_median, to_2D, local_correlations, movie_iterator, compute_residuals

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)
    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1

#%%
fnames = '/home/nel/caiman_data/example_movies/demoMovie/demoMovie.tif'
#fnames = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/N.01.01/mov_N.01.01.tif"
# path_ROIs = download_demo(folder, 'demo_voltage_imaging_ROIs.hdf5')
# mask = load(path_ROIs)
#A = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/N01_A.npy", allow_pickle=True)[()]
#b = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/N01_b.npy", allow_pickle=True)[()]
# A = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/k53_A.npy",allow_pickle=True)[()]
# b = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/k53_b.npy",allow_pickle=True)[()]
#mask = np.concatenate((A.toarray()[:,-500:],b),axis=1)
# mask = None

fr = 30                         # sample rate of the movie
ROIs = None                     # a 3D matrix contains all region of interests

mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
num_frames_init =   500         # number of frames used for initialization
num_frames_total =  2000        # estimated total number of frames for processing, this is used for generating matrix to store data
offline_batch_size = 5          # number of frames for one batch to perform offline motion correction
batch_size= 1                   # number of frames processing at the same time using gpu 
flip = False                    # whether to flip signal to find spikes   
detrend = True                  # whether to remove the slow trend in the fluorescence data            
dc_param = 0.9995
do_deconvolve = True            # If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
ms = [5, 5]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
                                # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
n_split = 1                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
nb = 2                          # number of background components
trace_with_neg=False             # return trace with negative components (noise) if True; otherwise the trace is cutoff at 0
                
options = {
    'fnames': fnames,
    'fr': fr,
    'ROIs': ROIs,
    'mode': mode, 
    'num_frames_init': num_frames_init, 
    'num_frames_total':num_frames_total,
    'offline_batch_size': offline_batch_size,
    'batch_size':batch_size,
    'flip': flip,
    'detrend': detrend,
    'dc_param': dc_param,            
    'do_deconvolve': do_deconvolve,
    'ms': ms,
    'hals_movie': hals_movie,
    'center_dims':center_dims, 
    'n_split': n_split,
    'nb' : nb, 
    'trace_with_neg':trace_with_neg}

mov = cm.load(fnames, subindices=range(num_frames_init))
# fnames_init = fnames.split('.')[0] + '_init.tif'
# mov.save(fnames_init)

# run caiman initialization. User might need to change the parameters 
# inside the file to get good initialization result
# caiman_file = run_caiman_init(fnames_init)

# load results of initialization
#caiman_file = '/home/nel/caiman_data/example_movies/demoMovie/memmap__d1_60_d2_80_d3_1_order_C_frames_1500__init.hdf5'
#cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
#estimates = cnm2.estimates
# template = cnm2.estimates.template
# Cn = cnm2.estimates.Cn
template= np.median(mov[:1500], axis=0)

#%%
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
# run motion correction on GPU on the initialization movie
mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov[:100], template, fio.params.mc_nnls['offline_batch_size'], min_mov=mov.min())             
plt.plot(shifts_fiola)