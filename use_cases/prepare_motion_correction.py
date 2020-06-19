#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:07:15 2020

@author: @caichangjia
"""
#%%
import cv2
import glob
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import os


try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.utils.utils import download_demo, download_model


#%%
#movie_folder = '/home/nel/data/voltage_data/Marton/454597/Cell_0/40x_patch1/movie'
#movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_3/40x_1xtube_10A2/movie'
#movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_3/40x_1xtube_10A3/movie'
#movie_folder = '/home/nel/data/voltage_data/Marton/462149/Cell_1/40x_1xtube_10A1/movie'#1kHZ
#movie_folder = '/home/nel/data/voltage_data/Marton/462149/Cell_1/40x_1xtube_10A2/movie'#1kHZ
#movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A5/movie'#EPSP
#movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A6/movie'
#movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A7/movie'
"""
movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_4/40x_1xtube_10A4/movie'
movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_6/40x_1xtube_10A10/movie'
movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_6/40x_1xtube_10A11/movie'
movie_folder = '/home/nel/data/voltage_data/Marton/466769/Cell_0/40x_1xtube_10A_1/movie'
movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A8/movie' # EPSP
movie_folder = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A9/movie' # EPSP
movie_folder = '/home/nel/data/voltage_data/Marton/462149/Cell_3/40x_1xtube_10A3/movie' # 1KHZ
movie_folder = '/home/nel/data/voltage_data/Marton/462149/Cell_3/40x_1xtube_10A4/movie' # 1KHZ
"""
movie_folders = ['/home/nel/data/voltage_data/Marton/456462/Cell_4/40x_1xtube_10A4/movie',
                 '/home/nel/data/voltage_data/Marton/456462/Cell_6/40x_1xtube_10A10/movie',
                 '/home/nel/data/voltage_data/Marton/456462/Cell_6/40x_1xtube_10A11/movie',
                 '/home/nel/data/voltage_data/Marton/466769/Cell_0/40x_1xtube_10A_1/movie',
                 '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A8/movie',
                 '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A9/movie',
                 '/home/nel/data/voltage_data/Marton/462149/Cell_3/40x_1xtube_10A3/movie',
                 '/home/nel/data/voltage_data/Marton/462149/Cell_3/40x_1xtube_10A4/movie', 
                 '/home/nel/data/voltage_data/Marton/466769/Cell_2/40x_1xtube_10A_6/movie',
                 '/home/nel/data/voltage_data/Marton/466769/Cell_2/40x_1xtube_10A_4/movie',
                 '/home/nel/data/voltage_data/Marton/466769/Cell_3/40x_1xtube_10A_8/movie'][8:]

# %% start a cluster for parallel processing
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

#%% 
for movie_folder in movie_folders:
    fnames = [os.path.join(movie_folder, i) for i in sorted(os.listdir(movie_folder)) if '.tif' in i]
    #dataset dependent parameters
    # dataset dependent parameters
    fr = 400                                        # sample rate of the movie
                                                   
    # motion correction parameters
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt = (3, 3)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5)                             # maximum allowed rigid shift
    strides = (48, 48)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)                             # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = volparams(params_dict=opts_dict)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run correction
    mc.motion_correct(save_movie=True)
    
    # %% MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries
    
    # memory map the file in order 'C'
    fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                                    add_to_mov=border_to_0, dview=dview)  # exclude border






















