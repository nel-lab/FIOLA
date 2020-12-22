#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:01:46 2020
viola simulation
@author: @caichangjia
"""
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time, sleep
from threading import Thread

from nmf_support import normalize
from violaparams import violaparams
from viola import VIOLA
import scipy.io
from skimage.io import imread
from match_spikes import match_spikes_greedy, compute_F1

#%%
def run_viola(fnames, path_ROIs, fr=400, options=None):
    # Load movie and ROIs
    file_dir = os.path.split(fnames)[0]
    if '.hdf5' in fnames:
        with h5py.File(fnames,'r') as h5:
            mov = np.array(h5['mov'])
    elif '.tif' in fnames:
        mov = imread(fnames)
    else:
        print('do not support this movie format')
    with h5py.File(path_ROIs,'r') as h5:
        ROIs = np.array(h5['mov'])

    #%% setting params
    # dataset dependent parameters
    fnames = ''
    fr = fr
    ROIs = ROIs
    border_to_0 = 2
    flip = True
    num_frames_init = 10000
    num_frames_total=20000
    thresh_range= [3, 4]
    erosion=0 
    update_bg = True
    use_spikes=True 
    initialize_with_gpu=False
    adaptive_threshold=True
    filt_window=15
    
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
    
    if options is not None:
        print('using external options')
        opts.change_params(params_dict=options)
    else:
        print('not using external options')
    
    #%% process offline for init
    vio = VIOLA(params=opts)
    vio.fit(mov[:num_frames_init])
    
    #%% process online
    scope = [num_frames_init, num_frames_total]
    vio.pipeline.load_frame_thread = Thread(target=vio.pipeline.load_frame, daemon=True, args=(mov[scope[0]:scope[1], :, :],))
    vio.pipeline.load_frame_thread.start()
    start = time()
    vio.fit_online()
    sleep(0.1) # wait finish
    print(f'total time online: {time()-start}')
    print(f'time per frame  online: {(time()-start)/(scope[1]-scope[0])}')

    plt.plot(vio.pipeline.saoz.trace[0])

    
    #%%
    vio.compute_estimates()
    
    #%% save
    save_name = f'viola_result_init_{num_frames_init}_bg_{update_bg}_use_spikes_False_small_mask'
    np.save(os.path.join(file_dir, 'viola', save_name), vio.estimates)
    
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    
    