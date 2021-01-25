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

from viola.nmf_support import normalize
from viola.violaparams import violaparams
from viola.viola import VIOLA
import scipy.io
from skimage.io import imread
from viola.match_spikes import match_spikes_greedy, compute_F1

#%%
def run_viola(fnames, path_ROIs, fr=400, online_gpu=True, options=None):
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
    num_frames_total = mov.shape[0]

    border_to_0 = 2
    flip = True
    thresh_range= [3, 4]
    erosion=0 
    hals_movie='hp_thresh'
    use_rank_one_nmf=True
    semi_nmf=False
    update_bg = True
    use_spikes=True 
    initialize_with_gpu=False
    adaptive_threshold=True
    filt_window=15
    
    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'ROIs': ROIs,
        'num_frames_total': num_frames_total,
        'border_to_0': border_to_0,
        'flip': flip,
        'thresh_range': thresh_range,
        'erosion':erosion, 
        'hals_movie': hals_movie,
        'use_rank_one_nmf': use_rank_one_nmf,
        'semi_nmf': semi_nmf,
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
    print(f'Movie shape: {mov.shape}')
    print(f'ROIs shape: {ROIs.shape}')
    print(f'Number of frames for initialization: {opts.data["num_frames_init"]}')    
    print(f'Total Number of frames for processing: {opts.data["num_frames_total"]}')
    sleep(5)
    vio = VIOLA(params=opts)
    vio.fit(mov[:opts.data['num_frames_init']])
    
    #%% process online
    if online_gpu == True:
        vio.pipeline.load_frame_thread = Thread(target=vio.pipeline.load_frame, 
                                                daemon=True, 
                                                args=(mov[opts.data['num_frames_init']:opts.data['num_frames_total'], :, :],))
        vio.pipeline.load_frame_thread.start()
        start = time()
        vio.fit_online()
        sleep(5) # wait finish
        print(f'total time online: {time()-start}')
        print(f'time per frame  online: {(time()-start)/(opts.data["num_frames_total"]-opts.data["num_frames_init"])}')
    else:
        vio.fit_without_gpu(mov)

    #%%
    vio.compute_estimates()
    plt.plot(normalize(vio.estimates.t_s[0]))
    #plt.hlines(vio.saoz.thresh[0, 0], 0, 30000)
    #print(vio.saoz.thresh_factor)
    
    #%% save
    save_name = f'viola_result_online_gpu_{online_gpu}_init_{opts.data["num_frames_init"]}' \
        f'_bg_{opts.mc_nnls["update_bg"]}_use_spikes_{opts.mc_nnls["use_spikes"]}' \
        f'_hals_movie_{opts.mc_nnls["hals_movie"]}' \
        f'_adaptive_threshold_{opts.spike["adaptive_threshold"]}' \
        f'_do_scale_{opts.spike["do_scale"]}'
    np.save(os.path.join(file_dir, 'viola', save_name), vio.estimates)
    
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    
