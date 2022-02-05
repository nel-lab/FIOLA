#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:01:46 2020
fiola simulation
@author: @caichangjia
"""
import caiman as cm
import glob
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.python.client import device_lib
from time import time, sleep
from threading import Thread
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import match_spikes_greedy, normalize, compute_F1, movie_iterator, load, to_2D
import scipy.io
from skimage.io import imread


logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)
    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1


#%%
def run_fiola(fnames, path_ROIs, fr=400, options=None):
    #%%
    file_dir = os.path.split(fnames)[0]
    num_frames_init = options['num_frames_init']
    num_frames_total = options['num_frames_total']
    mode = options['mode']
    logging.info('Loading Movie')
    mov = cm.load(fnames, subindices=range(num_frames_init))
    mask = load(path_ROIs)
    template = np.median(mov, 0)

    #%% Run FIOLA
    #example motion correction
    motion_correct = True
    #example source separation
    do_nnls = True
    #%% Mot corr only
    if motion_correct:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        # run motion correction on GPU on the initialization movie
        mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], min_mov=mov.min())             
        plt.plot(shifts_fiola)
    else:    
        mc_nn_mov = mov
    
    #%% NNLS only
    if do_nnls:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        if mode == 'voltage':
            A = scipy.sparse.coo_matrix(to_2D(mask, order='F')).T
            fio.fit_hals(mc_nn_mov, A)
            Ab = fio.Ab # Ab includes spatial masks of all neurons and background
        else:
            Ab = np.hstack((estimates.A.toarray(), estimates.b))
            
        trace_fiola, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch_size']) 
        plt.plot(trace_fiola.T)
        
    #%% Set up online pipeline
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    if mode == 'voltage': # not thoroughly tested and computationally intensive for large files, it will estimate the baseline
        fio.fit_hals(mc_nn_mov, A)
        Ab = fio.Ab
    else:
        Ab = np.hstack((estimates.A.toarray(), estimates.b))
    Ab = Ab.astype(np.float32)
        
    fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())
    #%% run online
    time_per_step = np.zeros(num_frames_total-num_frames_init)
    traces = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
    start = time()
        
    for idx, memmap_image in movie_iterator(fnames, num_frames_init, num_frames_total, batch_size=1):
        if idx % 100 == 0:
            print(idx)        
        fio.fit_online_frame(memmap_image)   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i        
        traces[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx]
        time_per_step[idx-num_frames_init] = (time()-start)
    
    traces = traces.T
    logging.info(f'total time online: {time()-start}')
    logging.info(f'time per frame online: {(time()-start)/(num_frames_total-num_frames_init)}')
    plt.plot(np.diff(time_per_step),'.')
    #%% Visualize result
    fio.compute_estimates()
    fio.view_components(template)
    
    # import pdb
    # pdb.set_trace()
    
    #%% save some interesting data
    # if True:
    #     np.savez(fnames[:-4]+'_fiola_result_v3.0.npz', time_per_step=time_per_step, traces=traces, 
    #          caiman_file = caiman_file, 
    #          fnames_exp = fnames, 
    #          estimates = fio.estimates)
    save_name = f'fiola_result_num_layers_{fio.params.mc_nnls["num_layers"]}_trace_with_neg_{fio.params.mc_nnls["trace_with_neg"]}_v3.1'
    np.save(os.path.join(file_dir, 'viola', save_name), fio.estimates)
    
    
    
    #%% save
    # save_name = f'fiola_result_online_gpu_{online_gpu}_init_{opts.data["num_frames_init"]}' \
    #     f'_bg_{opts.mc_nnls["update_bg"]}_use_spikes_{opts.mc_nnls["use_spikes"]}' \
    #     f'_hals_movie_{opts.mc_nnls["hals_movie"]}_semi_nmf_{opts.mc_nnls["semi_nmf"]}' \
    #     f'_adaptive_threshold_{opts.spike["adaptive_threshold"]}' \
    #     f'_do_scale_{opts.spike["do_scale"]}_freq_{opts.spike["freq"]}'\
    #     f'_filt_window_{opts.spike["filt_window"]}_minimal_thresh_{opts.spike["minimal_thresh"]}'\
    #     f'_template_window_{opts.spike["template_window"]}_v2.1'
    #np.save(os.path.join(file_dir, 'fiola', save_name), vio.estimates)
    
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    
