# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:48:17 2020
Pipeline for online analysis of voltage imaging data
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
@author: @agiovann, @caichangjia, @cynthia
"""
import h5py
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import pyximport
pyximport.install()
import os
import scipy.io
import sys
sys.path.append('/home/nel/CODE/VIOLA')
from time import time, sleep
from threading import Thread

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass
#
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import normalize, normalize_piecewise, match_spikes_greedy, compute_F1, load, signal_filter, extract_spikes
#sys.path.append('/media/nel/storage/Code/NEL_LAB/fiola/use_cases')
#sys.path.append(os.path.abspath('/Users/agiovann/SOFTWARE/fiola'))
#from use_cases.test_run_fiola import run_fiola # must be in use_cases folder
from paper_reproduction.Fig4_5.test_run_fiola import run_fiola


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name_id', type=int, required=True)
parser.add_argument('--num_layers', type=int, required=True)
args = parser.parse_args()


#%%
def main(name_id=None, num_layers=None):
    mode = ['overlapping', 'non_overlapping', 'positron'][1]
    dropbox_folder = '/media/nel/storage/NEL-LAB Dropbox/'
    #dropbox_folder = '/Users/agiovann/Dropbox/'
    
    if mode == 'overlapping':
        ROOT_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/data/voltage_data/simulation/overlapping'
        SAVE_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/result/test_simulations/overlapping'
        names = [f'viola_sim3_{i}' for i in range(1, 19)]
        #names = [f'viola_sim3_{i}' for i in range(4, 7)]
        #names = [f'viola_sim6_{i}' for i in range(2, 20, 3)]
    
    elif mode == 'non_overlapping':
        ROOT_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/data/voltage_data/simulation/non_overlapping'
        SAVE_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/result/test_simulations/non_overlapping'
        #names = [f'viola_sim5_{i}' for i in range(1, 8)]
        names = [f'viola_sim5_{i}' for i in range(2, 8, 2)]
        #names = [f'viola_sim7_{i}' for i in range(2, 9)]
        
    elif mode == 'positron':
        ROOT_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/data/voltage_data/simulation/test/sim4_positron'
        SAVE_FOLDER = dropbox_folder+'NEL/Papers/VolPy_online/result/test_simulations/test/sim4_positron'
        names = [f'viola_sim4_{i}' for i in range(1, 13)]
    
    name = names[name_id]
       
    #%%
    mode = 'voltage'
    num_frames_init = 10000
    num_frames_total = 75000
    border_to_0 = 2
    flip = True
    use_rank_one_nmf=False
    hals_movie='hp_thresh'
    semi_nmf=False
    update_bg = True
    use_spikes= False
    batch_size=1
    center_dims=None
    initialize_with_gpu=True
    do_scale = False
    adaptive_threshold=True
    filt_window=15
    minimal_thresh=3
    step=2500
    template_window=2
    trace_with_neg=False
    #num_layers = 30
        
    options = {
        'mode': mode, 
        'border_to_0': border_to_0,
        'flip': flip,
        'num_frames_init': num_frames_init, 
        'num_frames_total': num_frames_total, 
        'use_rank_one_nmf': use_rank_one_nmf,
        'hals_movie': hals_movie,
        'semi_nmf':semi_nmf,  
        'update_bg': update_bg,
        'use_spikes':use_spikes, 
        'batch_size':batch_size,
        'initialize_with_gpu':initialize_with_gpu,
        'do_scale': do_scale,
        'adaptive_threshold': adaptive_threshold,
        'filt_window': filt_window,
        'minimal_thresh': minimal_thresh,
        'step': step, 
        'template_window':template_window, 
        'num_layers': num_layers, 
        'trace_with_neg':trace_with_neg}
    
    #%%
    fnames = os.path.join(ROOT_FOLDER, name, name+'.hdf5')
    print(f'NOW PROCESSING: {fnames}')
    path_ROIs = os.path.join(ROOT_FOLDER, name, 'viola', 'ROIs_gt.hdf5')
    run_fiola(fnames, path_ROIs, fr=400, options=options)
    
if __name__ == "__main__":
    main(name_id=args.name_id, num_layers=args.num_layers)

