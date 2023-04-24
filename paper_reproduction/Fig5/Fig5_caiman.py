#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using CaImAn Online (OnACID).
The demo demonstates the analysis of a sequence of files using the CaImAn online
algorithm. The steps include i) motion correction, ii) tracking current 
components, iii) detecting new components, iv) updating of spatial footprints.
The script demonstrates how to construct and use the params and online_cnmf
objects required for the analysis, and presents the various parameters that
can be passed as options. A plot of the processing time for the various steps
of the algorithm is also included.
@author: Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing the data used in this demo.
"""

import glob
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import download_demo

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

# %%
def main():
    pass # For compatibility between running under Spyder and the CLI

# %%  download and list all files to be processed
    fnames_256 = ["/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_256/k53_256.tif"]
    fnames_512 = ["/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_512/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00001.tif"]
    fnames_1024 = ["/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_1024/k53_1024.tif"]
    fnames_all = [fnames_256, fnames_512, fnames_1024]
    fnames = fnames_all[0]
    # fnames = ["/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_256/k53_256.tif"]
# %%   Set up some parameters

    fr = 15  # frame rate (Hz)
    update_num_comps = False
    # update_freq = np.inf
    # num_times_comp_updated = 0
    # minibatch_suff_stat = np.inf
    use_cnn = True
    cnn_lowest=100000
    SNR_lowest = np.inf
    rval_lowest=1.1
    decay_time = 0.5  # approximate length of transient event in seconds
    gSig = (3, 3)  # expected half size of neurons
    p = 1  # order of AR indicator dynamics
    min_SNR = np.inf   # minimum SNR for accepting new components
    ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
    gnb = 2  # number of background components
    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
    mot_corr = True  # flag for online motion correction
    pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
    max_shifts_online = np.ceil(10.).astype('int')  # maximum allowed shift during motion correction
    sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
    rval_thr = 10000  # source correlation threshold for candidate components
    # set up some additional supporting parameters needed for the algorithm
    # (these are default values but can change depending on dataset properties)
    init_batch = 1500  # number of frames for initialization (presumably from the first file)
    K = 500  # initial number of components
    epochs = 1  # number of passes over the data
    show_movie = False # show the movie as the data gets processed

    params_dict = {'fnames': fnames,
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                    'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': mot_corr,
                   'init_batch': init_batch,
                   'init_method': 'bare',
                   'normalize': True,
                   'sniper_mode': sniper_mode,
                   'use_cnn': use_cnn,
                   'min_cnn_thr': 10000,
                   'cnn_lowest': cnn_lowest,
                   'SNR_lowest': SNR_lowest,
                   'rval_lowest': rval_lowest,
                    'update_num_comps': update_num_comps,
                    'use_dense': True,
                    # 'update_freq': update_freq,
                    # 'num_times_comp_updated': num_times_comp_updated,
                    # 'minibatch_suff_stat': minibatch_suff_stat,
                   'K': K,
                   'epochs': epochs,
                   'max_shifts_online': max_shifts_online,
                   'pw_rigid': pw_rigid,
                   'dist_shape_update': True,
                   'min_num_trial': 10,
                   'show_movie': show_movie}
#%% run iteratively
    import time
    for fnames in fnames_all:
        params_dict["fnames"] = fnames
        if "1024" in fnames[0]:
            dim = "1024"
            params_dict["gSig"] = (12,12)
        else:
            dim = "512"
            params_dict["gSig"] = (6,6)
        for neurs in [100, 200, 500]:
            params_dict["K"] = neurs
            opts = cnmf.params.CNMFParams(params_dict=params_dict)
            for i in range(3):
                cnm = cnmf.online_cnmf.OnACID(params=opts)
                cnm.fit_online()
                T_motion = 1e3*np.array(cnm.t_motion)
                T_detect = 1e3*np.array(cnm.t_detect)
                T_shapes = 1e3*np.array(cnm.t_shapes)
                T_track = 1e3*np.array(cnm.t_online) - T_motion - T_shapes
                plt.figure()
                plt.stackplot(np.arange(len(T_motion)-1), T_motion[1:], T_track[1:])
                plt.legend(labels=['motion', 'tracking', 'detect', 'shapes'], loc=2)
                plt.title('Processing time allocation')
                plt.xlabel('Frame #')
                plt.ylabel('Processing time [ms]')
                base_file = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/28mar_finalsub_MULTIPLE/cm"
                neurs = str(cnm.estimates.C.shape[0])
                save_obj = { 
                    "T_motion": T_motion, 
                    "fiola": 1000*np.diff(cnm.fiola_timing),
                    "T_shapes": T_shapes,
                    "T_track": T_track
                }
                np.save(base_file+"_"+dim+"_"+neurs+"_"+ str(i)+".npy",save_obj)
                np.save(base_file+"params_"+dim+"_"+neurs+"_"+ str(i)+".npy",params_dict)
                time.sleep(100)
                # np.save(base_file+"_"+dim+"_"+neurs+"_"+ str(i)+"_C.npy", cnm.estimates.C)

# %% fit online once

    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()

# %% plot contours (this may take time)
    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    images = cm.load(fnames)
    Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    cnm.estimates.plot_contours(img=Cn, display_numbers=False)

# %% view components
    cnm.estimates.view_components(img=Cn)

# %% plot timing performance (if a movie is generated during processing, timing
# will be severely over-estimated)

    T_motion = 1e3*np.array(cnm.t_motion)
    T_detect = 1e3*np.array(cnm.t_detect)
    T_shapes = 1e3*np.array(cnm.t_shapes)
    T_track = 1e3*np.array(cnm.t_online) - T_motion - T_shapes# - T_detect
    plt.figure()
    plt.stackplot(np.arange(len(T_motion)), T_motion, T_track, T_shapes)
    plt.legend(labels=['motion', 'tracking', 'shapes'], loc=2)
    plt.title('Processing time allocation')
    plt.xlabel('Frame #')
    plt.ylabel('Processing time [ms]')
    print(np.mean(T_motion)+ np.mean(T_track)+ np.mean(T_shapes))
    print(np.mean(T_motion), np.mean(T_track), np.mean(T_shapes))
#%% save data
    base_file = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/7apr/cm"
    dim = str(512)
    neurs = str(cnm.estimates.C.shape[0])
    save_obj = { 
        "T_motion": T_motion, 
        "T_detect": T_detect,
        "T_shapes": T_shapes,
        "T_track": T_track
    }
    np.save(base_file+"_"+dim+"_"+neurs+".npy",save_obj)
    np.save(base_file+"params_"+dim+"_"+neurs+".npy",params_dict)
    np.save(base_file+"_"+dim+"_"+neurs+"_C.npy", cnm.estimates.C)