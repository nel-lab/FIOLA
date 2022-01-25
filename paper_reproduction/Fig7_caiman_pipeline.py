#!/usr/bin/env python

"""
Complete demo pipeline for processing two photon calcium imaging data using the
CaImAn batch algorithm. The processing pipeline included motion correction,
source extraction and deconvolution. The demo shows how to construct the
params, MotionCorrect and cnmf objects and call the relevant functions. You
can also run a large part of the pipeline with a single method (cnmf.fit_file)
See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline.ipynb)
Dataset couresy of Sue Ann Koay and David Tank (Princeton University)

This demo pertains to two photon data. For a complete analysis pipeline for
one photon microendoscopic data see demo_pipeline_cnmfE.py

copyright GNU General Public License v2.0
authors: @agiovann and @epnev
"""

import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time

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
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.summary_images import local_correlations_movie_offline

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)

#%%
def run_caiman_fig7(fnames, pw_rigid=False, motion_correction_only=False, K=5):
    start = time()    
    # fnames = ['/media/nel/DATA/fiola/R2_20190219/3000/mov_R2_20190219T210000_3000.hdf5']
    # fnames = ['/media/nel/storage/fiola/R6_20200210T2100/mov_R6_20200210T2100.hdf5']
    # mm = cm.load(fnames, subindices=slice(0, 3000, 1))
    # mm.play(fr=100, gain=0.1)
    # mm.save('/media/nel/storage/fiola/R6_20200210T2100/3000/mov_R6_20200210T2100_3000.hdf5')
    # fnames = ['/media/nel/storage/fiola/R6_20200210T2100/3000/mov_R6_20200210T2100_3000.hdf5']
    # fnames = ['/media/nel/storage/fiola/R6_20200210T2100/mov_R6_20200210T2100.hdf5']
    # #folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/calcium_data/images_J123'
    #fnames = ['/media/nel/DATA/fiola/dandi_3000.tif']
#%% First setup some parameters for data and motion correction
    # dataset dependent parameters
    fr = 30             # imaging rate in frames per second
    decay_time = 0.4    # length of a typical transient in seconds
    dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (20., 20.)       # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

    # motion correction parameters
    #pw_rigid = False       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    mc_dict = {
        'fnames': fnames,
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

    opts = params.CNMFParams(params_dict=mc_dict)

# %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = False

    if display_images:
        m_orig = cm.load_movie_chain(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

# %% Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)
    
    if motion_correction_only:
        return 1

# %% compare with original movie
    if display_images:
        m_orig = cm.load_movie_chain(fnames)
        m_els = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=30, q_max=99.5, magnification=1)  # press q to exit

# %% MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries

    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name=f'memmap_pw_rigid_{pw_rigid}', order='C',dview=dview,
                               border_to_0=border_to_0)  # exclude borders

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

# %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%  parameters for source extraction and deconvolution    
    p = 1                    # order of the autoregressive system
    gnb = 2                 # number of global background components
    merge_thr = 0.95         # merging tWhreshold, max correlation allowed
    rf = 15
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 10         # amount of overlap between the patches in pixels
    #K = 5                    # number of components per patch
    gSig = [4, 4]            # expected half size of neurons in pixels
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
    ssub = 2                 # spatial subsampling during initialization
    tsub = 2                 # temporal subsampling during intialization

    # parameters for component evaluation
    opts_dict = {'fnames': fnames,
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

    opts.change_params(params_dict=opts_dict);
# %% RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0). If you want to have
    # deconvolution within each patch change params.patch['p_patch'] to a
    # nonzero value

    #opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

# %% ALTERNATE WAY TO RUN THE PIPELINE AT ONCE
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
    #  cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #  cnm1.fit_file(motion_correct=True)

# %% plot contours of found components
    Cns = local_correlations_movie_offline(mc.mmap_file[0],
                                           remove_baseline=True, window=1000, stride=1000,
                                           winSize_baseline=100, quantil_min_baseline=10,
                                           dview=dview)
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0
    # cnm.estimates.plot_contours(img=Cn)
    # plt.title('Contour plots of found components')
#%% save results
    cnm.estimates.Cn = Cn
    #cnm.save(fname_new[:-5]+'_5_5_K_8_init.hdf5')

# %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm2 = cnm.refit(images, dview=dview)
    # %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 1.6  # signal to noise ratio for accepting a component
    rval_thr = 0.8  # space correlation threshold for accepting a component
    cnn_thr = 0.99  # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

    cnm2.params.set('quality', {'decay_time': decay_time,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': False,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest})
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    # %% PLOT COMPONENTS
    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

    # %% VIEW TRACES (accepted and rejected)

    if display_images:
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components_bad)
        
    #%%
    #sort = np.argsort(cnm2.estimates.SNR_comp[cnm2.estimates.idx_components_bad])[::-1]
    #%% update object with selected components
    #cnm2.estimates.select_components(use_object=True)
    #%% Extract DF/F values
    #cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    #%% Show final traces
    #cnm2.estimates.view_components(img=Cn)
    #%%
    cnm2.estimates.Cn = Cn
    cnm2.save(cnm2.mmap_file[:-5] + 'non_rigid_K_5.hdf5')
    end = time()
    print(end-start)
    print(cnm2.estimates.A.shape)
    print(len(cnm2.estimates.idx_components))
    #%% reconstruct denoised movie (press q to exit)
    # if display_images:
    #     rec = cnm2.estimates.play_movie(images, q_max=99.9, gain_res=1,frame_range=slice(0, 1000, 1),
    #                               magnification=0.5,
    #                               bpx=0, 
    #                               include_bck=False, 
    #                               display=False)  # background not shown


    # #%%
    # from caiman.source_extraction.cnmf.cnmf import load_CNMF
    # cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8.hdf5')
    # cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/3000/memmap_pw_rigid_True_d1_796_d2_512_d3_1_order_C_frames_3000_non_rigid_K_5.hdf5')
    # cnm2 = load_CNMF('/media/nel/storage/fiola/R6_20200210T2100/3000/memmap_pw_rigid_True_d1_796_d2_512_d3_1_order_C_frames_3000_non_rigid_K_5.hdf5')
    #%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
# if __name__ == "__main__":
#     main()
