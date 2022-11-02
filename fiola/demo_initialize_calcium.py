#!/usr/bin/env python

"""
Demo for initializaing calcium spatial footprints. 
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
from caiman.source_extraction.cnmf.utilities import get_file_size


logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)
#%%    
def run_caiman_init(fnames, pw_rigid = True, max_shifts=[6, 6], gnb=2, K = 5, gSig = [4, 4]):
    """
    Run caiman for initialization.
    
    Parameters
    ----------
    fnames : string
        file name
    pw_rigid : bool, 
        flag to select rigid vs pw_rigid motion correction. The default is True.
    max_shifts: list
        maximum shifts allowed for x axis and y axis. The default is [6, 6].
    gnb : int
        number of background components. The default is 2.
    K : int
        number of components per patch. The default is 5.
    gSig : list
        expected half size of neurons in pixels. The default is [4, 4].

    Returns
    -------
    output_file : string
        file with caiman output

    """
    c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
    
    timing = {}
    timing['start'] = time()

    # dataset dependent parameters
    display_images = False
    
    fr = 30             # imaging rate in frames per second
    decay_time = 0.4    # length of a typical transient in seconds
    dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    # motion correction parameters
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
        'border_nan': 'copy',
    }
    
    opts = params.CNMFParams(params_dict=mc_dict)
    
    # Motion correction and memory mapping
    time_init = time()
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0)  # exclude borders
    #
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    #  restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    #
    f_F_mmap = mc.mmap_file[0]
    Cns = local_correlations_movie_offline(f_F_mmap,
                                       remove_baseline=True, window=1000, stride=1000,
                                       winSize_baseline=100, quantil_min_baseline=10,
                                       dview=dview)
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0
    # if display_images: 
       
    plt.imshow(Cn,vmax=0.5)
    #   parameters for source extraction and deconvolution
    p = 1                    # order of the autoregressive system
    merge_thr = 0.85         # merging threshold, max correlation allowed
    rf = 15
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6          # amount of overlap between the patches in pixels
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
    ssub = 2                     # spatial subsampling during initialization
    tsub = 2                     # temporal subsampling during intialization
    
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
    #  RUN CNMF ON PATCHES
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    #  COMPONENT EVALUATION
    min_SNR = 1.0  # signal to noise ratio for accepting a component
    rval_thr = 0.75  # space correlation threshold for accepting a component
    cnn_thr = 0.3  # threshold for CNN based classifier
    cnn_lowest = 0.0 # neurons with cnn probability lower than this value are rejected
    
    cnm.params.set('quality', {'decay_time': decay_time,
                           'min_SNR': min_SNR,
                           'rval_thr': rval_thr,
                           'use_cnn': False,
                           'min_cnn_thr': cnn_thr,
                           'cnn_lowest': cnn_lowest})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    print(len(cnm.estimates.idx_components))
    time_patch = time()
    #
    if display_images:
        cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)
    #
    cnm.estimates.select_components(use_object=True)
    # RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm2 = cnm.refit(images, dview=dview)
    time_end = time() 
    print(time_end- time_init)
    #  COMPONENT EVALUATION
    min_SNR = 1.1  # signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    cnn_thr = 0.15  # threshold for CNN based classifier
    cnn_lowest = 0.0 # neurons with cnn probability lower than this value are rejected
    
    cnm2.params.set('quality', {'decay_time': decay_time,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': False,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest})
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    print(len(cnm2.estimates.idx_components))
    
    #  PLOT COMPONENTS
    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
    #  VIEW TRACES (accepted and rejected)
    if display_images:
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components_bad)
    # update object with selected components
    cnm2.estimates.select_components(use_object=True)
    # Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    # Show final traces
    #cnm2.estimates.view_components(img=Cn)
    #
    cnm2.mmap_F = f_F_mmap 
    cnm2.estimates.Cn = Cn
    cnm2.estimates.template = mc.total_template_rig
    cnm2.estimates.shifts = mc.shifts_rig
    save_name = cnm2.mmap_file[:-5] + '_caiman_init.hdf5'
    
    timing['end'] = time()
    print(timing)
    cnm2.save(save_name)
    print(save_name)
    output_file = save_name
    # STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    plt.close('all')        
    return output_file
