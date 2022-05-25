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
from caiman.source_extraction.cnmf.utilities import get_file_size
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fnames', type=str, required=True)
parser.add_argument('--num_frames_init', type=int, required=True)
parser.add_argument('--K', type=int, required=True)
args = parser.parse_args()

#%%    
def run_caiman_init(fnames=None, num_frames_init=None, K=None, min_SNR2=None):
    fnames = '/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000.hdf5'
    num_frames_init = 0
    K = 8
    min_SNR2 = 2
    pw_rigid = False
    # for dandi data only
    if num_frames_init > 0:
        folder = fnames.rsplit('/', 1)[0] + f'/{num_frames_init}/'
        mov = cm.load(fnames, subindices=range(num_frames_init))
        fnames_init = folder + fnames.rsplit('/', 1)[1][:-5] + f'_init_{num_frames_init}.tif'
        mov.save(fnames_init)
        fnames = fnames_init
    
    # general pipeline
    print(fnames)
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
    max_shift_um = (12., 12.)       # maximum shift in umgg
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    # motion correction parameters
    #pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
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
    gnb = 2                  # number of global background components
    merge_thr = 0.85         # merging threshold, max correlation allowed
    rf = 15
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6          # amount of overlap between the patches in pixels
    #K = 5                    # number of components per patch
    gSig = [4, 4]            # expected half size of neurons in pixels
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
    #for min_SNR in [1.1, 1.3, 1.5, 1.8, 2.0]:
    #min_SNR2 = 2.0  # signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    cnn_thr = 0.15  # threshold for CNN based classifier
    cnn_lowest = 0.0 # neurons with cnn probability lower than this value are rejected
    
    cnm2.params.set('quality', {'decay_time': decay_time,
                                'min_SNR': min_SNR2,
                                'rval_thr': rval_thr,
                                'use_cnn': False,
                                'min_cnn_thr': cnn_thr,
                                'cnn_lowest': cnn_lowest})
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    print(len(cnm2.estimates.idx_components))
    # save_name = cnm2.mmap_file[:-5] + f'_v3.13.hdf5'
    # cnm2.save(save_name)
    # np.save(folder + f'min_snr_{min_SNR}_comp_{len(cnm2.estimates.idx_components)}.npy', cnm2.estimates.idx_components)
        
    #  PLOT COMPONENTS
    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components, display_numbers=False, cmap='gray')
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
    cnm2.estimates.view_components(img=Cn)
    #
    cnm2.mmap_F = f_F_mmap 
    cnm2.estimates.Cn = Cn
    cnm2.estimates.template = mc.total_template_rig
    cnm2.estimates.shifts = mc.shifts_rig
    save_name = cnm2.mmap_file[:-5] + '_v3.13.hdf5'
    
    timing['end'] = time()
    print(timing)
    cnm2.estimates.timing = timing
    cnm2.save(save_name)
    print(save_name)
    output_file = save_name
    # STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    plt.close('all')        

if __name__ == "__main__":
    run_caiman_init(fnames=args.fnames, num_frames_init=args.num_frames_init, K=args.K, min_SNR2=args.min_SNR2)
    print(args)