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
                    level=logging.INFO)# %%
def main():
    pass # For compatibility between running under Spyder and the CLI

# %%  download and list all files to be processed
    from time import time
    start = time()
    fnames = ['/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000.hdf5']
    # your list of files should look something like this
    logging.info(fnames)
#%% original 
    fr = 15  # frame rate (Hz)
    decay_time = 0.4  # approximate length of transient event in seconds
    gSig = (4, 4)  # expected half size of neurons
    p = 1  # order of AR indicator dynamics
    min_SNR = 2   # minimum SNR for accepting new components
    ds_factor = 2  # spatial downsampling factor (increases speed but may lose some fine structure)
    gnb = 2  # number of background components
    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
    mot_corr = True  # flag for online motion correction
    pw_rigid = True  # flag for pw-rigid motion correction (slower but potentially more accurate)
    max_shifts_online = np.ceil(5).astype('int')  # maximum allowed shift during motion correction
    sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
    rval_thr = 0.85  # soace correlation threshold for candidate components
    # set up some additional supporting parameters needed for the algorithm
    # (these are default values but can change depending on dataset properties)
    init_batch = 100  # number of frames for initialization (presumably from the first file)
    K = 20  # initial number of components
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
                   'K': K,
                   'epochs': epochs,
                   'max_shifts_online': max_shifts_online,
                   'pw_rigid': pw_rigid,
                   'dist_shape_update': True,
                   'min_num_trial': 10,
                   'show_movie': show_movie}
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
# %% fit online
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()
    #cnm.save(os.path.splitext(fnames[0])[0]+'_caiman_online_results_dec_lag_v3.14.hdf5')


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
    T_track = 1e3*np.array(cnm.t_online) - T_motion - T_detect - T_shapes
    plt.figure()
    plt.stackplot(np.arange(len(T_motion)), T_motion, T_track, T_detect, T_shapes)
    plt.legend(labels=['motion', 'tracking', 'detect', 'shapes'], loc=2)
    plt.title('Processing time allocation')
    plt.xlabel('Frame #')
    plt.ylabel('Processing time [ms]')
    plt.show()
    
    end  = time()
#%%
base_file = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/k53_"
saveDict = {"T_motion":T_motion,"T_detect":T_detect,"T_shapes": T_shapes, "T_track": T_track}
np.save(base_file+ "cm_1024_nnls_time.npy", saveDict, allow_pickle=True)
#%% RUN IF YOU WANT TO VISUALIZE THE RESULTS (might take time)
    c, dview, n_processes = \
        cm.cluster.setup_cluster(backend='local', n_processes=None,
                                 single_thread=False)
    if opts.online['motion_correct']:
        shifts = cnm.estimates.shifts[-cnm.estimates.C.shape[-1]:]
        if not opts.motion['pw_rigid']:
            memmap_file = cm.motion_correction.apply_shift_online(images, shifts,
                                                        save_base_name='MC')
        else:
            mc = cm.motion_correction.MotionCorrect(fnames, dview=dview,
                                                    **opts.get_group('motion'))

            mc.y_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
            mc.x_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
            memmap_file = mc.apply_shifts_movie(fnames, rigid_shifts=False,
                                                save_memmap=True,
                                                save_base_name='MC')
    else:  # To do: apply non-rigid shifts on the fly
        memmap_file = images.save(fnames[0][:-4] + 'mmap')
    cnm.mmap_file = memmap_file


    #cnm = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mov_R2_20190219T210000_caiman_online_results_v3.0_new_params.hdf5')
    memmap_file = '/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_.mmap'
    Yr, dims, T = cm.load_memmap(memmap_file)

    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    min_SNR = 2.0  # peak SNR for accepted components (if above this, acept)
    rval_thr = 0.85  # space correlation threshold (if above this, accept)
    use_cnn = False  # use the CNN classifier
    min_cnn_thr = 0.99  # if cnn classifier predicts below this value, reject
    cnn_lowest = 0.1  # neurons with cnn probability lower than this value are rejected

    cnm.params.set('quality',   {'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': use_cnn,
                                'min_cnn_thr': min_cnn_thr,
                                'cnn_lowest': cnn_lowest})

    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    cnm.estimates.Cn = Cn
    cnm.save(os.path.splitext(fnames[0])[0]+'_caiman_online_results_v3.8.hdf5')

    dview.terminate()
    end  = time()

#%%
aa = np.where(np.array(cnm.time_neuron_added)[:, 1] < 10000)[0]
bb = cnm.estimates.idx_components
print(len(np.intersect1d(aa, bb)))
#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
