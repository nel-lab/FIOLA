#!/usr/bin/env python
"""
Demo pipeline for processing voltage imaging data. The processing pipeline
includes motion correction, memory mapping, segmentation, denoising and source
extraction. The demo shows how to construct the params, MotionCorrect and VOLPY 
objects and call the relevant functions. See inside for detail.

Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
author: @caichangjia
"""
import cv2
import glob
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
import matplotlib.pyplot as plt



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

# %%  Load demo movie and ROIs
#fnames = download_demo('demo_voltage_imaging.hdf5', 'volpy')  # file path to movie file (will download if not present)
#path_ROIs = download_demo('demo_voltage_imaging_ROIs.hdf5', 'volpy')  # file path to ROIs file (will download if not present)
movie_folder = ['/home/nel/data/voltage_data/Marton/456462/Cell_4/40x_1xtube_10A4/movie', #good
             '/home/nel/data/voltage_data/Marton/456462/Cell_6/40x_1xtube_10A10/movie', #not bad
             '/home/nel/data/voltage_data/Marton/456462/Cell_6/40x_1xtube_10A11/movie', # bad
             '/home/nel/data/voltage_data/Marton/466769/Cell_0/40x_1xtube_10A_1/movie', # bad
             '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A8/movie', # ok
             '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A9/movie', # not good
             '/home/nel/data/voltage_data/Marton/462149/Cell_3/40x_1xtube_10A3/movie', # good
             '/home/nel/data/voltage_data/Marton/462149/Cell_3/40x_1xtube_10A4/movie', # bad
             '/home/nel/data/voltage_data/Marton/466769/Cell_2/40x_1xtube_10A_6/movie',
             '/home/nel/data/voltage_data/Marton/466769/Cell_2/40x_1xtube_10A_4/movie',
             '/home/nel/data/voltage_data/Marton/466769/Cell_3/40x_1xtube_10A_8/movie'][10] 
fnames = [os.path.join(movie_folder, i) for i in sorted(os.listdir(movie_folder)) if '.tif' in i] 
#path_ROIs = '/home/nel/data/voltage_data/Marton/456462/Cell_3/40x_1xtube_10A2/movie/ROIs.hdf5'

#%% dataset dependent parameters
# dataset dependent parameters
fr = 1000                                       # sample rate of the movie
                                               
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

# %% start a cluster for parallel processing
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
# %% SEGMENTATION
# create summary images
folder = sorted(os.listdir(movie_folder)) 
mmap_file = [os.path.join(movie_folder, [file for file in folder if '_F_frames' in file][0])] 
fname_new = os.path.join(movie_folder, [file for file in folder if '_C_frames' in file][-1]) 
m = cm.load(fname_new)
img = mean_image(mmap_file[0], window = 1000, dview=dview)
img = (img-np.mean(img))/np.std(img)

gaussian_blur = False        # Use gaussian blur when the quality of corr image(Cn) is bad
Cn = local_correlations_movie_offline(mmap_file[0], fr=fr, window=fr*4, 
                                      stride=fr*4, winSize_baseline=fr, 
                                      remove_baseline=True, gaussian_blur=gaussian_blur,
                                      dview=dview).max(axis=0)
img_corr = (Cn-np.mean(Cn))/np.std(Cn)
summary_image = np.stack([img, img, img_corr], axis=2).astype(np.float32) 
plt.imshow(img)

#%% three methods for segmentation
methods_list = ['manual_annotation',        # manual annotation needs user to prepare annotated datasets same format as demo ROIs 
                'quick_annotation',         # quick annotation annotates data with simple interface in python
                'maskrcnn' ]                # maskrcnn is a convolutional network trained for finding neurons using summary images
method = methods_list[1]
if method == 'manual_annotation':                
    with h5py.File(path_ROIs, 'r') as fl:
        ROIs = fl['mov'][()]  

elif method == 'quick_annotation':           
    ROIs = utils.quick_annotation(img, min_radius=10, max_radius=20)

elif method == 'maskrcnn':                 # Important!! make sure install keras before using mask rcnn
    weights_path = download_model('mask_rcnn')
    ROIs = utils.mrcnn_inference(img=summary_image, size_range=[12, 22],
                                 weights_path=weights_path, display_result=True) # size parameter decides size range of masks to be selected
 
#%%
cm.movie(ROIs).save(os.path.join(movie_folder, 'ROIs.hdf5'))
# %% restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=2, single_thread=False, maxtasksperchild=1)

# %% parameters for trace denoising and spike extraction
ROIs = ROIs                                   # region of interests
index = list(range(len(ROIs)))                # index of neurons
weights = None                                # reuse spatial weights 

context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
flip_signal = True                            # Important!! Flip signal or not, True for Voltron indicator, False for others
hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
threshold_method = 'adaptive_threshold'                   # 'simple' or 'adaptive_threshold'
min_spikes= 10                                # minimal spikes to be found
threshold = 3.5                               # threshold for finding spikes, increase threshold to find less spikes
do_plot = False                               # plot detail of spikes, template for the last iteration
ridge_bg= 0.5                               # ridge regression regularizer strength for background removement
sub_freq = 20                                 # frequency for subthreshold extraction
weight_update = 'ridge'                       # 'ridge' or 'NMF' for weight update
n_iter = 2

opts_dict={'fnames': fname_new,
           'ROIs': ROIs,
           'index': index,
           'weights': weights,
           'context_size': context_size,
           'flip_signal': flip_signal,
           'hp_freq_pb': hp_freq_pb,
           'threshold_method': threshold_method,
           'min_spikes':min_spikes,
           'threshold': threshold,
           'do_plot':do_plot,
           'ridge_bg':ridge_bg,
           'sub_freq': sub_freq,
           'weight_update': weight_update,
           'n_iter': n_iter}

opts.change_params(params_dict=opts_dict);          

#%% TRACE DENOISING AND SPIKE DETECTION
vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
vpy.fit(n_processes=n_processes, dview=dview)

#%% visualization
display_images = True
if display_images:
    print(np.where(vpy.estimates['locality'])[0])    # neurons that pass locality test
    idx = np.where(vpy.estimates['locality'] > 0)[0]
    utils.view_components(vpy.estimates, img_corr, idx)

# %% STOP CLUSTER and clean up log files
cm.stop_server(dview=dview)
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
