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
from metrics import metric

#%% files for processing
n_neurons = ['1', '2', 'many'][0]

if n_neurons in ['1', '2']:
    movie_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/',
                   '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron',
                   '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/'][1]
    
    movie_lists = ['454597_Cell_0_40x_patch1_mc.tif', '456462_Cell_3_40x_1xtube_10A2_mc.tif',
                 '456462_Cell_3_40x_1xtube_10A3_mc.tif', '456462_Cell_5_40x_1xtube_10A5_mc.tif',
                 '456462_Cell_5_40x_1xtube_10A6_mc.tif', '456462_Cell_5_40x_1xtube_10A7_mc.tif', 
                 '462149_Cell_1_40x_1xtube_10A1_mc.tif', '462149_Cell_1_40x_1xtube_10A2_mc.tif',
                 '456462_Cell_4_40x_1xtube_10A4_mc.tif', '456462_Cell_6_40x_1xtube_10A10_mc.tif',
                 '456462_Cell_5_40x_1xtube_10A8_mc.tif', '456462_Cell_5_40x_1xtube_10A9_mc.tif', 
                 '462149_Cell_3_40x_1xtube_10A3_mc.tif', '466769_Cell_2_40x_1xtube_10A_6_mc.tif',
                 '466769_Cell_2_40x_1xtube_10A_4_mc.tif', '466769_Cell_3_40x_1xtube_10A_8_mc.tif']
    
    frate_all = np.array([400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 995.02, 400.8 ,
       400.8 , 400.8 , 400.8 , 400.8 , 995.02, 995.02, 995.02, 995.02])

    fnames = [os.path.join(movie_folder, file) for file in movie_lists[:16]]

    combined_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons',
                    '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons'][0]
    
    combined_lists = ['neuron0&1_x[1, -1]_y[1, -1].tif', 
                   'neuron0&1_x[2, -2]_y[2, -2].tif', 
                   'neuron1&2_x[4, -2]_y[4, -2].tif', 
                   'neuron1&2_x[6, -2]_y[8, -2].tif']
elif n_neurons == 'many':
    movie_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data'][0]
    
    movie_lists = ['demo_voltage_imaging_mc.hdf5', 
                   'FOV4_50um.hdf5']
#%%    
all_f1_scores = []
all_prec = []
all_rec = []
all_corr_subthr = []
all_thresh = []
all_snr = []
compound_f1_scores = []
compound_prec = []
compound_rec = []
v_sg = []
all_set = np.arange(0, 16)
test_set = np.array([2, 6, 10, 14])
training_set = np.array([ 0,  1,  3,  4,  5,  7,  8,  9, 11, 12, 13, 15])
mouse_fish_set = np.array([16, 17, 18])

#%%
for kk in test_set:
    file_set = [kk]
    name = movie_lists[file_set[0]]
    fr = frate_all[kk]
    fname = os.path.join(movie_folder, name)
    path_ROIs = os.path.join(movie_folder, name[:-7]+'_ROI.hdf5')

#%% dataset dependent parameters
    # dataset dependent parameters
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
        'fnames': fname,
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

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
    # Run correction
    mc.motion_correct(save_movie=True)

# %% MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries
    
    # memory map the file in order 'C'
    fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                                    add_to_mov=border_to_0, dview=dview)  # exclude border

# %% SEGMENTATION
    # create summary images
    img = mean_image(mc.mmap_file[0], window = 1000, dview=dview)
    img = (img-np.mean(img))/np.std(img)
    
    gaussian_blur = False        # Use gaussian blur when the quality of corr image(Cn) is bad
    Cn = local_correlations_movie_offline(mc.mmap_file[0], fr=int(fr), window=int(fr)*4, 
                                          stride=int(fr)*4, winSize_baseline=int(fr), 
                                          remove_baseline=True, gaussian_blur=gaussian_blur,
                                          dview=dview).max(axis=0)
    img_corr = (Cn-np.mean(Cn))/np.std(Cn)
    summary_image = np.stack([img, img, img_corr], axis=2).astype(np.float32) 
    
    #%% three methods for segmentation
    methods_list = ['manual_annotation',        # manual annotation needs user to prepare annotated datasets same format as demo ROIs 
                    'quick_annotation',         # quick annotation annotates data with simple interface in python
                    'maskrcnn' ]                # maskrcnn is a convolutional network trained for finding neurons using summary images
    method = methods_list[0]
    if method == 'manual_annotation':                
        with h5py.File(path_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]  
            if ROIs.shape[0] != 1:
                ROIs = ROIs[np.newaxis,:]

    elif method == 'quick_annotation':           
        ROIs = utils.quick_annotation(img, min_radius=4, max_radius=8)

    elif method == 'maskrcnn':                 # Important!! make sure install keras before using mask rcnn
        weights_path = download_model('mask_rcnn')
        ROIs = utils.mrcnn_inference(img=summary_image, weights_path=weights_path, display_result=True)
            
# %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=True, maxtasksperchild=1)

# %% parameters for trace denoising and spike extraction
    ROIs = ROIs                                   # region of interests
    index = list(range(len(ROIs)))                # index of neurons
    weights = None                                # reuse spatial weights 

    context_size = 30                             # number of pixels surrounding the ROI to censor from the background PCA
    censor_size = 8
    flip_signal = True                            # Important!! Flip signal or not, True for Voltron indicator, False for others
    hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
    threshold_method = 'adaptive_threshold'                   # 'simple' or 'adaptive_threshold'
    min_spikes= 10                                # minimal spikes to be found
    threshold = 3.5                               # threshold for finding spikes, increase threshold to find less spikes
    do_plot = False                               # plot detail of spikes, template for the last iteration
    ridge_bg= 0.5                               # ridge regression regularizer strength for background removement
    sub_freq = 20                                 # frequency for subthreshold extraction
    weight_update = 'ridge'                       # 'ridge' or 'NMF' for weight update
    
    opts_dict={'fnames': fname_new,
               'ROIs': ROIs,
               'index': index,
               'weights': weights,
               'context_size': context_size,
               'censor_size': censor_size,
               'flip_signal': flip_signal,
               'hp_freq_pb': hp_freq_pb,
               'threshold_method': threshold_method,
               'min_spikes':min_spikes,
               'threshold': threshold,
               'do_plot':do_plot,
               'ridge_bg':ridge_bg,
               'sub_freq': sub_freq,
               'weight_update': weight_update}

    opts.change_params(params_dict=opts_dict);          

#%% TRACE DENOISING AND SPIKE DETECTION
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)
    
    #idx = np.where(vpy.estimates['locality'] > 0)[0]
    #utils.view_components(vpy.estimates, img_corr, idx)
    
    
#%%
    for idx, k in enumerate(list(file_set)):
        indexes = vpy.estimates['spikes'].flatten()  
        name_traces = '/'.join(fnames[k].split('/')[:-2] + ['one_neuron_result', 
                                       fnames[k].split('/')[-1][:-7]+'_output.npz'])
        # F1 score
        dict1 = np.load(name_traces, allow_pickle=True)
        indexes = np.delete(indexes, np.where(indexes >= dict1['v_t'].shape[0])[0])
        
        dict1_v_sp_ = dict1['v_t'][indexes]
        v_sg.append(dict1['v_sg'])
            
        for i in range(len(dict1['sweep_time']) - 1):
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
        dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
        
        precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned\
                            = metric(dict1['sweep_time'], dict1['e_sg'], 
                                  dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                  dict1['v_sg'], dict1_v_sp_ , 
                                  dict1['v_t'], dict1['v_sub'],save=False)
        p = len(e_match)/len(v_spike_aligned)
        r = len(e_match)/len(e_spike_aligned)
        f = (2 / (1 / p + 1 / r))
    
        print(np.array(F1).round(2).mean())
        all_f1_scores.append(np.array(F1).round(2))
        all_prec.append(np.array(precision).round(2))
        all_rec.append(np.array(recall).round(2))
        compound_f1_scores.append(f)                
        compound_prec.append(p)
        compound_rec.append(r)
    print(f'average_F1:{np.mean([np.nanmean(fsc) for fsc in all_f1_scores])}')
    #print(f'average_sub:{np.nanmean(all_corr_subthr,axis=0)}')
    print(f'F1:{np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)}')
    print(f'prec:{np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)}'); 
    print(f'rec:{np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)}')
    print(f'average_compound_f1:{np.mean(np.array(compound_f1_scores)).round(3)}')
    print(f'compound_f1:{np.array(compound_f1_scores).round(2)}')
    print(f'compound_prec:{np.array(compound_prec).round(2)}')
    print(f'compound_rec:{np.array(compound_rec).round(2)}')
    
    
    save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result'
    np.savez(os.path.join(save_folder, 'spike_detection_volpy_'+movie_lists[kk][:-7]+'_output.npz'), estimates=vpy.estimates)
#%%
dict1 = {}
dict1['average_F1'] = np.mean([np.nanmean(fsc) for fsc in all_f1_scores])
dict1['F1'] = np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)
dict1['prec'] = np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)
dict1['rec'] = np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)
dict1['average_compound_f1'] = np.mean(np.array(compound_f1_scores)).round(3)
dict1['compound_f1'] = np.array(compound_f1_scores).round(2)
dict1['compound_prec'] = np.array(compound_prec).round(2)
dict1['compound_rec'] =  np.array(compound_rec).round(2)
dict1['snr'] = np.array(all_snr).round(2).T

np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/volpy_training.npy', dict1)

















