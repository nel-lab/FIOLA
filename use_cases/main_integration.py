#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:30:11 2020
Use hals algorithm to refine spatial components extracted by rank-1 nmf. 
Use nnls with gpu for signal extraction 
Use online template matching (saoz) for spike extraction
@author: @agiovann, @caichangjia, @cynthia
"""
#%%
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
from scipy.optimize import nnls    
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from skimage import measure
from sklearn.decomposition import NMF

from fiola.utilities import signal_filter, to_3D, to_2D, bin_median, hals, select_masks, normalize, nmf_sequential, OnlineFilter, match_spikes_greedy, compute_F1
from fiola.metrics import metric
from time import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

#%% files for processing
string = os.getcwd().split('/')
BASE_FOLDER = os.path.join('/'+string[1], string[2], 'NEL-LAB Dropbox/NEL/Papers/')

n_neurons = ['1', '2', 'many', 'test'][3]

if n_neurons in ['1', '2']:
    movie_folder = [os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/one_neuron/'),
                   os.path.join(BASE_FOLDER, 'VolPy/Marton/video_small_region/')][0]
    
    movie_lists = ['454597_Cell_0_40x_patch1', '456462_Cell_3_40x_1xtube_10A2',
                 '456462_Cell_3_40x_1xtube_10A3', '456462_Cell_5_40x_1xtube_10A5',
                 '456462_Cell_5_40x_1xtube_10A6', '456462_Cell_5_40x_1xtube_10A7', 
                 '462149_Cell_1_40x_1xtube_10A1', '462149_Cell_1_40x_1xtube_10A2',
                 '456462_Cell_4_40x_1xtube_10A4', '456462_Cell_6_40x_1xtube_10A10',
                 '456462_Cell_5_40x_1xtube_10A8', '456462_Cell_5_40x_1xtube_10A9', 
                 '462149_Cell_3_40x_1xtube_10A3', '466769_Cell_2_40x_1xtube_10A_6',
                 '466769_Cell_2_40x_1xtube_10A_4', '466769_Cell_3_40x_1xtube_10A_8', 
                 '09282017Fish1-1', '10052017Fish2-2', 'Mouse_Session_1']
    
    frate_all = np.array([400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 995.02, 400.8 ,
       400.8 , 400.8 , 400.8 , 400.8 , 995.02, 995.02, 995.02, 995.02,
       300.  , 300.  , 400.  ])
    
    fnames = [os.path.join(movie_folder, file) for file in movie_lists]
    
    combined_folder = [os.path.join(BASE_FOLDER, 'VolPy/Marton/overlapping_neurons'),
                    '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons'][0]
    
    combined_lists = ['neuron0&1_x[1, -1]_y[1, -1].tif', 
                   'neuron0&1_x[2, -2]_y[2, -2].tif', 
                   'neuron1&2_x[4, -2]_y[4, -2].tif', 
                   'neuron1&2_x[6, -2]_y[8, -2].tif']
elif n_neurons == 'many':
    movie_folder = [os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/original_data/multiple_neurons'), 
                    os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/simulation/test')][0]
   
    movie_lists = ['demo_voltage_imaging_mc.hdf5', 
                   'FOV4_50um_mc.hdf5',
                   '06152017Fish1-2_portion.hdf5', 
                   'FOV4_50um.hdf5', 
                   'viola_sim1_1.hdf5']
    
elif n_neurons == 'test':
    movie_folder = [os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/simulation/overlapping/viola_sim3_1'),
                    os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/simulation/overlapping/viola_sim3_2'),
                    os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/simulation/overlapping/viola_sim3_3'),
                    os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/simulation/overlapping/viola_sim3_5'),
                    os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/simulation/overlapping/viola_sim3_18'),
                    os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/simulation/non_overlapping/viola_sim2_7'),
                    os.path.join(BASE_FOLDER, 'VolPy_online/data/voltage_data/simulation/non_overlapping/viola_sim5_7')][0]
    movie_lists = ['viola_sim3_1.hdf5',
                   'viola_sim3_2.hdf5',
                   'viola_sim3_3.hdf5',
                   'viola_sim3_5.hdf5',
                   'viola_sim3_18.hdf5',   # overlapping 8 neurons
                   'viola_sim2_7.hdf5',
                   'viola_sim5_7.hdf5']   # non-overlapping 50 neurons

#%% Choosing datasets
#movie = base_folder + dataset + dataset + ".hdf5"
#rois = base_folder + dataset + dataset + "_ROIs.hdf5"
if n_neurons == '1':
    file_set = [-3]
    name = movie_lists[file_set[0]]
    belong_Marton = True
    if ('Fish' in name) or ('Mouse' in name):
        belong_Marton = False
    frate = frate_all[file_set[0]]
    mov = imread(os.path.join(movie_folder, name, name+'_mc.tif'))
    with h5py.File(os.path.join(movie_folder, name, name+'_ROI.hdf5'),'r') as h5:
        mask = np.array(h5['mov'])
    if mask.shape[0] != 1:
        mask = mask[np.newaxis,:]

elif n_neurons == '2':
    file_set = [0, 1]
    name = combined_lists[0]
    frate = 400
    mov = imread(os.path.join(combined_folder, name))
    with h5py.File(os.path.join(combined_folder, name[:-4]+'_ROIs.hdf5'),'r') as h5:
       mask = np.array(h5['mov'])

elif n_neurons == 'many':
    # name = movie_lists[1]
    frate = 400
    with h5py.File(movie,'r') as h5:
       mov = np.array(h5['mov'])
    with h5py.File(rois,'r') as h5:
       mask = np.array(h5['mov'])
    #mask = mask.transpose([0, 2, 1])
       
elif n_neurons == 'test':
    name = movie_lists[0]
    frate = 400
    with h5py.File(os.path.join(movie_folder, name),'r') as h5:
       mov = np.array(h5['mov'])
    with h5py.File(os.path.join(movie_folder, 'viola', 'ROIs_gt.hdf5'),'r') as h5:
       mask = np.array(h5['mov'])
    
"""
with h5py.File(os.path.join(movie_folder, name),'r') as h5:
       mov = np.array(h5['mov'])
    with h5py.File(os.path.join(movie_folder, 'viola', 'ROIs_gt.hdf5'),'r') as h5:
       mask = np.array(h5['mov'])
"""    

# Preliminary processing
# Remove border pixel of the motion corrected movie
"""
border_pixel = 2
mov[:, :border_pixel, :] = mov[:, border_pixel:border_pixel + 1, :]
mov[:, -border_pixel:, :] = mov[:, -border_pixel-1:-border_pixel, :]
mov[:, :, :border_pixel] = mov[:, :, border_pixel:border_pixel + 1]
mov[:, :, -border_pixel:] = mov[:, :, -border_pixel-1:-border_pixel]
"""
    
# original movie !!!!
flip = True
if flip == True:
    y = to_2D(-mov).copy()
else:
    y = to_2D(mov).copy()
use_signal_filter = True   
if use_signal_filter:  
    y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
 
do_plot = True
if do_plot:
    plt.figure()
    plt.imshow(mov[0])
    plt.figure()
    if n_neurons == 'many' or 'test':
        plt.imshow(mask.sum(0))    
    else:            
        for i in range(mask.shape[0]):
            plt.imshow(mask[i], alpha=0.5)
# mask = mask.transpose([0,2,1])


# Use nmf sequentially to extract all neurons in the region
num_frames_init = mov.shape[0]
use_rank_one_nmf = False
hals_movie = ['hp_thresh', 'hp', 'orig'][0]
hals_orig = False
if hals_movie=='hp_thresh':
    y_input = np.maximum(y_filt[:num_frames_init], 0).T
elif hals_movie=='hp':
    y_input = y_filt[:num_frames_init].T
else:
    y_input = -y[:num_frames_init].T
    hals_orig = True

if use_rank_one_nmf:
    y_seq = y_filt[:num_frames_init,:].copy()
    std = [np.std(y_filt[:, np.where(mask_2D[i]>0)[0]].mean(1)) for i in range(len(mask_2D))]
    seq = np.argsort(std)[::-1]
    print(f'sequence of rank1-nmf: {seq}')
    W, H = nmf_sequential(y_seq, mask=mask, seq=seq, small_mask=True)
    nA = np.linalg.norm(H)
    H = H/nA
    W = W*nA
else:
    mask_2D = to_2D(mask)
    nA = np.linalg.norm(mask_2D)
    H = mask_2D/nA
    W = (y_input.T@H.T)   


# Use hals to optimize masks
#from nmf_support import hals_init_spikes
# to make fish work one needs semi-nmf, input is high-passed movie
update_bg = True
use_spikes = False
semi_nmf = False

H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y.shape[1],1)) / y.shape[1],
                         np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, semi_nmf=semi_nmf,
                         update_bg=update_bg, use_spikes=use_spikes, hals_orig=hals_orig, fr=frate)

if do_plot:
    plt.figure();plt.imshow(H_new.sum(axis=1).reshape(mov.shape[1:], order='F'));plt.colorbar()
    plt.figure();plt.imshow(b.reshape(mov.shape[1:], order='F'));plt.colorbar()

if update_bg:
     H_new = np.hstack((H_new, b))

# normalization will enable gpu-nnls extracting bg signal 
H_new = H_new / norm(H_new, axis=0)

np.save(base_folder + dataset + dataset + "_H_new_full", H_new)
#%% Motion correct and use NNLS to extract signals
# You can skip rank 1-nmf, hals step if H_new is saved
#np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data//multiple_neurons/FOV4_50um_H_new.npy', H_new)
# H_new = np.load(os.path.join(movie_folder, name[:-8]+'_H_new.npy'))
use_GPU = True
use_batch = True
if use_GPU:
    mov_in = mov 
    Ab = H_new.astype(np.float32)
    template = bin_median(mov_in, exclude_nans=False)
    center_dims =(template.shape[0], template.shape[1])
    #center_dims =(128, 128)
    if not use_batch:
        from fiola.pipeline_gpu import Pipeline, get_model
        b = mov[0].reshape(-1, order='F')
        x0 = nnls(Ab,b)[0][:,None]
        AtA = Ab.T@Ab
        Atb = Ab.T@b
        n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
        theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
        theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
        mc0 = mov_in[0:1,:,:, None]
        
        model = get_model(template, center_dims, Ab, 30, ms_h=0, ms_w=0)
        model.compile(optimizer='rmsprop', loss='mse')
        spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_1, mov_in[:,:,:20000])
        traces_viola = spike_extractor.get_traces(10000)

    else:
    #FOR BATCHES:
        batch_size = 200
        num_components = Ab.shape[-1]
        template = bin_median(mov_in, exclude_nans=False)
        b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')
        x0=[]
        for i in range(batch_size):
            x0.append(nnls(Ab,b[:,i])[0])
        x0 = np.array(x0).T
        AtA = Ab.T@Ab
        Atb = Ab.T@b
        n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
        theta_2 = (Atb/n_AtA).astype(np.float32)

        from fiola.batch_gpu import Pipeline, get_model
        model_batch = get_model(template, center_dims, Ab, num_components, batch_size, ms_h=0, ms_w=0)
        model_batch.compile(optimizer = 'rmsprop',loss='mse')
        mc0 = mov_in[0:batch_size, :, :, None][None, :]
        x_old, y_old = np.array(x0[None,:]), np.array(x0[None,:])
        spike_extractor = Pipeline(model_batch, x_old, y_old, mc0, theta_2, mov_in, num_components, batch_size)
        spikes_gpu = spike_extractor.get_traces(mov.shape[0])
        traces_viola = []
        for spike in spikes_gpu:
            for i in range(batch_size):
                traces_viola.append([spike[:,:,i]])

    traces_viola = np.array(traces_viola).squeeze().T
    trace_all = traces_viola.copy()

else:
    #Use nnls to extract signal for neurons or not and filter
    fe = slice(0,None)
    trace_nnls = np.array([nnls(H_new,yy)[0] for yy in (-y)[fe]])
    trace_all = trace_nnls.T.copy() 


#%% Viola spike extraction, result is in the estimates object
if True:    
    trace = trace_all[:].copy()
    if len(trace.shape) == 1:
        trace = trace[None, :]
    saoz = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                  detrend=True, flip=True, 
                                  frate=frate, thresh_range=[2.8, 5.0], 
                                  adaptive_threshold=True, online_filter_method='median_filter',
                                  template_window=2, filt_window=15, minimal_thresh=2.8, mfp=0.1, step=2500, do_plot=False)
    saoz.fit(trace[:, :20000], num_frames=trace.shape[1])
    for n in range(20000, trace.shape[1]):
        saoz.fit_next(trace[:, n: n+1], n)
    saoz.compute_SNR()
    saoz.reconstruct_signal()
    print(f'thresh:{saoz.thresh}')
    print(f'SNR: {saoz.SNR}')
    print(f'Mean_SNR: {np.array(saoz.SNR).mean()}')
    print(f'Spikes based on mask sequence: {(saoz.index>0).sum(1)}')
    estimates = saoz
    estimates.spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz.index])
    weights = H_new.reshape((mov.shape[1], mov.shape[2], H_new.shape[1]), order='F')
    weights = weights.transpose([2, 0, 1])
    estimates.weights = weights

    #SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/data/voltage_data/ephys_voltage/10052017Fish2-2'
    #save_path = os.path.join(SAVE_FOLDER, 'viola', f'viola_update_bg_{update_bg}_use_spikes_{use_spikes}')
    #np.save(save_path, estimates)    
    
    
#%%
    plt.plot(normalize(saoz.trace[0]))    
    plt.plot(normalize(saoz.t_d[0]))
    plt.plot(normalize(saoz.t0[0]))
    plt.plot(normalize(saoz.t_sub[0]))
    #plt.plot(normalize(saoz.t0[0]-saoz.t_sub[0])); 
    plt.plot(normalize(saoz.t_s[0]))
    plt.xlim([0, 10000])

    plt.plot(normalize(saoz.t0[0]-saoz.t_sub[0]))
    plt.plot(normalize(saoz.t0[0]))
    
#%%
    tt = saoz.t0[0].copy()
    tt.shape
    t_new = []
    step = 5000
    for i in range(np.ceil(len(tt)/step).astype(np.int16)):
        if (i+1)*5000 > len(tt):
            ttt = tt[i*5000:]
        else:
            ttt = tt[i*5000:(i+1)*5000]
        t_new.append(normalize(ttt))
    plt.plot(np.hstack(t_new))

#%%
    np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction/viola_sim5_7_nnls_result.npy', trace_all)
    
    
#%%
    ROOT_FOLDER = movie_folder
    gt_path = os.path.join(ROOT_FOLDER, name, name+'_output.npz')
    dict1 = np.load(gt_path, allow_pickle=True)
    length = dict1['v_sg'].shape[0]    
    mode = 'viola'    
    if mode == 'viola':
        """
        #vi_folder = os.path.join(ROOT_FOLDER, name, 'viola')
        #vi_files = sorted([file for file in os.listdir(vi_folder) if 'filt_window' not in file and 'v2.1' in file])# and '24000' in file])
        #if len(vi_files) == 0:
        #    vi_files = sorted([file for file in os.listdir(vi_folder) if 'v2.0' in file and 'thresh_factor' in file])# and '24000' in file])
        #print(f'files number: {len(vi_files)}')
        #if len(vi_files) != 1:
            raise Exception('file number greater than 1')
            vi_files = [file for file in vi_files if '15000' in file]
        vi_file = vi_files[0]
        vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
        """
        #vi_spatial = saoz.H_new.copy()
        vi = saoz
        vi_temporal = vi.t_s.copy().flatten()
        vi_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vi.index])[0]
        #thr.append(vi.thresh_factor[0])
        
        n_cells = 1
        vi_result = {'F1':[], 'precision':[], 'recall':[]}        
        rr = {'F1':[], 'precision':[], 'recall':[]}        
        vi_spikes = np.delete(vi_spikes, np.where(vi_spikes >= dict1['v_t'].shape[0])[0])        
        
        dict1_v_sp_ = dict1['v_t'][vi_spikes]
    
    elif mode == 'volpy':
        v_folder = os.path.join(ROOT_FOLDER, name, 'volpy')
        v_files = sorted([file for file in os.listdir(v_folder)])
        print(f'files number: {len(v_files)}')
        v_file = v_files[0]
        v = np.load(os.path.join(v_folder, v_file), allow_pickle=True).item()
        
        v_spatial = v['weights'][0]
        v_temporal = v['ts'][0]
        v_spikes = v['spikes'][0]        
        v_spikes = np.delete(v_spikes, np.where(v_spikes >= dict1['v_t'].shape[0])[0])
        dict1_v_sp_ = dict1['v_t'][v_spikes]
     
    if 'Cell' in name:
        for i in range(len(dict1['sweep_time']) - 1):
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
        dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
    
    from viola.metrics import metric
    precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned, spnr\
                        = metric(name, dict1['sweep_time'], dict1['e_sg'], 
                              dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                              vi_temporal, dict1_v_sp_ , 
                              dict1['v_t'], dict1['v_sub'],init_frames=20000, save=False)
        
    p = len(e_match)/len(v_spike_aligned)
    r = len(e_match)/len(e_spike_aligned)
    f = (2 / (1 / p + 1 / r))


#%%
plt.plot(dict1['e_t'], dict1['e_sg'])
plt.vlines(dict1['e_sp'], -30, 20)

#%%
plt.plot(dict1['v_t'], vi_temporal)
plt.plot(vi.thresh[0])

#%%
vi_temporal = vi.t.copy().flatten()
plt.plot(dict1['v_t'], vi_temporal)
plt.vlines(dict1['e_sp'], np.max(vi_temporal), np.max(vi_temporal)*1.1)
#%%
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    from viola.running_statistics import non_symm_median_filter
    tt = saoz.t0[0].copy()
    m_15 = median_filter(tt, 15)
    m_13 = median_filter(tt, 13)
    """
    w = [8, 4]
    mm = tt.copy()
    for i in range(len(tt)):
        if i > w[0] & i < len(tt) - w[1]:
            mm[i] = np.median(tt[i - w[0] : i + w[1] + 1])
    """
    mm = non_symm_median_filter(tt, [8, 4])        
    #s_15 = savgol_filter(tt, 15, polyorder=1)
    #s_9 = savgol_filter(tt, 9, polyorder=1)
    plt.plot(tt); 
    plt.plot(m_15); plt.plot(m_13);plt.plot(mm); plt.xlim([0, 3000])
    
    
    
#%% Load simulation groundtruth
    import scipy.io
    vi_result_all = []
    gt_files = [file for file in os.listdir(movie_folder) if 'SimResults' in file]
    gt_file = gt_files[0]
    gt = scipy.io.loadmat(os.path.join(movie_folder, gt_file))
    gt = gt['simOutput'][0][0]['gt']
    spikes = gt['ST'][0][0][0]

    
#%% Compute F1 score for spike extraction
    n_cells = spikes.shape[0]
    rr = {'F1':[], 'precision':[], 'recall':[]}
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        s2 = estimates.spikes[idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
        F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
        rr['F1'].append(F1)
        rr['precision'].append(precision)
        rr['recall'].append(recall)  
        
    fig, ax =  plt.subplots(1,3)
    ax[0].boxplot(rr['F1']); plt.title('F1')
    ax[1].boxplot(rr['precision']); plt.title('precision')
    ax[2].boxplot(rr['recall']); plt.title('recall')


#%%
    
result = {}
for ff in [[7, 3], [6, 3], [5, 3], [4, 3], [3, 3]]:
    for tem in [2, 0]:
        trace = trace_all[:].copy()
        if len(trace.shape) == 1:
            trace = trace[None, :]
        saoz = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                      detrend=True, flip=True, 
                                      frate=frate, thresh_range=[2.8, 5.0], 
                                      adaptive_threshold=True, online_filter_method='median_filter',
                                      template_window=tem, filt_window=ff, minimal_thresh=3, mfp=0.1, step=2500, do_plot=False)
        saoz.fit(trace[:, :10000], num_frames=trace.shape[1])
        for n in range(10000, trace.shape[1]):
            saoz.fit_next(trace[:, n: n+1], n)
        saoz.compute_SNR()
        saoz.reconstruct_signal()
        print(f'thresh:{saoz.thresh}')
        print(f'SNR: {saoz.SNR}')
        print(f'Mean_SNR: {np.array(saoz.SNR).mean()}')
        print(f'Spikes based on mask sequence: {(saoz.index>0).sum(1)}')
        estimates = saoz
        estimates.spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz.index])
        weights = H_new.reshape((mov.shape[1], mov.shape[2], H_new.shape[1]), order='F')
        weights = weights.transpose([2, 0, 1])
        estimates.weights = weights        
        
        t_range = [10000, 75000]
        n_cells = spikes.shape[0]
        rr = {'F1':[], 'precision':[], 'recall':[]}
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s1 = s1[np.logical_and(s1>=t_range[0], s1<t_range[1])]
            s2 = estimates.spikes[idx]
            s2 = s2[np.logical_and(s2>=t_range[0], s2<t_range[1])]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)  
            
        fig, ax =  plt.subplots(1,3)
        ax[0].boxplot(rr['F1']); plt.title('F1')
        ax[1].boxplot(rr['precision']); plt.title('precision')
        ax[2].boxplot(rr['recall']); plt.title('recall')
        print(np.mean(rr['F1']))
        
        result[f'filt_window{ff}, template_window{tem}'] = np.mean(rr['F1'])
        
        print(result)
        print('**************************************')
    
#%%
    step = 2500
    plt.plot(saoz.t_s[0])
    for idx, tt in enumerate(saoz.thresh[0]):
        if idx == 0:
            plt.hlines(tt, 0, 30000)
        else:
            plt.hlines(tt, 30000 + (idx -1) * step, 30000 + idx * step)
    
#%%
ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron'
select = np.array(range(1))[:]
names = movie_lists.copy()
f1_scores = []                
prec = []
rec = []
thr = []
for idx, name in enumerate(np.array(names)[select]):
    gt_path = os.path.join(ROOT_FOLDER, name, name+'_output.npz')
    dict1 = np.load(gt_path, allow_pickle=True)
    length = dict1['v_sg'].shape[0]    
    
    vi_temporal = saoz.t_s.copy()
    vi_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz.index])[0]
    #thr.append(vi.thresh_factor[0])
    
    n_cells = 1
    vi_result = {'F1':[], 'precision':[], 'recall':[]}        
    rr = {'F1':[], 'precision':[], 'recall':[]}        
    vi_spikes = np.delete(vi_spikes, np.where(vi_spikes >= dict1['v_t'].shape[0])[0])        
    dict1_v_sp_ = dict1['v_t'][vi_spikes]

    if 'Cell' in name:
        for i in range(len(dict1['sweep_time']) - 1):
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
        dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
    
    precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned\
                        = metric(name, dict1['sweep_time'], dict1['e_sg'], 
                              dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                              dict1['v_sg'], dict1_v_sp_ , 
                              dict1['v_t'], dict1['v_sub'],init_frames=20000, save=False)
        
    p = len(e_match)/len(v_spike_aligned)
    r = len(e_match)/len(e_spike_aligned)
    f = (2 / (1 / p + 1 / r))

    f1_scores.append(f)                
    prec.append(p)
    rec.append(r)
print(f1_scores)    


#%%
m = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/Mouse_Session_1/volpy/volpy_Session_adaptive_threshold_4_ridge_bg_0.01.npy' , allow_pickle=True)
#m = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/10052017Fish2-2/volpy/volpy_registere_adaptive_threshold_4_ridge_bg_0.01.npy' , allow_pickle=True)
#m = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/09282017Fish1-1/volpy/volpy_registere_adaptive_threshold_4_ridge_bg_0.01.npy', allow_pickle=True)
plt.figure(); plt.plot(normalize(m.item()['ts'][2]))  # 2,0,7
#plt.plot(normalize(ts))
plt.plot(normalize(saoz.t_s[0]))
#plt.plot(normalize(aa))
#%%
plt.plot(saoz.t_s[0]);
plt.hlines(saoz.thresh[0,0], 0, 20000)



    
#%%    
##############################################################################################################
    
#%% fix decay use df/f
    y = trace_all[0].copy()
    box_pts = 100
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    plt.plot(y_smooth)    
    
    yy = y/y_smooth
    plt.plot(yy[100:79000])
    
#%%
    tt = saoz.t_s[0].copy()
    spikes = estimates.spikes[0]
    
    t_rec = np.zeros(trace.shape)
    spk_avg = []
    t_curve = np.zeros(tt.shape)
    t_curve[spikes[0]] = tt[spikes[0]].mean()
    for i in range(len(spikes)):
        if i > 0:
            #t_curve[spikes[i]] = np.percentile(tt[spikes[np.max((0, i-100)) : i]], 80)
            t_curve[spikes[i]] = tt[spikes[i]]
    plt.plot(tt)
    #plt.plot(saoz.t_rec[0])
    plt.plot(t_curve)
    
            
    
#%%
        
        
    for idx in range(trace.shape[0]):
        if spikes.size > 0:
            t_rec[idx, spikes] = 1
            t_rec[idx] = np.convolve(t_rec[idx], np.flip(PTA[idx]), 'same')   
    
    
    
    
#%%
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')
popt, pcov = curve_fit(func, xdata, ydata)
#array([ 2.55423706,  1.35190947,  0.47450618])
plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    
    
    #%%
    vpy = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation/test/volpy_viola_sim1_1_adaptive_threshold.npy', allow_pickle=True).item()        
    idx = 0
    plt.plot(normalize(vpy['t'][idx]))
    plt.plot(normalize(saoz_5000.t0[np.argsort(seq)][idx]))
    
    #%%
    trace = trace_all[:].copy()
    saoz_5000 = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                      detrend=True, flip=True, 
                                      frate=frate, thresh_range=[2.8, 5.0], 
                                      filt_window=15, mfp=0.1)
    saoz_5000.fit(trace[:, :5000], num_frames=trace.shape[1])
    for n in range(5000, trace.shape[1]):
        saoz_5000.fit_next(trace[:, n: n+1], n)
    saoz_5000.compute_SNR()
    saoz_5000.reconstruct_signal()
#%%
    trace = trace_all[:].copy()
    saoz_20000 = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                      detrend=True, flip=True, 
                                      frate=frate, thresh_range=[2.8, 5.0], 
                                      mfp=0.2)
    saoz_20000.fit(trace[:, :20000], num_frames=trace.shape[1])
    saoz_20000.compute_SNR()
    saoz_20000.reconstruct_signal()
    
    
    #%% I want to see distribution of thresh 
    x = saoz_5000.thresh_factor.flatten()[:50]
    y = saoz_20000.thresh_factor.flatten()[:50]
    #plt.scatter(x+np.random.rand(50)/50 , y+np.random.rand(55)/50) 
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)
    x_pred = np.arange(2.5, 4, 0.1)[:, np.newaxis]    
    y_pred = lr.predict(x_pred)
    plt.scatter(x+np.random.rand(50)/50 , y+np.random.rand(50)/50) 
    plt.plot(x_pred, y_pred)    
    
    plt.figure(); plt.hist(x); plt.hist(y, color='r'); plt.legend(['5000', '20000']);plt.title('distribution of threshold')
    # Conclusion: running offline with 20000 frames yield more consistent thresh distribution result
    
    #%% I want to see trace produced by 5000 and 20000
    idx = 1
    x1 = saoz_5000.t_s[idx]
    plt.plot(x1)
    x2 = saoz_20000.t_s[idx]
    plt.plot(x2)
    # trace does not look bad to me
    
    #%%
    for idx in range(1,4):
        plt.figure(); plt.plot(saoz_5000.t[idx]); plt.plot(saoz_5000.t_sub[idx]); #plt.plot(saoz_5000.t_d[idx])  
    #%%
    s1 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_5000.index])
    s2 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_20000.index])

    n_cells = 50
    from match_spikes import match_spikes_greedy, compute_F1
    rr = {'F1':[], 'precision':[], 'recall':[], 'TP':[], 'FP':[], 'FN':[]}
    for idx in range(n_cells):
        ss1 = spikes[seq][idx].flatten()
        ss2 = s1[idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(ss1, ss2, max_dist=4)
        F1, precision, recall, TP, FP, FN = compute_F1(ss1, ss2, idx1_greedy, idx2_greedy)   
        for keys, values in rr.items():
            rr[keys].append(eval(keys))
    
    fig, ax = plt.subplots(1, 3)
    plt.suptitle('viola initialized 5000 frames')
    ax[0].boxplot(rr['F1']); ax[0].set_title('F1')
    ax[1].boxplot(rr['precision']); ax[1].set_title('precision')
    ax[2].boxplot(rr['recall']); ax[2].set_title('recall')
    plt.tight_layout()
    
    print(f'Mean: {np.array(rr["F1"]).mean()}, Median: {np.median(np.array(rr["F1"]))}')
    
    rr1 = rr.copy()
    
    #%%
    n_cells = 50
    from match_spikes import match_spikes_greedy, compute_F1
    rr = {'F1':[], 'precision':[], 'recall':[], 'TP':[], 'FP':[], 'FN':[]}
    for idx in range(n_cells):
        ss1 = spikes[seq][idx].flatten()
        ss2 = s2[idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(ss1, ss2, max_dist=4)
        F1, precision, recall, TP, FP, FN = compute_F1(ss1, ss2, idx1_greedy, idx2_greedy)   
        for keys, values in rr.items():
            rr[keys].append(eval(keys))
    
    fig, ax = plt.subplots(1, 3)
    plt.suptitle('viola initialized 5000 frames')
    ax[0].boxplot(rr['F1']); ax[0].set_title('F1')
    ax[1].boxplot(rr['precision']); ax[1].set_title('precision')
    ax[2].boxplot(rr['recall']); ax[2].set_title('recall')
    plt.tight_layout()
    
    print(f'Mean: {np.array(rr["F1"]).mean()}, Median: {np.median(np.array(rr["F1"]))}')
    
    rr2 = rr.copy()
    
    #%% see spikes
    idx = 1
    x1 = saoz_5000.t0[idx]
    x1_sub = saoz_5000.t_sub[idx]
    plt.plot(x1)
    plt.plot(x1_sub)
    x2 = saoz_20000.t0[idx]
    x2_sub = saoz_20000.t_sub[idx]
    plt.plot(x2)
    plt.plot(x2_sub)
    plt.legend(['5000', '5000_sub', '20000', '20000_sub'])
    h = np.max([np.max(x1), np.max(x2)])
    plt.scatter(s1[idx], np.ones((len(s1[idx])))*h, color='blue')
    plt.scatter(s2[idx], np.ones((len(s2[idx])))*h+0.5, color='green')
    plt.vlines(spikes[seq][idx], 0, h+0.5)
    
    print(f'{[{keys:rr1[keys][idx]} for keys in rr1]}')    
    print(f'{[{keys:rr2[keys][idx]} for keys in rr2]}')    
    
    
    
    
    #%%
    # threshold is not that important
    # online filter suspicious to me
    # online filter is not good, it produces more FP
    
    #%%
    from match_spikes import match_spikes_greedy, compute_F1
    s2 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_20000.index])
        
    filt_windows = np.arange(9, 25, 2)
    mfps = np.arange(0.005,0.05, 0.005)
    rr_all = {}
    for filt_window in filt_windows:
        for mfp in mfps:
            trace = trace_all[:].copy()
            saoz_5000 = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                              detrend=True, flip=True, 
                                              frate=frate, thresh_range=[2.8, 5.0], 
                                              filt_window=filt_window, mfp=mfp)
            saoz_5000.fit(trace[:, :5000], num_frames=trace.shape[1])
            for n in range(5000, trace.shape[1]):
                saoz_5000.fit_next(trace[:, n: n+1], n)
            #saoz_5000.compute_SNR()
            #saoz_5000.reconstruct_signal()
            s1 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_5000.index])
            n_cells = 50
            rr = {'F1':[], 'precision':[], 'recall':[], 'TP':[], 'FP':[], 'FN':[]}
            for idx in range(n_cells):
                ss1 = spikes[seq][idx].flatten()
                ss2 = s1[idx]
                idx1_greedy, idx2_greedy = match_spikes_greedy(ss1, ss2, max_dist=4)
                F1, precision, recall, TP, FP, FN = compute_F1(ss1, ss2, idx1_greedy, idx2_greedy)   
                for keys, values in rr.items():
                    rr[keys].append(eval(keys))
            
            """
            fig, ax = plt.subplots(1, 3)
            plt.suptitle('viola initialized 5000 frames')
            ax[0].boxplot(rr['F1']); ax[0].set_title('F1')
            ax[1].boxplot(rr['precision']); ax[1].set_title('precision')
            ax[2].boxplot(rr['recall']); ax[2].set_title('recall')
            plt.tight_layout()
            """
            rr_all[(filt_window, mfp)]= rr.copy()
    
    #%%
    F1_mean = [np.median(np.array(rr['F1'])) for rr in rr_all.values()]
    plt.plot(filt_windows, F1_mean)    
    
    #%%
    mat = np.zeros((len(filt_windows), len(mfps)))
    for idx_x, filt_window in enumerate(filt_windows):
        for idx_y, mfp in enumerate(mfps):
            rr = rr_all[(filt_window, mfp)]
            mat[idx_x, idx_y] = np.median(np.array(rr['F1']))
    
    plt.title('Heatmap for F1 score'); plt.imshow(mat); plt.colorbar(); plt.yticks(list(range(len(filt_windows))), filt_windows)
    plt.xticks(list(range(len(mfps))), mfps); plt.xlabel('mfps'); plt.ylabel('filt_windows')
    


    #%% use adaptive threshold  
    from match_spikes import match_spikes_greedy, compute_F1
    s2 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_20000.index])
        
    filt_windows = np.array([15])
    rr_all = {}
    for filt_window in filt_windows:
        trace = trace_all[:].copy()
        saoz_5000 = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                          detrend=True, flip=True, 
                                          frate=frate, thresh_range=[2.8, 5.0], 
                                          filt_window=filt_window, mfp=0.2, do_plot=False)
        saoz_5000.fit(trace[:, :5000], num_frames=trace.shape[1])
        for n in range(5000, trace.shape[1]):
            saoz_5000.fit_next(trace[:, n: n+1], n)
        #saoz_5000.compute_SNR()
        #saoz_5000.reconstruct_signal()
        s1 = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz_5000.index])
        n_cells = 50
        rr = {'F1':[], 'precision':[], 'recall':[], 'TP':[], 'FP':[], 'FN':[]}
        for idx in range(n_cells):
            ss1 = spikes[seq][idx].flatten()
            ss2 = s1[idx]
            idx1_greedy, idx2_greedy = match_spikes_greedy(ss1, ss2, max_dist=4)
            F1, precision, recall, TP, FP, FN = compute_F1(ss1, ss2, idx1_greedy, idx2_greedy)   
            for keys, values in rr.items():
                rr[keys].append(eval(keys))
        
        fig, ax = plt.subplots(1, 3)
        plt.suptitle('viola adaptive threshold initialized 5000 frames')
        ax[0].boxplot(rr['F1']); ax[0].set_title('F1')
        ax[1].boxplot(rr['precision']); ax[1].set_title('precision')
        ax[2].boxplot(rr['recall']); ax[2].set_title('recall')
        plt.tight_layout()
        
        rr_all[(filt_window, mfp)]= rr.copy()      

#%%
    mat = np.zeros((len(filt_windows), len([1])))
    for idx_x, filt_window in enumerate(filt_windows):
        rr = rr_all[(filt_window, mfp)]
        mat[idx_x, 0] = np.median(np.array(rr['F1']))
    
    plt.title('Heatmap for F1 score adaptive threshold'); plt.imshow(mat); plt.colorbar(); plt.yticks(list(range(len(filt_windows))), filt_windows)
    plt.xticks(list(range(len(mfps))), mfps); plt.xlabel('mfps'); plt.ylabel('filt_windows')
        
    # two problems, online filtering problem and spike extraction problem
    # filtering problem: small filt_window (9 frames as default, that means 4 frames lag) works much worse than (15 frames)
    # threshold problem: adaptive threshold works well; mfp the smaller the better (0.01 best, 0.2 default), not suitable in this case
    # after changing filt_window size to 15, changing to adaptive threshold method, F1 score 0.935 
    #%% Load VolPy result
    name_estimates = ['demo_voltage_imaging_estimates.npy', 'FOV4_50um_estimates.npz', 
                      '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_06152017Fish1-2.npy',
                      '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_IVQ32_S2_FOV1.npy', 
                      '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_FOV4_35um.npy' ]
    #name_estimates = [os.path.join(movie_folder, name) for name in name_estimates]
    estimates = np.load(name_estimates[4], allow_pickle=True).item()
    
    #%% Check neurons
    idx = 5
    idx_volpy = seq[idx]
    idx = np.where(seq==idx_volpy)[0][0]
    spikes_online = list(set(saoz.index[idx]) - set([0]))
    #plt.figure();plt.imshow(H_new[:,idx].reshape(mov.shape[1:], order='F'));plt.colorbar()
    plt.figure();plt.imshow(H_new[:,idx].reshape(mov.shape[1:], order='F'))
    plt.figure()
    plt.plot(saoz.t0[idx], color='blue', label=f'online_{len(spikes_online)}')
    plt.plot(normalize(saoz.t[idx]), color='red', label=f'online_t_s')
    plt.plot(normalize(estimates['t'][idx_volpy]), color='orange', label=f'volpy_{estimates["spikes"][idx_volpy].shape[0]}')
    plt.plot(normalize(saoz.t_rec[idx]), color='red', label=f'online_t_rec')
    plt.plot(saoz.t_sub[idx], color='red', label=f'online_t_sub')
    plt.vlines(spikes_online, -1.6, -1.2, color='blue', label='online_spikes')
    plt.vlines(estimates['spikes'][idx_volpy], -1.8, -1.4, color='orange', label='volpy_spikes')
    print(len(list(set(saoz.index[idx]))))
    plt.legend()   

    #%% Traces
    idx_list = np.where((saoz.index_track> 50))[0]
    idx_volpy  = seq[idx_list]
    #idx_volpy = np.where(np.array([len(estimates['spikes'][k]) for k in range(len(estimates['spikes']))])>30)[0]
    #idx_list = np.array([np.where(seq==idx_volpy[k])[0][0] for k in range(idx_volpy.size)])
    length = idx_list.size
    fig, ax = plt.subplots(idx_list.size,1)
    colorsets = plt.cm.tab10(np.linspace(0,1,10))
    colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]
    scope=[0, 20000]
    
    for n, idx in enumerate(idx_list):
        idx_volpy = seq[idx]
        ax[n].plot(np.arange(scope[1]), normalize(estimates['t'][idx_volpy]), 'c', linewidth=0.5, color='orange', label='volpy')
        ax[n].plot(np.arange(scope[1]), normalize(saoz.t_s[idx, :scope[1]]), 'c', linewidth=0.5, color='blue', label='viola')
        #ax[n].plot(np.arange(20000), normalize(saoz.t_s[idx, :20000]), 'c', linewidth=0.5, color='red', label='viola')
        #ax[n].plot(normalize(saoz.t_s[idx]), color='blue', label=f'viola_t_s')
    
        spikes_online = list(set(saoz.index[idx]) - set([0]))
        ax[n].vlines(spikes_online, 1.2, 1.5, color='red', label='viola_spikes')
        ax[n].vlines(estimates['spikes'][idx_volpy], 1.5, 1.8, color='black', label='volpy_spikes')
    
        if n<length-1:
            ax[n].get_xaxis().set_visible(False)
            ax[n].spines['right'].set_visible(False)
            ax[n].spines['top'].set_visible(False) 
            ax[n].spines['bottom'].set_visible(False) 
            ax[n].spines['left'].set_visible(False) 
            ax[n].set_yticks([])
        
        if n==length-1:
            ax[n].legend()
            ax[n].spines['right'].set_visible(False)
            ax[n].spines['top'].set_visible(False)  
            ax[n].spines['left'].set_visible(True) 
            ax[n].set_xlabel('Frames')
        ax[n].set_ylabel('o')
        ax[n].get_yaxis().set_visible(True)
        ax[n].yaxis.label.set_color(colorsets[np.mod(n,9)])
        ax[n].set_xlim(scope)
            
    plt.tight_layout()
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/whole_FOV/traces.pdf')
 
    #%% Spatial contours
    Cn = mov[0]
    vmax = np.percentile(Cn, 99)
    vmin = np.percentile(Cn, 5)
    plt.figure()
    plt.imshow(Cn, interpolation='None', vmax=vmax, vmin=vmin, cmap=plt.cm.gray)
    plt.title('Neurons location')
    d1, d2 = Cn.shape
    #cm1 = com(mask.copy().reshape((N,-1), order='F').transpose(), d1, d2)
    colors='yellow'
    for n, idx in enumerate(idx_list):
        contours = measure.find_contours(mask[seq[idx]], 0.5)[0]
        plt.plot(contours[:, 1], contours[:, 0], linewidth=1, color=colorsets[np.mod(n,9)])
        #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/whole_FOV/spatial_masks.pdf')
    
    #%%
    saoz.seq = seq
    saoz.H = H_new
    np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/viola_FOV4_35um', saoz)
#%% Extract spikes and compute F1 score
if n_neurons in ['1', '2']:
    for idx, k in enumerate(list(file_set)):
        name_traces = '/'.join(fnames[k].split('/')[:-2] + ['one_neuron_result', 
                                   fnames[k].split('/')[-1][:-7]+'_output.npz'])
        # F1 score
        dict1 = np.load(name_traces, allow_pickle=True)
        length = dict1['v_sg'].shape[0]
        
        trace = trace_all[seq[idx]:seq[idx]+1, :].copy()
        #trace = dc_blocked[np.newaxis,:].copy()
        saoz = SignalAnalysisOnlineZ(do_scale=True, thresh_range=[2.8, 5], frate=frate, robust_std=False, detrend=True, flip=True)
        #saoz.fit(trr_postf[:20000], len())
        #trace=dict1['v_sg'][np.newaxis, :]
        saoz.fit(trace_all[:20000], trace_all.shape[1])
        for n in range(20000, trace_all.shape[1]):
            saoz.fit_next(trace_all[:, n:n+1], n)
        saoz.compute_SNR()
        indexes = np.array(list(set(saoz.index[0]) - set([0])))
        thresh = saoz.thresh_factor[0, 0]
        snr = saoz.SNR
        indexes = np.delete(indexes, np.where(indexes >= dict1['v_t'].shape[0])[0])
        
        dict1_v_sp_ = dict1['v_t'][indexes]
        v_sg.append(dict1['v_sg'])
            
        if belong_Marton:
            for i in range(len(dict1['sweep_time']) - 1):
                dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
        
        precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned\
                            = metric(dict1['sweep_time'], dict1['e_sg'], 
                                  dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                  dict1['v_sg'], dict1_v_sp_ , 
                                  dict1['v_t'], dict1['v_sub'],save=False, belong_Marton=belong_Marton)
        
        p = len(e_match)/len(v_spike_aligned)
        r = len(e_match)/len(e_spike_aligned)
        f = (2 / (1 / p + 1 / r))
print(np.array(F1).round(2).mean())
all_f1_scores.append(np.array(F1).round(2))
all_prec.append(np.array(precision).round(2))
all_rec.append(np.array(recall).round(2))
all_thresh.append(thresh)
all_snr.append(snr)
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
print(f'snr:{np.array(all_snr).round(2)}')
dict2 = {}
dict2['trace'] = saoz.trace
dict2['indexes'] = sorted(indexes)
dict2['t_s'] = saoz.t_s
dict2['snr'] = saoz.SNR
dict2['sub'] = saoz.sub
dict2['template'] = saoz.PTA
dict2['thresh'] = saoz.thresh
dict2['thresh_factor'] = saoz.thresh_factor
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result'
np.save(os.path.join(save_folder, 'spike_detection_saoz_'+ name[:-7]  +'_output'), dict2)
        
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

np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/saoz_training.npy', dict1)
   
#%%
if n_neurons == '1':
#plt.plot(dict1['v_t'], saoz.trace.flatten())
plt.plot(dict1['v_t'], normalize(saoz.trace.flatten()))
plt.plot(dict1['e_t'], normalize(dict1['e_sg'])/10)
plt.vlines(dict1_v_sp_, -2, -1)    


elif n_neurons == '2':
show_frames = 80000 
#%matplotlib auto
t1 = normalize(trace_all[0])
t2 = normalize(trace_all[1])
t3 = normalize(v_sg[0])
t4 = normalize(v_sg[1])
plt.plot(dict1['v_t'][:show_frames], t1[:show_frames] + 0.5, label='neuron1')
plt.plot(dict1['v_t'][:show_frames], t2[:show_frames], label='neuron2')
plt.plot(dict1['v_t'][:show_frames], t3[:show_frames] + 0.5, label='gt1')
plt.plot(dict1['v_t'][:show_frames], t4[:show_frames], label='gt2')
#plt.plot(dict1['e_t'], t4-3, label='ele')
#plt.vlines(dict1['e_sp'], -3, -2.5, color='black')
plt.legend()
  
#%%   
print(f'average_F1:{np.mean([np.nanmean(fsc) for fsc in all_f1_scores])}')
print(f'F1:{np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)}')
print(f'prec:{np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)}'); 
print(f'rec:{np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)}')

#%% corr
corr = []
#file_set = np.arange(0,8)
names = []
file_set = np.array([7])
for idx, k in enumerate(list(file_set)):
    name_traces = '/'.join(fnames[k].split('/')[:-2] + ['one_neuron_result', 
                               fnames[k].split('/')[-1][:-7]+'_output.npz'])
    # F1 score
    dict1 = np.load(name_traces, allow_pickle=True)
    corr.append(np.corrcoef(dict1['e_sub'][:20000], dict1['v_sub'][:20000])[0, 1])
    names.append(fnames[k].split('/')[-1][:13])
    
#%%
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

plt.bar(file_set, corr)    
plt.xticks(file_set, names, rotation='vertical', fontsize=4)  
plt.ylabel('corrcoef')  
np.array(corr).mean()
plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Backup/2020-06-29-Jannelia-meeting/bar_plot.pdf')
#%%
plt.figure()
plt.plot(normalize(dict1['e_sub'][:20000]), label='ephys sub')
plt.plot(normalize(dict1['v_sub'][:20000]), label='volpy sub')
#np.corrcoef(dict1['e_sub'][:20000], dict1['v_sub'][:20000])
plt.legend()
plt.xlim([0,3000])
plt.ylim([-1,1])
plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Backup/2020-06-29-Jannelia-meeting/dataset_7.pdf')

#%% reconstruction
shape = mask.shape[1:]
scope = [0, 1000]
saoz.reconstruct_movie(H_new, shape, scope)

#%%
mov_new = saoz.mov_rec.copy()
play(mov_new, fr=400, q_min=0.01)        
#mv_bl = mv.computeDFF(secsWindow=0.1)[0]

#%%
y = to_2D(-mov[scope[0]:scope[1]]).copy()
y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
mov_detrend = to_3D(y_filt, shape=mov[scope[0]:scope[1]].shape)
play(mov_detrend, q_min=80, q_max=98)

#%%
from viola.spikepursuit import denoise_spikes
t = W.flatten().copy()
t0 = t.copy()
recon = np.zeros((y_input.T.shape[0], y_input.T.shape[1]+1))
recon[:,0] = 1
recon[:,1:] = y_input.copy().T

ts, spikes, t_rec, templates, low_spikes, thresh = denoise_spikes(t, 
        window_length=5, fr=400,  hp_freq=1, threshold_method='simple', pnorm=0.5, 
        threshold=3.5, min_spikes=10, do_plot=True)

for _ in range(3):
    tr = t_rec.copy()
    
    from sklearn.linear_model import Ridge
    #Ri = Ridge(alpha=9212983, fit_intercept=True, solver='lsqr')
    Ri = Ridge(alpha=0, fit_intercept=True, solver='lsqr')

    Ri.fit(recon, tr)
    weights = Ri.coef_
    weights[0] = Ri.intercept_
    
    plt.figure(); plt.imshow(weights[1:].reshape([30,30],order='F')); plt.show()
    # update the signal            
    t = np.matmul(recon, weights)
    t = t - np.mean(t)
    
    """
    b = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr').fit(Ub, t).coef_
    t = t - np.matmul(Ub, b)
    """
    # correct shrinkage
    t = np.double(t * np.mean(t0[spikes]) / np.mean(t[spikes]))
    
    # estimate spike times
    ts, spikes, t_rec, templates, low_spikes, thresh = denoise_spikes(t, 
            window_length=8, fr=400,  hp_freq=1, threshold_method='simple', pnorm=0.5, 
            threshold=3.5, min_spikes=10, do_plot=True)
    
#%%
t = trace_all.copy()
t = -signal_filter(t, 15, 400)
plt.plot(t[-1])

#%%
t1 = t.copy()   # classic
t2 = t.copy()   # orig, use_spikes True
t3 = t.copy()    # hp_thresh, use_spikes True
t4 = t.copy()
t5 = t.copy()

#%%
plt.plot(normalize(t1[0])); plt.plot(normalize(t2[0])); 
plt.plot(normalize(t3[0])); 
plt.plot(normalize(t4[0])); plt.plot(normalize(t5[0]))

plt.plot(normalize(t4[0])); plt.plot(normalize(t1[0])); 
plt.legend(['classic', 'original use_spikes True', 'hp_thresh, use_spikes True'])
#%%
t = trace_all.flatten()
t = -signal_filter(t, 1/3, 400)
ts, spikes, t_rec, templates, low_spikes, thresh = denoise_spikes(t, 
            window_length=8, fr=400,  hp_freq=1, threshold_method='simple', pnorm=0.5, 
            threshold=3.5, min_spikes=10, do_plot=True)
plt.plot(ts)

t = trace_all.flatten()
t = -signal_filter(t, 5, 400)
plt.plot(t)



#%%
    plt.figure(); plt.plot(trace_nnls[:,0]); plt.figure(); plt.plot(traces_viola)
    plt.figure(); plt.plot(trace_nnls[:,0]); plt.plot(traces_viola[0])