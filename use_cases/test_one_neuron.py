#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:30:11 2020
Use hals algorithm to refine spatial components extracted by rank-1 nmf. 
Use nnls with gpu for signal extraction 
Use sao for spike extraction
@author: @agiovann, @caichangjia, @cynthia
"""
#%%
from caiman_functions import signal_filter, to_3D, to_2D
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import nnls    
from signal_analysis_online import SignalAnalysisOnline, SignalAnalysisOnlineZ
from sklearn.decomposition import NMF
from metrics import metric
from nmf_support import hals, select_masks, normalize
from skimage.io import imread
import h5py
from running_statistics import OnlineFilter
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
                 '466769_Cell_2_40x_1xtube_10A_4_mc.tif', '466769_Cell_3_40x_1xtube_10A_8_mc.tif', 
                 '09282017Fish1-1_mc.tif', '10052017Fish2-2_mc.tif', 'Mouse_Session_1_mc.tif']
    
    frate_all = np.array([400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 995.02, 400.8 ,
       400.8 , 400.8 , 400.8 , 400.8 , 995.02, 995.02, 995.02, 995.02,
       300.  , 300.  , 400.  ])
    
    fnames = [os.path.join(movie_folder, file) for file in movie_lists]
    
    combined_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons',
                    '/home/andrea/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/overlapping_neurons'][0]
    
    combined_lists = ['neuron0&1_x[1, -1]_y[1, -1].tif', 
                   'neuron0&1_x[2, -2]_y[2, -2].tif', 
                   'neuron1&2_x[4, -2]_y[4, -2].tif', 
                   'neuron1&2_x[6, -2]_y[8, -2].tif']
elif n_neurons == 'many':
    movie_folder = ['/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data'][1]
    
    movie_lists = ['demo_voltage_imaging_mc.hdf5', 
                   'FOV4_50um_mc.hdf5']
    
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
#%% Choosing datasets
for i in training_set:
    if n_neurons == '1':
        file_set = [i]
        name = movie_lists[file_set[0]]
        belong_Marton = True
        if ('Fish' in name) or ('Mouse' in name):
            belong_Marton = False
        frate = frate_all[file_set[0]]
        mov = imread(os.path.join(movie_folder, name))
        with h5py.File(os.path.join(movie_folder, name[:-7]+'_ROI.hdf5'),'r') as h5:
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
        name = movie_lists[0]
        frate = 400
        with h5py.File(os.path.join(movie_folder, name),'r') as h5:
           mov = np.array(h5['mov'])
        with h5py.File(os.path.join(movie_folder, name[:-8]+'_ROIs.hdf5'),'r') as h5:
           mask = np.array(h5['mov'])
    
    
    #%%
    # original movie
    y = to_2D(-mov).copy()     
    y_filt = signal_filter(y.T,freq = 1/3, fr=frate).T
    y_filt = y_filt 
    
    do_plot = True
    if do_plot:
        plt.figure()
        plt.imshow(mov[0])
        plt.figure()
        if n_neurons == 'many':
            plt.imshow(mask.sum(0))    
        else:            
            for i in range(mask.shape[0]):
                plt.imshow(mask[i], alpha=0.5)
    
    #%% Use nmf sequentially to extract all neurons in the region
    num_frames_init = 20000
    y_seq = y_filt[:num_frames_init,:].copy()
    W_tot = []
    H_tot = []
    mask_2D = to_2D(mask)
    std = [np.std(y_filt[:, np.where(mask_2D[i]>0)[0]].mean(1)) for i in range(len(mask_2D))]
    seq = np.argsort(std)[::-1]
    print(f'sequence of rank1-nmf: {seq}')
    
    for i in seq:
        model = NMF(n_components=1, init='nndsvd', max_iter=200, verbose=False)
        y_temp, _ = select_masks(y_seq, mov[:num_frames_init].shape, mask=mask[i])
        W = model.fit_transform(np.maximum(y_temp,0))
        H = model.components_
        y_seq = y_seq - W@H
        W_tot.append(W)
        H_tot.append(H)
        if do_plot:
            plt.figure();plt.plot(W);
            plt.figure();plt.imshow(H.reshape(mov.shape[1:], order='F'));plt.colorbar()
    H = np.vstack(H_tot)
    W = np.hstack(W_tot)
    #%% Use hals to optimize masks
    update_bg = False
    y_input = np.maximum(y_filt[:num_frames_init], 0)
    y_input =to_3D(y_input, shape=(num_frames_init,mov.shape[1],mov.shape[2]), order='F').transpose([1,2,0])
    
    H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y.shape[1],1)) / y.shape[1],
                                 np.random.rand(1,num_frames_init), bSiz=None, maxIter=3, 
                                 update_bg=update_bg, use_spikes=True)
    plt.close('all')
    if do_plot:
        for i in range(mask.shape[0]):
            plt.figure();plt.imshow(H_new[:,i].reshape(mov.shape[1:], order='F'));plt.colorbar()
        
    #%% Motion correct and use NNLS to extract signals
    #import tensorflow as tf
    
    use_GPU = False
    use_batch = False
    if use_GPU:
        mov_in = mov 
        Ab = H_new.astype(np.float32)
        template = np.median(mov_in, axis=0)
        center_dims = (template.shape[0], template.shape[1])
        if not use_batch:
    
            b = to_2D(mov).T[:, 0]
            x0 = nnls(Ab,b)[0][:,None]
            AtA = Ab.T@Ab
            Atb = Ab.T@b
            n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
            theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
            mc0 = mov_in[0:1,:,:, None]
    
            from pipeline_gpu import Pipeline, get_model
            model = get_model(template, center_dims, Ab, 30)
            model.compile(optimizer='rmsprop', loss='mse')
            spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, mov_in[:,:,:100000])
            traces_viola = spike_extractor.get_spikes(20000)
    
        else:
        #FOR BATCHES:
            batch_size = 20
            num_frames = 1800
            num_components = Ab.shape[-1]
    
            template = np.median(mov_in, axis=0)
            
            b = to_2D(mov).T[:, 0:batch_size]
            x0=[]
            for i in range(batch_size):
                x0.append(nnls(Ab,b[:,i])[0])
            x0 = np.array(x0).T
            AtA = Ab.T@Ab
            Atb = Ab.T@b
            n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
            theta_2 = (Atb/n_AtA).astype(np.float32)
    
            from batch_gpu import Pipeline, get_model
    
            model_batch = get_model(template, center_dims, Ab, num_components, batch_size)
            model_batch.compile(optimizer = 'rmsprop',loss='mse')
    
            mc0 = mov_in[0:batch_size, :, :, None][None, :]
            x_old, y_old = np.array(x0[None,:]), np.array(x0[None,:])
            spike_extractor = Pipeline(model_batch, x_old, y_old, mc0, theta_2, mov_in, num_components, batch_size)
            spikes_gpu = spike_extractor.get_spikes(1000)
            traces_viola = []
            for spike in spikes_gpu:
                for i in range(batch_size):
                    traces_viola.append([spike[:,:,i]])
    
        traces_viola = np.array(traces_viola).squeeze().T
        traces_viola = signal_filter(traces_viola,freq = 1/3, fr=frate).T
        traces_viola -= np.median(traces_viola, 0)[np.newaxis, :]
        traces_viola = -traces_viola.T
        trace_all = np.hstack([traces_viola,np.zeros((traces_viola.shape[0],1))])
    
    else:
        #Use nnls to extract signal for neurons or not and filter
        from running_statistics import OnlineFilter       
        from time import time
        fe = slice(0,None)
        if update_bg:
            trace_nnls = np.array([nnls(np.hstack((H_new, b)),yy)[0] for yy in (-y)[fe]])
        else:
            trace_nnls = np.array([nnls(H_new,yy)[0] for yy in (-y)[fe]])
        filter_method = ['offline', 'butter', 'dc_blocker'][2]
        if filter_method == 'offline':
            trace_nnls = signal_filter(trace_nnls.T,freq = 1/3, fr=frate).T
            trace_nnls -= np.median(trace_nnls, 0)[np.newaxis, :]
            trace_nnls = -trace_nnls.T
            trace_all = trace_nnls 
        elif filter_method == 'butter': # filter online
            freq = 1/3
            trace_all = trace_nnls.T
            onFilt = OnlineFilter(freq=freq, fr=frate, mode='high')
            trace_filt = np.zeros_like(trace_all)
            trace_filt[:,:20000] = onFilt.fit(trace_all[:,:20000])    
            time0 = time()
            for i in range(20000, trace_all.shape[-1]):
                trace_filt[:,i] = onFilt.fit_next(trace_all[:,i])
            print(time()-time0)
            trace_all -= np.median(trace_all.T, 0)[np.newaxis, :].T
            trace_all = -trace_filt
        elif filter_method == 'dc_blocker':
            trace = trace_nnls.T
            trace_filt = np.zeros(trace.shape)        
            for tp in range(trace.shape[1]):
                if tp > 0:
                    trace_filt[:, tp] = trace[:, tp] - trace[:, tp - 1] + 0.995 * trace_filt[:, tp - 1]
            trace_all = -trace_filt
            trace_all = trace_all - np.median(trace_all)
    
    #%%
    # idxv = 0
    # plt.plot(np.hstack([traces_viola[idxv],0])-trace_nnls[idxv])
    # plt.plot(np.hstack([traces_viola[idxv],0])-trace_nnls[idxv])
    # #%%
    # plt.plot(trace_nnls[idxv])
    # plt.plot(traces_viola[idxv])
    
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
            saoz = SignalAnalysisOnlineZ()
            #sao.fit(trr_postf[:20000], len())
            #trace=dict1['v_sg'][np.newaxis, :]
            saoz = SignalAnalysisOnlineZ(frate=frate, robust_std=False, do_scale=True)
            saoz.fit(trace_all[:, :20000], trace_all.shape[1])
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
    #plt.plot(dict1['v_t'], sao.trace.flatten())
    plt.plot(dict1['v_t'], normalize(sao.trace.flatten()))
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
