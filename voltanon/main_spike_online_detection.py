#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 08:27:50 2020

@author: agiovann
"""

#%% import library
import os
import matplotlib.pyplot as plt
import numpy as np
from running_statistics import estimate_running_std
from signal_analysis_online import SignalAnalysisOnline
from metrics import metric
from visualization import plot_marton
#%%
base_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new',
               '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new'][1]
lists = ['454597_Cell_0_40x_patch1_output.npz', '456462_Cell_3_40x_1xtube_10A2_output.npz',
             '456462_Cell_3_40x_1xtube_10A3_output.npz', '456462_Cell_5_40x_1xtube_10A5_output.npz',
             '456462_Cell_5_40x_1xtube_10A6_output.npz', '456462_Cell_5_40x_1xtube_10A7_output.npz', 
             '462149_Cell_1_40x_1xtube_10A1_output.npz', '462149_Cell_1_40x_1xtube_10A2_output.npz', ]
file_list = [os.path.join(base_folder, file)for file in lists]

temp = file_list[0].split('/')[-1].split('_')

pr= []
re = []
F = []
sub = []
N_opt = [2,2,2,2,2,2]
all_f1_scores = []
all_prec = []
all_rec = []

all_corr_subthr = []

mode = ['online', 'minimum', 'percentile', 'v_sub', 'low_pass', 'double'][0]

for k in np.array(list(range(0, 8))):
    if (k == 6) or (k==3):
        thresh_height = None
    else:
        thresh_height = None
    dict1 = np.load(file_list[k], allow_pickle=True)
    img = dict1['v_sg']
    #img /= estimate_running_std(img, q_min=0.1, q_max=99.9)
    #std_estimate = np.diff(np.percentile(img,[75,25]))/100
    ### I comment it due to it will influence peak distribution
    #for i in range(len(dict1['sweep_time']) - 1):
    #    idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]), dict1['v_t']<dict1['sweep_time'][i+1][0])])[1]
    #    img[idx_to_rem] = np.random.normal(0,1,len(idx_to_rem))*std_estimate
    
    #for i in range(len(dict1['sweep_time']) - 1):
    #    idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]-1), dict1['v_t']<dict1['sweep_time'][i][-1]-0.85)])[1]
    #    img[idx_to_rem] = np.random.normal(0,1,len(idx_to_rem))*std_estimate
#    
#    sub_1 = estimate_subthreshold(img, thres_STD=5,  kernel_size=21)
#    all_corr_subthr.append([np.corrcoef(normalize(dict1['e_sub']),normalize(sub_1))[0,1],np.corrcoef(normalize(dict1['e_sub']),normalize(dict1['v_sub']))[0,1]])
    
    frate = 1/np.median(np.diff(dict1['v_t']))
    perc_window = 50
    perc_stride = 25
    if mode == 'v_sub':
        signal_subthr = dict1['v_sub']
    elif mode == 'online':
        signal_subthr = None
    elif mode == 'percentile':
        perc = np.array([np.percentile(el,20) for el in rolling_window(img.T[None,:], perc_window, perc_stride)])
        signal_subthr =  cv2.resize(perc, (1,img.shape[0]),cv2.INTER_CUBIC).squeeze()
    elif mode == 'minimum':
        minima = np.array([np.min(el) for el in rolling_window(img.T[None,:], 10, 5)])
        signal_subthr = cv2.resize(minima, (1,img.shape[0]),interpolation = cv2.INTER_CUBIC).squeeze()
    elif mode == 'low_pass':
        if ((k == 6) or (k==7)):
            signal_subthr = signal_filter(dict1['v_sg'], 15, fr=1000, order=5, mode='low')
        else:
            signal_subthr = signal_filter(dict1['v_sg'], 15, fr=400, order=5, mode='low')
    elif mode == 'double':
        if ((k == 6) or (k==7)):
            subthr1 = signal_filter(dict1['v_sg'], 10, fr=1000, order=5, mode='low')
        else:
            subthr1 = signal_filter(dict1['v_sg'], 10, fr=400, order=5, mode='low')
            
        perc = np.array([np.percentile(el,20) for el in rolling_window((img-subthr1).T[None,:], perc_window, perc_stride)])
#        signal_subthr = np.concatenate([np.zeros(15),perc,np.zeros(14)]) #cv2.resize(perc, (1,img.shape[0])).squeeze()
        subthr2 =  cv2.resize(perc, (1,img.shape[0]),cv2.INTER_CUBIC).squeeze()
        signal_subthr = subthr1 + subthr2        
    
    if signal_subthr is not None:    
        signal_no_subthr = img -  signal_subthr
#    signal_no_subthr = dict1['v_sg'] - dict1['v_sub']
    
    #indexes, erf, z_signal = find_spikes(img, signal_no_subthr=signal_no_subthr, 
    #                                     thres_STD=4.5, thres_STD_ampl=4, min_dist=1, 
    #                                     N=2, win_size=20000, stride=5000, 
    #                                     spike_before=3, spike_after=4, 
    #                                     bidirectional=False)
    
    #indexes = find_spikes_tm(img, signal_subthr, thresh_height)
    #indexes = find_spikes_rh(img, thresh_height)[0]
    #indexes = find_spikes_rh_online(img, thresh_height, window=10000, step=5000)
    img = img.astype(np.float32)
    sao = SignalAnalysisOnline(thresh_STD=None)
    #trace = img[np.newaxis, :]
    trace = np.array([img for i in range(50)])
    sao.fit(trace[:, :20000], num_frames=100000)
    for n in range(20000, img.shape[0]):
        sao.fit_next(trace[:, n: n+1], n)
    indexes = np.array((list(set(sao.index[0]) - set([0]))))  
    
    dict1_v_sp_ = dict1['v_t'][indexes]
    
    range_run = estimate_running_std(np.diff(img).squeeze(), 20000, 5000, q_min=0.000001, q_max=99.999999)
    std_run = estimate_running_std(np.diff(img).squeeze(), 20000, 5000, q_min=25, q_max=75)
    plt.plot(range_run/std_run)
    for i in range(len(dict1['sweep_time']) - 1):
        dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
    dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
#    
#    dict1_v_sp_ = dict1['v_sp']
#    precision, recall, F1, sub_corr, e_match, v_match, mean_time = metric(dict1['sweep_time'], dict1['e_sg'], 
#                                                                          dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
#                                                                          dict1['v_sg'], dict1['v_sp'], 
#                                                                          dict1['v_t'], dict1['v_sub'],save=False)
    precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned = metric(dict1['sweep_time'], dict1['e_sg'], 
                                                                          dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                                                          dict1['v_sg'], dict1_v_sp_ , 
                                                                          dict1['v_t'], dict1['v_sub'],save=False)
    
    
    
    all_f1_scores.append(np.nanmean(np.array(F1)).round(2))
    all_prec.append(np.nanmean(np.array(precision)).round(2))
    all_rec.append(np.nanmean(np.array(recall)).round(2))
     
    continue
    
    plot_marton(e_spike_aligned, v_spike_aligned, e_match, v_match, mean_time, precision, recall, F1, sub_corr)

#%%
print(f'average_F1:{np.mean([np.mean(fsc) for fsc in all_f1_scores])}')
print(f'average_sub:{np.mean(all_corr_subthr,axis=0)}')
print(f'F1:{np.array([np.mean(fsc) for fsc in all_f1_scores]).round(2)}')
print(f'prec:{np.array([np.mean(fsc) for fsc in all_prec]).round(2)}'); 
print(f'rec:{np.array([np.mean(fsc) for fsc in all_rec]).round(2)}')