#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:58:55 2020
files for loading and analyzing proccessed Marton's data
@author: caichangjia
"""
#%% import library
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from signal_analysis_online import SignalAnalysisOnline
from metrics import metric
import os
from spike_extraction_routines import estimate_subthreshold_signal, extract_exceptional_events, find_spikes
from running_statistics import estimate_running_std
#%%
base_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new',
               '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new'][1]
lists = ['454597_Cell_0_40x_patch1_output.npz', '456462_Cell_3_40x_1xtube_10A2_output.npz',
             '456462_Cell_3_40x_1xtube_10A3_output.npz', '456462_Cell_5_40x_1xtube_10A5_output.npz',
             '456462_Cell_5_40x_1xtube_10A6_output.npz', '456462_Cell_5_40x_1xtube_10A7_output.npz', 
             '462149_Cell_1_40x_1xtube_10A1_output.npz', '462149_Cell_1_40x_1xtube_10A2_output.npz', ]
file_list = [os.path.join(base_folder, file)for file in lists]

all_f1_scores = []
all_prec = []
all_rec = []
all_corr_subthr = []

mode = 'minimum'
mode = 'percentile'
#mode = 'v_sub'
#mode_spikes = 'anomaly'
#mode_spikes = 'exceptionality'
mode_spikes = 'multi_peak'
if mode_spikes == 'anomaly':
    N=0
    thres_STD = 5
    only_rising = False
    normalize_signal = True 
elif mode_spikes == 'exceptionality':
    N=1
    thres_STD = 3.5    
    only_rising = True
    normalize_signal = True

elif mode_spikes == 'multi_peak': 
    thres_STD = None  
    only_rising = None
    N = None
    normalize_signal = False
else:
    raise Exception()
    
perc_subthr = 20
all_results = []
thresh_factor=[]
scale = []

for thres_STD in [thres_STD]:#range(23,24,1):
    res_dict = dict()
    all_f1_scores = []
    all_prec = []
    all_rec = []
    all_corr_subthr = []
    for file in file_list[0:8]:
        dict1 = np.load(file, allow_pickle=True)
        spike_purs = False
        if spike_purs:
            dict1_v_sp_ = dict1['e_sp']
        else:
            img = dict1['v_sg']            
            #img /= estimate_running_std(img, q_min=0.1, q_max=99.9, win_size=20000, stride=5000)
            print(f'time between frame:{np.diff(dict1["v_t"])[0]}')
            
            idx_to_remove_estimate = []
            for i in range(len(dict1['sweep_time']) - 1):
                idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]), dict1['v_t']<dict1['sweep_time'][i+1][0])])[1]
               # img[idx_to_rem] = np.random.normal(0,1,len(idx_to_rem))*std_estimate
                idx_to_remove_estimate.append(idx_to_rem)

            for i in range(len(dict1['sweep_time']) - 1):
                idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]-1), dict1['v_t']<dict1['sweep_time'][i][-1]-0.85)])[1]
                idx_to_remove_estimate.append(idx_to_rem)
                
            idx_good_estimate = np.setdiff1d(range(len(img)),np.concatenate(idx_to_remove_estimate))
                                        
            subthreshold_signal = estimate_subthreshold_signal(img, mode='percentile', perc_subthr=perc_subthr,
                                                               perc_window=50, perc_stride=25, thres_STD=5, 
                                                               kernel_size=21,return_nans=False)
            
            
            all_corr_subthr.append([np.corrcoef((dict1['e_sub']),
                                                (subthreshold_signal))[0,1],
                                                np.corrcoef((dict1['e_sub']),
                                                (dict1['v_sub']))[0,1]])
            
            frate = 1/np.median(np.diff(dict1['v_t']))
            signal_no_subthr = img - subthreshold_signal
        #    signal_no_subthr = dict1['v_sg'] - dict1['v_sub']
            
            indexes, erf, z_signal, estimator = find_spikes(img[:], signal_no_subthr=signal_no_subthr, 
                                                 mode=mode_spikes, only_rising=only_rising, normalize_signal=normalize_signal, 
                                                 samples_covariance=10000, thres_STD=thres_STD, #thres_STD=3.5, 
                                                 thres_STD_ampl=4, min_dist=1, 
                                                 N=N, win_size=20000, stride=5000)
            
            #thresh_factor.append(erf)
            #scale.append(z_signal)
            
            indexes = np.setdiff1d(indexes, np.concatenate(idx_to_remove_estimate))
            dict1_v_sp_ = dict1['v_t'][indexes]
            
            for i in range(len(dict1['sweep_time']) - 1):
                dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
            
            precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned\
                                = metric(dict1['sweep_time'], dict1['e_sg'], 
                                      dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                      dict1['v_sg'], dict1_v_sp_ , 
                                      dict1['v_t'], dict1['v_sub'],save=False)
            
            if False:
                plt.figure()
                plt.plot(dict1['e_sp'],[20]*len(dict1['e_sp']),'k|')
                erf = np.log(npdf)
                plt.plot(dict1['v_t'][1:],-erf,'r-')
                plt.plot(dict1_v_sp_,[18]*len(dict1_v_sp_),'c|')
                plt.plot(dict1['v_t'],normalize(img)*2,'g-')
                FN = list(set(dict1_v_sp_)-set(v_match))
                FP = list(set(dict1['e_sp'])-set(e_match))
                plt.plot(FP,[22]*len(FP),'y*')
                plt.plot(FN,[21]*len(FN),'m*')
            print(np.array(F1).round(2).mean())
            all_f1_scores.append(np.array(F1).round(2))
            all_prec.append(np.array(precision).round(2))
            all_rec.append(np.array(recall).round(2))

    
    res_dict['all_f1_scores'] = all_f1_scores
    res_dict['all_prec'] = all_prec
    res_dict['all_rec'] = all_rec
    res_dict['thres_STD'] = thres_STD
    print(f'average_F1:{np.mean([np.nanmean(fsc) for fsc in all_f1_scores])}')
    print(f'average_sub:{np.nanmean(all_corr_subthr,axis=0)}')
    print(f'F1:{np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)}')
    print(f'prec:{np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)}'); 
    print(f'rec:{np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)}')

    all_results.append(res_dict)

#%%
[plt.plot(scale[i], label=str(i)) for i in range(len(scale))]
plt.legend()

#%%
all_f1_mat = []
for res in all_results:
    all_f1_scores = res['all_f1_scores']
    all_prec = res['all_prec']
    all_rec = res['all_rec']
#    print(f'average_F1:{np.mean([np.mean(fsc) for fsc in all_f1_scores])}')
#    print(res['thres_STD'])
#    print(f'average_F1:{np.mean([np.mean(fsc) for fsc in all_prec])}')
#    print(f'average_F1:{np.mean([np.mean(fsc) for fsc in all_rec])}')
#    print(f'average_sub:{np.mean(all_corr_subthr,axis=0)}')
    print(f'F1:{np.array([np.mean(fsc) for fsc in all_f1_scores]).round(2)}')
    all_f1_mat.append(np.array([np.mean(fsc) for fsc in all_f1_scores]))
#    print(f'prec:{np.array([np.mean(fsc) for fsc in all_prec]).round(2)}'); 
#    print(f'rec:{np.array([np.mean(fsc) for fsc in all_rec]).round(2)}')
    plt.figure()
    plt.subplot(1,3,1)
    [plt.plot(fsc,'-.') for fsc in res['all_f1_scores']]
    plt.subplot(1,3,2)
    [plt.plot(fsc,'-.') for fsc in  res['all_prec']]
    plt.subplot(1,3,3)
    [plt.plot(fsc,'-.') for fsc in  res['all_rec']]
    plt.legend([fl[-30:-11] for fl in file_list])
#%%
if False:
    #%%
    #indexes = peakutils.indexes(np.diff(img), thres=0.18, min_dist=3, thres_abs=True)
    eph = (dict1['e_sg']-np.mean(dict1['e_sg']))/(np.max(dict1['e_sg'])-np.min(dict1['e_sg']))
    #img = (dict1['v_sg']-np.mean(dict1['v_sg']))/np.max(dict1['v_sg'])
    plt.figure()
    FN = list(set(dict1_v_sp_)-set(v_match))
    FP = list(set(dict1['e_sp'])-set(e_match))
    plt.plot(dict1['e_sp'],[1.1]*len(dict1['e_sp']),'k.')
    plt.plot(dict1['v_sp'],[1.08]*len(dict1['v_sp']),'g.')
    plt.plot(FP,[1.06]*len(FP),'c|')
    plt.plot(FN,[1.04]*len(FN),'b|')
    plt.plot(dict1_v_sp_,[1.02]*len(dict1_v_sp_),'r.')
    plt.plot(dict1['v_t'][1:],np.diff(img)/np.max(np.diff(img)),'.-')
    plt.plot(dict1['e_t'],eph/np.max(eph), color='k')
    plt.plot(dict1['v_t'],img/np.max(img),'-')
    #%%
    plt.plot(dict1['e_sp'],[30]*len(dict1['e_sp']),'k.')
    erf = np.log(npdf)
    plt.plot(dict1['v_t'][1:],-erf,'r-')
#    plt.plot(dict1['v_t'],subs/3)
    
    #%%
    plt.plot(dict1['v_t'], img, '.-')
    plt.legend()
    plt.vlines(e_spike_aligned, img.min()-0.5, img.min(), color='black')
    plt.vlines(v_spike_aligned, img.min()-1, img.min()-0.5, color='red')
    
    #%%
    plt.plot(signal_no_subthr)

    from scipy import zeros, signal, random
    
    data = img
#    b, a = scipy.signal.butter(2, 0.1)
#    filtered = scipy.signal.lfilter(b, a, data)
#    result= data - filtered
##    nans, x= nan_helper(data)
##    data[nans]= np.intezrp(x(nans), x(~nans), data[~nans])
##    data = scipy.signal.medfilt(data, kernel_size=21)
##    data = img-data
    b = signal.butter(15, 0.01, btype='lowpass', output='sos')
    z = signal.sosfilt_zi(b)
    result, z = signal.sosfilt(b, data, zi=z)
#    plt.plot(result)
    result = zeros(data.size)
    for i, x in enumerate(data):
        result[i], z = signal.sosfilt(b, [x], zi=z)

        