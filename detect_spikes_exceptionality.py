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
import peakutils

#%% relevant functions
def distance_spikes(s1, s2, max_dist):
    """ Define distance matrix between two spike train.
    Distance greater than maximum distance is assigned one.
    """    
    D = np.ones((len(s1), len(s2)))
    for i in range(len(s1)):
        for j in range(len(s2)):
            if np.abs(s1[i] - s2[j]) > max_dist:
                D[i, j] = 1
            else:
                D[i, j] = (np.abs(s1[i] - s2[j]))/5/max_dist
    return D

def find_matches(D):
    """ Find matches between two spike train by solving linear assigment problem.
    Delete matches where their distance is greater than maximum distance
    """
    index_gt, index_method = linear_sum_assignment(D)
    del_list = []
    for i in range(len(index_gt)):
        if D[index_gt[i], index_method[i]] == 1:
            del_list.append(i)
    index_gt = np.delete(index_gt, del_list)
    index_method = np.delete(index_method, del_list)
    return index_gt, index_method

def spike_comparison(i, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist, save=False):
    e_sg = e_sg[np.where(np.multiply(e_t>=scope[0], e_t<=scope[1]))[0]]
    e_sg = (e_sg - np.mean(e_sg))/(np.max(e_sg)-np.min(e_sg))*np.max(v_sg)
    e_sp = e_sp[np.where(np.multiply(e_sp>=scope[0], e_sp<=scope[1]))[0]]
    e_t = e_t[np.where(np.multiply(e_t>=scope[0], e_t<=scope[1]))[0]]
    #plt.plot(e_t, e_sg, label='ephys', color='blue')
    #plt.plot(e_sp, np.max(e_sg)*1.1*np.ones(e_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
    
    v_sg = v_sg[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_sp = v_sp[np.where(np.multiply(v_sp>=scope[0], v_sp<=scope[1]))[0]]
    v_t = v_t[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    #plt.plot(v_t, v_sg, label='ephys', color='blue')
    #plt.plot(v_sp, np.max(v_sg)*1.1*np.ones(v_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
    
    # Distance matrix and find matches
    D = distance_spikes(s1=e_sp, s2=v_sp, max_dist=max_dist)
    index_gt, index_method = find_matches(D)
    match = [e_sp[index_gt], v_sp[index_method]]
    height = np.max(np.array(e_sg.max(), v_sg.max()))
    
    # Calculate measures
    TP = len(index_gt)
    FP = len(v_sp) - TP
    FN = len(e_sp) - TP
    try:    
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        if v_sp == 0:
            precision = 1
        else:    
            precision = 0

    try:    
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 1

    try:
        F1 = 2 * (precision * recall) / (precision + recall) 
    except ZeroDivisionError:
        F1 = 0

    print('precision:',precision)
    print('recall:',recall)
    print('F1:',F1)      
    if save:
        plt.figure()
        plt.plot(e_t, e_sg, color='b', label='ephys')
        plt.plot(e_sp, 1.2*height*np.ones(e_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
        plt.plot(v_t, v_sg, color='orange', label='VolPy')
        plt.plot(v_sp, 1.4*height*np.ones(len(v_sp)),color='orange', marker='.', ms=2, fillstyle='full', linestyle='none')
        for j in range(len(index_gt)):
            plt.plot((e_sp[index_gt[j]], v_sp[index_method[j]]),(1.25*height, 1.35*height), color='gray',alpha=0.5, linewidth=1)
        ax = plt.gca()
        ax.locator_params(nbins=7)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.legend(prop={'size': 6})
        plt.tight_layout()
        plt.savefig(f'{volpy_path}/spike_sweep{i}_{vpy.params.volspike["threshold_method"]}.pdf')
    return precision, recall, F1, match

# Compute subthreshold correlation coefficents
def sub_correlation(i, v_t, e_sub, v_sub, scope, save=False):
    e_sub = e_sub[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_sub = v_sub[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_t = v_t[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    corr = np.corrcoef(e_sub, v_sub)[0,1]
    if save:
        plt.figure()
        plt.plot(v_t, e_sub)
        plt.plot(v_t, v_sub)   
        plt.savefig(f'{volpy_path}/spike_sweep{i}_subthreshold.pdf')
    return corr

def metric(sweep_time, e_sg, e_sp, e_t, e_sub, v_sg, v_sp, v_t, v_sub, save=False):
    precision = []
    recall = []
    F1 = []
    sub_corr = []
    mean_time = []
    e_match = []
    v_match = []
    
    for i in range(len(sweep_time)):
        print(f'sweep{i}')
        if i == 0:
            scope = [np.ceil(max([e_t.min(), v_t.min()])), sweep_time[i][-1]]
        elif i == len(sweep_time) - 1:
            scope = [sweep_time[i][0], np.floor(min([e_t.max(), v_t.max()]))]
        else:
            scope = [sweep_time[i][0], sweep_time[i][-1]]
        mean_time.append(1 / 2 * (scope[0] + scope[-1]))
        
        pr, re, F, match = spike_comparison(i, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist=0.05, save=save)
        corr = sub_correlation(i, v_t, e_sub, v_sub, scope, save=save)
        precision.append(pr)
        recall.append(re)
        F1.append(F)
        sub_corr.append(corr)
        e_match.append(match[0])
        v_match.append(match[1])
    
    e_match = np.concatenate(e_match)
    v_match = np.concatenate(v_match)

    return precision, recall, F1, sub_corr, e_match, v_match, mean_time

#%%
from caiman.base.movies import rolling_window
from functools import partial
import cv2
import scipy
from scipy.interpolate import interp1d
import scipy

def estimate_running_std(signal_in, win_size=20000, stride=5000, 
                         idx_exclude=None, q_min=25, q_max=75):
    """
    Function to estimate ROBUST runnning std
    
    Args:
        win_size: int
            window used to compute running std to normalize signals when 
            compensating for photobleaching
            
        stride: int
            corresponding stride to win_size
            
        idx_exclude: iterator
            indexes to exclude when computing std
        
        q_min: float
            lower percentile for estimation of signal variability (do not change)
        
        q_max: float
            higher percentile for estimation of signal variability (do not change)
        
        
    Returns:
        std_run: ndarray
            running standard deviation
    
    """
    if idx_exclude is not None:
        signal = signal_in[np.setdiff1d(range(len(signal_in)), idx_exclude)]        
    else:
        signal = signal_in
    iter_win = rolling_window(signal[None,:],win_size,stride)
    myperc = partial(np.percentile, q=[q_min,q_max], axis=-1)
    res = np.array(list(map(myperc,iter_win))).T.squeeze()
    iqr = (res[1]-res[0])/1.35
    std_run = cv2.resize(iqr,signal_in[None,:].shape).squeeze()
    return std_run

def extract_exceptional_events(z_signal, thres_STD=5, N=2, min_dist=1, bidirectional=False):
    """
    Extract peaks that are stastically significant over estimated noise
    
    
    N: int
        window used to compute exceptionality (higher frame rates than 1Khz
        MIGHT need larger N) 
    
    thres_STD: float
            threshold related to z scored signal 
        
    min_dist: int
        min distance between peaks
        
    bidirectional: bool
            whether to build an error function that accounts for the direction
            of signal (it does not seem to help using this)
            
    Returns:
        indexes: list
            indexes of inferred spikes 

        erf: ndarray
            float representing the exceptionality of the trace over N points
    
    """
    if bidirectional:        
        erf = scipy.special.log_ndtr(-np.abs(z_signal))
    else:
        erf = scipy.special.log_ndtr(-z_signal)
    erf = np.cumsum(erf)
    erf[N:] -= erf[:-N]
    indexes = peakutils.indexes(-erf, thres=thres_STD, min_dist=min_dist, thres_abs=True)
    return indexes, erf
    
#%%     
def find_spikes(signal, signal_no_subthr=None, thres_STD=5, thres_STD_ampl=4, 
                min_dist=1, N=2, win_size=20000, stride=5000, spike_before=3, 
                spike_after=4, q_min=25, q_max=75, bidirectional=False):
    """
    Function that extracts spike from np.diff(signal). In general the only 
    parameters that should be adapted are thres_STD and 
    @TODO: extract subthreshold signal as well
    
    Args:
        signal: ndarray
            fluorescence signal after detrending
        
        signal_no_subthr: ndarray
            signal without subthreshold activity 
            
        thres_STD: float
            threshold related to z scored signal 
            
        thres_STD_ampl: float
            threshold related to z scored signal without subthreshold activity
            
        min_dist: int
            min distance between spikes
            
        N: int
            window used to compute exceptionality (higher frame rates than 1Khz
            MIGHT need larger N) 
            
        win_size: int
            window used to compute running std to normalize signals when 
            compensating for photobleaching
            
        stride: int
            corresponding stride to win_size
            
        spike_before: int
            how many sample to remove before a spike when removing the spike
            from the trace
        
        spike_after: int
            how many sample to remove after a spike when removing the spike
            from the trace        
            
        q_min: float
            lower percentile for estimation of signal variability (do not change)
        
        q_max: float
            higher percentile for estimation of signal variability (do not change)
            
        bidirectional: bool
            whether to build an error function that accounts for the direction
            of signal (it does not seem to help using this)
            
        
    Returns:
        indexes: list
            indexes of inferred spikes 

        erf: ndarray
            float representing the exceptionality of the trace over N points
        
        z_signal: ndarray
            z scored signal
        
    """
    
    signal = np.diff(signal)
    std_run = estimate_running_std(signal, win_size, stride, q_min=q_min, q_max=q_max)
    z_signal = signal/std_run
    index_exceptional,_ = extract_exceptional_events(z_signal, thres_STD=thres_STD, N=N, min_dist=min_dist, bidirectional=bidirectional)    
    index_remove = np.concatenate([index_exceptional+ii for ii in range(-spike_before,spike_after)])
    std_run = estimate_running_std(signal, win_size, stride,idx_exclude=index_remove, q_min=q_min, q_max=q_max)
    z_signal = signal/std_run 
    indexes, erf = extract_exceptional_events(z_signal, thres_STD=thres_STD, N=N, min_dist=min_dist, bidirectional=bidirectional)
    # remove spikes that are not large peaks in the original signal
    if signal_no_subthr is not None:
        signal_no_subthr /= estimate_running_std(signal_no_subthr, 20000, 5000, 
                                                 q_min=q_min, q_max=q_max)
        
        indexes = np.intersect1d(indexes,np.where(signal_no_subthr[1:]>thres_STD_ampl))
        
    return indexes, erf, z_signal
#%%
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def estimate_subthreshold(signal, thres_STD = 5, spike_before=3, spike_after=4, kernel_size=21):
  delta_sig = np.diff(signal)
  index_exceptional, erf, z_sig = find_spikes(delta_sig, thres_STD=thres_STD)
  index_remove = np.concatenate([index_exceptional+ii for ii in range(-spike_before,spike_after)])
  sig_sub = signal.copy()
  sig_sub[np.minimum(index_remove,len(signal)-1)] = np.nan
  nans, x= nan_helper(sig_sub)
  sig_sub[nans]= np.interp(x(nans), x(~nans), sig_sub[~nans])
  sig_sub = scipy.signal.medfilt(sig_sub, kernel_size=kernel_size)
  return sig_sub

def normalize(ss):
    aa = (ss-np.min(ss))/(np.max(ss)-np.min(ss))
    aa -= np.median(aa)
    aa /= estimate_running_std(aa)
    return aa
#%%
file_list = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/454597_Cell_0_40x_patch1_output.npz', 
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_3_40x_1xtube_10A2_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_3_40x_1xtube_10A3_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A5_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A7_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/462149_Cell_1_40x_1xtube_10A1_output.npz', 
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/462149_Cell_1_40x_1xtube_10A2_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A6_output.npz']

fig = plt.figure(figsize=(12,12))
temp = file_list[0].split('/')[-1].split('_')
fig.suptitle(f'subject_id:{temp[0]}  Cell number:{temp[2]}')

pr= []
re = []
F = []
sub = []
N_opt = [2,2,2,2,2,2]
#new_Thr_opt = [8.5,8, 5.5, 10.5, 6.5, 10.5, 6.5, 6.5]
#new_F1 = [0.97,0.96, 0.82, 0.6, 0.39, 0.86, 0.67, 0.76]
#
#Thr_opt = [8.5,8.5, 5.5, 10.5, 6.5, 10.5, 6.5 ]
#F1 = [0.97, 0.96, 0.82, 0.6, 0.39, 0.86, 0.67]
all_f1_scores = []
all_corr_subthr = []

for file in file_list[:]:
    dict1 = np.load(file, allow_pickle=True)
    img = dict1['v_sg']
    print(np.diff( dict1['v_t']))
    std_estimate = np.diff(np.percentile(img,[75,25]))/4
    for i in range(len(dict1['sweep_time']) - 1):
        idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]), dict1['v_t']<dict1['sweep_time'][i+1][0])])[1]
        img[idx_to_rem] = np.random.normal(0,1,len(idx_to_rem))*std_estimate
    
    for i in range(len(dict1['sweep_time']) - 1):
        idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]-1), dict1['v_t']<dict1['sweep_time'][i][-1]-0.85)])[1]
        img[idx_to_rem] = np.random.normal(0,1,len(idx_to_rem))*std_estimate
    
    sub_1 = estimate_subthreshold(img, thres_STD=5)
    all_corr_subthr.append([np.corrcoef(normalize(dict1['e_sub']),normalize(sub_1))[0,1],np.corrcoef(normalize(dict1['e_sub']),normalize(dict1['v_sub']))[0,1]])
#    delta_img = np.diff(img)
    signal_no_subthr = dict1['v_sg'] - dict1['v_sub']     
    indexes, erf, z_signal = find_spikes(img, signal_no_subthr=signal_no_subthr, 
                                         thres_STD=5, thres_STD_ampl=4, min_dist=1, 
                                         N=2, win_size=20000, stride=5000, 
                                         spike_before=3, spike_after=4, 
                                         bidirectional=False)
    
    dict1_v_sp_ = dict1['v_t'][indexes]
    
#    range_run = estimate_running_std(delta_img, 20000, 5000, q_min=0.000001, q_max=99.999999)
#    std_run = estimate_running_std(delta_img, 20000, 5000, q_min=25, q_max=75)
#    plt.plot(range_run/std_run)
    for i in range(len(dict1['sweep_time']) - 1):
        dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
    dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
#    
#    dict1_v_sp_ = dict1['v_sp']
#    precision, recall, F1, sub_corr, e_match, v_match, mean_time = metric(dict1['sweep_time'], dict1['e_sg'], 
#                                                                          dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
#                                                                          dict1['v_sg'], dict1['v_sp'], 
#                                                                          dict1['v_t'], dict1['v_sub'],save=False)
    precision, recall, F1, sub_corr, e_match, v_match, mean_time = metric(dict1['sweep_time'], dict1['e_sg'], 
                                                                          dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                                                          dict1['v_sg'], dict1_v_sp_ , 
                                                                          dict1['v_t'], dict1['v_sub'],save=False)
    
    
    
    all_f1_scores.append(np.array(F1).mean().round(2))
    continue 
    pr.append(np.array(precision).mean().round(2))
    re.append(np.array(recall).mean().round(2))
    F.append(np.array(F1).mean().round(2))
    sub.append(np.array(sub_corr).mean().round(2))
    ax1 = fig.add_axes([0.05, 0.8, 0.9, 0.15])
    
    
    e_fr = np.unique(np.floor(dict1['e_sp']), return_counts=True)
    v_fr = np.unique(np.floor(dict1_v_sp_), return_counts=True)
    ax1.plot(e_fr[0], e_fr[1], color='black')
    ax1.plot(v_fr[0], v_fr[1], color='g')
    ax1.legend(['ephys','voltage'])
    ax1.set_ylabel('Firing Rate (Hz)')
    
    
    ax2 = fig.add_axes([0.05, 0.6, 0.9, 0.15])
    ax2.vlines(list(set(dict1_v_sp_)-set(v_match)), 2.75,3.25, color='red')
    ax2.vlines(dict1_v_sp_, 1.75,2.25, color='green')
    ax2.vlines(dict1['e_sp'], 0.75,1.25, color='black')
    ax2.vlines(list(set(dict1['e_sp'])-set(e_match)), -0.25,0.25, color='red')
    plt.yticks(np.arange(4), ['False Negative', 'Ephys', 'Voltage', 'False Positive'])
    
    ax3 = fig.add_axes([0.05, 0.2, 0.9, 0.35])
    ax3.plot(mean_time, precision, 'o-', c='blue')
    ax3.plot(mean_time, recall, 'o-', c='orange')
    ax3.plot(mean_time, F1, 'o-', c='green')
    
    
    ax4 = fig.add_axes([0.05, 0, 0.9, 0.15])
    ax4.plot(mean_time, sub_corr, 'o-', c='blue')
    #plt.savefig(f'{volpy_path}/metric_{vpy.params.volspike["threshold_method"]}.pdf', bbox_inches='tight')
if False:    
    ax3.legend([f'precision:{pr}', f'recall: {re}', f'F1: {F}'])
    ax4.legend([f'corr:{sub}'])
#%%
print(np.mean(all_f1_scores))
print(np.mean(all_corr_subthr,axis=0))
#%%
if False:
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
    plt.plot(dict1['v_t'][1:],-erf/np.max(-erf),'r-')
    plt.plot(dict1['v_t'],subs/3)
    
#%%
if False:
    sub_1 = estimate_subthreshold(img, thres_STD=3.5)
    plt.plot(normalize(dict1['e_sub']),'k')
    plt.plot(normalize(dict1['v_sub']))
    plt.plot(normalize(sub_1))    