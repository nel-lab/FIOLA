#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:58:55 2020
files for loading and analyzing proccessed Marton's data
@author: caichangjia
"""
#%% import library
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
import peakutils
from scipy import signal
from scipy.signal import savgol_filter
from time import time
from caiman.components_evaluation import mode_robust_fast
from scipy.signal import argrelextrema
from scipy import stats
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
    spike = [e_sp, v_sp]
    match = [e_sp[index_gt], v_sp[index_method]]
    height = np.max(np.array(e_sg.max(), v_sg.max()))
    
    # Calculate measures
    TP = len(index_gt)
    FP = len(v_sp) - TP
    FN = len(e_sp) - TP
    
    if len(e_sp) == 0:
        F1 = np.nan
        precision = np.nan
        recall = np.nan
    else:
        try:    
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
    
        recall = TP / (TP + FN)
    
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
    return precision, recall, F1, match, spike

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
    e_spike_aligned = []
    v_spike_aligned = []
    
    for i in range(len(sweep_time)):
        print(f'sweep{i}')
        if i == 0:
            scope = [max([e_t.min(), v_t.min()]), sweep_time[i][-1]]
        elif i == len(sweep_time) - 1:
            scope = [sweep_time[i][0], min([e_t.max(), v_t.max()])]
        else:
            scope = [sweep_time[i][0], sweep_time[i][-1]]
        mean_time.append(1 / 2 * (scope[0] + scope[-1]))
        
        pr, re, F, match, spike = spike_comparison(i, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist=0.05, save=save)
        corr = sub_correlation(i, v_t, e_sub, v_sub, scope, save=save)
        precision.append(pr)
        recall.append(re)
        F1.append(F)
        sub_corr.append(corr)
        e_match.append(match[0])
        v_match.append(match[1])
        e_spike_aligned.append(spike[0])
        v_spike_aligned.append(spike[1])
        
    e_match = np.concatenate(e_match)
    v_match = np.concatenate(v_match)
    e_spike_aligned = np.concatenate(e_spike_aligned)
    v_spike_aligned = np.concatenate(v_spike_aligned)


    return precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned

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
        #indexes = np.where(signal_no_subthr[1:]>thres_STD_ampl)
        
    return indexes, erf, z_signal
#%%
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def estimate_subthreshold(signal, thres_STD = 5, spike_before=3, spike_after=4, kernel_size=21, return_nans=False):
  delta_sig = np.diff(signal)
  index_exceptional, erf, z_sig = find_spikes(delta_sig, thres_STD=thres_STD)
  index_remove = np.concatenate([index_exceptional+ii for ii in range(-spike_before,spike_after)])
  sig_sub = signal.copy()
  sig_sub[np.minimum(index_remove,len(signal)-1)] = np.nan
  if return_nans:
      return sig_sub
  else:
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
#def compute_thresh(peak_height, prev_thresh=None, delta_max=np.inf):
#    mode = mode_robust_fast(peak_height)
#    peak_height -= mode
#    std = np.abs(2*np.percentile(peak_height,25))/1.35
#    kernel = stats.gaussian_kde(peak_height)
#    x_val = np.arange(0,np.max(peak_height),0.05)-mode
#    pdf = kernel(x_val)
#    
#    
#    return thresh   
#%%
def compute_thresh(peak_height, prev_thresh=None, delta_max=0.03, number_maxima_before=1):
        kernel = stats.gaussian_kde(peak_height)
        x_val = np.linspace(0,np.max(peak_height),1000)
        pdf = kernel(x_val)
        second_der = np.diff(pdf,2)
        mean = np.mean(peak_height)
        min_idx = argrelextrema(kernel(x_val), np.less)

        minima = x_val[min_idx]
        minima = minima[minima>mean]
        minima_2nd = argrelextrema(second_der, np.greater)
        minima_2nd = x_val[minima_2nd]

        if prev_thresh is None:
            delta_max = np.inf
            prev_thresh = mean
            plt.figure()
                        
        
        thresh = prev_thresh 

        if (len(minima)>0) and (np.abs(minima[0]-prev_thresh)< delta_max):
            thresh = minima[0]
            mnt = (minima_2nd-thresh)
            mnt = mnt[mnt<0]
            thresh += mnt[np.maximum(-len(mnt)+1,-number_maxima_before)]             
        
          
        plt.figure()
        plt.plot(x_val, pdf,'c')    
#        plt.plot(x_val[1:], np.diff(pdf,1)*50,'k')  
        plt.plot(x_val[2:],second_der*500,'r')  

        plt.plot(thresh,0, '*')   
#        plt.ylim([-.2,1])
        plt.pause(0.1)
        
        print(thresh)
        return thresh
#%%
def find_spikes_rh_online(t, thresh_height=4, window=10000, step=5000):
    """ Find spikes based on the relative height of peaks online
    Args:
        t: 1-D array
            one dimensional signal
            
        thresh height: int
            selected threshold
            
        window: int
            window for updating statistics including median, thresh_sub, std
            
        step: int
            step for updating statistics including median, thresh_sub, std

            
    Returns:
        index: 1-D array
            index of spikes
    """
    
    def compute_std(peak_height, median):
        data = peak_height - median
        ff1 = -data * (data < 0)
        Ns = np.sum(ff1 > 0)
        std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
        return std      

    
    _, thresh_sub_init, thresh_init, peak_height, median_init = find_spikes_rh(t[:20000], thresh_height)
    t_init = time()
    window_length = 2
    peak_height = np.array([])
    index = []
    median = [median_init]
    thresh_sub = [thresh_sub_init]
    thresh = [thresh_init]
    ts = np.zeros(t.shape)
    time_all = [] 
    
    for i in range(len(t)):
        if i > 2 * window_length:  
            ts[i] = t[i] - median[-1]
            # Estimate thresh_sub
            if (i > window) and (i % step == 0):
                tt = -ts[i - window : i][ts[i - window : i] < 0]  
                thresh_sub.append(np.percentile(tt, 99))
                print(f'{i} frames processed')
            
            if (i > window) and (i % step == 100):
                tt = t[i - window : i]  
                median.append(np.percentile(tt, 50))
            
            if (i > window) and (i % step == 200):
                length = len(peak_height)
                if length % 2 == 0:
                    temp_median = peak_height[int(length / 2) - 1]
                else:
                    temp_median = peak_height[int((length-1) / 2)]
                
                if thresh_height is None:
                    thresh.append(compute_thresh(peak_height, thresh[-1]))                    
                else:
                    thresh.append(compute_std(peak_height, temp_median) * thresh_height)
                
            # decide if two frames ago it is a peak
            temp = ts[i - 2] - ts[(i - 4) : (i+1)]
            if (temp[1] > 0) and (temp[3] > 0):
                left = np.max((temp[0], temp[1]))
                right = np.max((temp[3], temp[4]))
                height = np.single(1 / (1 / left + 0.7 / right))
                indices = np.searchsorted(peak_height, height)
                peak_height = np.insert(peak_height, indices, height)
                
                if ts[i - 2] > thresh_sub[-1]:
                    if height > thresh[-1]:
                        index.append(i - 2)
        t_c = time()
        time_all.append(t_c)
                
    print(f'{1000*(time() - t_init) / t.shape[0]} ms per frame')  
    print(f'{1000*np.diff(time_all).max()} ms for maximum per frame')  
    
    #plt.figure()
    #plt.plot(np.diff(np.array(time_all)))
    
    return index

#%%
def find_spikes_rh(t, thresh_height):
    """ Find spikes based on the relative height of peaks
    Args:
        t: 1-D array
            one dimensional signal
            
        thresh height: int
            selected threshold
            
    Returns:
        index: 1-D array
            index of spikes
    """
        
    # List peaks based on their relative peak heights
    #t = img.copy()
    median = np.median(t) 
    t = t - median
    window_length = 2
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    index = signal.find_peaks(t, height=None)[0]
    index = index[np.logical_and(index > (-window[0]), index < (len(t) - window[-1]))]
    matrix = t[index[:, np.newaxis]+window]
    left = np.maximum((matrix[:,2] - matrix[:,1]), (matrix[:,2] - matrix[:,0]))  
    right = np.maximum((matrix[:,2] - matrix[:,3]), (matrix[:,2] - matrix[:,4]))  
#    peak_height = 1 / (1 / left + 0.7 / right)
    peak_height = 1 / (1 / left + 0.7 / right)
    
    # Thresholding
    data = peak_height - np.median(peak_height)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    if thresh_height is not None:
        thresh = thresh_height * std
    else:
        thresh = compute_thresh(peak_height)
        
    index = index[peak_height > thresh]

    # Only select peaks above subthreshold
    tt = -t[t<0]
    thresh_sub = np.percentile(tt, 99)
    index_sub = np.where(t > thresh_sub)[0]
    index = np.intersect1d(index,index_sub)
    
    #plt.hist(peak_height, 500)
    #plt.vlines(thresh, peak_height.min(), peak_height.max(), color='red')
    #plt.figure()
    #plt.plot(dict1['v_t'], t); plt.vlines(dict1['v_t'][index], t.min()-5, t.min(), color='red');
    #plt.vlines(dict1['e_sp'], t.min()-10, t.min()-5, color='black') 
    
    return index, thresh_sub, thresh, peak_height, median


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

mode = ['minimum', 'percentile', 'v_sub', 'low_pass', 'double'][2]

for k in np.array(list(range(0,8))):
    if (k == 6) or (k==3):
        thresh_height = None
    else:
        thresh_height = 6
    dict1 = np.load(file_list[k], allow_pickle=True)
    img = dict1['v_sg']
    img /= estimate_running_std(img, q_min=0.1, q_max=99.9)
    std_estimate = np.diff(np.percentile(img,[75,25]))/100
    ### I comment it due to it will influence peak distribution
    #for i in range(len(dict1['sweep_time']) - 1):
    #    idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]), dict1['v_t']<dict1['sweep_time'][i+1][0])])[1]
    #    img[idx_to_rem] = np.random.normal(0,1,len(idx_to_rem))*std_estimate
    
    #for i in range(len(dict1['sweep_time']) - 1):
    #    idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]-1), dict1['v_t']<dict1['sweep_time'][i][-1]-0.85)])[1]
    #    img[idx_to_rem] = np.random.normal(0,1,len(idx_to_rem))*std_estimate
#    
    sub_1 = estimate_subthreshold(img, thres_STD=5,  kernel_size=21)
    all_corr_subthr.append([np.corrcoef(normalize(dict1['e_sub']),normalize(sub_1))[0,1],np.corrcoef(normalize(dict1['e_sub']),normalize(dict1['v_sub']))[0,1]])
    
    frate = 1/np.median(np.diff(dict1['v_t']))
    perc_window = 50
    perc_stride = 25
    if mode == 'v_sub':
        signal_subthr = dict1['v_sub']
    elif mode == 'percentile':
        perc = np.array([np.percentile(el,20) for el in rolling_window(img.T[None,:], perc_window, perc_stride)])
#        signal_subthr = np.concatenate([np.zeros(15),perc,np.zeros(14)]) #cv2.resize(perc, (1,img.shape[0])).squeeze()
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
    
    sao = SignalAnalysisOnline()
    img = img[np.newaxis, :]
    sao.fit(img[:, :20000])
    for n in range(20000, img.shape[1]):
        sao.fit_next(img[:, n: n+1], n)
    indexes = sao.index[0]    
    
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
    precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned = metric(dict1['sweep_time'], dict1['e_sg'], 
                                                                          dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                                                          dict1['v_sg'], dict1_v_sp_ , 
                                                                          dict1['v_t'], dict1['v_sub'],save=False)
    
    
    
    all_f1_scores.append(np.nanmean(np.array(F1)).round(2))
    all_prec.append(np.nanmean(np.array(precision)).round(2))
    all_rec.append(np.nanmean(np.array(recall)).round(2))
     
    continue
    pr.append(np.nanmean(np.array(precision)).round(2))
    re.append(np.nanmean(np.array(recall)).round(2))
    F.append(np.nanmean(np.array(F1)).round(2))
    sub.append(np.array(sub_corr).mean().round(2))
    
    fig = plt.figure()
    ax1 = fig.add_axes([0.05, 0.8, 0.9, 0.15])
    e_fr = np.unique(np.floor(e_spike_aligned), return_counts=True)
    v_fr = np.unique(np.floor(v_spike_aligned), return_counts=True)
    ax1.plot(e_fr[0], e_fr[1], color='black')
    ax1.plot(v_fr[0], v_fr[1], color='g')
    ax1.legend(['ephys','voltage'])
    ax1.set_ylabel('Firing Rate (Hz)')
    
    
    ax2 = fig.add_axes([0.05, 0.6, 0.9, 0.15])
    ax2.vlines(list(set(v_spike_aligned)-set(v_match)), 2.75,3.25, color='red')
    ax2.vlines(v_spike_aligned, 1.75,2.25, color='green')
    ax2.vlines(e_spike_aligned, 0.75,1.25, color='black')
    ax2.vlines(list(set(e_spike_aligned)-set(e_match)), -0.25,0.25, color='red')
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
all_f1_scores.append(np.array(F1).round(2))
all_prec.append(np.array(precision).round(2))
all_rec.append(np.array(recall).round(2))

#%%
print(f'average_F1:{np.mean([np.mean(fsc) for fsc in all_f1_scores])}')
print(f'average_sub:{np.mean(all_corr_subthr,axis=0)}')
print(f'F1:{np.array([np.mean(fsc) for fsc in all_f1_scores]).round(2)}')
print(f'prec:{np.array([np.mean(fsc) for fsc in all_prec]).round(2)}'); 
print(f'rec:{np.array([np.mean(fsc) for fsc in all_rec]).round(2)}')


#%%
[plt.plot(fsc,'-.') for fsc in all_rec]
plt.legend(lists)
#plt.plot(dict1['v_t'], dict1['v_sg'], '.-', color='blue')#;plt.plot(dict1['v_t'], dict1['v_sg']-signal_subthr);
#plt.plot(dict1['v_t'][:-1], np.diff(dict1['v_sg']-signal_subthr)-10, '.-', color='orange')
#plt.plot(dict1['e_t'], dict1['e_sg'])
plt.plot(dict1['v_t'], img, '.-')
#plt.plot(dict1['v_t'], t_s)
#plt.plot(dict1['v_t'], signal_subthr, label='quan');
#plt.plot(dict1['v_t'], dict1['v_sub'], label='vsub');
#plt.plot(dict1['v_t'], signal_subthr, label='subthr');
#plt.plot(dict1['v_t'], img - signal_subthr, label='processed');
#plt.plot(dict1['v_t'], t_s, label='tm');

#plt.plot(dict1['v_t'], sub_20, label='low_pass_20');
#plt.plot(dict1['v_t'], sub_15, label='low_pass_15');
plt.legend()
plt.vlines(e_spike_aligned, img.min()-5, img.min(), color='black')
plt.vlines(v_spike_aligned, img.min()-10, img.min()-5, color='red')


#%%
#print(lists) 

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

#%%
def find_spikes_tm(img, signal_subthr, thresh_height=3.5):
    from caiman.source_extraction.volpy.spikepursuit import signal_filter
    from caiman.source_extraction.volpy.spikepursuit import get_thresh
    from scipy import signal
    t = img - signal_subthr
    t = t - np.median(t)
    std_run = estimate_running_std(t, 20000, 5000, q_min=25, q_max=75)
    t = t/std_run
    
    window_chunk = 30000
    T = len(t)
    n = int(np.floor(T / window_chunk))
    mv_std = []
    index = []
    
    window_chunk_length = 2
    for i in range(n):
        if i < n - 1:
            data = t[window_chunk * i: window_chunk * i + window_chunk]
        else:
            data = t[window_chunk * i:]
    
        data = data - np.median(data)
        ff1 = -data * (data < 0)
        Ns = np.sum(ff1 > 0)
        std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
        thresh = 4 * std
        idx = signal.find_peaks(data, height=thresh)[0]
        
        #plt.plot(t); plt.vlines(idx, t.min()-5, t.min(), color='red')
        
        window = np.int64(np.arange(-window_length, window_length + 1, 1))
        idx = idx[np.logical_and(idx > (-window[0]), idx < (len(data) - window[-1]))]
        PTD = data[(idx[:, np.newaxis] + window)]
        #PTA = np.mean(PTD, 0)
        PTA = np.median(PTD, 0)
        PTA = PTA
        #plt.figure()
        #plt.plot(PTA)
        #templates = PTA
      
        t_s = np.convolve(data, np.flipud(PTA), 'same')
        t_s = t_s / t_s.max() * data.max()
        data = t_s
        data = data - np.median(data)
        pks2 = data[signal.find_peaks(data, height=None)[0]]
        thresh2, falsePosRate, detectionRate, _ = get_thresh(pks2, clip=0, pnorm=0.5, min_spikes=30)
        index.append(signal.find_peaks(data, height=thresh2)[0] + i * window_chunk)
    index = np.concatenate(index)
    """
    plt.figure()
    plt.hist(pks2, 500)
    plt.axvline(x=thresh2, c='r')
    plt.title('after matched filter')
    plt.tight_layout()
    plt.show()    
    """
    return index    

#%%    
def find_spikes_tm(img, signal_subthr, thresh_height=3.5):
    from caiman.source_extraction.volpy.spikepursuit import signal_filter
    from caiman.source_extraction.volpy.spikepursuit import get_thresh
    from scipy import signal
    t = img - signal_subthr
    t = t - np.median(t)
    std_run = estimate_running_std(t, 20000, 5000, q_min=25, q_max=75)
    t = t/std_run
    
    window_length = 2
    data = t[:20000]
    data = data - np.median(data)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    thresh = 4 * std
    index = signal.find_peaks(data, height=thresh)[0]
    
    #plt.plot(t); plt.vlines(index, t.min()-5, t.min(), color='red')
    
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    index = index[np.logical_and(index > (-window[0]), index < (len(data) - window[-1]))]
    PTD = data[(index[:, np.newaxis] + window)]
    #PTA = np.mean(PTD, 0)
    PTA = np.median(PTD, 0)
    PTA = PTA
    #plt.figure()
    #plt.plot(PTA)
    #templates = PTA
    
    """
    plt.figure()
    plt.plot(np.transpose(PTD), c=[0.5, 0.5, 0.5])
    plt.plot(PTA, c='black', linewidth=2)
    plt.title('Peak-triggered average')
    plt.show()
    """
  
    t_s = np.convolve(t, np.flipud(PTA), 'same')
    t_s = t_s / t_s.max() * t.max()
    
    """
    data = t_s
    data = data - np.median(data)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    thresh = thresh_height * std
    index = signal.find_peaks(data, height=thresh)[0]
    #plt.plot(t+5, label='orig'); plt.plot(t_s, label='template');plt.legend()
    """
    
    window = 30000
    T = len(t_s)
    n = int(np.floor(T / window))
    mv_std = []
    index = []
    
    for i in range(n):
        if i < n - 1:
            data = t_s[window * i: window * i + window]
        else:
            data = t_s[window * i:]
        data = data - np.median(data)
        ff1 = -data * (data < 0)
        Ns = np.sum(ff1 > 0)
        mv_std.append(np.sqrt(np.divide(np.sum(ff1**2), Ns)))
        thresh2 = thresh_height * mv_std[-1]
        spike = signal.find_peaks(data, height=thresh2)[0] + i *window
        index.append(spike)
    index = np.concatenate(index) 
    """
    for i in range(n):
        if i < n - 1:
            data = t_s[window * i: window * i + window]
        else:
            data = t_s[window * i:]
        data = data - np.median(data)
        pks2 = data[signal.find_peaks(data, height=None)[0]]
        thresh2, falsePosRate, detectionRate, _ = get_thresh(pks2, clip=0, pnorm=0.5, min_spikes=30)
        
        index.append(signal.find_peaks(data, height=thresh2)[0] + i * window)
    index = np.concatenate(index)
    
    plt.figure()
    plt.hist(pks2, 500)
    plt.axvline(x=thresh2, c='r')
    plt.title('after matched filter')
    plt.tight_layout()
    plt.show()    
    """
    return index

#%%
plt.plot(img, '.-')
#%%
plt.plot(img); plt.plot(signal_subthr);plt.plot(dict1['v_sub'])
plt.plot(t, label='orig');plt.plot(t_s, label='processed');plt.hlines(thresh2, 0, len(t));plt.legend()

#%%
from caiman.source_extraction.volpy.spikepursuit import signal_filter
sub_20 = signal_filter(dict1['v_sg'], 20, fr=400, order=5, mode='low')
sub_15 = signal_filter(dict1['v_sg'], 15, fr=400, order=5, mode='low')
sub_10 = signal_filter(dict1['v_sg'], 10, fr=400, order=5, mode='low')
sub_15_order3 = signal_filter(dict1['v_sg'], 15, fr=400, order=3, mode='low')
sub_15_order1 = signal_filter(dict1['v_sg'], 15, fr=400, order=1, mode='low')
sub_5 = signal_filter(dict1['v_sg'], 5, fr=400, order=5, mode='low')
sub_1 = signal_filter(dict1['v_sg'], 1, fr=400, order=5, mode='low')
 
 
#%%
plt.plot(img);
#plt.plot(signal_subthr, label='subthr');
plt.plot(dict1['v_sub'], label='vsub');
#plt.plot(minimal, label='minimal');
#plt.plot(sub_20, label='low_pass_20');
plt.plot(sub_10, label='low_pass_1');

#plt.plot(sub_1, label='low_pass_1');
perc = np.array([np.percentile(el,20) for el in rolling_window((img-sub_10).T[None,:], perc_window, perc_stride)])
#        signal_subthr = np.concatenate([np.zeros(15),perc,np.zeros(14)]) #cv2.resize(perc, (1,img.shape[0])).squeeze()
signal_subthr =  cv2.resize(perc, (1,img.shape[0]),cv2.INTER_CUBIC).squeeze()
plt.plot(sub_10+signal_subthr, label='remove twice')

#plt.plot(sub_15_order3, label='low_pass_15_3');
#plt.plot(sub_15_order1, label='low_pass_15_1');
#plt.plot(minimum, label='minimum')
#plt.plot(img - sub_15-signal_subthr, label='remove twice')
#plt.plot(minimal, label='minimal')
#plt.plot(sub_5, label='low_pass_5');
perc = np.array([np.percentile(el,20) for el in rolling_window((img).T[None,:], perc_window, perc_stride)])
#        signal_subthr = np.concatenate([np.zeros(15),perc,np.zeros(14)]) #cv2.resize(perc, (1,img.shape[0])).squeeze()
signal_subthr =  cv2.resize(perc, (1,img.shape[0]),cv2.INTER_CUBIC).squeeze()
plt.plot(signal_subthr, label='remove once')

plt.legend()
