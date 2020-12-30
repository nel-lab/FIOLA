#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:01:15 2020
Online template matching
@author: caichangjia
"""
from .caiman_functions import get_thresh
from .caiman_functions import signal_filter
from .running_statistics import estimate_running_std
from .running_statistics import OnlineFilter
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def find_spikes_tm(img, freq, frate, do_scale=False, robust_std=False, 
                   adaptive_threshold=True, thresh_range=[3.5,5.0], mfp=0.2, do_plot=False):
    """
    Parameters
    ----------
    img : ndarray
        Input one dimensional detrended signal.
    freq : float
        Frequency for subthreshold extraction.
    frate : float
        Movie frame rate
    do_scale : bool, optional
        Whether scale the signal or not. The default is False.
    robust_std : bool, optional
        Whether to use robust way to estimate noise. The default is False.
    thresh_range : list, optional
        Range of threshold factor. Real threshold is threshold factor 
        multiply by the estimated noise level. The default is [3.5,5.0].
    mfp : float, optional
        Maximum estimated false positive. An upper bound for estimated false 
        positive rate based on noise. Higher value means more FP and less FN. 
        The default is 0.2.
    do_plot: bool, optional
        Whether to plot or not.

    Returns
    -------
    index : ndarray
        An array of spike times.
    thresh2 : float
        Threshold after template matching.
    PTA : ndarray
        Spike template. 
    t0 : ndarray
        Normalized trace.
    t : ndarray
        Trace removed from subthreshold.
    t_s : ndarray
        Trace after template matching.
    sub : ndarray
        Trace of subtrehsold.
    median : float
        Median of the original trace
    scale : float
        Scale of the original trace based on 1 percentile.
    thresh_factor : float
        A value determine the threshold level.
    median2 : float
        Median of the trace after template matching. 
    """
    # Normalize and remove subthreshold
    median = np.median(img)
    scale = -np.percentile(img-median, 1)
    if do_scale == True:
        t0 = (img - median) / scale
    else:
        t0 = img.copy() - median
    sub = signal_filter(t0, freq, frate, order=1, mode='low')
    t = t0 - sub
    
    # First time thresholding
    data = t.copy()
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    thresh = 4 * std
    idx = signal.find_peaks(data, height=thresh)[0]
    
    # Form a template
    window_length = 2    
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    idx = idx[np.logical_and(idx > (-window[0]), idx < (len(data) - window[-1]))]
    PTD = data[(idx[:, np.newaxis] + window)]
    PTA = np.median(PTD, 0)
    
    # Template matching, second time thresholding
    t_s = np.convolve(data, np.flipud(PTA), 'same')
    data = t_s
    median2 = np.median(data)
    data = data - median2
    ff1 = -data * (data < 0)
    if robust_std:
        std = -np.percentile(data, 25) * 2 /1.35
    else:
        Ns = np.sum(ff1 > 0)
        std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    if robust_std:
        thresh_list = np.arange(thresh_range[0], thresh_range[1], 0.1)    
    else:
        thresh_list = np.arange(thresh_range[0], thresh_range[1], 0.1)

    # Select best threshold based on estimated false positive rate
    if adaptive_threshold:
        pks2 = t_s[signal.find_peaks(t_s, height=None)[0]]
        try:
            thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=0.25, min_spikes=10)  # clip=0 means no clipping
            thresh_factor = thresh2 / std
        except:
            print('Adaptive threshold fails, automatically choose thresh factor to be 3.5')
            thresh_factor = 3.5
            thresh2 = thresh_factor * std
    else:
        for thresh_factor in thresh_list:
            thresh_temp = thresh_factor * std
            n_peaks = len(signal.find_peaks(data, height=thresh_temp)[0])    
            n_false = len(signal.find_peaks(ff1, height=thresh_temp)[0])
            #print(thresh_factor)
            ##try:
            #    print(f'n_false/n_peaks:{n_false / n_peaks}')
            #except:
            #    print(f'n_peaks equals 0')    
            if n_peaks == 0:
                thresh_factor = 3.5
                thresh2 = thresh_factor * std
                break
            if n_false / n_peaks < mfp:
                thresh2 = thresh_temp
                break
            if thresh_factor == thresh_list[-1]:
                thresh2 = thresh_temp
    index = signal.find_peaks(data, height=thresh2)[0]
    print(f'final threshhold equals: {thresh2/std}')
    
    # plot signal, threshold, template and peak distribution
    if do_plot:
        plt.figure()
        plt.plot(t_s)
        plt.hlines(thresh2, 0, len(t_s))
        plt.title(signal)
        plt.show()
        
        plt.figure();
        plt.plot(PTA);
        plt.title('template')
        plt.show()

        plt.figure()
        plt.hist(pks2, 500)
        plt.axvline(x=thresh2, c='r')
        plt.title('distribution of spikes')
        plt.tight_layout()
        plt.show()    

    return index, thresh2, PTA, t0, t, t_s, sub, median, scale, thresh_factor, median2    

"""
def find_spikes_tm_online(img, frate, freq=15, do_scale=False):
    window = 10000 
    step = 5000
    frames_init = 20000
    index_init, thresh_init, PTA, t0_init, t_init, t_s_init, sub_init, median_init, scale_init, thresh_factor, median2_init = find_spikes_tm(img[:frames_init], freq, frate, do_scale)
    t0 = np.zeros(img.shape)
    t = np.zeros(img.shape)
    t_s = np.zeros(img.shape)
    sub_online = np.zeros(img.shape)
    t0[:len(t_init)] = t0_init
    t[:len(t_init)] = t_init
    t_s[:len(t_init)] = t_s_init
    sub_online[:len(t_init)] = sub_init
    index = index_init.copy()
    median = [median_init]
    scale = [scale_init]
    thresh = [thresh_init]
    median2 = [median2_init]
    
    filt = OnlineFilter(freq, frate, order=1, mode='low')
    filt.fit(t0[np.newaxis, :frames_init])
    for tp in range(frames_init, len(img)):
        if do_scale == True:
            t0[tp] = (img[tp] - median[-1]) / scale[-1]
        else:    
            t0[tp] = img[tp] - median[-1]
        sub_online[tp] = filt.fit_next(t0[np.newaxis, tp])
        t[tp] = t0[tp] - sub_online[tp] 
        if tp > frames_init + 1:
            temp = t[tp - 4 : tp + 1]
            t_s[tp - 2] = np.convolve(temp, np.flipud(PTA), mode='valid')
            t_s[tp - 2] = t_s[tp - 2] - median2
        if t_s[tp - 2] > thresh[-1]:
            index = np.append(index, tp - 2)            
        if (tp >= 2.5 * window) and (tp % step == 0):
            tt = img[tp - int(2.5 * window) : tp]  
            median.append(np.percentile(tt, 50))
            scale.append(-np.percentile(tt - median[-1], 1))
        if (tp > 2.5 * window) and (tp % step == 100):
            tt = t_s[tp - int(window) : tp]  
            data = tt
            data = data - np.median(data)
            ff1 = -data * (data < 0)
            Ns = np.sum(ff1 > 0)
            std = np.sqrt(np.divide(np.sum(ff1**2), Ns))
            thresh.append(std * thresh_factor)
            
    plt.figure()
    plt.plot(img)
    plt.title('orig')
    plt.show()
    plt.figure()
    plt.plot(t_s)    
    plt.hlines(thresh, 0, len(img))
    plt.title('thresh')
    plt.show()
    #plt.legend()
    
    return index, thresh_factor, sub_online    
"""




