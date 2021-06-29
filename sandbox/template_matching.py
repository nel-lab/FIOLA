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
from .running_statistics import OnlineFilter, non_symm_median_filter
from .spikepursuit import adaptive_thresh

from scipy.ndimage import median_filter
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def find_spikes_tm(img, freq, frate, do_scale=False, filt_window=15, template_window=2, robust_std=False, 
                   adaptive_threshold=True, thresh_range=[3.5,5.0], minimal_thresh=3.0, mfp=0.2, do_plot=False):
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
    minimal_thresh :
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
    
    #sub = signal_filter(t0, freq, frate, order=1, mode='low')
    #filt_window=15
    if type(filt_window) is list:
        sub = non_symm_median_filter(t0, filt_window)                
    else:
        sub = median_filter(t0, size=filt_window)
    t = t0 - sub

    # First time thresholding
    data = t.copy()
    if do_plot:
        plt.figure(); plt.plot(data);plt.show()
    pks = data[signal.find_peaks(data, height=None)[0]]
    thresh, _, _, _ = adaptive_thresh(pks, clip=100, pnorm=0.25, min_spikes=10)
    idx = signal.find_peaks(data, height=thresh)[0]
    
    # Form a template
    if template_window > 0:
        window_length = template_window
    else:
        window_length = 2 
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    idx = idx[np.logical_and(idx > (-window[0]), idx < (len(data) - window[-1]))]
    PTD = data[(idx[:, np.newaxis] + window)]
    PTA = np.median(PTD, 0)
    
    if template_window > 0:
        # Template matching, second time thresholding
        t_s = np.convolve(data, np.flipud(PTA), 'same')
        data = t_s.copy()
    else:
        print('skip template matching')
        data = t.copy()
        t_s = t.copy()

    median2 = np.median(data)
    data = data - median2
    if robust_std:
        std = -np.percentile(data, 25) * 2 /1.35
    else:
        ff1 = -data * (data < 0)
        Ns = np.sum(ff1 > 0)
        std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    thresh_list = np.arange(thresh_range[0], thresh_range[1], 0.1)    

    # Select best threshold based on estimated false positive rate
    if adaptive_threshold:
        pks2 = t_s[signal.find_peaks(t_s, height=None)[0]]
        try:
            thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=0.5, min_spikes=10)  # clip=0 means no clipping
            thresh_factor = thresh2 / std
            if thresh_factor < minimal_thresh:
                print(f'Adaptive threshold factor is lower than minimal theshold: {minimal_thresh}, choose thresh factor to be {minimal_thresh}')
                thresh_factor = minimal_thresh
            thresh2 = thresh_factor * std
        except:
            print('Adaptive threshold fails, automatically choose thresh factor to be 3')
            thresh_factor = 3
            thresh2 = thresh_factor * std
    else:
        for thresh_factor in thresh_list:
            thresh_temp = thresh_factor * std
            n_peaks = len(signal.find_peaks(data, height=thresh_temp)[0])    
            n_false = len(signal.find_peaks(ff1, height=thresh_temp)[0])
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
    print(f'###final threshhold equals: {thresh2/std}###')
                
    #import pdb
    #pdb.set_trace()
    # plot signal, threshold, template and peak distribution
    if do_plot:
        plt.figure()
        plt.plot(t)
        plt.hlines(thresh, 0, len(t_s))
        plt.title('signal before template matching')
        plt.show()
        #plt.pause(3)

        plt.figure()
        plt.plot(t_s)
        plt.hlines(thresh2, 0, len(t_s))
        plt.title('signal after template matching')
        plt.show()
        #plt.pause(5)
        
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
        plt.pause(5)
        
    peak_to_std = data[index] / std    
    peak_level = data[index[index>300]] / std
    peak_level = np.percentile(peak_level, 95)

    return index, thresh2, PTA, t0, t, t_s, sub, median, scale, thresh_factor, median2, std, peak_to_std, peak_level 
