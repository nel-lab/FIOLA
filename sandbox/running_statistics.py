#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 08:07:20 2020

@author: agiovann
"""
from .caiman_functions import signal_filter
import cv2
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
#import scipy
#from scipy.interpolate import interp1d
from scipy import stats
from scipy.signal import argrelextrema, butter, sosfilt, sosfilt_zi

class OnlineFilter(object):
    def __init__(self, freq, fr, order=3, mode='high'):
        '''
        Object encapsulating Online filtering for spike extraction traces
        Args:
            freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
        '''
        self.freq = freq
        self.fr = fr
        self.mode = mode
        self.order=order
        self.normFreq = freq / (fr / 2)        
        self.filt = butter(self.order, self.normFreq, self.mode, output='sos')         
        self.z_init = sosfilt_zi(self.filt)
        
    
    def fit(self, sig):
        """
        Online filter initialization and running offline

        Parameters
        ----------
        sig : ndarray
            input signal to initialize
        num_frames_buf : int
            frames to use for buffering

        Returns
        -------
        sig_filt: ndarray
            filtered signal

        """
        sig_filt = signal_filter(sig, freq=self.freq, fr=self.fr, order=self.order, mode=self.mode)
        # result_init, z_init = signal.sosfilt(b, data[:,:20000], zi=z)
        self.z_init = np.repeat(self.z_init[:,None,:], sig.shape[0], axis=1)
    
        #sos_all = np.zeros(sig_filt.shape)

        for i in range(0,sig.shape[-1]-1):
            _ , self.z_init = sosfilt(self.filt, np.expand_dims(sig[:,i], axis=1), zi=self.z_init)
            
        return sig_filt 

    def fit_next(self, sig):
        
        sig_filt, self.z_init = sosfilt(self.filt, np.expand_dims(sig,axis=1), zi=self.z_init)
        return sig_filt.squeeze()

        
def rolling_window(ndarr, window_size, stride):   
        """
        generates efficient rolling window for running statistics
        Args:
            ndarr: ndarray
                input pixels in format pixels x time
            window_size: int
                size of the sliding window
            stride: int
                stride of the sliding window
        Returns:
                iterator with views of the input array
                
        """
        for i in range(0,ndarr.shape[-1]-window_size-stride+1,stride): 
            yield ndarr[:,i:np.minimum(i+window_size, ndarr.shape[-1])]
            
        if i+stride != ndarr.shape[-1]:
           yield ndarr[:,i+stride:]

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

def compute_std(peak_height):
    data = peak_height - np.median(peak_height)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    return std

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
    
    thresh = prev_thresh 

    if (len(minima)>0) and (np.abs(minima[0]-prev_thresh)< delta_max):
        thresh = minima[0]
        mnt = (minima_2nd-thresh)
        mnt = mnt[mnt<0]
        thresh += mnt[np.maximum(-len(mnt)+1,-number_maxima_before)]
    #else:
    #    thresh = 100
        
    thresh_7 = compute_std(peak_height) * 7.5
    
    """
    print(f'previous thresh: {prev_thresh}')
    print(f'current thresh: {thresh}')  
    """
    plt.figure()
    plt.plot(x_val, pdf,'c')    
    plt.plot(x_val[2:],second_der*500,'r')  
    plt.plot(thresh,0, '*')   
    plt.vlines(thresh_7, 0, 2, color='r')
    plt.pause(0.1)
    
    return thresh

def non_symm_median_filter(t, filt_window):
    m = t.copy()
    for i in range(len(t)):
        if (i > filt_window[0]) and (i < len(t) - filt_window[1]):
            m[i] = np.median(t[i - filt_window[0] : i + filt_window[1] + 1])
    return m
