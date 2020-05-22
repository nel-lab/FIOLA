#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:12:44 2020

@author: @caichangjia
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import stats
from scipy.signal import argrelextrema
from time import time

#%%
sao = SignalAnalysisOnline()
sao.fit(trace[:, :20000])
for n in range(20000, 100000):
    sao.fit_next(trace[:, n: n+1], n)

#%%
class SignalAnalysisOnline(object):
    def __init__(self, params=None, estimates=None, dview=None):
        #self.params = params
        #self.estimates = estimates
        self.t_detect = []
        self.thresh_height = None
        self.window = 10000
        self.step = 5000
        
    def fit(self, trace):
        #trace = self.estimates.trace
        self.trace = trace
        self.trace_rm = np.zeros((trace.shape)) 
        self.median = np.array(np.zeros((trace.shape[0],1)))
        self.thresh = np.array(np.zeros((trace.shape[0],1)))
        self.thresh_sub = np.array(np.zeros((trace.shape[0],1)))
        self.index = [np.array([], dtype=int) for i in range(trace.shape[0])]
        self.peak_height = [np.array([], dtype=int) for i in range(trace.shape[0])]
        for idx, tr in enumerate(trace):        
            self.trace_rm[idx], self.index[idx], self.thresh_sub[idx], \
            self.thresh[idx], self.peak_height[idx], self.median[idx] \
            = find_spikes_rh(tr, self.thresh_height)
        return self

    def fit_next(self, trace_in, n):
        t_start = time()
        # Estimate thresh_sub, median, thresh        
        if (n > self.window) and (n % self.step == 0):
            self.update_thresh_sub()
            print(f'now processing frame{n}')
        if (n > self.window) and (n % self.step == 100):
            self.update_median()
        if (n > self.window) and (n % self.step == 200):
            self.update_thresh()
        self.trace, self.trace_rm, self.index, self.peak_height = find_spikes_rh_multiple \
        (self.trace, self.trace_rm, trace_in, self.median, self.thresh, \
         self.thresh_sub, self.index, self.peak_height)
        self.t_detect.append(time() - t_start)
        return self       

    def update_thresh_sub(self):
        thresh_sub_new = np.zeros((self.trace.shape[0], 1))
        for idx, tr in enumerate(self.trace_rm):
            tt = -tr[-self.window:][tr[-self.window:] < 0]
            thresh_sub_new[idx] = np.percentile(tt, 99)
        self.thresh_sub = np.append(self.thresh_sub, thresh_sub_new, axis=1)
        return self    
    
    def update_median(self):
        tt = self.trace[:, -self.window:]
        self.median = np.append(self.median, np.percentile(tt, 50, axis=1)[:, np.newaxis], axis=1)
        return self

    def update_thresh(self):
        thresh_new = np.zeros((self.trace.shape[0], 1))
        for idx, peaks in enumerate(self.peak_height):
            if self.thresh_height is None:
                thresh_new[idx] = compute_thresh(peaks, self.thresh[idx, -1])
            else:
                thresh_new[idx] = compute_std(peaks) * self.thresh_height
        self.thresh = np.append(self.thresh, thresh_new, axis=1)
        return self

#%%
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
        #plt.figure()
                    
    
    thresh = prev_thresh 

    if (len(minima)>0) and (np.abs(minima[0]-prev_thresh)< delta_max):
        thresh = minima[0]
        mnt = (minima_2nd-thresh)
        mnt = mnt[mnt<0]
        thresh += mnt[np.maximum(-len(mnt)+1,-number_maxima_before)]             
    
    """   
    plt.figure()
    plt.plot(x_val, pdf,'c')    
    plt.plot(x_val[1:], np.diff(pdf,1)*50,'k')  
    plt.plot(x_val[2:],second_der*500,'r')  

    plt.plot(thresh,0, '*')   
    plt.ylim([-.2,1])
    plt.pause(0.1)
    """
    return thresh            
       
def find_spikes_rh_multiple(t, t_rm, t_in, median, thresh, thresh_sub, index, peak_height):
    t = np.append(t, t_in, axis=1)
    t_in = t_in - median[:, -1:]
    t_rm = np.append(t_rm, t_in, axis=1)
    temp = t_rm[:, -3:-2] - t_rm[:, -5:]
    
    left = np.max((temp[:,0:1], temp[:,1:2]), axis=0)
    right = np.max((temp[:,3:4], temp[:,4:5]), axis=0)
    height = np.single(1 / (1 / left + 0.7 / right))
    indices = np.where(np.logical_and((temp[:,1] > 0), (temp[:,3] > 0)))[0]
    for idx in indices:
        peak_height[idx] = np.append(peak_height[idx], height[idx])
        if (t_rm[idx, -3] > thresh_sub[idx, -1]) and (height[idx] > thresh[idx, -1]):
            index[idx] = np.append(index[idx], (t.shape[1] - 2))            
    return t, t_rm, index, peak_height

def find_spikes_rh(t, thresh_height=None):
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
    
    return t, index, thresh_sub, thresh, peak_height, median
    
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

    
    _, thresh_sub_init, thresh_init, peak_height, median_init = find_spikes_rh(t[:], thresh_height)
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
    
    