#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:12:44 2020

@author: @caichangjia @andrea.giovannucci
"""
#%%
#import matplotlib.pyplot as plt
import numpy as np
#from scipy import signal
#from scipy import stats
#from scipy.signal import argrelextrema
from time import time
from spike_extraction_routines import find_spikes_rh, find_spikes_rh_multiple
from running_statistics import compute_std, compute_thresh

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
        self.scale = np.array(np.zeros((trace.shape[0],1)))
        self.thresh = np.array(np.zeros((trace.shape[0],1)))
        self.thresh_sub = np.array(np.zeros((trace.shape[0],1)))
        self.index = [np.array([], dtype=int) for i in range(trace.shape[0])]
        self.peak_height = [np.array([], dtype=int) for i in range(trace.shape[0])]
        for idx, tr in enumerate(trace):        
            self.trace_rm[idx], self.index[idx], self.thresh_sub[idx], \
            self.thresh[idx], self.peak_height[idx], self.median[idx], self.scale[idx] \
            = find_spikes_rh(tr, self.thresh_height)
        return self

    def fit_next(self, trace_in, n):
        t_start = time()
        # Estimate thresh_sub, median, thresh        
        if (n > self.window):
            if (n % self.step == 0):
                self.update_median_scale()
                print(f'now processing frame{n}')
            if (n % self.step == 100):
                self.update_thresh_sub()
            if (n % self.step == 200):
                self.update_thresh()
                
        self.trace, self.trace_rm, self.index, self.peak_height = find_spikes_rh_multiple \
        (self.trace, self.trace_rm, trace_in, self.median, self.scale, self.thresh, \
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
    
    def update_median_scale(self):
        tt = self.trace[:, np.int(-self.window*2.5):]
        self.median = np.append(self.median, np.percentile(tt, 50, axis=1)[:, np.newaxis], axis=1)
        self.scale = np.append(self.scale, np.percentile(tt, 99.9, axis=1)[:, np.newaxis], axis=1)
        return self

    def update_thresh(self):
        thresh_new = np.zeros((self.trace.shape[0], 1))
        for idx, peaks in enumerate(self.peak_height):
            if self.thresh_height is None:
                thresh_new[idx] = compute_thresh(peaks[-15000:], self.thresh[idx, -1])
            else:
                thresh_new[idx] = compute_std(peaks) * self.thresh_height
        self.thresh = np.append(self.thresh, thresh_new, axis=1)
        return self


    
    
       
    
    