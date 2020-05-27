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
    def __init__(self, params=None, estimates=None, dview=None, thresh_STD=None):
        #self.params = params
        #self.estimates = estimates
        self.t_detect = []
        self.thresh_factor = thresh_STD
        self.window = 10000
        self.step = 5000
        
    def fit(self, trace, num_frames):
        #trace = self.estimates.trace
        self.trace = np.zeros((trace.shape[0], num_frames), dtype=np.float32)
        self.trace[:, :trace.shape[1]] = trace
        self.trace_rm = np.zeros((trace.shape[0], num_frames), dtype=np.float32) 
        self.median = np.zeros((trace.shape[0],1), dtype=np.float32)
        self.scale = np.zeros((trace.shape[0],1), dtype=np.float32)
        self.thresh = np.zeros((trace.shape[0],1), dtype=np.float32)
        self.thresh_sub = np.zeros((trace.shape[0],1), dtype=np.float32)
        #self.index = [np.array([], dtype=int) for i in range(trace.shape[0])]
        self.index = np.zeros((trace.shape[0], num_frames), dtype=np.int32)
        self.index_track = np.zeros((trace.shape[0]), dtype=np.int32)
        self.peak_height = np.zeros((trace.shape[0], num_frames), dtype=np.float32)
        self.peak_height_track = np.zeros((trace.shape[0]), dtype=np.int32)
        self.SNR = np.zeros((trace.shape[0]), dtype=np.float32)
        #self.thresh_factor
        t_start = time()
        for idx, tr in enumerate(trace):        
            self.trace_rm[idx, :trace.shape[1]], index_init, self.thresh_sub[idx], \
            self.thresh[idx], peak_height_init, self.median[idx], self.scale[idx], self.thresh_factor \
            = find_spikes_rh(tr, self.thresh_factor, do_scale=True)             
            self.index_track[idx] = index_init.shape[0]
            self.peak_height_track[idx] = peak_height_init.shape[0]
            self.index[idx, :self.index_track[idx]] = index_init
            self.peak_height[idx, :self.peak_height_track[idx]] = peak_height_init
        self.t_detect = self.t_detect + [(time() - t_start) / trace.shape[1]] * trace.shape[1] 
        return self

    def fit_next(self, trace_in, n):
        self.n = n
        if self.n % self.step ==0:
            print(self.n)
        t_start = time()
        # Estimate thresh_sub, median, thresh        
        if (n >= 2.5 * self.window):
            for idx in range(self.trace.shape[0]):
                if (n % self.step == idx * 4 ):
                    self.update_median(idx)
                if (n % self.step == idx * 4 + 1):
                    self.update_scale(idx)
                #if (n % self.step == idx * 4 + 2):
                    #self.update_thresh(idx)
                if (n % self.step == idx * 4 + 3):
                    self.update_thresh_sub(idx)
        self.trace, self.trace_rm, self.index, self.peak_height, self.index_track, self.peak_height_track = find_spikes_rh_multiple \
        (self.trace, self.trace_rm, trace_in, self.median, self.scale, self.thresh, \
         self.thresh_sub, self.index, self.peak_height, self.index_track, self.peak_height_track, n)
        self.t_detect.append(time() - t_start)
        return self       

    def update_median(self, idx):
        tt = self.trace[idx, self.n - np.int(self.window*2.5):self.n]
        if idx == 0:
            self.median = np.append(self.median, self.median[:, -1:], axis=1)
        self.median[idx, -1] = np.median(tt)
        return self
    
    def update_scale(self, idx):
        tt = self.trace[idx, self.n - np.int(self.window*2.5):self.n]
        #tt = self.trace[idx, :self.n]
        if idx == 0:
            self.scale = np.append(self.scale, self.scale[:, -1:], axis=1)
        self.scale[idx, -1] = np.percentile(tt, 99)
        return self

    def update_thresh(self, idx):
        peaks = self.peak_height[idx]
        if idx == 0:
            self.thresh = np.append(self.thresh, self.thresh[:, -1:], axis=1)
        if self.thresh_factor is None:
            thresh_new = compute_thresh(peaks[:self.peak_height_track[idx]], self.thresh[idx, -1])
        else:
            thresh_new = compute_std(peaks[:self.peak_height_track[idx]]) * self.thresh_factor
        self.thresh[idx, -1] = thresh_new
        return self

    def update_thresh_sub(self, idx):
        tt = -self.trace_rm[idx, (self.n - self.window):self.n][self.trace_rm[idx, (self.n - self.window):self.n] < 0]
        if idx == 0:
            self.thresh_sub = np.append(self.thresh_sub, self.thresh_sub[:, -1:], axis=1)
        self.thresh_sub[idx, -1] = np.percentile(tt, 99)
        return self

    def compute_SNR(self):
        for idx in range(self.trace.shape[0]):
            mean_height = self.trace_rm[idx][self.index[idx][self.index[idx] > 0]].mean()
            #mean_height = self.peak_height[idx][self.peak_height[idx] > 0].mean()
            std = compute_std(self.trace_rm[idx][self.trace_rm[idx] != 0])
            snr = mean_height / std
            self.SNR[idx] = snr
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    
    