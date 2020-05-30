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
    def __init__(self, thresh_STD=None, window = 10000, step = 5000, do_scale=True,
                 percentile_thr_sub=99):
        '''
        Object encapsulating Online Spike extraction from input traces
        Args:
            thresh_STD: float  
                threshold for spike selection, if None computed automatically
            window: int
                window over which to compute the running statistics
            window: int
                stride over which to compute the running statistics                
            do_scale: Bool
                whether to scale the input trace or not
            percentile_thr_sub: float
                percentile used as a threshold to decide when to accept a spike
        '''
        self.t_detect = []
        self.thresh_factor = thresh_STD
        self.window = window
        self.step = step
        self.do_scale = do_scale
        self.percentile_thr_sub=percentile_thr_sub
        
    def fit(self, trace_in, num_frames):
        """
        Method for computing spikes in offline mode, and initializer for the online mode
        Args:
            trace_in: ndarray
                the vector (num neurons x time steps) to use to simulate the online process
        """
        #trace = self.estimates.trace
        nn, tm = trace_in.shape # number of neurons and time steps
        # contains all the extracted fluorescence traces
        self.trace = np.zeros((nn, num_frames), dtype=np.float32) 
        self.trace[:, :tm] = trace_in
        # contains @todo
        self.trace_rm = np.zeros((nn, num_frames), dtype=np.float32) 
        # running statistics
        self.median = np.zeros((nn,1), dtype=np.float32)
        self.scale = self.median.copy()
        self.thresh =self.median.copy()
        self.thresh_sub = self.median.copy()
        # contains @todo
        self.index = np.zeros((nn, num_frames), dtype=np.int32)
        self.peak_height = np.zeros((nn, num_frames), dtype=np.float32)
        self.index_track = np.zeros((nn), dtype=np.int32)        
        self.peak_height_track = np.zeros((nn), dtype=np.int32)
        self.SNR = np.zeros((nn), dtype=np.float32)
        t_start = time()
        # initialize for each neuron: @todo parallelize
        for idx, tr in enumerate(trace_in):        
            output_list = find_spikes_rh(tr, self.thresh_factor, do_scale=self.do_scale)                         
            self.trace_rm[idx, :tm] = output_list[0]
            index_init = output_list[1]
            self.thresh_sub[idx] = output_list[2] 
            self.thresh[idx] = output_list[3]
            peak_height_init = output_list[4]
            self.median[idx] = output_list[5]
            self.scale[idx] = output_list[6]
            self.thresh_factor = output_list[7]
            
            self.index_track[idx] = index_init.shape[0]
            self.peak_height_track[idx] = peak_height_init.shape[0]
            self.index[idx, :self.index_track[idx]] = index_init
            self.peak_height[idx, :self.peak_height_track[idx]] = peak_height_init
        
        self.t_detect = self.t_detect + [(time() - t_start) / tm] * trace_in.shape[1] 
        
        return self

    def fit_next(self, trace_in, n):
        '''
        Method to incrementally estimate the spikes at each time step
        Args:
            trace_in: ndarray
                vector containing the fluorescence value at a single time point
            n: time step    
        '''        
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
                         
                                          
        self.trace[:, n:(n + 1)] = trace_in
        #scale and remove mean
        trace_in -= self.median[:, -1:]
        trace_in /= self.scale[:, -1:]
        self.trace_rm[:, n:(n + 1)] = trace_in
        # @todo
        temp = self.trace_rm[:, (n - 2):(n - 1)] - self.trace_rm[:, (n - 4):(n + 1)]    
        left = np.max((temp[:,0:1], temp[:,1:2]), axis=0)
        right = np.max((temp[:,3:4], temp[:,4:5]), axis=0)
        height = np.single(1 / (1 / left + 0.7 / right))
        
        indices = np.where(np.logical_and((temp[:,1] > 0), (temp[:,3] > 0)))[0]
        
        for idx in indices:
            self.peak_height[idx, self.peak_height_track[idx]] = height[idx]
            self.peak_height_track[idx] += 1
            if (self.trace_rm[idx, n - 2] > self.thresh_sub[idx, -1]) and (height[idx] > self.thresh[idx, -1]):
                self.index[idx, self.index_track[idx]] = (n - 2)     
                self.index_track[idx] +=1        
        
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
        self.thresh_sub[idx, -1] = np.percentile(tt, self.percentile_thr_sub)
        return self

    def compute_SNR(self):
        for idx in range(self.trace.shape[0]):
            mean_height = self.trace_rm[idx][self.index[idx][self.index[idx] > 0]].mean()
            #mean_height = self.peak_height[idx][self.peak_height[idx] > 0].mean()
            std = compute_std(self.trace_rm[idx][self.trace_rm[idx] != 0])
            snr = mean_height / std
            self.SNR[idx] = snr
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    
    