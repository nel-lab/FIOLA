#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:12:44 2020
Viola currently using SignalAnalysisOnlineZ object for spike extraction
SignalAnalysisOnlineZ object is based on template matching method 
SignalAnalysisOnine object analyze signal based on two threshold: one based on 
peak height distribution, the other based on subthreshold height.
@author: @caichangjia @andrea.giovannucci
"""
#%%
import numpy as np
from time import time
from spike_extraction_routines import find_spikes_rh
from running_statistics import compute_std, compute_thresh
from template_matching import find_spikes_tm
from running_statistics import OnlineFilter

#%%
class SignalAnalysisOnlineZ(object):
    def __init__(self, window = 10000, step = 5000, detrend=True, flip=True, 
                 do_scale=False, robust_std=False, adaptive_threshold=True, frate=400, freq=15, 
                 thresh_range=[3.5, 5], mfp=0.2, online_filter_method = 'median_filter',
                 filt_window = 15, do_plot=False):
        '''
        Object encapsulating Online Spike extraction from input traces
        Args:
            window: int
                window over which to compute the running statistics
            step: int
                stride over which to compute the running statistics
            detrend: bool
                whether to remove the trend due to photobleaching 
            flip: bool
                whether to flip the signal after removing trend.
                True for voltron indicator. False for others
            do_scale: bool
                whether to scale the input trace or not
            robust_std: bool
                whether to use robust way to estimate noise
            frate: float
                movie frame rate
            freq: float
                frequency for removing subthreshold activity
            adaptive_threshold: bool
                whether to use adaptive threshold method for deciding threshold level
            thresh_range: list
                Range of threshold factor. Real threshold is threshold factor 
                multiply by the estimated noise level. The default is [3.5,5.0].
            mfp : float
                Maximum estimated false positive. An upper bound for estimated false 
                positive rate based on noise. Higher value means more FP and less FN. 
                The default is 0.2.
            online_filter_method: str
                Use 'median_filter' or 'butter' for doing online filter. 'median filter'
                is more accurate but also introduce more lags
            filt_window: int
                window size for the median filter                
            do_plot: bool
                Whether to plot or not.
                
        '''
        self.t_detect = []
        self.window = window
        self.step = step
        self.detrend = detrend
        self.flip = flip
        self.do_scale = do_scale
        self.robust_std = robust_std
        self.freq = freq
        self.frate = frate
        self.adaptive_threshold = adaptive_threshold
        self.thresh_range = thresh_range
        self.mfp = mfp
        self.online_filter_method = online_filter_method
        self.filt_window = filt_window
        self.do_plot = do_plot
        self.filt = OnlineFilter(freq, frate, order=1, mode='low')
        
        
    def fit(self, trace_in, num_frames):
        """
        Method for computing spikes in offline mode, and initializer for the online mode
        Args:
            trace_in: ndarray
                the vector (num neurons x time steps) used to initialize the online process
            num_frames: int
                total number of frames for processing including both initialization and online processing
        """
        nn, tm = trace_in.shape # number of neurons and time steps for initialization
        # contains all the extracted fluorescence traces
        self.nn = nn
        self.frames_init = tm
        self.trace = np.zeros((nn, num_frames), dtype=np.float32) 
        self.t_d = np.zeros((nn, num_frames), dtype=np.float32) 
        # contains @todo
        self.t0 = np.zeros((nn, num_frames), dtype=np.float32) 
        self.t = np.zeros((nn, num_frames), dtype=np.float32) 
        self.t_s = np.zeros((nn, num_frames), dtype=np.float32) 
        self.t_sub = np.zeros((nn, num_frames), dtype=np.float32) 

        # running statistics
        self.median = np.zeros((nn,1), dtype=np.float32)
        self.scale = self.median.copy()
        self.thresh =self.median.copy()
        self.median2 = self.median.copy()
        # contains @todo
        self.index = np.zeros((nn, num_frames), dtype=np.int32)
        self.index_track = np.zeros((nn), dtype=np.int32)        
        self.SNR = np.zeros((nn), dtype=np.float32)
        self.PTA = np.zeros((nn, 5), dtype=np.float32)
        self.thresh_factor = np.zeros((nn, 1), dtype=np.float32)

        t_start = time()
        # initialize for each neuron: @todo parallelize
        if self.flip:
            self.trace[:, :tm] =  - trace_in.copy()
        else:
            self.trace[:, :tm] = trace_in.copy()
       
        if self.detrend:
            for tp in range(trace_in.shape[1]):
                if tp > 0:
                    self.t_d[:, tp] = self.trace[:, tp] - self.trace[:, tp - 1] + 0.995 * self.t_d[:, tp - 1]
        else:
            self.t_d = self.trace.copy()
        
        for idx, tr in enumerate(self.t_d[:, :tm]):  
            print(idx)
            output_list = find_spikes_tm(tr, self.freq, self.frate, self.do_scale, 
                                         self.robust_std, self.adaptive_threshold, self.thresh_range, 
                                         self.mfp, self.do_plot)
            self.index_track[idx] = output_list[0].shape[0]
            self.index[idx, :self.index_track[idx]] = output_list[0]
            self.thresh[idx] = output_list[1]
            self.PTA[idx] = output_list[2]
            self.t0[idx, :tm] = output_list[3]
            self.t[idx, :tm] = output_list[4]
            self.t_s[idx, :tm] = output_list[5]
            self.t_sub[idx, :tm] = output_list[6]
            self.median[idx] = output_list[7]
            self.scale[idx] = output_list[8]
            self.thresh_factor[idx] = output_list[9]
            self.median2[idx] = output_list[10]
        self.filt.fit(self.t0[:, :tm])
        self.t_detect = self.t_detect + [(time() - t_start) / tm] * trace_in.shape[1]         
        return self

    #@profile
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
        
        # Update running statistics
        res = n % self.step        
        if ((n > 3.0 * self.window) and (res < 3 * self.nn)):
            idx = int(res / 3)
            temp = res - 3 * idx
            if temp == 0:
                self.update_median(idx)
            elif temp == 1:
                self.update_scale(idx)
            elif temp == 2:
                self.update_thresh(idx)

        # Detrend, normalize and remove subthreshold
        if self.flip:
            self.trace[:, n:(n + 1)] = - trace_in.copy()
        else:
            self.trace[:, n:(n + 1)] = trace_in.copy()
        if self.detrend:
            self.t_d[:, n:(n + 1)] = self.trace[:, n:(n + 1)] - self.trace[:, (n - 1):n] + 0.995 * self.t_d[:, (n - 1):n]
        else:
            self.t_d[:, n:(n + 1)] = self.trace[:, n:(n + 1)]
        temp = self.t_d[:, n:(n+1)].copy()        
        temp -= self.median[:, -1:]
        if self.do_scale == True:
            temp /= self.scale[:, -1:]
        self.t0[:, n:(n + 1)] = temp.copy()
        
        sub_index = self.n - int((self.filt_window - 1) / 2)
        if self.online_filter_method == 'median_filter':
            if self.n >= self.frames_init + (self.filt_window - 1) / 2:
                temp = self.t0[:, self.n - self.filt_window + 1: self.n + 1]
                self.t_sub[:, sub_index] = np.median(temp, 1)
                self.t[:, sub_index:sub_index + 1] = self.t0[:, sub_index:sub_index + 1] - self.t_sub[:, sub_index:sub_index + 1]
            
            if self.n > self.frames_init + 1 + (self.filt_window - 1) / 2:                                      
                temp = self.t[:, sub_index - 4 : sub_index + 1]                          # time n-2 do template matching 
                self.t_s[:, sub_index - 2] = (temp * self.PTA).sum(axis=1)
                self.t_s[:, sub_index - 2] = self.t_s[:, sub_index - 2] - self.median2[:, -1]
            
            # Find spikes above threshold
            if self.n > self.frames_init +2 + (self.filt_window - 1) / 2:                                        # time n-3 confirm spikes
                temp = self.t_s[:, sub_index - 4: sub_index -1].copy()
                idx_list = np.where((temp[:, 1] > temp[:, 0]) * (temp[:, 1] > temp[:, 2]) * (temp[:, 1] > self.thresh[:, -1]))[0]
                if idx_list.size > 0:
                    self.index[idx_list, self.index_track[idx_list]] = sub_index - 3
                    self.index_track[idx_list] +=1
        elif self.online_filter_method == 'butter':
            self.t_sub[:, n] = self.filt.fit_next(self.t0[:, n])
            self.t[:, n:(n + 1)] = self.t0[:, n:(n + 1)] - self.t_sub[:, n:(n + 1)]  # time n remove subthreshold
            
            # Template matching
            if self.n > self.frames_init + 1:                                      
                temp = self.t[:, self.n - 4 : self.n + 1]                          # time n-2 do template matching 
                self.t_s[:, self.n - 2] = (temp * self.PTA).sum(axis=1)
                self.t_s[:, self.n - 2] = self.t_s[:, self.n - 2] - self.median2[:, -1]
            
            # Find spikes above threshold
            if self.n > self.frames_init +2:                                        # time n-3 confirm spikes
                temp = self.t_s[:, self.n - 4: self.n -1].copy()
                idx_list = np.where((temp[:, 1] > temp[:, 0]) * (temp[:, 1] > temp[:, 2]) * (temp[:, 1] > self.thresh[:, -1]))[0]
                if idx_list.size > 0:
                    self.index[idx_list, self.index_track[idx_list]] = self.n - 3
                    self.index_track[idx_list] +=1
        self.t_detect.append(time() - t_start)
        return self
        
    def update_median(self, idx):
        tt = self.t_d[idx, self.n - np.int(self.window*2.5):self.n]
        if idx == 0:
            self.median = np.append(self.median, self.median[:, -1:], axis=1)
        self.median[idx, -1] = np.median(tt)
        return self
    
    def update_scale(self, idx):
        tt = self.t_d[idx, self.n - np.int(self.window*2.5):self.n]
        #tt = self.trace[idx, :self.n]
        if idx == 0:
            self.scale = np.append(self.scale, self.scale[:, -1:], axis=1)
        self.scale[idx, -1] = - np.percentile(tt, 1)
        #self.scale[idx, -1] = np.percentile(tt, 99)
        
        return self

    def update_thresh(self, idx):
        if idx == 0:
            self.thresh = np.append(self.thresh, self.thresh[:, -1:], axis=1)
        tt = self.t_s[idx, self.n - self.window:self.n]
        if self.robust_std:
            thresh_new = (-np.percentile(tt, 25) * 2 / 1.35) * self.thresh_factor[idx]
        else:    
            thresh_new = compute_std(tt) * self.thresh_factor[idx] * (0.989) ** int(self.n / self.window)
        self.thresh[idx, -1] = thresh_new
        return self
        
    def compute_SNR(self):
        for idx in range(self.trace.shape[0]):
            sg = np.percentile(self.t[idx], 99)
            std = np.percentile(self.t[idx], 75) *2 / 1.35
            snr = sg / std
            self.SNR[idx] = snr
        return self
            
    def reconstruct_signal(self):
        self.t_rec = np.zeros(self.trace.shape)
        for idx in range(self.trace.shape[0]):
            spikes = np.array(list(set(self.index[idx])-set([0])))
            if spikes.size > 0:
                self.t_rec[idx, spikes] = 1
                self.t_rec[idx] = np.convolve(self.t_rec[idx], np.flip(self.PTA[idx]), 'same')   #self.scale[idx,0]   
        return self
                
    def reconstruct_movie(self, A, shape, scope):
        self.A = A.reshape((shape[0], shape[1], -1), order='F')
        self.C = self.t_rec[:, 0:1000]
        self.mov_rec = np.dot(self.A, self.C).transpose((2,0,1))    
        return self
 
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
            step: int
                stride over which to compute the running statistics                
            do_scale: Bool
                whether to scale the input trace or not
            percentile_thr_sub: float
                percentile used as a threshold to decide when to accept a spike
        '''
        self.t_detect = []
        self.thresh_STD = thresh_STD
        self.thresh_factor = thresh_STD
        self.window = window
        self.step = step
        self.do_scale = do_scale
        self.percentile_thr_sub=percentile_thr_sub
        
    def fit(self, trace_in, num_frames, frate, freq = 1/3):
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
            output_list = find_spikes_rh(tr, self.thresh_factor, do_scale=self.do_scale, thresh_percentile=self.percentile_thr_sub)                         
            self.trace_rm[idx, :tm] = output_list[0]
            index_init = output_list[1]
            self.thresh_sub[idx] = output_list[2] 
            self.thresh[idx] = output_list[3]
            peak_height_init = output_list[4]
            self.median[idx] = output_list[5]
            self.scale[idx] = output_list[6]
            self.thresh_factor = None
            
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
                if (n % self.step == idx * 4 + 2):
                    self.update_thresh(idx)
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
        if self.thresh_STD is None:
            thresh_new = compute_thresh(peaks[:self.peak_height_track[idx]], self.thresh[idx, -1])
        else:
            thresh_new = compute_std(peaks[:self.peak_height_track[idx]]) * self.thresh_STD
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
            
         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    
    