#!/usr/bin/env python
"""
Fiola uses SignalAnalysisOnlineZ object for online spike extraction which
is based on template matching method 
@author: @caichangjia @andrea.giovannucci
"""
import logging
from math import exp, log
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from scipy.ndimage import median_filter
from scipy import signal
from time import time, sleep
from threading import Thread

from fiola.utilities import compute_std, compute_thresh, get_thresh, signal_filter, estimate_running_std, OnlineFilter, non_symm_median_filter, adaptive_thresh
from fiola.oasis import par_fit_next_AR1, par_fit_next_AR2, reconstruct_AR1, reconstruct_AR2


class SignalAnalysisOnlineZ(object):
    def __init__(self, mode='voltage', window = 10000, step = 5000, flip=True, detrend=True, dc_param=0.995, do_deconvolve=True,
                 do_scale=False, template_window=2, robust_std=False, adaptive_threshold=True, fr=400, freq=15, 
                 minimal_thresh=3.0, online_filter_method = 'median_filter', filt_window = 15, do_plot=False,
                 p=1, nb=0):
        '''
        Object encapsulating Online Spike extraction from input traces
        Args:
            mode: str
                voltage or calcium fluorescence indicator. Note that calcium deconvolution is not implemented here.
            window: int
                window over which to compute the running statistics
            step: int
                stride over which to compute the running statistics
            flip: bool
                whether to flip the signal after removing trend.
                True for voltron indicator. False for others
            detrend: bool
                whether to remove the slow trend in the fluorescence data            
            dc_param: float
                DC blocker parameter for removing the slow trend in the fluorescence data. It is usually between
                0.99 and 1. Higher value will remove less trend. No detrending will perform if detrend=False.
            do_deconvolve: bool
                If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
            do_scale: bool
                whether to scale the input trace or not
            template_window: int
                half window size for template matching. Will not perform template matching
                if 0.                
            robust_std: bool
                whether to use robust way to estimate noise
            adaptive_threshold: bool
                whether to use adaptive threshold method for deciding threshold level.
                Currently it should always be True.
            fr: float
                movie frame rate
            freq: float
                frequency for removing subthreshold activity
            minimal_thresh: float
                minimal threshold allowed for adaptive threshold. Threshold lower than minimal_thresh
                will be adjusted to minimal_thresh.
            online_filter_method: str
                Use 'median_filter' or 'median_filter_no_lag' for doing online filter. 'median filter'
                is more accurate but also introduce more lags
            filt_window: int or list
                window size for the median filter. The median filter is not symmetric when the two values 
                in the list is not the same(for example [8, 4] means using 8 past frames and 4 future frames for 
                                            removing the median of the current frame)                 
            do_plot: bool
                Whether to plot or not.
            p: int
                order of the AR process for calcium deconvolution
                no deconvolution is performed if p=0
            nb: int
                number of background components
        '''
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)
                logging.debug(f'{name}, {value}')
        self.t_detect = []
        self.t_update = []
        
        self.update_q = Queue()
        self.update_q.put(0)
        self.update_thread = Thread(target=self.update_statistics_thread, daemon=True)
        self.update_thread.start()
        
    def fit(self, trace_in, num_frames):
        """
        Method for computing spikes in offline mode, and initializer for the online mode
        Args:
            trace_in: ndarray
                the vector (num neurons x time steps) used to initialize the online process
            num_frames: int
                total number of frames for processing including both initialization and online processing
        """
        nn, tm = np.array(trace_in).shape # number of neurons and time steps for initialization
        self.nn = nn
        self.frames_init = tm
        self.n = tm        
        self.N = self.nn - self.nb # N number of neurons, excluding background components when doing deconvolution/spike detection
        
        if self.step < 3 * self.N:
            raise Exception('too many neurons for updating statistics, please increase parameter step')
        
        if self.mode == 'voltage':
            # contains all the extracted fluorescence traces
            self.trace =  np.zeros((self.nn, num_frames), dtype=np.float32)
            self.t_d, self.t0, self.t, self.t_s, self.t_sub, self.trace_deconvolved = (np.zeros((self.N, num_frames), dtype=np.float32) for _ in range(6))
            
            # contains running statistics
            self.median, self.scale, self.thresh, self.median2, self.std = (np.zeros((self.N,1), dtype=np.float32) for _ in range(5))
    
            # contains spike time
            self.index = np.zeros((self.N, num_frames), dtype=np.int32)
            self.index_track = np.zeros((self.N), dtype=np.int32)        
            self.peak_to_std = np.zeros((self.N, num_frames), dtype=np.float32)
            self.SNR, self.thresh_factor, self.peak_level = (np.zeros((self.N, 1), dtype=np.float32) for _ in range(3))
            if self.template_window > 0:
                self.PTA = np.zeros((self.N, 2*self.template_window+1), dtype=np.float32)
            else:
                self.PTA = np.zeros((self.N, 5), dtype=np.float32)
            
            t_start = time()
            # initialize for each neuron: @todo parallelize
            if self.flip:
                self.trace[:, :tm] = -trace_in.copy()
            else:
                self.trace[:, :tm] = trace_in.copy()
           
            if self.detrend:
                self.t_d[:, 0] = self.trace[:self.N, 0]
                for tp in range(trace_in.shape[1]):
                    if tp > 0:
                        self.t_d[:, tp] = self.trace[:self.N, tp] - self.trace[:self.N, tp - 1] + self.dc_param * self.t_d[:, tp - 1]
            else:
                self.t_d = self.trace[:self.N, :].copy()
                
            if self.do_deconvolve:
                for idx, tr in enumerate(self.t_d[:, :tm]):  
                    output_list = find_spikes_tm(tr, self.freq, self.fr, self.do_scale, self.filt_window, self.template_window,
                                                     self.robust_std, self.adaptive_threshold, self.minimal_thresh, self.do_plot)
                    self.index_track[idx] = output_list[0].shape[0]
                    self.index[idx, :self.index_track[idx]], self.thresh[idx], self.PTA[idx], self.t0[idx, :tm], \
                        self.t[idx, :tm], self.t_s[idx, :tm], self.t_sub[idx, :tm], self.median[idx], self.scale[idx], \
                            self.thresh_factor[idx], self.median2[idx], self.std[idx], \
                                self.peak_to_std[idx, :self.index_track[idx]], self.peak_level[idx] = output_list
                    self.trace_deconvolved[idx][self.index[idx]] = 1
                    self.trace_deconvolved[idx, 0] = 0 # exclude the first frame
            else:
                logging.info('skipping deconvolution')
                                    
            self.t_detect = self.t_detect + [(time() - t_start) / tm] * trace_in.shape[1]         
        elif self.mode == 'calcium':
            self.trace = np.zeros((nn, num_frames), dtype=np.float32)
            self.t_d = np.zeros((self.N, num_frames), dtype=np.float32)
            self.trace[:, :tm] = trace_in.copy()
            
            if self.flip:
                raise Exception('flipping signal is not supported for calcium imaging')
                
            if self.detrend:
                self.t_d[:, 0] = self.trace[:self.N, 0]
                for tp in range(tm):
                    if tp > 0:
                        self.t_d[:, tp] = self.trace[:self.N, tp] - self.trace[:self.N, tp - 1] + self.dc_param * self.t_d[:, tp - 1]
                self.median = np.median(trace_in[:self.N], axis=1)
                self.t_d = self.t_d - self.median[:, np.newaxis]
            else:
                self.t_d = self.trace[:self.N].copy()
                self.median = np.median(trace_in[:self.N], axis=1)
                self.t_d = self.t_d - self.median[:, np.newaxis]
            
            t_start = time()
            #idx=0
            if self.do_deconvolve: 
                #print(idx,time())
                #idx+=1
                if self.p>0: # calcium deconvolution
                    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
                    self.trace_deconvolved = np.zeros((self.N, num_frames), dtype=np.float32)
                    results_foopsi = map(lambda t: constrained_foopsi(t, p=self.p), self.t_d[:self.N, :tm])
                    if self.p==1:
                        self.b, self.lam, self.g = np.array([[r[1], r[-1], r[3][0]] for r in
                                                            results_foopsi], dtype=np.float32).T
                        self._lg = np.log(self.g)
                        self._bl = self.b + self.lam*(1-self.g)
                        self._v, self._w = np.zeros((2, self.N, 50), dtype=np.float32)
                        self._t, self._l = np.zeros((2, self.N, 50), dtype=np.int32)
                        self._i = np.zeros(self.N, dtype=np.int32)  # number of pools (spikes)
                        n = 0
                        for y in self.t_d[:self.N, :tm].T:
                            par_fit_next_AR1(y-self._bl, self.trace_deconvolved, self._lg,
                                              self._v, self._w, self._t, self._l, self._i, n)
                            n += 1
                            tmp = self._v.shape[1]
                            if self._i.max() >= tmp:
                                vw = np.zeros((2, self.N, tmp+50), dtype=np.float32)
                                tl = np.zeros((2, self.N, tmp+50), dtype=np.int32)
                                vw[:,:,:tmp] = self._v, self._w
                                tl[:,:,:tmp] = self._t, self._l
                                self._v,self._w = vw
                                self._t,self._l = tl
                    elif self.p==2:
                        self.b, self.lam, g1, g2 = np.array([[r[1], r[-1], *r[3]] for r in
                                                            results_foopsi], dtype=np.float32).T
                        self.g = np.transpose([g1,g2])
                        self._bl = self.b + self.lam*(1-self.g.sum(1))
                        # precompute
                        self._d = (g1 + np.sqrt(g1*g1 + 4*g2)) / 2
                        self._r = (g1 - np.sqrt(g1*g1 + 4*g2)) / 2
                        self._g11 = ((np.exp(np.outer(np.log(self._d), np.arange(1, 1001))) -
                                        np.exp(np.outer(np.log(self._r), np.arange(1, 1001)))) / 
                                        (self._d - self._r)[:,None]).astype(np.float32)
                        self._g12 = np.zeros_like(self._g11)
                        self._g12[:,1:] = g2[:,None] * self._g11[:,:-1]
                        self._g11g11 = np.cumsum(self._g11 * self._g11, axis=1)
                        self._g11g12 = np.cumsum(self._g11 * self._g12, axis=1)
                        n = 0
                        # initialize
                        self._y = np.empty((self.N, num_frames), dtype=np.float32)
                        self._v, self._w = np.zeros((2, self.N, 50), dtype=np.float32)
                        self._t, self._l = np.zeros((2, self.N, 50), dtype=np.int32)
                        self._i = np.zeros(self.N, dtype=np.int32)  # number of pools (spikes)
                        # process
                        for yt in self.t_d[:self.N].T:
                            self._y[:,n] = yt-self._bl
                            par_fit_next_AR2(self._y, self.trace_deconvolved, self._d,
                                              self._g11, self._g12, self._g11g11, self._g11g12,
                                              self._v, self._w, self._t, self._l, self._i,n)
                            n +=1
                            tmp = self._v.shape[1]
                            if self._i.max()>=tmp:
                                vw = np.zeros((2, self.N, tmp+50), dtype=np.float32)
                                tl = np.zeros((2, self.N, tmp+50), dtype=np.int32)
                                vw[:,:,:tmp] = self._v, self._w
                                tl[:,:,:tmp] = self._t, self._l
                                self._v,self._w = vw
                                self._t,self._l = tl
            else:
                logging.info('skipping deconvolution')
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
        t_start = time() 

        if self.mode == 'voltage':
            # Detrend, normalize 
            if self.flip:
                self.trace[:, n:(n + 1)] = - trace_in.copy()
            else:
                self.trace[:, n:(n + 1)] = trace_in.copy()
            if self.detrend:
                self.t_d[:, n:(n + 1)] = self.trace[:self.N, n:(n + 1)] - self.trace[:self.N, (n - 1):n] + self.dc_param * self.t_d[:, (n - 1):n]
            else:
                self.t_d[:, n:(n + 1)] = self.trace[:self.N, n:(n + 1)]
                                
            if self.do_deconvolve:
                temp = self.t_d[:, n:(n+1)].copy()        
                temp -= self.median[:, -1:]
                if self.do_scale == True:
                    temp /= self.scale[:, -1:]
                self.t0[:, n:(n + 1)] = temp.copy()        
    
                # remove subthreshold
                if self.online_filter_method == 'median_filter':
                    if type(self.filt_window) is int:
                        sub_index = self.n - int((self.filt_window - 1) / 2)        
                        lag = int((self.filt_window - 1) / 2)  
                        if self.n >= self.frames_init + lag:
                            temp = self.t0[:, self.n - self.filt_window + 1: self.n + 1]
                            self.t_sub[:, sub_index] = np.median(temp, 1)
                            self.t[:, sub_index:sub_index + 1] = self.t0[:, sub_index:sub_index + 1] - self.t_sub[:, sub_index:sub_index + 1]
                    elif type(self.filt_window) is list:
                        sub_index = self.n - self.filt_window[1]
                        lag = self.filt_window[1]  
                        if self.n >= self.frames_init + lag:
                            temp = self.t0[:, self.n - sum(self.filt_window) : self.n + 1]
                            self.t_sub[:, sub_index] = np.median(temp, 1)
                            self.t[:, sub_index:sub_index + 1] = self.t0[:, sub_index:sub_index + 1] - self.t_sub[:, sub_index:sub_index + 1]
                        
                elif self.online_filter_method == 'median_filter_no_lag':
                    sub_index = self.n        
                    lag = 0
                    if self.n >= self.frames_init:
                        temp = np.zeros((self.N, self.filt_window))
                        temp[:, :int((self.filt_window - 1) / 2) + 1] = self.t0[:, self.n - int((self.filt_window - 1) / 2): self.n + 1]
                        temp[:, int((self.filt_window - 1) / 2) + 1:] = np.flip(self.t0[:, self.n - int((self.filt_window - 1) / 2): self.n])
                        self.t_sub[:, sub_index] = np.median(temp, 1)
                        self.t[:, sub_index:sub_index + 1] = self.t0[:, sub_index:sub_index + 1] - self.t_sub[:, sub_index:sub_index + 1]
                
                if self.template_window > 0:
                    if self.n >= self.frames_init + lag + self.template_window:                                      
                        temp = self.t[:, sub_index - 2*self.template_window : sub_index + 1]                         
                        self.t_s[:, sub_index - self.template_window] = (temp * self.PTA).sum(axis=1)
                        self.t_s[:, sub_index - self.template_window] = self.t_s[:, sub_index - self.template_window] - self.median2[:, -1]
                else:
                    if self.n >= self.frames_init + lag + self.template_window:                                      
                        self.t_s[:, sub_index] = self.t[:, sub_index]
                        
                # Find spikes above threshold
                if self.n >= self.frames_init + lag + self.template_window + 1:
                    temp = self.t_s[:, sub_index - self.template_window - 2: sub_index -self.template_window + 1].copy()
                    idx_list = np.where((temp[:, 1] > temp[:, 0]) * (temp[:, 1] > temp[:, 2]) * (temp[:, 1] > self.thresh[:, -1]))[0]
                    if idx_list.size > 0:
                        self.index[idx_list, self.index_track[idx_list]] = sub_index - self.template_window - 1
                        self.peak_to_std[idx_list, self.index_track[idx_list]] = self.t_s[idx_list, sub_index - self.template_window - 1] /self.std[idx_list, -1]
                        self.index_track[idx_list] += 1  
                        self.trace_deconvolved[idx_list, sub_index - self.template_window - 1] = 1

        elif self.mode == 'calcium':
            self.trace[:, n:(n+1)] = trace_in.copy()  
            
            if self.flip:
                raise Exception('flipping signal is not supported for calcium imaging')
                
            if self.detrend:
                self.t_d[:, n:(n + 1)] = self.trace[:self.N, n:(n + 1)] - self.trace[:self.N, (n - 1):n] + self.dc_param * self.t_d[:, (n - 1):n]
                self.t_d[:, n:(n + 1)] = self.t_d[:, n:(n + 1)] - self.median[:, np.newaxis]
            else:
                self.t_d[:, n:(n + 1)] = self.trace[:self.N, n:(n + 1)]
                self.t_d[:, n:(n + 1)] = self.t_d[:, n:(n + 1)] - self.median[:, np.newaxis]

            if self.do_deconvolve:
                if self.p>0: # deconvolve/denoise
                    N = len(self._bl)
                    if self.p==1:
                        par_fit_next_AR1(self.t_d[:N,n]-self._bl, self.trace_deconvolved, self._lg,
                                          self._v, self._w, self._t, self._l, self._i, n)
                        tmp = self._v.shape[1]
                        if self._i.max() >= tmp:
                            vw = np.zeros((2, N, tmp+50), dtype=np.float32)
                            tl = np.zeros((2, N, tmp+50), dtype=np.int32)
                            vw[:,:,:tmp] = self._v, self._w
                            tl[:,:,:tmp] = self._t, self._l
                            self._v,self._w = vw
                            self._t,self._l = tl
                    elif self.p==2:
                        self._y[:,n] = self.t_d[:N,n]-self._bl
                        par_fit_next_AR2(self._y, self.trace_deconvolved, self._d,
                                          self._g11, self._g12, self._g11g11, self._g11g12,
                                          self._v, self._w, self._t, self._l, self._i,n)
                        tmp = self._v.shape[1]
                        if self._i.max()>=tmp:
                            vw = np.zeros((2, N, tmp+50), dtype=np.float32)
                            tl = np.zeros((2, N, tmp+50), dtype=np.int32)
                            vw[:,:,:tmp] = self._v, self._w
                            tl[:,:,:tmp] = self._t, self._l
                            self._v,self._w = vw
                            self._t,self._l = tl
        self.t_detect.append(time() - t_start)
        if self.do_deconvolve:
            self.update_q.put(n)
        return self
    
    def update_statistics(self, n):
        if self.mode == 'voltage':
            res = n % self.step        
            if ((n > 3.0 * self.window) and (res < 3 * self.N)):
                idx = int(res / 3)   # index of neuron waiting for updating
                temp = res - 3 * idx
                if temp == 0:
                    self.update_median(idx)
                elif temp == 1:
                    if self.do_scale:
                        self.update_scale(idx)
                elif temp == 2:
                    self.update_thresh(idx)
        return self
    
    def update_statistics_thread(self):
        self.flag_update=0  
        while True:
            n = self.update_q.get() 
            if self.flag_update > 0:
                self.update_statistics(n)                     
            self.flag_update = self.flag_update + 1
            sleep(1e-5)
            self.t_update.append(time())
    
    def update_median(self, idx):
        tt = self.t_d[idx, self.n - np.int(self.window*2.5):self.n]
        if idx == 0:
            self.median = np.append(self.median, self.median[:, -1:], axis=1)
        self.median[idx, -1] = np.median(tt)
        return self
    
    def update_scale(self, idx):
        tt = self.t_d[idx, self.n - np.int(self.window*2.5):self.n]
        if idx == 0:
            self.scale = np.append(self.scale, self.scale[:, -1:], axis=1)
        self.scale[idx, -1] = - np.percentile(tt, 1)
        return self
        
    def update_thresh(self, idx):
        # add a new column of threshold
        if idx == 0:
            self.thresh = np.append(self.thresh, self.thresh[:, -1:], axis=1)   
            self.thresh_factor = np.append(self.thresh_factor, self.thresh_factor[:, -1:], axis=1)   
            self.peak_level = np.append(self.peak_level, self.peak_level[:, -1:], axis=1)   
            self.std = np.append(self.std, self.std[:, -1:], axis=1)   
        tt = self.t_s[idx, self.n - np.int(self.window * 2.5):self.n]
     
        if self.index[idx, self.index_track[idx]-100] > 300:       
            temp = self.peak_to_std[idx, self.index_track[idx]-100:self.index_track[idx]]
        else:
            temp = self.peak_to_std[idx, np.where(self.index[idx, :]>300)[0][0]: self.index_track[idx]]
        peak_level = np.percentile(temp, 95)
        thresh_factor = self.thresh_factor[idx, 0] * peak_level / self.peak_level[idx, -1]
        
        if thresh_factor >= self.minimal_thresh:
            if thresh_factor <= self.thresh_factor[idx, -1]:
                self.thresh_factor[idx, -1] = thresh_factor   # thresh_factor never increase
        else:
            self.thresh_factor[idx, -1] = self.minimal_thresh
        
        if self.robust_std:
            thresh_new = (-np.percentile(tt, 25) * 2 / 1.35) * self.thresh_factor[idx]
        else:    
            std = compute_std(tt)
            thresh_new =  std* self.thresh_factor[idx, -1]
        self.std[idx, -1] = std
        self.thresh[idx, -1] = thresh_new        
        return self
        
    def compute_SNR(self):
        for idx in range(self.trace.shape[0]):
            sg = np.percentile(self.t_s[idx], 99)
            std = compute_std(self.t_s[idx])
            snr = sg / std
            self.SNR[idx] = snr
        return self

    def reconstruct_signal(self):
        if self.mode == 'voltage':
            self.t_rec = np.zeros(self.trace.shape)
            for idx in range(self.N):
            #     spikes = np.array(list(set(self.index[idx])-set([0])))
            #     if spikes.size > 0:
            #         self.t_rec[idx, spikes] = 1
                self.t_rec[idx] = np.convolve(self.trace_deconvolved[idx], np.flip(self.PTA[idx]), 'same')   #self.scale[idx,0]
        elif self.mode == 'calcium' and self.p > 0:
            T = self.trace.shape[1]
            N = len(self._v)
            self.trace_denoised = np.zeros_like(self.trace_deconvolved)
            if self.p==1:
                reconstruct_AR1(self.trace_deconvolved, self.trace_denoised, self.g,
                                self._v, self._w, self._t, self._i)
            elif self.p==2:
                reconstruct_AR2(self.trace_denoised, self._d, self.g,
                                self._v, self._w, self._t, self._l, self._i)
        return self

    def reconstruct_movie(self, A, shape, scope):
        self.A = A.reshape((shape[0], shape[1], -1), order='F')
        self.C = self.t_rec[:, scope[0]:scope[1]]
        self.mov_rec = np.dot(self.A, self.C).transpose((2,0,1))    
        return self

def find_spikes_tm(img, freq, fr, do_scale=False, filt_window=15, template_window=2, robust_std=False, 
                   adaptive_threshold=True, minimal_thresh=3.0, do_plot=False):
    """
    Offline template matching methods to find peaks, decide threshold
    Parameters
    ----------
    img : ndarray
        Input one dimensional detrended signal.
    freq : float
        Frequency for subthreshold extraction.
    fr : float
        Movie frame rate
    do_scale : bool, optional
        Whether scale the signal or not. The default is False.
    filt_window: int or list
        window size for the median filter. The median filter is not symmetric when the two values 
        in the list is not the same(for example [8, 4] means using 8 past frames and 4 future frames for 
                                    removing the median of the current frame)                 
    template_window: int
        half window size for template matching. Will not perform template matching
        if 0.                
    robust_std: bool
        whether to use robust way to estimate noise
    adaptive_threshold: bool
        whether to use adaptive threshold method for deciding threshold level.
        Currently it should always be True.
    minimal_thresh: float
        minimal threshold allowed for adaptive threshold. Threshold lower than minimal_thresh
        will be adjusted to minimal_thresh.
    do_plot: bool
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
    std: float
        Estimated standard deviation
    peak_to_std: float
        Spike height devided by std
    peak_level: float
        95 percentile of the peak_to_std. 
        It is updated to compensate for the photobleaching effect.
    """
    # Normalize and remove subthreshold
    median = np.median(img)
    scale = -np.percentile(img-median, 1)
    if do_scale == True:
        t0 = (img - median) / scale
    else:
        t0 = img.copy() - median
    
    #sub = signal_filter(t0, freq, fr, order=1, mode='low')
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
        logging.info('skipping template matching')
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
        
    # Select best threshold based on estimated false positive rate
    if adaptive_threshold:
        pks2 = t_s[signal.find_peaks(t_s, height=None)[0]]
        try:
            thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=0.5, min_spikes=10)  # clip=0 means no clipping
            thresh_factor = thresh2 / std
            if thresh_factor < minimal_thresh:
                logging.warning(f'adaptive threshold factor is lower than minimal theshold: {minimal_thresh}, choose thresh factor to be {minimal_thresh}')
                thresh_factor = minimal_thresh
            thresh2 = thresh_factor * std
        except:
            logging.warning('adaptive threshold fails, automatically choose thresh factor to be 3')
            thresh_factor = 3
            thresh2 = thresh_factor * std
    else:
        raise ValueError('method not supported')

    index = signal.find_peaks(data, height=thresh2)[0]
    logging.info(f'final threshhold: {thresh2/std}')
                
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
        plt.title('spike distribution')
        plt.tight_layout()
        plt.show()  
        
    peak_to_std = data[index] / std    
    try:
        peak_level = data[index[index>300]] / std # remove peaks in first three hundred frames to improve robustness
    except:
        peak_level = data[index] / std 

    peak_level = np.percentile(peak_level, 95)

        
        

    return index, thresh2, PTA, t0, t, t_s, sub, median, scale, thresh_factor, median2, std, peak_to_std, peak_level 
