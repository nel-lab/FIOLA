#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 08:09:48 2020

@author: agiovann
"""
from caiman.base.movies import rolling_window
#from functools import partial
import cv2
import scipy
#from scipy.interpolate import interp1d
#import scipy
import numpy as np
#from scipy.signal import argrelextrema
from time import time
from running_statistics import estimate_running_std
import peakutils
from scipy import signal
from running_statistics import compute_thresh
from sklearn.covariance import EllipticEnvelope
from scipy.stats import multivariate_normal
#%%
def extract_exceptional_events(z_signal, input_erf=False, thres_STD=5, N=2, min_dist=1, bidirectional=False):
    """
    Extract peaks that are stastically significant over estimated noise
    
    
    N: int
        window used to compute exceptionality (higher frame rates than 1Khz
        MIGHT need larger N) 
        
    input_erf: bool
        if True it means that the inpur is already an error function
    
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
    if input_erf:
        erf = z_signal
    else:
        if bidirectional:        
            erf = scipy.special.log_ndtr(-np.abs(z_signal))
        else:
            erf = scipy.special.log_ndtr(-z_signal)
    
    if N>0:
        
        erf = np.cumsum(erf)
        erf[N:] -= erf[:-N]
        # compensate for delay
        erf = np.roll(erf,-N+1)
    #indexes = np.where(-erf>=thres_STD)[0]
    indexes = peakutils.indexes(-erf, thres=thres_STD, min_dist=min_dist, thres_abs=True)
    return indexes, erf
#%%
    
#%%     
#%%        
def find_spikes(signal_orig, signal_no_subthr=None, normalize_signal=False, thres_STD=3.5, thres_STD_ampl=4, 
                mode='anomaly', only_rising=False, samples_covariance=10000, min_dist=1, N=2, 
                win_size=20000, stride=5000, spike_before=3, spike_after=4,
                q_min=25, q_max=75, bidirectional=False):
    """
    Function that extracts spikes from np.diff(signal). In general the only 
    parameters that should be adapted are thres_STD ('anomaly' and 'exceptionality')
    and thres_STD_ampl (only for 'exceptionality')
    
    Args:
        signal: ndarray
            fluorescence signal after detrending
        
        signal_no_subthr: ndarray, None
            signal without subthreshold activity, required for mode 'anomaly'
            if None not used 

        thres_STD: float
            threshold related to z scored signal or anomaly probability
            
        thres_STD_ampl: float
            threshold related to z scored amplitude signal without subthreshold activity
            
        mode: str
            'anomaly': use anomaly detection using robust covariate estimates 
                       from 2D Gaussian, requires signal_no_subthresh
            'exceptionality': using single dimension gaussian on difference of 
                       signal
            
        samples_covariance: int
            number of samples used to estimate the covariance for anomaly detection
        
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
            
        estimator: sklearn.covariance.EllipticEnvelope object
            estimator for anomaly
            
    """

    signal = np.diff(signal_orig)
    if normalize_signal:
        std_run = estimate_running_std(signal, win_size, stride, q_min=q_min, q_max=q_max)
        z_signal = signal/std_run
        index_exceptional,aa = extract_exceptional_events(z_signal, thres_STD=thres_STD, N=N, min_dist=min_dist, bidirectional=bidirectional)    
        index_remove = np.concatenate([index_exceptional+ii for ii in range(-spike_before,spike_after)])
        std_run = estimate_running_std(signal, win_size, stride,idx_exclude=index_remove, q_min=q_min, q_max=q_max)
        z_signal = signal/std_run 
        
        if signal_no_subthr is not None:
            signal_no_subthr /= estimate_running_std(signal_no_subthr, win_size, stride, 
                                                     q_min=q_min, q_max=q_max)
    else:
        z_signal = signal
        
    if mode == 'exceptionality':
        indexes, erf = extract_exceptional_events(z_signal, thres_STD=thres_STD,
                                                  N=N, min_dist=min_dist, 
                                                  bidirectional=bidirectional)
    
        #remove spikes that are not large peaks in the original signal
        if signal_no_subthr is not None:
            indexes = np.intersect1d(indexes,np.where(signal_no_subthr[1:]>thres_STD_ampl))
        
        estimator = None
    elif mode == 'anomaly':
        estimator = EllipticEnvelope().fit(np.vstack([z_signal[:samples_covariance], signal_no_subthr[1:samples_covariance+1]]).T)
        
        rv = multivariate_normal(estimator.location_, estimator.covariance_)
        npdf = (1-rv.cdf(np.vstack([z_signal[:],signal_no_subthr[1:]]).T))
        erf = np.log(npdf)
        
        if only_rising:
            erf = erf*(z_signal>0)
                
        indexes, erf = extract_exceptional_events(erf, input_erf=True, thres_STD=thres_STD,
                                                  N=N, min_dist=min_dist, 
                                                  bidirectional=bidirectional)
    elif mode == 'multi_peak':
        t, indexes, thresh_sub, thresh, peak_height, median, scale = \
                                    find_spikes_rh(signal_orig, do_scale=False)
        
    return indexes, erf, z_signal, estimator
#%%
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


#def estimate_subthreshold(signal, thres_STD = 5, spike_before=3, spike_after=4, kernel_size=21, return_nans=False):
#  delta_sig = np.diff(signal)
#  index_exceptional, erf, z_sig = find_spikes(delta_sig, thres_STD=thres_STD)
#  index_remove = np.concatenate([index_exceptional+ii for ii in range(-spike_before,spike_after)])
#  sig_sub = signal.copy()
#  sig_sub[np.minimum(index_remove,len(signal)-1)] = np.nan
#  if return_nans:
#      return sig_sub
#  else:
#      nans, x= nan_helper(sig_sub)
#      sig_sub[nans]= np.interp(x(nans), x(~nans), sig_sub[~nans])
#      sig_sub = scipy.signal.medfilt(sig_sub, kernel_size=kernel_size)
#      return sig_sub
  
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
def find_spikes_rh(t, thresh_height=None, window_length = 2, 
                           witgh_descending_peak=0.7,
                           do_scale=True):
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
    median = np.median(t) 
    t = t - median
    if do_scale:
        scale = np.percentile(t, 99.9)  
        t = t / scale
    else: 
        scale = None
        
    window_length = 2
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    index = signal.find_peaks(t, height=None)[0]
    index = index[np.logical_and(index > (-window[0]), index < (len(t) - window[-1]))]
    matrix = t[index[:, np.newaxis]+window]
    left = np.maximum((matrix[:,2] - matrix[:,1]), (matrix[:,2] - matrix[:,0]))  
    right = np.maximum((matrix[:,2] - matrix[:,3]), (matrix[:,2] - matrix[:,4]))  
    peak_height = 1 / (1 / left + witgh_descending_peak / right)
    
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
    

    return t, index, thresh_sub, thresh, peak_height, median, scale   

#def find_spikes_rh_noscale(t, thresh_height):
#    """ Find spikes based on the relative height of peaks
#    Args:
#        t: 1-D array
#            one dimensional signal
#            
#        thresh height: int
#            selected threshold
#            
#    Returns:
#        index: 1-D array
#            index of spikes
#    """
#        
#    # List peaks based on their relative peak heights
#    #t = img.copy()
#    median = np.median(t) 
#    t = t - median
#    window_length = 2
#    window = np.int64(np.arange(-window_length, window_length + 1, 1))
#    index = signal.find_peaks(t, height=None)[0]
#    index = index[np.logical_and(index > (-window[0]), index < (len(t) - window[-1]))]
#    matrix = t[index[:, np.newaxis]+window]
#    left = np.maximum((matrix[:,2] - matrix[:,1]), (matrix[:,2] - matrix[:,0]))  
#    right = np.maximum((matrix[:,2] - matrix[:,3]), (matrix[:,2] - matrix[:,4]))  
##    peak_height = 1 / (1 / left + 0.7 / right)
#    peak_height = 1 / (1 / left + 0.7 / right)
#    
#    # Thresholding
#    data = peak_height - np.median(peak_height)
#    ff1 = -data * (data < 0)
#    Ns = np.sum(ff1 > 0)
#    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
#    if thresh_height is not None:
#        thresh = thresh_height * std
#    else:
#        thresh = compute_thresh(peak_height)
#        
#    index = index[peak_height > thresh]
#
#    # Only select peaks above subthreshold
#    tt = -t[t<0]
#    thresh_sub = np.percentile(tt, 99)
#    index_sub = np.where(t > thresh_sub)[0]
#    index = np.intersect1d(index,index_sub)
#    
#    #plt.hist(peak_height, 500)
#    #plt.vlines(thresh, peak_height.min(), peak_height.max(), color='red')
#    #plt.figure()
#    #plt.plot(dict1['v_t'], t); plt.vlines(dict1['v_t'][index], t.min()-5, t.min(), color='red');
#    #plt.vlines(dict1['e_sp'], t.min()-10, t.min()-5, color='black') 
#    
#    return index, thresh_sub, thresh, peak_height, median

def find_spikes_rh_multiple(t, t_rm, t_in, median, scale, thresh, thresh_sub, index, peak_height):
    t = np.append(t, t_in, axis=1)
    t_in = (t_in - median[:, -1:]) / scale[:, -1:]
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

#%%


def estimate_subthreshold_signal(signal, mode='percentile', perc_subthr=20,
                                 perc_window=50, perc_stride=25, thres_STD = 5, 
                                 kernel_size=21,return_nans=False):
    
    """ Estimate the subthreshold signal with different methods
    Args:
        signal: ndarray
            fluorescence signal after detrending
        
        mode: str            
             'minimum': use minimum over window and with a stride, then interpolate
             'percentile': use lowest perc_subthr percentile  over window and with a stride, then interpolate
             'nospike_filter': remove spikes and then apply a median filter with window kernel_size (odd)

        perc_subthr: float
            lowest percentile to remove for subthreshold signal (only mode 
            'percentile')
        
        perc_window: int
            window over which to compute percentile (only mode 
            'percentile' and 'minimum')
            
        perc_stride: 
            stride to compute percentile (only mode 
            'percentile' and 'minimum')       

        thres_STD: float
            threshold related to z scored signal (only mode 
            'nospike_filter')
            
        return_nans: bool
            whether to return nan on the trace for missing spikes (only mode 
            'nospike_filter')
        
    Returns:
        signal_subthr: ndarray
            subthreshold signal
        
    """
    
    if mode == 'percentile':
        perc = np.array([np.percentile(el,20) for el in rolling_window(signal.T[None,:], perc_window, perc_stride)])
        signal_subthr =  cv2.resize(perc, (1,signal.shape[0]),cv2.INTER_CUBIC).squeeze()
    elif mode == 'minimum':
        minima = np.array([np.min(el) for el in rolling_window(signal.T[None,:], 15, 5)])
        signal_subthr = cv2.resize(minima, (1,signal.shape[0]),interpolation = cv2.INTER_CUBIC).squeeze()
    elif mode == 'nospike_filter':
        delta_sig = np.diff(signal)
        index_exceptional, erf, z_sig = find_spikes(delta_sig, thres_STD=thres_STD)
        index_remove = np.concatenate([index_exceptional+ii for ii in range(-spike_before,spike_after)])
        sig_sub = signal.copy()
        sig_sub[np.minimum(index_remove,len(signal)-1)] = np.nan
        if return_nans:
            signal_subthr = sig_sub
        else:
            nans, x= nan_helper(sig_sub)
            sig_sub[nans]= np.interp(x(nans), x(~nans), sig_sub[~nans])
            signal_subthr = scipy.signal.medfilt(sig_sub, kernel_size=kernel_size)                
        
    return signal_subthr          