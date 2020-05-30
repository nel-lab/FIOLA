#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:50:09 2019

@author: @caichangjia adapt based on Matlab code provided by Kaspar Podgorski and Amrita Singh
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import stats    
from scipy.ndimage.filters import gaussian_filter1d
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
from skimage.morphology import dilation
from skimage.morphology import disk
import cv2




def denoise_spikes(data, window_length, fr=400,  hp_freq=1, threshold_method='simple', 
                   min_spikes=5, threshold=3.5, last_round=False, do_plot=False):
    """ Function for finding spikes and the temporal filter given one dimensional signals.
        Use function whitened_matched_filter to denoise spikes. Two thresholding methods can be 
        chosen, 'simple' or 'adaptive thresholding'.

    Args:
        data: 1-d array
            one dimensional signal

        window_length: int
            length of window size for temporal filter

        fr: int
            number of samples per second in the video
            
        hp_freq: float
            high-pass cutoff frequency to filter the signal after computing the trace

        threshold_method: str
            'simple' or 'adaptive_threshold' method for thresholding signals
            'simple' method threshold based on estimated noise level 
            'adaptive_threshold' method threshold based on estimated peak distribution
            
        min_spikes: int
            minimal number of spikes to be detected

        threshold: float
            threshold for spike detection in 'simple' threshold method 
            The real threshold is the value multiply estimated noise level
            
        last_round: boolean
            if True no limit number of spikes to be found in the second round
            
        do_plot: boolean
            if Ture, will plot trace of signals and spiketimes, peak triggered
            average, histogram of heights
            
    Returns:
        datafilt: 1-d array
            signals after whitened matched filter

        spikes: 1-d array
            record of time of spikes

        t_rec: 1-d array
            recovery of original signals

        templates: 1-d array
            temporal filter which is the peak triggered average

        low_spikes: boolean
            True if number of spikes is smaller than 30
            
        thresh2: float
            real threshold in second round of spike detection 
    """
    # high-pass filter the signal to remove part of subthreshold activity
    data = data - np.median(data)
    data = signal_filter(data, hp_freq, fr, order=5)
    data = gaussian_filter1d(data, fr/500)          #fr/500
        
    low_spikes = False
    data = data - np.median(data)
    pks = data[signal.find_peaks(data, height=None, distance=int(fr/100))[0]]

    # find spikes    
    if threshold_method == 'simple':
        ff1 = -data * (data < 0)
        Ns = np.sum(ff1 > 0)
        std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
        thresh = 3.5 * std
        locs = signal.find_peaks(data, height=thresh, distance=int(fr/100))[0]
        if len(locs) < min_spikes:
            logging.warning(f'less than {min_spikes} spikes are found, pick top {min_spikes} spikes')
            thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
            locs = signal.find_peaks(data, height=thresh, distance=int(fr/100))[0]
            low_spikes = True
    elif threshold_method == 'adaptive_threshold':
        thresh, _, _, low_spikes = get_thresh(pks, 100, 0.25, min_spikes)
        locs = signal.find_peaks(data, height=thresh)[0]
    else:
        logging.warning("Error: threshold_method not found")
        raise Exception('Threshold_method not found!')

    # peak-traiggered average
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]
    PTD = data[(locs[:, np.newaxis] + window)]
    PTA = np.mean(PTD, 0)
    templates = PTA

    # whitened matched filter
    datafilt = whitened_matched_filter(data, locs, window)    
    datafilt = datafilt - np.median(datafilt)

    # spikes detected after filter
    pks2 = datafilt[signal.find_peaks(datafilt, height=None, distance=int(fr/100))[0]]
    if threshold_method == 'simple':
        ff1 = -datafilt * (datafilt < 0)
        Ns = np.sum(ff1 > 0)
        std2 = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
        thresh2 = threshold * std2
        spikes = signal.find_peaks(datafilt, height=thresh2, distance=int(fr/100))[0]
        if len(spikes) < min_spikes:
            low_spikes = True
            if last_round == True:
                logging.warning(f'last round: less than {min_spikes} spikes are found')
            else:
                logging.warning(f'less than {min_spikes} spikes are found, pick top {min_spikes} spikes')
                thresh2 = np.percentile(pks2, 100 * (1 - min_spikes / len(pks2)))
                spikes = signal.find_peaks(datafilt, height=thresh2, distance=int(fr/100))[0]
    elif threshold_method == 'adaptive_threshold':
        thresh2, falsePosRate, detectionRate, _ = get_thresh(pks2, clip=0, pnorm=0.5, min_spikes=min_spikes)  # clip=0 means no clipping
        spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    
    if len(spikes) > 0:
        t_rec = np.zeros(data.shape)
        t_rec[spikes] = 1
        t_rec = np.convolve(t_rec, PTA, 'same')   
        # filtering shrinks the data;
        # rescale so that the mean value at the peaks is same as in the input
        datafilt = datafilt * np.mean(data[spikes]) / np.mean(datafilt[spikes])
        thresh2 = thresh2 * np.mean(data[spikes]) / np.mean(datafilt[spikes])
    else:
        t_rec = np.zeros(data.shape)
        
    if do_plot:
        plt.figure()
        plt.subplot(211)
        plt.hist(pks, 500)
        plt.axvline(x=thresh, c='r')
        plt.title('raw data')
        plt.subplot(212)
        plt.hist(pks2, 500)
        plt.axvline(x=thresh2, c='r')
        plt.title('after matched filter')
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(np.transpose(PTD), c=[0.5, 0.5, 0.5])
        plt.plot(PTA, c='black', linewidth=2)
        plt.title('Peak-triggered average')
        plt.show()

        plt.figure()
        plt.subplot(211)
        plt.plot(data)
        plt.plot(locs, np.max(datafilt) * 1.1 * np.ones(locs.shape), color='r', marker='o', fillstyle='none',
                 linestyle='none')
        plt.plot(spikes, np.max(datafilt) * 1 * np.ones(spikes.shape), color='g', marker='o', fillstyle='none',
                 linestyle='none')
        plt.subplot(212)
        plt.plot(datafilt)
        plt.plot(locs, np.max(datafilt) * 1.1 * np.ones(locs.shape), color='r', marker='o', fillstyle='none',
                 linestyle='none')
        plt.plot(spikes, np.max(datafilt) * 1 * np.ones(spikes.shape), color='g', marker='o', fillstyle='none',
                 linestyle='none')
        plt.show()

    return datafilt, spikes, t_rec, templates, low_spikes, thresh2

def get_thresh(pks, clip, pnorm=0.5, min_spikes=30):
    """ Function for deciding threshold given heights of all peaks.

    Args:
        pks: 1-d array
            height of all peaks

        clip: int
            maximum number of spikes for producing templates

        pnorm: float, between 0 and 1, default is 0.5
            a variable deciding the amount of spikes chosen
            
        min_spikes: int
            minimal number of spikes to be detected

    Returns:
        thresh: float
            threshold for choosing spikes

        falsePosRate: float
            possibility of misclassify noise as real spikes

        detectionRate: float
            possibility of real spikes being detected

        low_spikes: boolean
            true if number of spikes is smaller than minimal value
    """
    # find median of the kernel density estimation of peak heights
    spread = np.array([pks.min(), pks.max()])
    spread = spread + np.diff(spread) * np.array([-0.05, 0.05])
    low_spikes = False
    pts = np.linspace(spread[0], spread[1], 2001)
    kde = stats.gaussian_kde(pks)
    f = kde(pts)    
    xi = pts
    center = np.where(xi > np.median(pks))[0][0]

    fmodel = np.concatenate([f[0:center + 1], np.flipud(f[0:center])])
    if len(fmodel) < len(f):
        fmodel = np.append(fmodel, np.ones(len(f) - len(fmodel)) * min(fmodel))
    else:
        fmodel = fmodel[0:len(f)]

    # adjust the model so it doesn't exceed the data:
    csf = np.cumsum(f) / np.sum(f)
    csmodel = np.cumsum(fmodel) / np.max([np.sum(f), np.sum(fmodel)])
    lastpt = np.where(np.logical_and(csf[0:-1] > csmodel[0:-1] + np.spacing(1), csf[1:] < csmodel[1:]))[0]
    if not lastpt.size:
        lastpt = center
    else:
        lastpt = lastpt[0]
    fmodel[0:lastpt + 1] = f[0:lastpt + 1]
    fmodel[lastpt:] = np.minimum(fmodel[lastpt:], f[lastpt:])

    # find threshold
    csf = np.cumsum(f)
    csmodel = np.cumsum(fmodel)
    csf2 = csf[-1] - csf
    csmodel2 = csmodel[-1] - csmodel
    obj = csf2 ** pnorm - csmodel2 ** pnorm
    maxind = np.argmax(obj)
    thresh = xi[maxind]

    if np.sum(pks > thresh) < min_spikes:
        low_spikes = True
        logging.warning(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
    elif ((np.sum(pks > thresh) > clip) & (clip > 0)):
        logging.warning(f'Selecting top {min_spikes} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))

    ix = np.argmin(np.abs(xi - thresh))
    falsePosRate = csmodel2[ix] / csf2[ix]
    detectionRate = (csf2[ix] - csmodel2[ix]) / np.max(csf2 - csmodel2)
    return thresh, falsePosRate, detectionRate, low_spikes


def whitened_matched_filter(data, locs, window):
    """
    Function for using whitened matched filter to the original signal for better
    SNR. Use welch method to approximate the spectral density of the signal.
    Rescale the signal in frequency domain. After scaling, convolve the signal with
    peak-triggered-average to make spikes more prominent.
    
    Args:
        data: 1-d array
            input signal

        locs: 1-d array
            spike times

        window: 1-d array
            window with size of temporal filter

    Returns:
        datafilt: 1-d array
            signal processed after whitened matched filter
    
    """
    N = np.ceil(np.log2(len(data)))
    censor = np.zeros(len(data))
    censor[locs] = 1
    censor = np.int16(np.convolve(censor.flatten(), np.ones([1, len(window)]).flatten(), 'same'))
    censor = (censor < 0.5)
    noise = data[censor]

    _, pxx = signal.welch(noise, fs=2 * np.pi, window=signal.get_window('hamming', 1000), nfft=2 ** N, detrend=False,
                          nperseg=1000)
    Nf2 = np.concatenate([pxx, np.flipud(pxx[1:-1])])
    scaling_vector = 1 / np.sqrt(Nf2)

    cc = np.pad(data.copy(),(0,np.int(2**N-len(data))),'constant')    
    dd = (cv2.dft(cc,flags=cv2.DFT_SCALE+cv2.DFT_COMPLEX_OUTPUT)[:,0,:]*scaling_vector[:,np.newaxis])[:,np.newaxis,:]
    dataScaled = cv2.idft(dd)[:,0,0]
    PTDscaled = dataScaled[(locs[:, np.newaxis] + window)]
    PTAscaled = np.mean(PTDscaled, 0)
    datafilt = np.convolve(dataScaled, np.flipud(PTAscaled), 'same')
    datafilt = datafilt[:len(data)]
    return datafilt


def signal_filter(sg, freq, fr, order=3, mode='high'):
    """
    Function for high/low passing the signal with butterworth filter
    
    Args:
        sg: 1-d array
            input signal
            
        freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
    Returns:
        sg: 1-d array
            signal after filtering            
    """
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg