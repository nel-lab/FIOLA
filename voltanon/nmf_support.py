#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:40:18 2020
This file includes function using nmf for voltage imaging online processing
@author: @agiovann and @caichangjia
"""
import cv2
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import numpy as np
import pylab as plt
import scipy.ndimage as nd
import scipy.sparse as spr

import caiman as cm
from caiman.base.movies import to_3D
from caiman.source_extraction.volpy.spikepursuit import denoise_spikes


def hals(Y, A, C, b, f, bSiz=3, maxIter=5, update_bg=True, use_spikes=False):
    """ Hierarchical alternating least square method for solving NMF problem
    Y = A*C + b*f
    Args:
       Y:      d1 X d2 [X d3] X T, raw data.
           It will be reshaped to (d1*d2[*d3]) X T in this
           function
       A:      (d1*d2[*d3]) X K, initial value of spatial components
       C:      K X T, initial value of temporal components
       b:      (d1*d2[*d3]) X nb, initial value of background spatial component
       f:      nb X T, initial value of background temporal component
       bSiz:   int or tuple of int
        blur size. A box kernel (bSiz X bSiz [X bSiz]) (if int) or bSiz (if tuple) will
        be convolved with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0.
       maxIter: int,
           maximum iteration of iterating HALS.
       update_bg: bool, 
           update background if True, otherwise background will be set zero
       use_spikes: bool, 
           if True the algorithm will detect spikes using VolPy offline 
           to optimize the spatial components A
           
    Returns:
        the updated A, C, b, f
    Authors:
        Johannes Friedrich, Andrea Giovannucci
    See Also:
        http://proceedings.mlr.press/v39/kimura14.pdf
    """
    # smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    nb = b.shape[1]  # number of background components
    if bSiz is not None:
        if isinstance(bSiz, (int, float)):
             bSiz = [bSiz] * len(dims)
        ind_A = nd.filters.uniform_filter(np.reshape(A,
                dims + (K,), order='F'), size=bSiz + [0])
        ind_A = np.reshape(ind_A > 1e-10, (np.prod(dims), K), order='F')
    else:
        ind_A = A>1e-10
    ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels
    def HALS4activity(Yr, A, C, iters=2):
        U = A.T.dot(Yr)
        V = A.T.dot(A) + np.finfo(A.dtype).eps
        for _ in range(iters):
            for m in range(len(U)):  # neurons and background
                C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                               V[m, m], 0, np.inf)
        return C
    def HALS4shape(Yr, A, C, iters=2):
        U = C.dot(Yr.T)
        V = C.dot(C.T) + np.finfo(C.dtype).eps
        for _ in range(iters):
            for m in range(K):  # neurons
                ind_pixels = np.squeeze(ind_A[:, m].toarray())
                A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                           ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                            V[m, m]), 0, np.inf)
            for m in range(nb):  # background
                A[:, K + m] = np.clip(A[:, K + m] + ((U[K + m] - V[K + m].dot(A.T)) /
                                                     V[K + m, K + m]), 0, np.inf)
        return A
    Ab = np.c_[A, b]
    Cf = np.r_[C, f.reshape(nb, -1)]
    for _ in range(maxIter):
        Cf = HALS4activity(np.reshape(
            Y, (np.prod(dims), T), order='F'), Ab, Cf)
        Cf_processed = Cf.copy()

        if not update_bg:
            Cf_processed[-1] = np.zeros(Cf_processed[-1].shape)

        if use_spikes:
            for i in range(Cf.shape[0]):
                if i != Cf.shape[0] - 1 : 
                    _, _, Cf_processed[i], _, _, _ = denoise_spikes(Cf[i], window_length=3, 
                                      threshold=4, threshold_method='adaptive_threshold')
        Cf = Cf_processed
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
        for i in range(Ab.shape[1]):
            plt.figure();plt.imshow(Ab[:, i].reshape(Y.shape[0:2], order='F'));plt.colorbar()
        
    return Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1)

def select_masks(Y, shape, mask=None):
    """ Select a mask for nmf
    Args:
        Y: d1*d2 X T
            input data
        shape: tuple, (d1, d2)
            FOV size
        mask: ndarray with dimension d1 X d2
            if None, you will manually select contours of neuron
            if the mask is not None, it will be dilated
    Returns:
        updated mask and Y
    """
    m = to_3D(Y,shape=shape, order='F')
    if mask is None:
        frame = m[:10000].std(axis=0)
        plt.figure()
        plt.imshow(frame, cmap=mpl_cm.Greys_r)
        pts = []    
        while not len(pts):
            pts = plt.ginput(0)
            plt.close()
            path = mpl_path.Path(pts)
            mask = np.ones(np.shape(frame), dtype=bool)
            for ridx, row in enumerate(mask):
                for cidx, pt in enumerate(row):
                    if path.contains_point([cidx, ridx]):
                        mask[ridx, cidx] = False
    else:
        mask = cv2.dilate(mask,np.ones((4,4),np.uint8),iterations = 1)
        mask = (mask < 1)
    Y = cm.movie((1.0 - mask)*m).to_2D() 
    plt.figure();plt.plot(((m * (1.0 - mask)).mean(axis=(1, 2))))
    return Y, mask 

def combine_datasets(movies, masks, num_frames, x_shifts=[3,-3], y_shifts=[3,-3], weights=None, shape=(15,15)):
    """ Combine two datasets to create manually overlapping neurons
    Args: 
        movies: list
            list of movies
        masks: list
            list of masks
        num_frames: int
            number of frames selected
        x_shifts, y_shifts: list
            shifts in x and y direction relative to the original movie
        weights: list
            weights of each movie
        shape: tuple
            shape of the new combined movie
    
    Returns:
        new_mov: ndarray
            new combined movie
        new_masks: list
            masks for neurons
    """
    new_mov = 0
    new_masks = []
    if weights is None:
        weights = [1/len(movies)]*len(movies)
    for mov, mask, x_shift, y_shift, weight in zip(movies, masks, x_shifts,y_shifts, weights):
        new_mask = cm.movie(mask)
        if mov.shape[1] != shape[0]:
            mov = mov.resize(shape[0]/mov.shape[1],shape[1]/mov.shape[2],1)
            new_mask = new_mask.resize(shape[0]/mask.shape[1],shape[1]/mask.shape[2], 1)
            
        if num_frames > mov.shape[0]:
            mov = np.concatenate((mov, np.zeros((num_frames - mov.shape[0], mov.shape[1], mov.shape[2]))), axis=0)
        new_mov += np.roll(mov[:num_frames]*weight, (x_shift, y_shift), axis=(1,2))
        new_mask = np.roll(new_mask, (x_shift, y_shift), axis=(1,2))
        new_masks.append(new_mask[0])   
        
    return new_mov, new_masks

def normalize(signal):
    """ Normalize the signal
    Args: 
        signal: ndarray
            input signal
    
    Returns:
        normalized signal
        
    """
    signal = (signal-np.percentile(signal,1, axis=0))/(np.percentile(signal,99, axis=0)-np.percentile(signal,1, axis=0))
    signal -= np.median(signal)
    return signal




