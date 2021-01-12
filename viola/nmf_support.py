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
from sklearn.decomposition import NMF
from .caiman_functions import to_3D, to_2D
from .spikepursuit import denoise_spikes
from viola.caiman_functions import signal_filter
import scipy
#%%
def mode_robust_fast(inputData, axis=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:

        def fnc(x):
            return mode_robust_fast(x)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        data = inputData.ravel()
        # The data need to be sorted for this to work
        data = np.sort(data)
        # Find the mode
        dataMode = _hsm(data)

    return dataMode

def _hsm(data):
    if data.size == 1:
        return data[0]
    elif data.size == 2:
        return data.mean()
    elif data.size == 3:
        i1 = data[1] - data[0]
        i2 = data[2] - data[1]
        if i1 < i2:
            return data[:2].mean()
        elif i2 > i1:
            return data[1:].mean()
        else:
            return data[1]
    else:

        wMin = np.inf
        N = data.size//2 + data.size % 2

        for i in range(0, N):
            w = data[i + N - 1] - data[i]
            if w < wMin:
                wMin = w
                j = i

        return _hsm(data[j:j + N])
#%%
def hals(Y, A, C, b, f, bSiz=3, maxIter=5, update_bg=True, use_spikes=False, frate=0):
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
                               V[m, m], -np.inf, np.inf)
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
    # Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
    for _ in range(maxIter):    
        Cf = HALS4activity(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
        Cf_processed = Cf.copy()

        if not update_bg:
            Cf_processed[-nb:] = np.zeros(Cf_processed[-nb:].shape)

        if use_spikes:
            for i in range(Cf.shape[0]):
                if i < Cf.shape[0] - nb: 
                    bl = scipy.ndimage.percentile_filter(Cf[i], 20, size=50)
                    tr = Cf[i] - bl
                    _, _, Cf_processed[i], _, _, _ = denoise_spikes(tr, window_length=3, clip=0, 
                                      threshold=3.0, threshold_method='simple', do_plot=False)
                    Cf_processed[i] = Cf_processed[i] + bl  
        Cf = Cf_processed
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
        
    return Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1)

def nmf_sequential(y_seq, mask, seq, small_mask=True):
    """ Use rank-1 nmf to sequentially extract neurons' spatial filters.
    
    Parameters
    ----------
    y_seq : ndarray, T * (# of pixel)
        Movie after detrend. It should be in 2D format.
    mask : ndarray, T * d1 * d2
        Masks of neurons
    seq : ndarray
        order of rank-1 nmf on neurons.  
    small_mask : bool, optional
        Use small context region when doing rank-1 nmf. The default is True.

    Returns
    -------
    W : ndarray
        Temporal components of neurons.
    H : ndarray
        Spatial components of neuron.

    """
    W_tot = []
    H_tot = []    
    for i in seq:
        print(f'now processing neuron {i}')
        model = NMF(n_components=1, init='nndsvd', max_iter=100, verbose=False)
        y_temp, _ = select_masks(y_seq, (y_seq.shape[0], mask.shape[1], mask.shape[2]), mask=mask[i])
        if small_mask:
            mask_dilate = cv2.dilate(mask[i],np.ones((4,4),np.uint8),iterations = 1)
            x0 = np.where(mask_dilate>0)[0].min()
            x1 = np.where(mask_dilate>0)[0].max()
            y0 = np.where(mask_dilate>0)[1].min()
            y1 = np.where(mask_dilate>0)[1].max()
            context_region = np.zeros(mask_dilate.shape)
            context_region[x0:x1+1, y0:y1+1] = 1
            context_region = context_region.reshape([-1], order='F')
            y_temp_small = y_temp[:, context_region>0]
            W = model.fit_transform(np.maximum(y_temp_small,0))
            H_small = model.components_
            #plt.figure(); plt.imshow(H_small.reshape([x1-x0+1, y1-y0+1], order='F')); plt.colorbar(); plt.show()
            H = np.zeros((1, y_temp.shape[1]))
            H[:, context_region>0] = H_small
        else:
            W = model.fit_transform(np.maximum(y_temp,0))
            H = model.components_
        y_seq = y_seq - W@H
        W_tot.append(W)
        H_tot.append(H)
    H = np.vstack(H_tot)
    W = np.hstack(W_tot)
        
    return W, H
    

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
    
    if mask is None:
        m = to_3D(Y,shape=shape, order='F')
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
        Y = to_2D((1.0 - mask)*m) 
    else:
        mask = cv2.dilate(mask,np.ones((4,4),np.uint8),iterations = 1)
        mask = (mask < 1)
        mask_2D = mask.reshape((mask.shape[0] * mask.shape[1]), order='F')
        Y = Y * (1.0 - mask_2D)
    # plt.figure();plt.plot(((m * (1.0 - mask)).mean(axis=(1, 2))))
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
            num_diff = num_frames - mov.shape[0]
            mov = np.concatenate((mov, mov[(mov.shape[0] - num_diff) : mov.shape[0], :, :]), axis=0)
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






