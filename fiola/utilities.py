#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:42:18 2021

@author: nel
"""
import cv2

from functools import partial
import logging

from matplotlib import path as mpl_path
import matplotlib.cm as mpl_cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import ifftshift
from numpy.fft import fftn, ifftn

from past.utils import old_div
import random

import scipy
from scipy import signal
from scipy import stats    
from scipy.ndimage.filters import gaussian_filter1d
from scipy.sparse.linalg import svds
from scipy import signal
from scipy import stats  
from scipy.optimize import linear_sum_assignment
import scipy.ndimage as nd
import scipy.sparse as spr
from scipy.signal import argrelextrema, butter, sosfilt, sosfilt_zi


from skimage.morphology import dilation
from skimage.morphology import disk
from sklearn.linear_model import Ridge
from sklearn.decomposition import NMF

import time
from typing import Any, Dict, List, Optional, Tuple

from fiola.external.cell_magic_wand import cell_magic_wand_single_point

#%% Below are functions for 3D motion correction
def local_correlations_movie(self,
                       eight_neighbours: bool = False,
                       swap_dim: bool = False,
                       frames_per_chunk: int = 1500,
                       do_plot: bool = False,
                       order_mean: int =1) -> np.ndarray:
    """Computes the correlation image (CI) for the input movie. If the movie has
    length more than 3000 frames it will automatically compute the max-CI
    taken over chunks of a user specified length.

        Args:
            self:  np.ndarray (3D or 4D)
                Input movie data in 3D or 4D format

            eight_neighbours: Boolean
                Use 8 neighbors if true, and 4 if false for 3D data (default = True)
                Use 6 neighbors for 4D data, irrespectively

            swap_dim: Boolean
                True indicates that time is listed in the last axis of Y (matlab format)
                and moves it in the front (default: False)

            frames_per_chunk: int
                Length of chunks to split the file into (default: 1500)

            do_plot: Boolean (False)
                Display a plot that updates the CI when computed in chunks

            order_mean: int (1)
                Norm used to average correlations over neighborhood (default: 1).

        Returns:
            rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

    """
    T = self.shape[0]
    Cn = np.zeros(self.shape[1:])
    if T <= 3000:
        Cn = local_correlations(np.array(self),
                                   eight_neighbours=eight_neighbours,
                                   swap_dim=swap_dim,
                                   order_mean=order_mean)
    else:

        n_chunks = T // frames_per_chunk
        for jj, mv in enumerate(range(n_chunks - 1)):
            logging.debug('number of chunks:' + str(jj) + ' frames: ' +
                          str([mv * frames_per_chunk, (mv + 1) * frames_per_chunk]))
            rho = local_correlations(np.array(self[mv * frames_per_chunk:(mv + 1) * frames_per_chunk]),
                                        eight_neighbours=eight_neighbours,
                                        swap_dim=swap_dim,
                                        order_mean=order_mean)
            Cn = np.maximum(Cn, rho)
            if do_plot:
                pl.imshow(Cn, cmap='gray')
                pl.pause(.1)

        logging.debug('number of chunks:' + str(n_chunks - 1) + ' frames: ' +
                      str([(n_chunks - 1) * frames_per_chunk, T]))
        rho = local_correlations(np.array(self[(n_chunks - 1) * frames_per_chunk:]),
                                    eight_neighbours=eight_neighbours,
                                    swap_dim=swap_dim,
                                    order_mean=order_mean)
        Cn = np.maximum(Cn, rho)
        if do_plot:
            pl.imshow(Cn, cmap='gray')
            pl.pause(.1)

    return Cn

def local_correlations(Y, eight_neighbours: bool = True, swap_dim: bool = True, order_mean=1) -> np.ndarray:
    """Computes the correlation image for the input dataset Y

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format
    
        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively
    
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

        order_mean: (undocumented)

    Returns:
        rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    rho = np.zeros(np.shape(Y)[1:])
    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    # yapf: disable
    if order_mean == 0:
        rho = np.ones(np.shape(Y)[1:])
        rho_h = rho_h
        rho_w = rho_w
        rho[:-1, :] = rho[:-1, :] * rho_h
        rho[1:,  :] = rho[1:,  :] * rho_h
        rho[:, :-1] = rho[:, :-1] * rho_w
        rho[:,  1:] = rho[:,  1:] * rho_w
    else:
        rho[:-1, :] = rho[:-1, :] + rho_h**(order_mean)
        rho[1:,  :] = rho[1:,  :] + rho_h**(order_mean)
        rho[:, :-1] = rho[:, :-1] + rho_w**(order_mean)
        rho[:,  1:] = rho[:,  1:] + rho_w**(order_mean)

    if Y.ndim == 4:
        rho_d = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
        rho[:, :, :-1] = rho[:, :, :-1] + rho_d
        rho[:, :, 1:] = rho[:, :, 1:] + rho_d

        neighbors = 6 * np.ones(np.shape(Y)[1:])
        neighbors[0]        = neighbors[0]        - 1
        neighbors[-1]       = neighbors[-1]       - 1
        neighbors[:,     0] = neighbors[:,     0] - 1
        neighbors[:,    -1] = neighbors[:,    -1] - 1
        neighbors[:,  :, 0] = neighbors[:,  :, 0] - 1
        neighbors[:, :, -1] = neighbors[:, :, -1] - 1

    else:
        if eight_neighbours:
            rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:,]), axis=0)
            rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:,]), axis=0)

            if order_mean == 0:
                rho_d1 = rho_d1
                rho_d2 = rho_d2
                rho[:-1, :-1] = rho[:-1, :-1] * rho_d2
                rho[1:,   1:] = rho[1:,   1:] * rho_d1
                rho[1:,  :-1] = rho[1:,  :-1] * rho_d1
                rho[:-1,  1:] = rho[:-1,  1:] * rho_d2
            else:
                rho[:-1, :-1] = rho[:-1, :-1] + rho_d2**(order_mean)
                rho[1:,   1:] = rho[1:,   1:] + rho_d1**(order_mean)
                rho[1:,  :-1] = rho[1:,  :-1] + rho_d1**(order_mean)
                rho[:-1,  1:] = rho[:-1,  1:] + rho_d2**(order_mean)

            neighbors = 8 * np.ones(np.shape(Y)[1:3])
            neighbors[0,   :] = neighbors[0,   :] - 3
            neighbors[-1,  :] = neighbors[-1,  :] - 3
            neighbors[:,   0] = neighbors[:,   0] - 3
            neighbors[:,  -1] = neighbors[:,  -1] - 3
            neighbors[0,   0] = neighbors[0,   0] + 1
            neighbors[-1, -1] = neighbors[-1, -1] + 1
            neighbors[-1,  0] = neighbors[-1,  0] + 1
            neighbors[0,  -1] = neighbors[0,  -1] + 1
        else:
            neighbors = 4 * np.ones(np.shape(Y)[1:3])
            neighbors[0,  :]  = neighbors[0,  :] - 1
            neighbors[-1, :]  = neighbors[-1, :] - 1
            neighbors[:,  0]  = neighbors[:,  0] - 1
            neighbors[:, -1]  = neighbors[:, -1] - 1

    # yapf: enable
    if order_mean == 0:
        rho = np.power(rho, 1. / neighbors)
    else:
        rho = np.power(np.divide(rho, neighbors), 1 / order_mean)

    return rho

def apply_shifts_dft(src_freq, shifts, diffphase, is_freq=True, border_nan=True):
    """
    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Args:
        apply shifts using inverse dft
        src_freq: ndarray
            if is_freq it is fourier transform image else original image
        shifts: shifts to apply
        diffphase: comes from the register_translation output
    """
    is3D = len(src_freq.shape) == 3
    if not is_freq:
        if is3D:
            src_freq = np.fft.fftn(src_freq)
        else:
            src_freq = np.dstack([np.real(src_freq), np.imag(src_freq)])
            src_freq = fftn(src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq[:, :, 0] + 1j * src_freq[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)

    if not is3D:
        shifts = shifts[::-1]
        nc, nr = np.shape(src_freq)
        Nr = ifftshift(np.arange(-np.fix(nr/2.), np.ceil(nr/2.)))
        Nc = ifftshift(np.arange(-np.fix(nc/2.), np.ceil(nc/2.)))
        Nr, Nc = np.meshgrid(Nr, Nc)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * 1. * Nr / nr - shifts[1] * 1. * Nc / nc))
    else:
        #shifts = np.array([*shifts[:-1][::-1],shifts[-1]])
        #import pdb
        #pdb.set_trace()
        shifts = np.array(list(shifts[:-1][::-1]) + [shifts[-1]])
        nc, nr, nd = np.array(np.shape(src_freq), dtype=float)
        Nr = ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.)))
        Nc = ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.)))
        Nd = ifftshift(np.arange(-np.fix(nd / 2.), np.ceil(nd / 2.)))
        Nr, Nc, Nd = np.meshgrid(Nr, Nc, Nd)
        Greg = src_freq * np.exp(-1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc -
                                  shifts[2] * Nd / nd))



    Greg = Greg.dot(np.exp(1j * diffphase))
    if is3D:
        new_img = np.real(np.fft.ifftn(Greg))
    else:
        Greg = np.dstack([np.real(Greg), np.imag(Greg)])
        new_img = ifftn(Greg)[:, :, 0]

    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shifts[:2])).astype(np.int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shifts[:2])).astype(np.int)
        if is3D:
            max_d = np.ceil(np.maximum(0, shifts[2])).astype(np.int)
            min_d = np.floor(np.minimum(0, shifts[2])).astype(np.int)
        if border_nan is True:
            new_img[:max_h, :] = np.nan
            if min_h < 0:
                new_img[min_h:, :] = np.nan
            new_img[:, :max_w] = np.nan
            if min_w < 0:
                new_img[:, min_w:] = np.nan
            if is3D:
                new_img[:, :, :max_d] = np.nan
                if min_d < 0:
                    new_img[:, :, min_d:] = np.nan
        elif border_nan == 'min':
            min_ = np.nanmin(new_img)
            new_img[:max_h, :] = min_
            if min_h < 0:
                new_img[min_h:, :] = min_
            new_img[:, :max_w] = min_
            if min_w < 0:
                new_img[:, min_w:] = min_
            if is3D:
                new_img[:, :, :max_d] = min_
                if min_d < 0:
                    new_img[:, :, min_d:] = min_
        elif border_nan == 'copy':
            new_img[:max_h] = new_img[max_h]
            if min_h < 0:
                new_img[min_h:] = new_img[min_h-1]
            if max_w > 0:
                new_img[:, :max_w] = new_img[:, max_w, np.newaxis]
            if min_w < 0:
                new_img[:, min_w:] = new_img[:, min_w-1, np.newaxis]
            if is3D:
                if max_d > 0:
                    new_img[:, :, :max_d] = new_img[:, :, max_d, np.newaxis]
                if min_d < 0:
                    new_img[:, :, min_d:] = new_img[:, :, min_d-1, np.newaxis]

    return new_img
#%% Below are functions for computing metrics for Marton's data
def distance_spikes(s1, s2, max_dist):
    """ Define distance matrix between two spike train.
    Distance greater than maximum distance is assigned one.
    """    
    D = np.ones((len(s1), len(s2)))
    for i in range(len(s1)):
        for j in range(len(s2)):
            if np.abs(s1[i] - s2[j]) > max_dist:
                D[i, j] = 1
            else:
                D[i, j] = (np.abs(s1[i] - s2[j]))/5/max_dist
    return D

def find_matches(D):
    """ Find matches between two spike train by solving linear assigment problem.
    Delete matches where their distance is greater than maximum distance
    """
    index_gt, index_method = linear_sum_assignment(D)
    del_list = []
    for i in range(len(index_gt)):
        if D[index_gt[i], index_method[i]] == 1:
            del_list.append(i)
    index_gt = np.delete(index_gt, del_list)
    index_method = np.delete(index_method, del_list)
    return index_gt, index_method

def spike_comparison(i, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist, save=False):
    e_sg = e_sg[np.where(np.multiply(e_t>=scope[0], e_t<=scope[1]))[0]]
    e_sg = (e_sg - np.mean(e_sg))/(np.max(e_sg)-np.min(e_sg))*np.max(v_sg)
    e_sp = e_sp[np.where(np.multiply(e_sp>=scope[0], e_sp<=scope[1]))[0]]
    e_t = e_t[np.where(np.multiply(e_t>=scope[0], e_t<=scope[1]))[0]]
    #plt.plot(e_t, e_sg, label='ephys', color='blue')
    #plt.plot(e_sp, np.max(e_sg)*1.1*np.ones(e_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
    
    v_sg = v_sg[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_sp = v_sp[np.where(np.multiply(v_sp>=scope[0], v_sp<=scope[1]))[0]]
    v_t = v_t[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    #plt.plot(v_t, v_sg, label='ephys', color='blue')
    #plt.plot(v_sp, np.max(v_sg)*1.1*np.ones(v_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
    
    # Distance matrix and find matches
    D = distance_spikes(s1=e_sp, s2=v_sp, max_dist=max_dist)
    index_gt, index_method = find_matches(D)
    spike = [e_sp, v_sp]
    match = [e_sp[index_gt], v_sp[index_method]]
    height = np.max(np.array(e_sg.max(), v_sg.max()))
    
    # Calculate measures
    TP = len(index_gt)
    FP = len(v_sp) - TP
    FN = len(e_sp) - TP
    
    if len(e_sp) == 0:
        F1 = np.nan
        precision = np.nan
        recall = np.nan
    else:
        try:    
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
    
        recall = TP / (TP + FN)
    
        try:
            F1 = 2 * (precision * recall) / (precision + recall) 
        except ZeroDivisionError:
            F1 = 0
 
    print('precision:',precision)
    print('recall:',recall)
    print('F1:',F1)      
    if save:
        plt.figure()
        plt.plot(e_t, e_sg, color='b', label='ephys')
        plt.plot(e_sp, 1.2*height*np.ones(e_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
        plt.plot(v_t, v_sg, color='orange', label='VolPy')
        plt.plot(v_sp, 1.4*height*np.ones(len(v_sp)),color='orange', marker='.', ms=2, fillstyle='full', linestyle='none')
        for j in range(len(index_gt)):
            plt.plot((e_sp[index_gt[j]], v_sp[index_method[j]]),(1.25*height, 1.35*height), color='gray',alpha=0.5, linewidth=1)
        ax = plt.gca()
        ax.locator_params(nbins=7)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.legend(prop={'size': 6})
        plt.tight_layout()
        plt.savefig(f'{volpy_path}/spike_sweep{i}_{vpy.params.volspike["threshold_method"]}.pdf')
    return precision, recall, F1, match, spike

def spnr_computation(i, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist, save=False):
    spnr = []
    e_sg = e_sg[np.where(np.multiply(e_t>=scope[0], e_t<=scope[1]))[0]]
    e_sg = (e_sg - np.mean(e_sg))/(np.max(e_sg)-np.min(e_sg))*np.max(v_sg)
    e_sp = e_sp[np.where(np.multiply(e_sp>=scope[0], e_sp<=scope[1]))[0]]
    e_t = e_t[np.where(np.multiply(e_t>=scope[0], e_t<=scope[1]))[0]]
    
    v_sg = normalize_piecewise(v_sg, step=5000)    
    v_sg = v_sg[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_sp = v_sp[np.where(np.multiply(v_sp>=scope[0], v_sp<=scope[1]))[0]]
    v_t = v_t[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    
    for e in e_sp:
        tp = np.where(np.abs(v_t - e) == np.min(np.abs(v_t - e)))[0][0]
        spnr.append(np.max(v_sg[tp-1:tp+2]))
    return spnr

# Compute subthreshold correlation coefficents
def sub_correlation(i, v_t, e_sub, v_sub, scope, save=False):
    e_sub = e_sub[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_sub = v_sub[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_t = v_t[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    corr = np.corrcoef(e_sub, v_sub)[0,1]
    if save:
        plt.figure()
        plt.plot(v_t, e_sub)
        plt.plot(v_t, v_sub)   
        plt.savefig(f'{volpy_path}/spike_sweep{i}_subthreshold.pdf')
    return corr

def metric(name, sweep_time, e_sg, e_sp, e_t, e_sub, v_sg, v_sp, v_t, v_sub, init_frames=20000, save=False):
    precision = []
    recall = []
    F1 = []
    sub_corr = []
    mean_time = []
    e_match = []
    v_match = []
    e_spike_aligned = []
    v_spike_aligned = []    
    spnr = []
    
    if 'Cell' in name: # belong Marton
        for i in range(len(sweep_time)):
            print(f'sweep{i}')
            if i == 0:
                scope = [max([e_t.min(), v_t.min()]), sweep_time[i][-1]]
            elif i == len(sweep_time) - 1:
                scope = [sweep_time[i][0], min([e_t.max(), v_t.max()])]
            else:
                scope = [sweep_time[i][0], sweep_time[i][-1]]
            mean_time.append(1 / 2 * (scope[0] + scope[-1]))
            
            # frames for initialization are not counted for F1 score            
            if v_t[init_frames] < scope[1]:
                if v_t[init_frames] < scope[0]:
                    pass
                else:
                    scope[0] = v_t[init_frames]               
                mean_time.append(1 / 2 * (scope[0] + scope[-1]))
                pr, re, F, match, spike = spike_comparison(i, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist=0.01, save=save)
                sp = spnr_computation(i, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist=0.01, save=save)
                corr = sub_correlation(i, v_t, e_sub, v_sub, scope, save=save)
                precision.append(pr)
                recall.append(re)
                F1.append(F)
                spnr.append(sp)
                sub_corr.append(corr)
                e_match.append(match[0])
                v_match.append(match[1])
                e_spike_aligned.append(spike[0])
                v_spike_aligned.append(spike[1])
            else:
                print(f'sweep{i} is used for initialization and not counted for F1 score')                
            
        e_match = np.concatenate(e_match)
        v_match = np.concatenate(v_match)
        e_spike_aligned = np.concatenate(e_spike_aligned)
        v_spike_aligned = np.concatenate(v_spike_aligned)
        spnr = np.mean(np.concatenate(spnr))
    else: 
        #import pdb
        #pdb.set_trace()
        scope = [e_t.min(), e_t.max()]
        scope[0] = v_t[init_frames]
        print(scope)
        
        if 'Mouse' in name:
            max_dist = 200  # 20000 Hz* 0.01s
        elif 'Fish' in name: 
            max_dist = 60 # 6000Hz * 0.01s

        pr, re, F, match, spike = spike_comparison(None, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist=max_dist, save=False)
        sp = spnr_computation(None, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist=0.01, save=save)
        precision.append(pr)
        recall.append(re)
        F1.append(F)
        spnr = np.mean(sp)
        e_match = match[0]
        v_match = match[1]
        e_spike_aligned = spike[0]
        v_spike_aligned = spike[1]


    return precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned, spnr

def compute_spnr(t1, t2, s1, s2, t_range, min_counts=10):
    t1 = normalize(t1)
    t2 = normalize(t2)
    both_found = np.intersect1d(s1, s2)
    both_found = both_found[np.logical_and(both_found>t_range[0], both_found<t_range[1])]
    print(len(both_found))
    spnr = [np.mean(t1[both_found]), np.mean(t2[both_found])]
    if len(both_found) < min_counts:
        spnr = [np.nan, np.nan]
    return spnr

#%% Below are functions for running statistics and online filtering
class OnlineFilter(object):
    def __init__(self, freq, fr, order=3, mode='high'):
        '''
        Object encapsulating Online filtering for spike extraction traces
        Args:
            freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
        '''
        self.freq = freq
        self.fr = fr
        self.mode = mode
        self.order=order
        self.normFreq = freq / (fr / 2)        
        self.filt = butter(self.order, self.normFreq, self.mode, output='sos')         
        self.z_init = sosfilt_zi(self.filt)
        
    
    def fit(self, sig):
        """
        Online filter initialization and running offline

        Parameters
        ----------
        sig : ndarray
            input signal to initialize
        num_frames_buf : int
            frames to use for buffering

        Returns
        -------
        sig_filt: ndarray
            filtered signal

        """
        sig_filt = signal_filter(sig, freq=self.freq, fr=self.fr, order=self.order, mode=self.mode)
        # result_init, z_init = signal.sosfilt(b, data[:,:20000], zi=z)
        self.z_init = np.repeat(self.z_init[:,None,:], sig.shape[0], axis=1)
    
        #sos_all = np.zeros(sig_filt.shape)

        for i in range(0,sig.shape[-1]-1):
            _ , self.z_init = sosfilt(self.filt, np.expand_dims(sig[:,i], axis=1), zi=self.z_init)
            
        return sig_filt 

    def fit_next(self, sig):
        
        sig_filt, self.z_init = sosfilt(self.filt, np.expand_dims(sig,axis=1), zi=self.z_init)
        return sig_filt.squeeze()

        
def rolling_window(ndarr, window_size, stride):   
        """
        generates efficient rolling window for running statistics
        Args:
            ndarr: ndarray
                input pixels in format pixels x time
            window_size: int
                size of the sliding window
            stride: int
                stride of the sliding window
        Returns:
                iterator with views of the input array
                
        """
        for i in range(0,ndarr.shape[-1]-window_size-stride+1,stride): 
            yield ndarr[:,i:np.minimum(i+window_size, ndarr.shape[-1])]
            
        if i+stride != ndarr.shape[-1]:
           yield ndarr[:,i+stride:]

def estimate_running_std(signal_in, win_size=20000, stride=5000, 
                         idx_exclude=None, q_min=25, q_max=75):
    """
    Function to estimate ROBUST runnning std
    
    Args:
        win_size: int
            window used to compute running std to normalize signals when 
            compensating for photobleaching
            
        stride: int
            corresponding stride to win_size
            
        idx_exclude: iterator
            indexes to exclude when computing std
        
        q_min: float
            lower percentile for estimation of signal variability (do not change)
        
        q_max: float
            higher percentile for estimation of signal variability (do not change)
        
        
    Returns:
        std_run: ndarray
            running standard deviation
    
    """
    if idx_exclude is not None:
        signal = signal_in[np.setdiff1d(range(len(signal_in)), idx_exclude)]        
    else:
        signal = signal_in
    iter_win = rolling_window(signal[None,:],win_size,stride)
    myperc = partial(np.percentile, q=[q_min,q_max], axis=-1)
    res = np.array(list(map(myperc,iter_win))).T.squeeze()
    iqr = (res[1]-res[0])/1.35
    std_run = cv2.resize(iqr,signal_in[None,:].shape).squeeze()
    return std_run

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
    
    thresh = prev_thresh 

    if (len(minima)>0) and (np.abs(minima[0]-prev_thresh)< delta_max):
        thresh = minima[0]
        mnt = (minima_2nd-thresh)
        mnt = mnt[mnt<0]
        thresh += mnt[np.maximum(-len(mnt)+1,-number_maxima_before)]
    #else:
    #    thresh = 100
        
    thresh_7 = compute_std(peak_height) * 7.5
    
    """
    print(f'previous thresh: {prev_thresh}')
    print(f'current thresh: {thresh}')  
    """
    plt.figure()
    plt.plot(x_val, pdf,'c')    
    plt.plot(x_val[2:],second_der*500,'r')  
    plt.plot(thresh,0, '*')   
    plt.vlines(thresh_7, 0, 2, color='r')
    plt.pause(0.1)
    
    return thresh

def non_symm_median_filter(t, filt_window):
    m = t.copy()
    for i in range(len(t)):
        if (i > filt_window[0]) and (i < len(t) - filt_window[1]):
            m[i] = np.median(t[i - filt_window[0] : i + filt_window[1] + 1])
    return m
#%% Below are functions for matching spikes
def compute_distances(s1, s2, max_dist):
    """
    Define a distance matrix of spikes.
    Distances greater than maximum distance are assigned one.

    Parameters
    ----------
    s1,s2 : ndarray
        Spikes time of two methods
    max_dist : int
        Maximum distance allowed between two matched spikes

    Returns
    -------
    D : ndarray
        Distance matrix between two spikes
    """
    D = np.ones((len(s1), len(s2)))
    for i in range(len(s1)):
        for j in range(len(s2)):
            if np.abs(s1[i] - s2[j]) > max_dist:
                D[i, j] = 1
            else:
                # 1.01 is to avoid two pairs of matches 'cross' each other
                D[i, j] = (np.abs(s1[i] - s2[j]))/5/max_dist ** 1.01 
    return D

def match_spikes_linear_sum(D):
    """
    Find matches among spikes by solving linear sum assigment problem.
    Delete matches where their distances are greater than the maximum distance.
    Parameters
    ----------
    D : ndarray
        Distance matrix between two spikes
        
    Returns
    -------
    idx1, idx2 : ndarray
        Matched spikes indexes

    """
    idx1, idx2 = linear_sum_assignment(D)
    del_list = []
    for i in range(len(idx1)):
        if D[idx1[i], idx2[i]] == 1:
            del_list.append(i)
    idx1 = np.delete(idx1, del_list)
    idx2 = np.delete(idx2, del_list)
    return idx1, idx2

def match_spikes_greedy(s1, s2, max_dist):
    """
    Match spikes using the greedy algorithm. Spikes greater than the maximum distance
    are never matched.
    Parameters
    ----------
    s1,s2 : ndarray
        Spike time of two methods
    max_dist : int
        Maximum distance allowed between two matched spikes

    Returns
    -------
    idx1, idx2 : ndarray
        Matched spikes indexes with respect to s1 and s2

    """
    l1 = list(s1.copy())
    l2 = list(s2.copy())
    idx1 = []
    idx2 = []
    temp1 = 0
    temp2 = 0
    while len(l1) * len(l2) > 0:
        if np.abs(l1[0] - l2[0]) <= max_dist:
            idx1.append(temp1)
            idx2.append(temp2)
            temp1 += 1
            temp2 += 1
            del l1[0]
            del l2[0]
        elif l1[0] < l2[0]:
            temp1 += 1
            del l1[0]
        elif l1[0] > l2[0]:
            temp2 += 1
            del l2[0]
    return idx1, idx2

def compute_F1(s1, s2, idx1, idx2):
    """
    Compute F1 scores, precision and recall.

    Parameters
    ----------
    s1,s2 : ndarray
        Spike time of two methods. Note we assume s1 as ground truth spikes.
    
    idx1, idx2 : ndarray
        Matched spikes indexes with respect to s1 and s2

    Returns
    -------
    F1 : float
        Measures of how well spikes are matched with ground truth spikes. 
        The higher F1 score, the better.
        F1 = 2 * (precision * recall) / (precision + recall)
    precision, recall : float
        Precision and recall rate of spikes matching.
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    """
    TP = len(idx1)
    FP = len(s2) - TP
    FN = len(s1) - TP
    
    if len(s1) == 0:
        F1 = np.nan
        precision = np.nan
        recall = np.nan
    else:
        try:    
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
        recall = TP / (TP + FN)
        try:
            F1 = 2 * (precision * recall) / (precision + recall) 
        except ZeroDivisionError:
            F1 = 0
            
    return F1, precision, recall

####################################################################################
# Below are functions for supporting nmf 
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

def hals(Y, A, C, b, f, bSiz=3, maxIter=5, semi_nmf=False, update_bg=True, use_spikes=False, hals_orig=False, fr=400):
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
       semi_nmf: bool,
           use semi-nmf (without nonnegative constraint on temporal traces) if True, 
           otherwise use nmf        
       update_bg: bool, 
           update background if True, otherwise background will be set zero
       use_spikes: bool, 
           if True the algorithm will detect spikes using VolPy offline 
           to optimize the spatial components A
       hals_orig: bool,
           if True the input matrix Y is from the original movie, otherwise the input matrix Y
           is thresholded at 0    
       fr: 
           frame rate of the movie           
           
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
    def HALS4activity(Yr, A, C, iters=2, semi_nmf=False):
        U = A.T.dot(Yr)
        V = A.T.dot(A) + np.finfo(A.dtype).eps
        for _ in range(iters):
            for m in range(len(U)):  # neurons and background
                if semi_nmf:
                    print('use semi-nmf')
                    C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                               V[m, m], -np.inf, np.inf)
                else:
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

    for thr_ in np.linspace(3.5,2.5,maxIter):    
        Cf = HALS4activity(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf, semi_nmf=semi_nmf)
        Cf_processed = Cf.copy()

        if not update_bg:
            Cf_processed[-nb:] = np.zeros(Cf_processed[-nb:].shape)

        if use_spikes:
            for i in range(Cf.shape[0]):
                if i < Cf.shape[0] - nb: 
                    if hals_orig:
                        bl = scipy.ndimage.percentile_filter(-Cf[i], 50, size=50)
                        tr = -Cf[i] - bl   
                        tr = tr-np.median(tr)
                        bl = bl+np.median(tr)
                    else:
                        bl = scipy.ndimage.percentile_filter(Cf[i], 50, size=50)
                        tr = Cf[i] - bl                    
                    
                    _, _, Cf_processed[i], _, _, _ = denoise_spikes(tr, window_length=3, clip=0, fr=fr,
                                      threshold=thr_, threshold_method='simple', do_plot=False, do_filter=False)
                    
                    if hals_orig:
                        Cf_processed[i] = -Cf_processed[i] - bl    
                    else:
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

def normalize(data):
    """ Normalize the data
    Args: 
        data: ndarray
            input data
    
    Returns:
        data_norm: ndarray
            normalized data        
    """
    data = data - np.median(data)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns))
    data_norm = data/std 
    #signal = (signal-np.percentile(signal,1, axis=0))/(np.percentile(signal,99, axis=0)-np.percentile(signal,1, axis=0))
    #signal -= np.median(signal)
    return data_norm

def normalize_piecewise(data, step=5000):
    """ Normalize the data every step frames
    Args: 
        data: ndarray
            input data
        step: int
            normalize the data every step of frames separately    
    Returns:
        data_norm: ndarray
            normalized data        
    """
    data_norm = []
    for i in range(np.ceil(len(data)/step).astype(np.int16)):
        if (i + 1)*step > len(data):
            d = data[i * step:]
        else:
            d = data[i * step : (i + 1) * step]
        data_norm.append(normalize(d))
    data_norm = np.hstack(data_norm)
    return data_norm

#######################################################################################
# Below are function from CaImAn 
def play(mov, fr=400, backend='opencv', magnification=1, interpolation=cv2.INTER_LINEAR, offset=0, gain=1, q_max=100, q_min=1):
    if q_max < 100:
        maxmov = np.nanpercentile(mov[0:10], q_max)
    else:
        maxmov = np.nanmax(mov)
    if q_min > 0:
        minmov = np.nanpercentile(mov[0:10], q_min)
    else:
        minmov = np.nanmin(mov)
        
    for iddxx, frame in enumerate(mov):
        if backend == 'opencv':
            if magnification != 1:
                frame = cv2.resize(frame, None, fx=magnification, fy=magnification, interpolation=interpolation)
            frame = (offset + frame - minmov) * gain / (maxmov - minmov)
            cv2.imshow('frame', frame)
            if cv2.waitKey(int(1. / fr * 1000)) & 0xFF == ord('q'):
                break
            
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    for i in range(10):
        cv2.waitKey(100)

def resize(mov_in, fx=1, fy=1, fz=1, interpolation=cv2.INTER_AREA):
        """
        Resizing caiman movie into a new one. Note that the temporal
        dimension is controlled by fz and fx, fy, fz correspond to
        magnification factors. For example to downsample in time by
        a factor of 2, you need to set fz = 0.5.

        Args:
            fx (float):
                Magnification factor along x-dimension

            fy (float):
                Magnification factor along y-dimension

            fz (float):
                Magnification factor along temporal dimension

        Returns:
            self (caiman movie)
        """
        T, d1, d2 = mov_in.shape
        d = d1 * d2
        elm = d * T
        max_els = 2**61 - 1    # the bug for sizes >= 2**31 is appears to be fixed now
        if elm > max_els:
            chunk_size = old_div((max_els), d)
            new_m: List = []
            logging.debug('Resizing in chunks because of opencv bug')
            for chunk in range(0, T, chunk_size):
                logging.debug([chunk, np.minimum(chunk + chunk_size, T)])
                m_tmp = mov_in[chunk:np.minimum(chunk + chunk_size, T)].copy()
                m_tmp = m_tmp.resize(fx=fx, fy=fy, fz=fz, interpolation=interpolation)
                if len(new_m) == 0:
                    new_m = m_tmp
                else:
                    new_m = timeseries.concatenate([new_m, m_tmp], axis=0)

            return new_m
        else:
            if fx != 1 or fy != 1:
                logging.debug("reshaping along x and y")
                t, h, w = mov_in.shape
                newshape = (int(w * fy), int(h * fx))
                mov = []
                logging.debug("New shape is " + str(newshape))
                for frame in mov_in:
                    mov.append(cv2.resize(frame, newshape, fx=fx, fy=fy, interpolation=interpolation))
                mov_in = np.asarray(mov)
            if fz != 1:
                logging.debug("reshaping along z")
                t, h, w = mov_in.shape
                mov_in = np.reshape(mov_in, (t, h * w))
                mov = cv2.resize(mov_in, (h * w, int(fz * t)), fx=1, fy=fz, interpolation=interpolation)
                mov = np.reshape(mov, (np.maximum(1, int(fz * t)), h, w))
                mov_in = np.asarray(mov)                

        return self       
           
def gaussian_blur_2D(movie_in,
                         kernel_size_x=5,
                         kernel_size_y=5,
                         kernel_std_x=1,
                         kernel_std_y=1,
                         borderType=cv2.BORDER_REPLICATE):
        """
        Compute gaussian blut in 2D. Might be useful when motion correcting

        Args:
            kernel_size: double
                see opencv documentation of GaussianBlur
            kernel_std_: double
                see opencv documentation of GaussianBlur
            borderType: int
                see opencv documentation of GaussianBlur

        Returns:
            self: ndarray
                blurred movie
        """
        movie_out = np.zeros_like(movie_in)
        for idx, fr in enumerate(movie_in):
            movie_out[idx] = cv2.GaussianBlur(fr,
                                         ksize=(kernel_size_x, kernel_size_y),
                                         sigmaX=kernel_std_x,
                                         sigmaY=kernel_std_y,
                                         borderType=borderType)

        return movie_out

def resize(mov_in, fx=1, fy=1, fz=1, interpolation=cv2.INTER_AREA):
        """
        Resizing caiman movie into a new one. Note that the temporal
        dimension is controlled by fz and fx, fy, fz correspond to
        magnification factors. For example to downsample in time by
        a factor of 2, you need to set fz = 0.5.

        Args:
            fx (float):
                Magnification factor along x-dimension

            fy (float):
                Magnification factor along y-dimension

            fz (float):
                Magnification factor along temporal dimension

        Returns:
            mov_in (caiman movie)
        """
        T, d1, d2 = mov_in.shape
        d = d1 * d2
        elm = d * T
        max_els = 2**61 - 1    # the bug for sizes >= 2**31 is appears to be fixed now
        if elm > max_els:
            chunk_size = old_div((max_els), d)
            new_m = []
            for chunk in range(0, T, chunk_size):
                m_tmp = mov_in[chunk:np.minimum(chunk + chunk_size, T)].copy()
                m_tmp = m_tmp.resize(fx=fx, fy=fy, fz=fz, interpolation=interpolation)
                if len(new_m) == 0:
                    new_m = m_tmp
                else:
                    new_m = np.concatenate([new_m, m_tmp], axis=0)

            return new_m
        else:
            if fx != 1 or fy != 1:
                t, h, w = mov_in.shape
                newshape = (int(w * fy), int(h * fx))
                mov = []
                for frame in mov_in:
                    mov.append(cv2.resize(frame, newshape, fx=fx, fy=fy, interpolation=interpolation))
                mov_out = np.asarray(mov)
            if fz != 1:
                t, h, w = mov_in.shape
                mov_in = np.reshape(mov_in, (t, h * w))
                mov = cv2.resize(mov_in, (h * w, int(fz * t)), fx=1, fy=fz, interpolation=interpolation)
                mov = np.reshape(mov, (np.maximum(1, int(fz * t)), h, w))
                mov_out = mov

        return mov_out

def to_2D(mov, order='F') -> np.ndarray:
        [T, d1, d2] = mov.shape
        d = d1 * d2
        return np.reshape(mov, (T, d), order=order)

def to_3D(mov2D, shape, order='F'):
    """
    transform a vectorized movie into a 3D shape
    """
    return np.reshape(mov2D, shape, order=order)

def bin_median(mat, window=10, exclude_nans=True):
    """ compute median of 3D array in along axis o by binning values
    Args:
        mat: ndarray
            input 3D matrix, time along first dimension
        window: int
            number of frames in a bin
    Returns:
        img:
            median image
    Raises:
        Exception 'Path to template does not exist:'+template
    """
    T, d1, d2 = np.shape(mat)
    if T < window:
        window = T
    num_windows = np.int(old_div(T, window))
    num_frames = num_windows * window
    if exclude_nans:
        img = np.nanmedian(np.nanmean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)
    else:
        img = np.median(np.mean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)
    return img

def bin_median_3d(self, window=10):
    """ compute median of 4D array in along axis o by binning values

    Args:
        mat: ndarray
            input 4D matrix, (T, h, w, z)

        window: int
            number of frames in a bin

    Returns:
        img:
            median image

    """
    T, d1, d2, d3 = np.shape(self)
    num_windows = np.int(old_div(T, window))
    num_frames = num_windows * window
    return np.nanmedian(np.nanmean(np.reshape(self[:num_frames], (window, num_windows, d1, d2, d3)), axis=0),
                        axis=0)

def nf_match_neurons_in_binary_masks(masks_gt,
                                     masks_comp,
                                     thresh_cost=.7,
                                     min_dist=10,
                                     print_assignment=False,
                                     plot_results=False,
                                     Cn=None,
                                     labels=['Session 1', 'Session 2'],
                                     cmap='gray',
                                     D=None,
                                     enclosed_thr=None,
                                     colors=['red', 'white']):
    """
    Match neurons expressed as binary masks. Uses Hungarian matching algorithm

    Args:
        masks_gt: bool ndarray  components x d1 x d2
            ground truth masks

        masks_comp: bool ndarray  components x d1 x d2
            mask to compare to

        thresh_cost: double
            max cost accepted

        min_dist: min distance between cm

        print_assignment:
            for hungarian algorithm

        plot_results: bool

        Cn:
            correlation image or median

        D: list of ndarrays
            list of distances matrices

        enclosed_thr: float
            if not None set distance to at most the specified value when ground truth is a subset of inferred

    Returns:
        idx_tp_1:
            indices true pos ground truth mask

        idx_tp_2:
            indices true pos comp

        idx_fn_1:
            indices false neg

        idx_fp_2:
            indices false pos

    """

    _, d1, d2 = np.shape(masks_gt)
    dims = d1, d2

    # transpose to have a sparse list of components, then reshaping it to have a 1D matrix red in the Fortran style
    A_ben = scipy.sparse.csc_matrix(np.reshape(masks_gt[:].transpose([1, 2, 0]), (
        np.prod(dims),
        -1,
    ), order='F'))
    A_cnmf = scipy.sparse.csc_matrix(np.reshape(masks_comp[:].transpose([1, 2, 0]), (
        np.prod(dims),
        -1,
    ), order='F'))

    # have the center of mass of each element of the two masks
    cm_ben = [scipy.ndimage.center_of_mass(mm) for mm in masks_gt]
    cm_cnmf = [scipy.ndimage.center_of_mass(mm) for mm in masks_comp]

    if D is None:
        #% find distances and matches
        # find the distance between each masks
        D = distance_masks([A_ben, A_cnmf], [cm_ben, cm_cnmf], min_dist, enclosed_thr=enclosed_thr)
        level = 0.98
    else:
        level = .98

    matches, costs = find_matches(D, print_assignment=print_assignment)
    matches = matches[0]
    costs = costs[0]

    #%% compute precision and recall
    TP = np.sum(np.array(costs) < thresh_cost) * 1.
    FN = np.shape(masks_gt)[0] - TP
    FP = np.shape(masks_comp)[0] - TP
    TN = 0

    performance = dict()
    performance['recall'] = old_div(TP, (TP + FN))
    performance['precision'] = old_div(TP, (TP + FP))
    performance['accuracy'] = old_div((TP + TN), (TP + FP + FN + TN))
    performance['f1_score'] = 2 * TP / (2 * TP + FP + FN)
    logging.debug(performance)
    #%%
    idx_tp = np.where(np.array(costs) < thresh_cost)[0]
    idx_tp_ben = matches[0][idx_tp]    # ground truth
    idx_tp_cnmf = matches[1][idx_tp]   # algorithm - comp

    idx_fn = np.setdiff1d(list(range(np.shape(masks_gt)[0])), matches[0][idx_tp])

    idx_fp = np.setdiff1d(list(range(np.shape(masks_comp)[0])), matches[1][idx_tp])

    idx_fp_cnmf = idx_fp

    idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp = idx_tp_ben, idx_tp_cnmf, idx_fn, idx_fp_cnmf

    if plot_results:
        try:   # Plotting function
            plt.rcParams['pdf.fonttype'] = 42
            font = {'family': 'Myriad Pro', 'weight': 'regular', 'size': 10}
            plt.rc('font', **font)
            lp, hp = np.nanpercentile(Cn, [5, 95])
            ses_1 = mpatches.Patch(color=colors[0], label=labels[0])
            ses_2 = mpatches.Patch(color=colors[1], label=labels[1])
            plt.subplot(1, 2, 1)
            plt.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
            [plt.contour(norm_nrg(mm), levels=[level], colors=colors[1], linewidths=1) for mm in masks_comp[idx_tp_comp]]
            [plt.contour(norm_nrg(mm), levels=[level], colors=colors[0], linewidths=1) for mm in masks_gt[idx_tp_gt]]
            if labels is None:
                plt.title('MATCHES')
            else:
                plt.title('MATCHES: ' + labels[1] + f'({colors[1][0]}), ' + labels[0] + f'({colors[0][0]})')
            plt.legend(handles=[ses_1, ses_2])
            plt.show()
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
            [plt.contour(norm_nrg(mm), levels=[level], colors=colors[1], linewidths=1) for mm in masks_comp[idx_fp_comp]]
            [plt.contour(norm_nrg(mm), levels=[level], colors=colors[0], linewidths=1) for mm in masks_gt[idx_fn_gt]]
            if labels is None:
                plt.title(f'FALSE POSITIVE ({colors[1][0]}), FALSE NEGATIVE ({colors[0][0]})')
            else:
                plt.title(labels[1] + f'({colors[1][0]}), ' + labels[0] + f'({colors[0][0]})')
            plt.legend(handles=[ses_1, ses_2])
            plt.show()
            plt.axis('off')
        except Exception as e:
            logging.warning("not able to plot precision recall: graphics failure")
            logging.warning(e)
    return idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp, performance

def distance_masks(M_s: List, cm_s: List[List], max_dist: float, enclosed_thr: Optional[float] = None) -> List:
    """
    Compute distance matrix based on an intersection over union metric. Matrix are compared in order,
    with matrix i compared with matrix i+1

    Args:
        M_s: tuples of 1-D arrays
            The thresholded A matrices (masks) to compare, output of threshold_components

        cm_s: list of list of 2-ples
            the centroids of the components in each M_s

        max_dist: float
            maximum distance among centroids allowed between components. This corresponds to a distance
            at which two components are surely disjoined

        enclosed_thr: float
            if not None set distance to at most the specified value when ground truth is a subset of inferred

    Returns:
        D_s: list of matrix distances

    Raises:
        Exception: 'Nan value produced. Error in inputs'

    """
    D_s = []

    for gt_comp, test_comp, cmgt_comp, cmtest_comp in zip(M_s[:-1], M_s[1:], cm_s[:-1], cm_s[1:]):

        # todo : better with a function that calls itself
        # not to interfer with M_s
        gt_comp = gt_comp.copy()[:, :]
        test_comp = test_comp.copy()[:, :]

        # the number of components for each
        nb_gt = np.shape(gt_comp)[-1]
        nb_test = np.shape(test_comp)[-1]
        D = np.ones((nb_gt, nb_test))

        cmgt_comp = np.array(cmgt_comp)
        cmtest_comp = np.array(cmtest_comp)
        if enclosed_thr is not None:
            gt_val = gt_comp.T.dot(gt_comp).diagonal()
        for i in range(nb_gt):
            # for each components of gt
            k = gt_comp[:, np.repeat(i, nb_test)] + test_comp
            # k is correlation matrix of this neuron to every other of the test
            for j in range(nb_test):   # for each components on the tests
                dist = np.linalg.norm(cmgt_comp[i] - cmtest_comp[j])
                                       # we compute the distance of this one to the other ones
                if dist < max_dist:
                                       # union matrix of the i-th neuron to the jth one
                    union = k[:, j].sum()
                                       # we could have used OR for union and AND for intersection while converting
                                       # the matrice into real boolean before

                    # product of the two elements' matrices
                    # we multiply the boolean values from the jth omponent to the ith
                    intersection = np.array(gt_comp[:, i].T.dot(test_comp[:, j]).todense()).squeeze()

                    # if we don't have even a union this is pointless
                    if union > 0:

                        # intersection is removed from union since union contains twice the overlaping area
                        # having the values in this format 0-1 is helpfull for the hungarian algorithm that follows
                        D[i, j] = 1 - 1. * intersection / \
                            (union - intersection)
                        if enclosed_thr is not None:
                            if intersection == gt_val[j] or intersection == gt_val[i]:
                                D[i, j] = min(D[i, j], 0.5)
                    else:
                        D[i, j] = 1.

                    if np.isnan(D[i, j]):
                        raise Exception('Nan value produced. Error in inputs')
                else:
                    D[i, j] = 1

        D_s.append(D)
    return D_s


def find_matches(D_s, print_assignment: bool = False) -> Tuple[List, List]:
    # todo todocument

    matches = []
    costs = []
    t_start = time.time()
    for ii, D in enumerate(D_s):
        # we make a copy not to set changes in the original
        DD = D.copy()
        if np.sum(np.where(np.isnan(DD))) > 0:
            logging.error('Exception: Distance Matrix contains NaN, not allowed!')
            raise Exception('Distance Matrix contains NaN, not allowed!')

        # we do the hungarian
        indexes = linear_sum_assignment(DD)
        indexes2 = [(ind1, ind2) for ind1, ind2 in zip(indexes[0], indexes[1])]
        matches.append(indexes)
        DD = D.copy()
        total = []
        # we want to extract those informations from the hungarian algo
        for row, column in indexes2:
            value = DD[row, column]
            if print_assignment:
                logging.debug(('(%d, %d) -> %f' % (row, column, value)))
            total.append(value)
        logging.debug(('FOV: %d, shape: %d,%d total cost: %f' % (ii, DD.shape[0], DD.shape[1], np.sum(total))))
        logging.debug((time.time() - t_start))
        costs.append(total)
        # send back the results in the format we want
    return matches, costs

def norm_nrg(a_):

    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order='F')
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx]**2)
    cumEn /= cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order='F')

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
    
    thresh_negative = -np.percentile(pks, 0.1)
    print(f'adaptive thresholding: {thresh}')
    print(f'thresholding negative value: {thresh_negative}')
    
    plt.figure(); plt.plot(pts, f); plt.plot(pts[:len(f)], fmodel); plt.vlines(thresh, 0, np.max(f), 'r')
    plt.vlines(thresh_negative, 0, np.max(f), 'b')
    plt.figure(); plt.plot(csf2); plt.plot(csmodel2)
    
    thresh = np.maximum(thresh, thresh_negative)

    
    """
    if np.sum(pks > thresh) < min_spikes:
        low_spikes = True
        logging.warning(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
    elif ((np.sum(pks > thresh) > clip) & (clip > 0)):
        logging.warning(f'Selecting top {min_spikes} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))
    """
    ix = np.argmin(np.abs(xi - thresh))
    falsePosRate = csmodel2[ix] / csf2[ix]
    detectionRate = (csf2[ix] - csmodel2[ix]) / np.max(csf2 - csmodel2)
    return thresh, falsePosRate, detectionRate, low_spikes

###############################################################################################################
# Below are functions from VolPy 
def quick_annotation(img, min_radius, max_radius, roughness=2):
    """ Quick annotation method in VolPy using cell magic wand plugin
    Args:
        img: 2-D array
            img as the background for selection
            
        min_radius: float
            minimum radius of the selection
            
        max_radius: float
            maximum raidus of the selection
            
        roughness: int
            roughness of the selection surface
            
    Return:
        ROIs: 3-D array
            region of interests 
            (# of components * # of pixels in x dim * # of pixels in y dim)
    """
    try:
        if __IPYTHON__:
            get_ipython().run_line_magic('matplotlib', 'auto')
    except NameError:
        pass

    def tellme(s):
        print(s)
        plt.title(s, fontsize=16)
        plt.draw()
        
    keep_select=True
    ROIs = []
    while keep_select:
        # Plot img
        plt.clf()
        plt.imshow(img, cmap='gray', vmax=np.percentile(img, 99))            
        if len(ROIs) == 0:
            pass
        elif len(ROIs) == 1:
            plt.imshow(ROIs[0], alpha=0.3, cmap='Oranges')
        else:
            plt.imshow(np.array(ROIs).sum(axis=0), alpha=0.3, cmap='Oranges')
        
        # Plot point and ROI
        tellme('Click center of neuron')
        center = plt.ginput(1)[0]
        plt.plot(center[0], center[1], 'r+')
        ROI = cell_magic_wand_single_point(img, (center[1], center[0]), 
                                           min_radius=min_radius, max_radius=max_radius, 
                                           roughness=roughness, zoom_factor=1)[0]
        plt.imshow(ROI, alpha=0.3, cmap='Reds')
    
        # Select or not
        tellme('Select? Key click for yes, mouse click for no')
        select = plt.waitforbuttonpress()
        if select:
            ROIs.append(ROI)
            tellme('You have selected a neuron. \n Keep selecting? Key click for yes, mouse click for no')
        else:
            tellme('You did not select a neuron \n Keep selecting? Key click for yes, mouse click for no')
        keep_select = plt.waitforbuttonpress()
        
    plt.close()        
    ROIs = np.array(ROIs)   
    
    try:
        if __IPYTHON__:
            get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    return ROIs

def denoise_spikes(data, window_length, fr=400,  hp_freq=1,  clip=100, threshold_method='adaptive_threshold', 
                   min_spikes=10, pnorm=0.5, threshold=3,  do_plot=True, do_filter=True):
    """ Function for finding spikes and the temporal filter given one dimensional signals.
        Use function whitened_matched_filter to denoise spikes. Two thresholding methods can be 
        chosen, simple or 'adaptive thresholding'.

    Args:
        data: 1-d array
            one dimensional signal

        window_length: int
            length of window size for temporal filter

        fr: int
            number of samples per second in the video
            
        hp_freq: float
            high-pass cutoff frequency to filter the signal after computing the trace
            
        clip: int
            maximum number of spikes for producing templates

        threshold_method: str
            adaptive_threshold or simple method for thresholding signals
            adaptive_threshold method threshold based on estimated peak distribution
            simple method threshold based on estimated noise level 
            
        min_spikes: int
            minimal number of spikes to be detected
            
        pnorm: float
            a variable deciding the amount of spikes chosen for adaptive threshold method

        threshold: float
            threshold for spike detection in simple threshold method 
            The real threshold is the value multiply estimated noise level

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
    # high-pass filter the signal for spike detection
    if do_filter:
        data = signal_filter(data, hp_freq, fr, order=5)
        data = data - np.median(data)    
    pks = data[signal.find_peaks(data, height=None)[0]]

    # first round of spike detection    
    if threshold_method == 'adaptive_threshold':
        thresh, _, _, low_spikes = adaptive_thresh(pks, clip, 0.25, min_spikes)
        locs = signal.find_peaks(data, height=thresh)[0]
    elif threshold_method == 'simple':
        thresh, low_spikes = simple_thresh(data, pks, clip, 3.5, min_spikes)
        locs = signal.find_peaks(data, height=thresh)[0]
    else:
        logging.warning("Error: threshold_method not found")
        raise Exception('Threshold_method not found!')

    # spike template
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]
    PTD = data[(locs[:, np.newaxis] + window)]
    PTA = np.median(PTD, 0)
    PTA = PTA - np.min(PTA)
    templates = PTA

    # whitened matched filtering based on spike times detected in the first round of spike detection
    datafilt = whitened_matched_filter(data, locs, window)    
    datafilt = datafilt - np.median(datafilt)

    # second round of spike detection on the whitened matched filtered trace
    pks2 = datafilt[signal.find_peaks(datafilt, height=None)[0]]
    if threshold_method == 'adaptive_threshold':
        thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=pnorm, min_spikes=min_spikes)  # clip=0 means no clipping
        spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    elif threshold_method == 'simple':
        thresh2, low_spikes = simple_thresh(datafilt, pks2, 0, threshold, min_spikes)
        spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    
    # compute reconstructed signals and adjust shrinkage
    t_rec = np.zeros(datafilt.shape)
    t_rec[spikes] = 1
    t_rec = np.convolve(t_rec, PTA, 'same')   
    factor = np.mean(data[spikes]) / np.mean(datafilt[spikes])
    datafilt = datafilt * factor
    thresh2_normalized = thresh2 * factor
        
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

    return datafilt, spikes, t_rec, templates, low_spikes, thresh2_normalized

def adaptive_thresh(pks, clip, pnorm=0.5, min_spikes=10):
    """ Adaptive threshold method for deciding threshold given heights of all peaks.

    Args:
        pks: 1-d array
            height of all peaks

        clip: int
            maximum number of spikes for producing templates

        pnorm: float, between 0 and 1, default is 0.5
            a variable deciding the amount of spikes chosen for adaptive threshold method
            
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
        logging.warning(f'Selecting top {clip} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))

    ix = np.argmin(np.abs(xi - thresh))
    falsePosRate = csmodel2[ix] / csf2[ix]
    detectionRate = (csf2[ix] - csmodel2[ix]) / np.max(csf2 - csmodel2)
    return thresh, falsePosRate, detectionRate, low_spikes


def simple_thresh(data, pks, clip, threshold=3.5, min_spikes=10):
    """ Simple threshold method for deciding threshold based on estimated noise level.

    Args:
        data: 1-d array
            the input trace
            
        pks: 1-d array
            height of all peaks

        clip: int
            maximum number of spikes for producing templates

        threshold: float
            threshold for spike detection in simple threshold method 
            The real threshold is the value multiply estimated noise level
    
        min_spikes: int
            minimal number of spikes to be detected

    Returns:
        thresh: float
            threshold for choosing spikes

        low_spikes: boolean
            true if number of spikes is smaller than minimal value
    """
    low_spikes = False
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    thresh = threshold * std
    locs = signal.find_peaks(data, height=thresh)[0]
    if len(locs) < min_spikes:
        logging.warning(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
        low_spikes = True
    elif ((len(locs) > clip) & (clip > 0)):
        logging.warning(f'Selecting top {clip} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))    
    return thresh, low_spikes


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
    sg = np.single(signal.filtfilt(b, a, sg, method='gust', padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg