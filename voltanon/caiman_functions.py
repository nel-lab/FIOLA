#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:32:42 2020

@author: andrea
"""

import numpy as np
from scipy import signal
from past.utils import old_div
import cv2

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
    print([order, normFreq, mode])
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg