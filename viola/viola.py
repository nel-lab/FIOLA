#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:23:20 2020
VIOLA object for online analysis of voltage imaging data. Including offline 
initialization of spatial masks and online analysis of voltage imaging data.
Please check violaparams.py for the explanation of parameters.
@author: @agiovann, @caichangjia, @cynthia
"""
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls    
from .signal_analysis_online import SignalAnalysisOnlineZ

from .caiman_functions import signal_filter, to_3D, to_2D, bin_median
from .nmf_support import hals, normalize, nmf_sequential
from .pipeline_gpu import get_model, Pipeline_overall, Pipeline

class VIOLA(object):
    def __init__(self, fnames=None, fr=None, ROIs=None,  
                 border_to_0=0, freq_detrend = 1/3, do_plot_init=True, erosion=0, 
                 update_bg=False, use_spikes=False, initialize_with_gpu=False, 
                 num_frames_total=100000, window = 10000, step = 5000, 
                 detrend=True, flip=True, do_scale=False, robust_std=False, freq=15, adaptive_threshold=True, 
                 thresh_range=[3.5, 5], mfp=0.2, filt_window=15, do_plot=False, params={}):
        if params is None:
            logging.warning("Parameters are not set from violaparams")
            raise Exception('Parameters are not set')
        else:
            self.params = params
        
    def fit(self, mov):
        print('Now start initialization of spatial footprint')
        border = self.params.mc_nnls['border_to_0']
        mask = self.params.data['ROIs']
        self.num_frames_init = len(mov)
        if border > 0:
            mov[:, :border, :] = mov[:, border:border + 1, :]
            mov[:, -border:, :] = mov[:, -border-1:-border, :]
            mov[:, :, :border] = mov[:, :, border:border + 1]
            mov[:, :, -border:] = mov[:, :, -border-1:-border]
        self.mov = mov

        if self.params.spike['flip'] == True:
            print('Flip movie for initialization')
            y = to_2D(-mov).copy() 
        else:
            print('Not flip movie for initialization')
            y = to_2D(mov).copy()

        y_filt = signal_filter(y.T,freq=self.params.mc_nnls['freq_detrend'], 
                               fr=self.params.data['fr']).T        
        
        if self.params.mc_nnls['do_plot_init']:
            plt.figure()
            plt.imshow(mov[0])
            plt.figure()
            plt.imshow(mask.sum(0))
            
        y_seq = y_filt.copy()
        
        if self.params.mc_nnls['erosion'] > 0:
            try:
                print('erode mask')
                kernel = np.ones((self.params.mc_nnls['erosion'], self.params.mc_nnls['erosion']),np.uint8)
                mask_new = np.zeros(mask.shape)
                for idx, mm in enumerate(mask):
                    mask_new[idx] = cv2.erode(mm,kernel,iterations = 1)
                mask = mask_new
            except:
                print('can not erode the mask')
        
        mask_2D = to_2D(mask)
        std = [np.std(y_filt[:, np.where(mask_2D[i]>0)[0]].mean(1)) for i in range(len(mask_2D))]
        seq = np.argsort(std)[::-1]
        self.seq = seq  
             
        print(f'sequence of rank1-nmf: {seq}')        
        W, H = nmf_sequential(y_seq, mask=mask, seq=seq, small_mask=True)
        if self.params.mc_nnls['do_plot_init']:
            plt.figure();plt.imshow(H.sum(axis=0).reshape(mov.shape[1:], order='F'));
            plt.colorbar();plt.title('Spatial masks after rank1 nmf')
        
        y_input = np.maximum(y_filt, 0)
        y_input = to_3D(y_input, shape=mov.shape, order='F').transpose([1,2,0])        
        H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y_filt.shape[1],1)) / y_filt.shape[1],
                                     np.random.rand(1,mov.shape[0]), bSiz=None, maxIter=3, 
                                     update_bg=self.params.mc_nnls['update_bg'], use_spikes=self.params.mc_nnls['use_spikes'])
        
        if self.params.mc_nnls['do_plot_init']:
            plt.figure();plt.imshow(H_new.sum(axis=1).reshape(mov.shape[1:], order='F'), 
                                    vmax=np.percentile(H_new.sum(axis=1), 99));
            plt.colorbar();plt.title('Spatial masks after hals');plt.show()
            plt.figure(); plt.imshow(b.reshape((100,100), order='F')); plt.show()
        
        if self.params.mc_nnls['update_bg']:
            H_new = np.hstack((H_new, b))
        self.H = H_new
 
        print('Now extract signal and spikes')            
        Ab = H_new.astype(np.float32)
        template = bin_median(mov, exclude_nans=False)
        center_dims = (template.shape[0], template.shape[1])

        b = mov[0].reshape(-1, order='F')
        x0 = nnls(Ab,b)[0][:,None]
        AtA = Ab.T@Ab
        Atb = Ab.T@b
        n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
        theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
        mc0 = mov[0:1,:,:, None]
        model = get_model(template, center_dims, Ab, 30)
        model.compile(optimizer='rmsprop', loss='mse')
        
        print('Extract traces for initialization')
        if self.params.mc_nnls['initialize_with_gpu']:
            spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, mov)
            traces_viola = spike_extractor.get_traces(mov.shape[0])
            traces_viola = np.array(traces_viola).squeeze().T
            trace = traces_viola.copy()
        else:
            fe = slice(0,None)
            if self.params.spike['flip'] == True:
                trace_nnls = np.array([nnls(H_new,yy)[0] for yy in (-y)[fe]])
            else:
                trace_nnls = np.array([nnls(H_new,yy)[0] for yy in (y)[fe]])
            trace = trace_nnls.T.copy() 
            
        print('Extract spikes for initialization')
        saoz = SignalAnalysisOnlineZ(thresh_range=self.params.spike['thresh_range'], 
                                     do_scale=self.params.spike['do_scale'], 
                                     freq=self.params.spike['freq'], detrend=self.params.spike['detrend'], 
                                     flip=self.params.spike['flip'], adaptive_threshold = self.params.spike['adaptive_threshold'],                                     
                                     filt_window=self.params.spike['filt_window'])
        saoz.fit(trace, num_frames=self.params.spike['num_frames_total'])              
        self.pipeline = Pipeline_overall(model, x0[None, :], x0[None, :], mc0, theta_2, saoz, len(mov))
    
    def fit_online(self):
        self.pipeline.get_spikes()
        
    def compute_estimates(self):
        self.estimates = self.pipeline.saoz
        self.estimates.seq = self.seq
        self.estimates.H = self.H
        self.estimates.params = self.params


        
            
        
        

        
        
        
        
        

            
            
        
        
        
        
        
    
