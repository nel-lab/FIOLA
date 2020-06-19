#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:23:20 2020
VIOLA object for online analysis of voltage imaging data. Including offline init 
of spatial masks and spike extraction algorithm, online analysis of voltage imaging
data.
@author: @agiovann, @caichangjia, @cynthia
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls    
from signal_analysis_online import SignalAnalysisOnlineZ


from caiman_functions import signal_filter, to_3D, to_2D, bin_median
from nmf_support import hals, normalize, nmf_sequential
from pipeline_gpu import get_model, Pipeline_overall


class VIOLA(object):
    def __init__(self, fnames=None, fr=None, ROIs=None, border_to_0=0, freq_detrend=1/3, do_plot_init=True, update_bg=False, num_frames_total=100000, window = 10000, step = 5000, 
                 detrend=True, flip=True, do_scale=False, robust_std=False, freq=15, 
                 thresh_range=[3.5, 5], mfp=0.2, do_plot=False, params={}):
        if params is None:
            logging.warning("Parameters are not set from violaparams")
            raise Exception('Parameters are not set')
        else:
            self.params = params
        self.estimates = {}
        
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
       
        y = to_2D(-mov).copy()
        y_filt = signal_filter(y.T,freq=self.params.mc_nnls['freq_detrend'], 
                               fr=self.params.data['fr']).T        
        
        if self.params.mc_nnls['do_plot_init']:
            plt.figure()
            plt.imshow(mov[0])
            plt.figure()
            plt.imshow(mask.sum(0))
            
        y_seq = y_filt.copy()
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
                                     update_bg=self.params.mc_nnls['update_bg'], use_spikes=True)
        if self.params.mc_nnls['do_plot_init']:
            plt.figure();plt.imshow(H_new.sum(axis=1).reshape(mov.shape[1:], order='F'), 
                                    vmax=np.percentile(H_new.sum(axis=1), 99));
            plt.colorbar();plt.title('Spatial masks after hals');plt.show()
        self.H = H_new

        print('Now extract signal and spikes')            
        fe = slice(0,None)
        if self.params.mc_nnls['update_bg']:
            trace = np.array([nnls(np.hstack((H_new, b)),yy)[0] for yy in (-y)[fe]])
        else:
            trace = np.array([nnls(H_new,yy)[0] for yy in (-y)[fe]])
        trace = trace.T
        
        saoz = SignalAnalysisOnlineZ(do_scale=True, freq=15, detrend=True, flip=True)
        saoz.fit(trace, num_frames=self.params.spike['num_frames_total'])              
        
        print('Now prepare template for online signal processing')
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
        self.pipeline = Pipeline_overall(model, x0[None, :], x0[None, :], mc0, theta_2, saoz, len(mov))
    
    def fit_online(self):
        self.pipeline.get_spikes()


        
            
        
        

        
        
        
        
        

            
            
        
        
        
        
        
    
