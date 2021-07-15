#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:23:20 2020
FIOLA object for online analysis of fluorescence imaging data. Including offline 
initialization of spatial masks and online analysis of voltage imaging data.
Please check violaparams.py for the explanation of parameters.
@author: @agiovann, @caichangjia, @cynthia
"""
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.optimize import nnls    
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from fiola.utilities import signal_filter, to_3D, to_2D, bin_median, hals, normalize, nmf_sequential

class FIOLA(object):
    def __init__(self, fnames=None, fr=None, ROIs=None, num_frames_init=10000, num_frames_total=20000, 
                 border_to_0=0, freq_detrend = 1/3, do_plot_init=True, erosion=0, 
                 hals_movie='hp_thresh', use_rank_one_nmf=True, semi_nmf=False,
                 update_bg=False, use_spikes=False, use_batch=True, batch_size=1, 
                 center_dims=None, initialize_with_gpu=False, 
                 window = 10000, step = 5000, detrend=True, flip=True, 
                 do_scale=False, template_window=2, robust_std=False, freq=15, adaptive_threshold=True, 
                 thresh_range=[3.5, 5], minimal_thresh=3.0, mfp=0.2, online_filter_method = 'median_filter',
                 filt_window = 15, do_plot=False, params={}):
        if params is None:
            logging.warning("Parameters are not set from fiolaparams")
            raise Exception('Parameters are not set')
        else:
            self.params = params
        
    def fit(self, mov):
        print('Now start initialization of spatial footprint')
        border = self.params.mc_nnls['border_to_0']
        mask = self.params.data['ROIs']
        
        if len(mask.shape) == 2:
           mask = mask[np.newaxis,:]
        if border > 0:
            mov[:, :border, :] = mov[:, border:border + 1, :]
            mov[:, -border:, :] = mov[:, -border-1:-border, :]
            mov[:, :, :border] = mov[:, :, border:border + 1]
            mov[:, :, -border:] = mov[:, :, -border-1:-border]
        self.mov = mov
        self.mask = mask
        
        if not np.all(mov.shape[1:] == mask.shape[1:]):
            raise Exception(f'Movie shape {mov.shape} does not match with masks shape {mask.shape}')

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
        
        hals_movie = self.params.mc_nnls['hals_movie']
        hals_orig = False
        if hals_movie=='hp_thresh':
            y_input = np.maximum(y_filt, 0).T
        elif hals_movie=='hp':
            y_input = y_filt.T
        elif hals_movie=='orig':
            y_input = -y.T
            hals_orig = True
        
        mask_2D = to_2D(mask)
        if self.params.mc_nnls['use_rank_one_nmf']:
            y_seq = y_filt.copy()
            std = [np.std(y_filt[:, np.where(mask_2D[i]>0)[0]].mean(1)) for i in range(len(mask_2D))]
            seq = np.argsort(std)[::-1]
            self.seq = seq  
                 
            print(f'sequence of rank1-nmf: {seq}')        
            W, H = nmf_sequential(y_seq, mask=mask, seq=seq, small_mask=True)
            nA = np.linalg.norm(H)
            H = H/nA
            W = W*nA
        else:
            nA = np.linalg.norm(mask_2D)
            H = mask_2D/nA
            W = (y_input.T@H.T)
            self.seq = np.array(range(mask_2D.shape[0]))

        if self.params.mc_nnls['do_plot_init']:
            plt.figure();plt.imshow(H.sum(axis=0).reshape(mov.shape[1:], order='F'));
            plt.colorbar();plt.title('Spatial masks before hals')
        
        print(f'HALS')
        H_new,W_new,b,f = hals(y_input, H.T, W.T, np.ones((y_filt.shape[1],1))/y_filt.shape[1],
                         np.random.rand(1,mov.shape[0]), bSiz=None, maxIter=3, semi_nmf=self.params.mc_nnls['semi_nmf'],
                         update_bg=self.params.mc_nnls['update_bg'], use_spikes=self.params.mc_nnls['use_spikes'],
                         hals_orig=hals_orig, fr=self.params.data['fr'])
       
        if self.params.mc_nnls['do_plot_init']:
            plt.figure();plt.imshow(H_new.sum(axis=1).reshape(mov.shape[1:], order='F'), 
                                    vmax=np.percentile(H_new.sum(axis=1), 99));
            plt.colorbar();plt.title('Spatial masks after hals');plt.show()
            plt.figure(); plt.imshow(b.reshape((self.mov.shape[1],self.mov.shape[2]), order='F')); plt.show()
        
        if self.params.mc_nnls['update_bg']:
            H_new = np.hstack((H_new, b))
        H_new = H_new / norm(H_new, axis=0)
        self.H = H_new
 
        if True:
            print('Now extract signal and spikes')     
            batch_size = self.params.mc_nnls['batch_size']
            Ab = H_new.astype(np.float32)
            num_components = Ab.shape[-1]
            template = bin_median(mov, exclude_nans=False)
            if self.params.mc_nnls['center_dims'] is None:
                center_dims = (mov.shape[1], mov.shape[2])
            else:
                center_dims = self.params.mc_nnls['center_dims'].copy()
    
            if not self.params.mc_nnls['use_batch']:
                from fiola.pipeline_gpu import get_model, Pipeline_overall, Pipeline
                b = mov[0].reshape(-1, order='F')
                x0 = nnls(Ab,b)[0][:,None]
                AtA = Ab.T@Ab
                Atb = Ab.T@b
                n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
                theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
                mc0 = mov[0:1,:,:, None]
                model = get_model(template, center_dims, Ab, 30, ms_h=0, ms_w=0)
                model.compile(optimizer='rmsprop', loss='mse')
            else:
                from fiola.batch_gpu import Pipeline, get_model, Pipeline_overall_batch
                b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')
                x0=[]
                for i in range(batch_size):
                    x0.append(nnls(Ab,b[:,i])[0])
                x0 = np.array(x0).T
                AtA = Ab.T@Ab
                Atb = Ab.T@b
                n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
                theta_2 = (Atb/n_AtA).astype(np.float32)
                model_batch = get_model(template, center_dims, Ab, num_components, batch_size, ms_h=0, ms_w=0)  # todo
                model_batch.compile(optimizer = 'rmsprop',loss='mse')
                mc0 = mov[0:batch_size, :, :, None][None, :]
                x_old, y_old = np.array(x0[None,:]), np.array(x0[None,:])  
                
            print('Extract traces for initialization')
            if self.params.mc_nnls['initialize_with_gpu']:
                if not self.params.mc_nnls['use_batch']:
                    spike_extractor = Pipeline(model, x0[None, :], x0[None, :], mc0, theta_2, mov)
                    traces_fiola = spike_extractor.get_traces(mov.shape[0])
                    traces_fiola = np.array(traces_fiola).squeeze().T
                    trace = traces_fiola.copy()
                else:
                    spike_extractor = Pipeline(model_batch, x_old, y_old, mc0, theta_2, mov, num_components, batch_size)
                    spikes_gpu = spike_extractor.get_traces(len(mov))
                    traces_fiola = []
                    for spike in spikes_gpu:
                        for i in range(batch_size):
                            traces_fiola.append([spike[:,:,i]])
                    traces_fiola = np.array(traces_fiola).squeeze().T
                    trace = traces_fiola.copy()
            else:
                fe = slice(0,None)
                if self.params.spike['flip'] == True:
                    trace_nnls = np.array([nnls(H_new,yy)[0] for yy in (-y)[fe]])
                else:
                    trace_nnls = np.array([nnls(H_new,yy)[0] for yy in (y)[fe]])
                trace = trace_nnls.T.copy() 

            if np.ndim(trace) == 1:
                trace = trace[None, :]
            print('Extract spikes for initialization')
            saoz = SignalAnalysisOnlineZ(window=self.params.spike['window'], step=self.params.spike['step'],
                                         detrend=self.params.spike['detrend'], flip=self.params.spike['flip'],                         
                                         do_scale=self.params.spike['do_scale'], template_window=self.params.spike['template_window'], 
                                         robust_std=self.params.spike['robust_std'], adaptive_threshold = self.params.spike['adaptive_threshold'],
                                         frate=self.params.data['fr'], freq=self.params.spike['freq'],
                                         thresh_range=self.params.spike['thresh_range'], minimal_thresh=self.params.spike['minimal_thresh'],
                                         mfp=self.params.spike['mfp'], online_filter_method = self.params.spike['online_filter_method'],                                        
                                         filt_window=self.params.spike['filt_window'], do_plot=self.params.spike['do_plot'])
            
            saoz.fit(trace, num_frames=self.params.data['num_frames_total'])              
            
            if not self.params.mc_nnls['use_batch']:
                self.pipeline = Pipeline_overall(model, x0[None, :], x0[None, :], mc0, theta_2, saoz, len(mov))
            else:
                self.pipeline = Pipeline_overall_batch(model_batch, x0[None, :], x0[None, :], mc0, theta_2, mov, num_components, batch_size, saoz, len(mov))
            
        
    def fit_online(self):
        self.pipeline.get_spikes()
        
    def fit_without_gpu(self, mov):
        border = self.params.mc_nnls['border_to_0']
        if border > 0:
            mov[:, :border, :] = mov[:, border:border + 1, :]
            mov[:, -border:, :] = mov[:, -border-1:-border, :]
            mov[:, :, :border] = mov[:, :, border:border + 1]
            mov[:, :, -border:] = mov[:, :, -border-1:-border]

        y = to_2D(mov).copy()
        fe = slice(0,None)
        trace_nnls = np.array([nnls(self.H,yy)[0] for yy in (y)[fe]])
        trace = trace_nnls.T.copy() 

        if len(trace.shape) == 1:
            trace = trace[None, :]
        saoz = SignalAnalysisOnlineZ(window=self.params.spike['window'], step=self.params.spike['step'],
                                     detrend=self.params.spike['detrend'], flip=self.params.spike['flip'],                         
                                     do_scale=self.params.spike['do_scale'], template_window=self.params.spike['template_window'], 
                                     robust_std=self.params.spike['robust_std'], adaptive_threshold = self.params.spike['adaptive_threshold'],
                                     frate=self.params.data['fr'], freq=self.params.spike['freq'],
                                     thresh_range=self.params.spike['thresh_range'], minimal_thresh=self.params.spike['minimal_thresh'],
                                     mfp=self.params.spike['mfp'], online_filter_method = self.params.spike['online_filter_method'],                                        
                                     filt_window=self.params.spike['filt_window'], do_plot=self.params.spike['do_plot'])
        saoz.fit(trace[:,:self.params.data['num_frames_init']], num_frames=self.params.data['num_frames_total'])   
        for n in range(self.params.data['num_frames_init'], trace.shape[1]):
            saoz.fit_next(trace[:, n: n+1], n)
        #self.pipeline.saoz = saoz
        self.saoz = saoz
        """
        saoz.compute_SNR()
        saoz.reconstruct_signal()
        print(f'thresh:{saoz.thresh}')
        print(f'SNR: {saoz.SNR}')
        print(f'Mean_SNR: {np.array(saoz.SNR).mean()}')
        print(f'Spikes based on mask sequence: {(saoz.index>0).sum(1)}')
        estimates = saoz
        estimates.spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz.index])
        weights = H_new.reshape((mov.shape[1], mov.shape[2], H_new.shape[1]), order='F')
        weights = weights.transpose([2, 0, 1])
        estimates.weights = weights
        """
        
    def compute_estimates(self):
        self.estimates = self.pipeline.saoz
        #self.estimates = self.saoz
        self.estimates.seq = self.seq
        self.estimates.H = self.H
        self.estimates.params = self.params


        
            
        
        

        
        
        
        
        

            
            
        
        
        
        
        
    
