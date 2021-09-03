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
import tensorflow as tf
import timeit
from fiola.gpu_mc_nnls import get_mc_model, get_nnls_model, get_model, Pipeline
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from fiola.utilities import signal_filter, to_3D, to_2D, bin_median, hals, normalize, nmf_sequential, local_correlations, quick_annotation

class FIOLA(object):
    def __init__(self, fnames=None, fr=None, ROIs=None, mode='voltage', init_method='masks', num_frames_init=10000, num_frames_total=20000, 
                 ms=[10,10], offline_batch_size=200, border_to_0=0, freq_detrend = 1/3, do_plot_init=False, erosion=0, 
                 hals_movie='hp_thresh', use_rank_one_nmf=False, semi_nmf=False,
                 update_bg=True, use_spikes=False, batch_size=1, use_fft=True, normalize_cc=True,
                 center_dims=None, num_layers=10, initialize_with_gpu=True, 
                 window = 10000, step = 5000, detrend=True, flip=True, 
                 do_scale=False, template_window=2, robust_std=False, freq=15, adaptive_threshold=True, 
                 thresh_range=[3.5, 5], minimal_thresh=3.0, mfp=0.2, online_filter_method = 'median_filter',
                 filt_window = 15, do_plot=False, params={}):
        # for documentation of parameters for class FIOLA, please check fiolaparams.py
        if params is None:
            logging.warning("Parameters are not set from fiolaparams")
            raise Exception('Parameters are not set')
        else:
            self.params = params
        
    def fit(self, mov_input, trace=None):
        self.mov_input = mov_input
        mask = self.params.data['ROIs']
        self.mask = mask
        
        # will perform CaImAn initialization outside the FIOLA object
        if self.params.data['init_method'] == 'caiman':
            if trace is None:
                raise ValueError('Need to input traces from CaImAn')
            mov = mov_input
            self.mov = mov
            mask_2D = mask.transpose([1,2,0]).reshape((-1, mask.shape[0]))
            Ab = mask_2D.copy()
            Ab = Ab / norm(Ab, axis=0)
            self.Ab = Ab            
        
        else:
            # perform motion correction before optimizing spatial footprints
            if self.params.mc_nnls['ms'] is None:
                logging.info('Skip motion correction')
                mov = self.mov_input.copy()
                self.params.mc_nnls['ms'] = [0,0]
            else:
                logging.info('Now start offline motion correction')
                template = bin_median(mov_input, exclude_nans=False)
                mov, self.shifts_offline, _ = self.fit_gpu_motion_correction(self.mov_input, template, self.params.mc_nnls['offline_batch_size'])
            
            logging.info('Now start initialization of spatial footprint')
            
            # quick annotation if no masks are provided
            self.corr = local_correlations(self.mov_input, swap_dim=False)
            if mask is None:
                logging.info('Start quick annotations')
                flag = 0
                while flag == 0:
                    mask = quick_annotation(self.corr, min_radius=5, max_radius=10).astype(np.float32)
                    if len(mask) > 0:
                        flag = 1
                        logging.info(f'You selected {len(mask)} components')
                    else:
                        logging.info(f"You didn't select any components, please reselect")
            
            if len(mask.shape) == 2:
               mask = mask[np.newaxis,:]

            border = self.params.mc_nnls['border_to_0']
            if border > 0:
                mov[:, :border, :] = mov[:, border:border + 1, :]
                mov[:, -border:, :] = mov[:, -border-1:-border, :]
                mov[:, :, :border] = mov[:, :, border:border + 1]
                mov[:, :, -border:] = mov[:, :, -border-1:-border]            
            if not np.all(mov.shape[1:] == mask.shape[1:]):
                raise Exception(f'Movie shape {mov.shape} does not match with masks shape {mask.shape}')

            self.mask = mask
            self.mov = mov
            
            if self.params.hals['hals_movie'] is not None:
                logging.info('Do HALS to optimize masks')
                self.fit_hals(mov, mask)
            else:
                logging.info('Weighted masks are given, no need to do HALS')
                mask_2D = to_2D(mask)
                Ab = mask_2D.T
                Ab = Ab / norm(Ab, axis=0)
                self.Ab = Ab
 
        logging.info('Now compile models for extracting signal and spikes')     
        self.Ab = self.Ab.astype(np.float32)
        template = bin_median(mov, exclude_nans=False)
        
        if self.params.data['init_method'] == 'caiman':
            pass            
        else:
            logging.info('Extract traces for initialization')
            if self.params.mc_nnls['initialize_with_gpu']:
                trace = self.fit_gpu_motion_correction_nnls(self.mov, template, 
                                                            batch_size=self.params.mc_nnls['offline_batch_size'], Ab=self.Ab)                
            else:
                logging.warning('Initialization without GPU')
                fe = slice(0,None)
                trace_nnls = np.array([nnls(self.Ab,yy)[0] for yy in self.mov[fe]])
                trace = trace_nnls.T.copy() 
    
            if np.ndim(trace) == 1:
                trace = trace[None, :]
        
        self.trace_init = trace
        
        logging.info('Extract spikes for initialization')
        saoz = SignalAnalysisOnlineZ(mode=self.params.data['mode'], window=self.params.spike['window'], step=self.params.spike['step'],
                                     detrend=self.params.spike['detrend'], flip=self.params.spike['flip'],                         
                                     do_scale=self.params.spike['do_scale'], template_window=self.params.spike['template_window'], 
                                     robust_std=self.params.spike['robust_std'], adaptive_threshold = self.params.spike['adaptive_threshold'],
                                     frate=self.params.data['fr'], freq=self.params.spike['freq'],
                                     thresh_range=self.params.spike['thresh_range'], minimal_thresh=self.params.spike['minimal_thresh'],
                                     mfp=self.params.spike['mfp'], online_filter_method = self.params.spike['online_filter_method'],                                        
                                     filt_window=self.params.spike['filt_window'], do_plot=self.params.spike['do_plot'])
        saoz.fit(trace, num_frames=self.params.data['num_frames_total'])              
    
        self.pipeline = Pipeline(self.params.data['mode'], self.mov, template, self.params.mc_nnls['batch_size'], self.Ab, saoz, 
                                 ms_h=self.params.mc_nnls['ms'][0], ms_w=self.params.mc_nnls['ms'][1], 
                                 use_fft=self.params.mc_nnls['use_fft'], normalize_cc=self.params.mc_nnls['normalize_cc'], 
                                 center_dims=self.params.mc_nnls['center_dims'], return_shifts=False, 
                                 num_layers=self.params.mc_nnls['num_layers'])
                                 
    def fit_online(self):
        self.pipeline.get_spikes()

    def fit_online_frame(self, frame):
        self.pipeline.load_frame(frame)
        self.pipeline.get_spikes()
        
    def compute_estimates(self):
        self.estimates = self.pipeline.saoz
        self.estimates.Ab = self.Ab
        if hasattr(self, 'seq'):
            self.estimates.seq = self.seq
            
    def fit_hals(self, mov, mask):
        if self.params.spike['flip'] == True:
            logging.info('Flip movie for initialization')
            y = to_2D(-mov).copy() 
        else:
            logging.info('Not flip movie for initialization')
            y = to_2D(mov).copy()

        y_filt = signal_filter(y.T,freq=self.params.hals['freq_detrend'], 
                               fr=self.params.data['fr']).T        
        
        if self.params.hals['do_plot_init']:
            plt.figure(); plt.imshow(mov.mean(0)); plt.title('Mean Image')
            plt.figure(); plt.imshow(mask.sum(0)); plt.title('Masks')
       
        if self.params.hals['erosion'] > 0:
            raise ValueError('Mask erosion is not supported now')
            # try:
            #     logging.info('erode mask')
            #     kernel = np.ones((self.params.mc_nnls['erosion'], self.params.mc_nnls['erosion']),np.uint8)
            #     mask_new = np.zeros(mask.shape)
            #     for idx, mm in enumerate(mask):
            #         mask_new[idx] = cv2.erode(mm,kernel,iterations = 1)
            #     mask = mask_new
            # except:
            #     logging.info('can not erode the mask')
        
        hals_orig = False
        if hals_movie=='hp_thresh':
            y_input = np.maximum(y_filt, 0).T
        elif hals_movie=='hp':
            y_input = y_filt.T
        elif hals_movie=='orig':
            y_input = -y.T
            hals_orig = True
    
        mask_2D = to_2D(mask)
        if self.params.hals['use_rank_one_nmf']:
            y_seq = y_filt.copy()
            std = [np.std(y_filt[:, np.where(mask_2D[i]>0)[0]].mean(1)) for i in range(len(mask_2D))]
            seq = np.argsort(std)[::-1]
            self.seq = seq                   
            logging.info(f'sequence of rank1-nmf: {seq}')        
            W, H = nmf_sequential(y_seq, mask=mask, seq=seq, small_mask=True)
            nA = np.linalg.norm(H)
            H = H/nA
            W = W*nA
        else:
            nA = np.linalg.norm(mask_2D)
            H = mask_2D/nA
            W = (y_input.T@H.T)
            self.seq = np.array(range(mask_2D.shape[0]))

        if self.params.hals['do_plot_init']:
            plt.figure();plt.imshow(H.sum(axis=0).reshape(mov.shape[1:], order='F'));
            plt.colorbar();plt.title('Spatial masks before HALS')
        
        A,C,b,f = hals(y_input, H.T, W.T, np.ones((y_filt.shape[1],1))/y_filt.shape[1],
                         np.random.rand(1,mov.shape[0]), bSiz=None, maxIter=3, semi_nmf=self.params.hals['semi_nmf'],
                         update_bg=self.params.hals['update_bg'], use_spikes=self.params.hals['use_spikes'],
                         hals_orig=hals_orig, fr=self.params.data['fr'])
       
        if self.params.hals['do_plot_init']:
            plt.figure();plt.imshow(A.sum(axis=1).reshape(mov.shape[1:], order='F'), vmax=np.percentile(A.sum(axis=1), 99));
            plt.colorbar();plt.title('Spatial masks after hals'); plt.show()
            plt.figure(); plt.imshow(b.reshape((mov.shape[1],mov.shape[2]), order='F')); plt.title('Background components');plt.show()
        
        if self.params.hals['update_bg']:
            Ab = np.hstack((A, b))
        else:
            Ab = A.copy()                    
        Ab = Ab / norm(Ab, axis=0)
        self.Ab = Ab        
        return self
        
    def fit_gpu_motion_correction(self, mov, template, batch_size):
        """
        Run GPU motion correction
    
        Parameters
        ----------
        mov : ndarray
            input movie
        template : ndarray
            the template used for motion correction
        batch_size : int
            number of frames used for motion correction each time. The default is 1.
        
        Returns
        -------
        mc_mov: ndarray
            motion corrected movie
        shifts: ndarray
            shifts in x and y respectively
        times: list
            time consumption for processing each batch
        """
        
        def generator():
            if len(mov) % batch_size != 0 :
                raise ValueError('batch_size needs to be a factor of frames of the movie')
            for idx in range(len(mov) // batch_size):
                yield{"m":mov[None, idx*batch_size:(idx+1)*batch_size,...,None]}
                     
        def get_frs():
            dataset = tf.data.Dataset.from_generator(generator, output_types={'m':tf.float32}, 
                                                     output_shapes={"m":(1, batch_size, dims[0], dims[1], 1)})
            return dataset
        
        times = []
        out = []
        flag = 1000
        index = 0
        dims = mov.shape[1:]
        mc_model = get_mc_model(template, batch_size, ms_h=self.params.mc_nnls['ms'][0], ms_w=self.params.mc_nnls['ms'][1], 
                                use_fft=self.params.mc_nnls['use_fft'], normalize_cc=self.params.mc_nnls['normalize_cc'], 
                                center_dims=self.params.mc_nnls['center_dims'], return_shifts=True)
        mc_model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(mc_model)
        
        logging.info('now start motion correction')
        start = timeit.default_timer()
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            out.append(i)
            times.append(timeit.default_timer()-start)
            index += 1    
            if index * batch_size >= flag:
                logging.info(f'processed {flag} frames')
                flag += 1000            
        
        logging.info('finish motion correction')
        logging.info(f'total timing:{times[-1]}')
        logging.info(f'average timing per frame:{times[-1] / len(mov)}')
        mc_mov = []; x_sh = []; y_sh = []
        for ou in out:
            keys = list(ou.keys())
            mc_mov.append(ou[keys[0]])
            x_sh.append(ou[keys[1]])
            y_sh.append(ou[keys[2]])
            
        mc_mov = np.vstack(mc_mov)
        mc_mov = mc_mov.reshape((-1, template.shape[0], template.shape[1]), order='F')
        shifts = np.vstack([np.array(x_sh).flatten(), np.array(y_sh).flatten()]).T
        
        return mc_mov, shifts, times
    
    def fit_gpu_nnls(self, mov, Ab, batch_size=1):
        """
        Run GPU NNLS for source extraction
    
        Parameters
        ----------
        mov: ndarray
            motion corrected movie
        Ab: ndarray (number of pixels * number of spatial footprints)
            spatial footprints for neurons and background        
        batch_size: int
            number of frames used for motion correction each time. The default is 1.
        num_layers: int
            number of iterations for performing nnls
        
        Returns
        -------
        trace: ndarray
            extracted temporal traces 
        """
        
        def generator():
            if len(mov) % batch_size != 0 :
                raise ValueError('batch_size needs to be a factor of frames of the movie')
            for idx in range(len(mov) // batch_size):
                yield {"m":mov[None, idx*batch_size:(idx+1)*batch_size,...,None], 
                       "y":y, "x":x, "k":[[0.0]]}
                
        def get_frs():
            dataset = tf.data.Dataset.from_generator(generator, 
                                                     output_types={"m": tf.float32,
                                                                   "y": tf.float32,
                                                                   "x": tf.float32,
                                                                   "k": tf.float32}, 
                                                     output_shapes={"m":(1, batch_size, dims[0], dims[1], 1),
                                                                    "y":(1, num_components, batch_size),
                                                                    "x":(1, num_components, batch_size),
                                                                    "k":(1, 1)})
            return dataset
        
        times = []
        out = []
        flag = 1000
        index = 0
        dims = mov.shape[1:]
        
        b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')         
        x0 = np.array([nnls(Ab,b[:,i])[0] for i in range(batch_size)]).T
        x, y = np.array(x0[None,:]), np.array(x0[None,:]) 
        num_components = Ab.shape[-1]
        
        nnls_model = get_nnls_model(dims, Ab, batch_size, self.params.mc_nnls['num_layers'])
        nnls_model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(nnls_model)
        
        logging.info('now start source extraction')
        start = timeit.default_timer()
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            out.append(i)
            times.append(timeit.default_timer()-start)
            index += 1    
            if index * batch_size >= flag:
                logging.info(f'processed {flag} frames')
                flag += 1000            
        
        logging.info('finish source extraction')
        logging.info(f'total timing:{times[-1]}')
        logging.info(f'average timing per frame:{times[-1] / len(mov)}')
        
        trace = []; 
        for ou in out:
            keys = list(ou.keys())
            trace.append(ou[keys[0]][0])        
        trace = np.hstack(trace)
        
        return trace

    def fit_gpu_motion_correction_nnls(self, mov, template, batch_size, Ab):
        """
        Run GPU motion correction and source extraction
    
        Parameters
        ----------
        mov: ndarray
            motion corrected movie
        template : ndarray
            the template used for motion correction
        batch_size: int
            number of frames used for motion correction each time. The default is 1.
        Ab: ndarray (number of pixels * number of spatial footprints)
            spatial footprints for neurons and background        
        
        Returns
        -------
        trace: ndarray
            extracted temporal traces 
        """
        
        def generator():
            if len(mov) % batch_size != 0 :
                raise ValueError('batch_size needs to be a factor of frames of the movie')
            for idx in range(len(mov) // batch_size):
                yield {"m":mov[None, idx*batch_size:(idx+1)*batch_size,...,None], 
                       "y":y, "x":x, "k":[[0.0]]}
                
        def get_frs():
            dataset = tf.data.Dataset.from_generator(generator, 
                                                     output_types={"m": tf.float32,
                                                                   "y": tf.float32,
                                                                   "x": tf.float32,
                                                                   "k": tf.float32}, 
                                                     output_shapes={"m":(1, batch_size, dims[0], dims[1], 1),
                                                                    "y":(1, num_components, batch_size),
                                                                    "x":(1, num_components, batch_size),
                                                                    "k":(1, 1)})
            return dataset
        
        times = []
        out = []
        flag = 1000
        index = 0
        dims = mov.shape[1:]
        
        b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')         
        x0 = np.array([nnls(Ab,b[:,i])[0] for i in range(batch_size)]).T
        x, y = np.array(x0[None,:]), np.array(x0[None,:]) 
        num_components = Ab.shape[-1]
        
        model = get_model(template, Ab, batch_size, 
                          ms_h=self.params.mc_nnls['ms'][0], ms_w=self.params.mc_nnls['ms'][1],
                          use_fft=self.params.mc_nnls['use_fft'], normalize_cc=self.params.mc_nnls['normalize_cc'], 
                          center_dims=self.params.mc_nnls['center_dims'], return_shifts=False, 
                          num_layers=self.params.mc_nnls['num_layers'])
        model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(model)
        
        logging.info('now start motion correction and source extraction')
        start = timeit.default_timer()
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            out.append(i)
            times.append(timeit.default_timer()-start)
            index += 1    
            if index * batch_size >= flag:
                logging.info(f'processed {flag} frames')
                flag += 1000            
        
        logging.info('finish motion correction and source extraction')
        logging.info(f'total timing:{times[-1]}')
        logging.info(f'average timing per frame:{times[-1] / len(mov)}')
        
        trace = []; 
        for ou in out:
            keys = list(ou.keys())
            trace.append(ou[keys[0]][0])        
        trace = np.hstack(trace)
        
        return trace
    
    
            
            
        
        

        
        
        
        
        

            
            
        
        
        
        
        
    
