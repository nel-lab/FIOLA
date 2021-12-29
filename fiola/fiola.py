#!/usr/bin/env python
"""
FIOLA object for online analysis of fluorescence imaging data. Including offline 
initialization and online analysis of voltage/calcium imaging data.
Please check fiolaparams.py for the explanation of parameters.
@author: @agiovann, @caichangjia, @cynthia
"""
import logging
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from numpy.linalg import norm
from scipy.optimize import nnls  
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True) # limit gpu memory
import timeit

from fiola.gpu_mc_nnls import get_mc_model, get_nnls_model, get_model, Pipeline
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from fiola.utilities import signal_filter, to_3D, to_2D, bin_median, hals, normalize, nmf_sequential, local_correlations, quick_annotation, HALS4activity

class FIOLA(object):
    def __init__(self, fnames=None, fr=400, ROIs=None, mode='voltage', init_method='binary_masks', num_frames_init=10000, num_frames_total=20000, 
                 ms=[10,10], offline_batch_size=200, border_to_0=0, freq_detrend = 1/3, do_plot_init=False, erosion=0, 
                 hals_movie='hp_thresh', use_rank_one_nmf=False, semi_nmf=False,
                 update_bg=True, use_spikes=False, batch_size=1, use_fft=True, normalize_cc=True,
                 center_dims=None, num_layers=10, n_split=1, initialize_with_gpu=True, 
                 window = 10000, step = 5000, detrend=True, flip=True, 
                 do_scale=False, template_window=2, robust_std=False, freq=15, adaptive_threshold=True, 
                 minimal_thresh=3.0, online_filter_method = 'median_filter',
                 filt_window = 15, do_plot=False, params={}):
        # please check fiolaparams.py for detailed documentation of parameters in class FIOLA
        if params is None:
            logging.warning("parameters are not set from fiolaparams")
            raise Exception('parameters are not set')
        else:
            self.params = params
        
    def fit(self, mov_input, mode, mask=None):
        """
        Offline method for doing motion correction, source extraction and spike detection(if needed).
        Prepare objects and parameters for the online analysis.

        Parameters
        ----------
        mov_input : ndarray
            input raw movie
        mode: str 
            ('voltage' or 'calcium')
        mask: 3D vector:
            if available a binary or weighted mask (output of CaImAn) can be provided
            
        """

        mov_input = mov_input.astype(np.float32)
        self.mov_input = mov_input
        if mov_input.min() < 0:
            self.min_mov = mov_input.min()
        else:
            self.min_mov = 0
        self.dims = mov_input.shape[1:]
        # subtract minimal of the movie to make nonnegative
        mov = self.mov_input.copy() - self.min_mov        
        # will perform CaImAn initialization outside the FIOLA object
        if mode == 'calcium':
            if mask is None:
                logging.info('beginning CaImAn initialization')                
                mov, trace, mask = self.fit_caiman_init(mov, self.params.mc_dict,  
                                                        self.params.opts_dict, self.params.quality_dict)               
                logging.info(f'found {mask.shape[0]} neurons')
                logging.info('CaImAn initialization complete')                               
                
            else:
                logging.info('beginning offline motion correction')
                template = bin_median(mov)
                # here we don't remove min_mov as it is removed before
                mov, self.shifts_offline, _ = self.fit_gpu_motion_correction(mov, template, self.params.mc_nnls['offline_batch_size'], 0) 
            self.corr = local_correlations(mov, swap_dim=False)                        
                        
        else: 
            # perform motion correction before optimizing spatial footprints
            if self.params.mc_nnls['ms'] is None:
                logging.info('skip motion correction')
                self.params.mc_nnls['ms'] = [0,0]
            else:
                logging.info('beginning offline motion correction')
                template = bin_median(mov)
                # here we don't remove min_mov as it is removed before
                mov, self.shifts_offline, _ = self.fit_gpu_motion_correction(mov, template, self.params.mc_nnls['offline_batch_size'], 0) 
            
            # quick annotation if no masks are provided
            self.corr = local_correlations(mov, swap_dim=False)
            logging.info('beginning initialization of spatial footprint')
            
            if mask is None:
                logging.info('beginning quick annotations')
                flag = 0
                while flag == 0:
                    mask = quick_annotation(self.corr, min_radius=5, max_radius=10).astype(np.float32)
                    if len(mask) > 0:
                        flag = 1
                        logging.info(f'selected {len(mask)} components')
                    else:
                        logging.warning("didn't select any components, please reselect")
        
        
        self.params.data['ROIs'] = mask             
            
        if len(mask.shape) == 2:
           mask = mask[np.newaxis,:]

        border = self.params.mc_nnls['border_to_0']
        if border > 0:
            mov[:, :border, :] = mov[:, border:border + 1, :]
            mov[:, -border:, :] = mov[:, -border-1:-border, :]
            mov[:, :, :border] = mov[:, :, border:border + 1]
            mov[:, :, -border:] = mov[:, :, -border-1:-border]            
        
        if not np.all(mov.shape[1:] == mask.shape[1:]):
            raise Exception(f'movie shape {mov.shape} does not match masks shape {mask.shape}')

        self.mask = mask
        self.mov = mov
        
        if self.params.hals['estimate_neuron_baseline']:                
            self.fit_hals(mov, mask, order='F')
            
        else:
            mask_2D = to_2D(mask, order='F') 
            Ab = mask_2D.copy()
            Ab = Ab / norm(Ab, axis=0)
            self.Ab = Ab             
 
        logging.info('compiling models for extracting signal and spikes')     
        self.Ab = self.Ab.astype(np.float32)
        template = bin_median(mov)       
       
        logging.info('extracting traces for initialization')
        if self.params.mc_nnls['initialize_with_gpu']:
            # here we don't remove min_mov as it is removed before
            trace = self.fit_gpu_motion_correction_nnls(self.mov, template, 
                                                        batch_size=self.params.mc_nnls['offline_batch_size'], 
                                                        min_mov=0, Ab=self.Ab)                
        else:
            logging.warning('initializing without GPU')
            fe = slice(0,None)
            trace_nnls = np.array([nnls(self.Ab,yy)[0] for yy in self.mov[fe]])
            trace = trace_nnls.T.copy() 

        if np.ndim(trace) == 1:
            trace = trace[None, :]        
        self.trace_init = trace
        self.Cf = trace

        logging.info('extracting spikes for initialization')
        saoz = self.fit_spike_extraction(trace)
    
        logging.info('compiling new models for online analysis')
        self.pipeline = Pipeline(self.params.data['mode'], self.mov, template, self.params.mc_nnls['batch_size'], self.Ab, saoz, 
                                 ms_h=self.params.mc_nnls['ms'][0], ms_w=self.params.mc_nnls['ms'][1], min_mov=self.min_mov,
                                 use_fft=self.params.mc_nnls['use_fft'], normalize_cc=self.params.mc_nnls['normalize_cc'], 
                                 center_dims=self.params.mc_nnls['center_dims'], return_shifts=False, 
                                 num_layers=self.params.mc_nnls['num_layers'])
        
        return self
                                 
    def fit_online(self):
        """
        process online
        """
        self.pipeline.get_spikes()
        return self

    def fit_online_frame(self, frame):
        """
        process the single input frame        

        Parameters
        ----------
        frame : ndarray
            input frame
        """
        self.pipeline.load_frame(frame)
        self.pipeline.get_spikes()
        return self
        
    def compute_estimates(self):
        """
        put computed results into estimates 
        """
        try:
            self.estimates = self.pipeline.saoz
        except:
            logging.warning('not using pipeline object; directly using saoz instead')
            self.estimates = self.saoz
        self.estimates.Ab = self.Ab
        if hasattr(self, 'seq'):
            self.estimates.seq = self.seq
        if self.params.data['mode'] == 'voltage':
            self.estimates.reconstruct_signal()
            
        return self
    
    def fit_caiman_init(self, mov, mc_dict, opts_dict, quality_dict, save_movie=True):
        """
        run caiman initialization for FIOLA
        
        Parameters
        ----------
        mov : ndarray
            input movie
        mc_dict : dict
            parameters for motion correction
        opts_dict : dict
            parameters for source extraction
        quality_dict : dict
            parameters for quality control
        save_movie : bool
            whether to save input movie or not. The default is True.

        Returns
        -------
        mov : ndarray
            motion corrected movie
        trace_init : ndarray
            output traces
        mask : ndarray
            spatial footprints

        """
        from fiola.caiman_init import run_caiman_init
        if save_movie == True:
            init_name = self.params.mc_dict['fnames'].split('.')[0] + f'_{mov.shape[0]}.tif'
        else:
            init_name = self.params.mc_dict['fnames']
        estimates = run_caiman_init(mov, init_name, mc_dict, opts_dict, quality_dict, save_movie=save_movie)
        estimates.plot_contours(img=estimates.Cn)
        plt.title('found components') 
        plt.show()
        mask = np.hstack((estimates.A.toarray(), estimates.b)).reshape([mov.shape[1], mov.shape[2], -1], order='F').transpose([2, 0, 1])
        trace_init = np.vstack((estimates.C, estimates.f))
        if np.ndim(trace_init) == 1:
            trace_init = trace_init[None, :]        
            
        
        return mov, trace_init, mask        
            
    def fit_hals(self, mov, mask, order='F'):
        """
        optimize binary masks to be weighted masks using HALS algorithm

        Parameters
        ----------
        mov : ndarray
            input movie
        mask : ndarray (binary)
            masks for each neuron

        """
        if self.params.spike['flip'] == True:
            logging.info('flipping movie for initialization')
            y = to_2D(-mov).copy() 
        else:
            logging.info('movie flip not requested')
            y = to_2D(mov).copy() 
            
        y_filt = signal_filter(y.T,freq=self.params.hals['freq_detrend'], 
                               fr=self.params.data['fr']).T        

        
        if self.params.hals['nb']>0:
            mask = mask[:-self.params.hals['nb']]
        
        if self.params.hals['do_plot_init']:
            plt.figure(); plt.imshow(mov.mean(0)); plt.title('Mean Image')
            plt.figure(); plt.imshow(mask.sum(0)); plt.title('Masks')
       
        if self.params.hals['erosion'] > 0:
            raise ValueError('mask erosion is not currently implemented')
        
        hals_orig = False
        if self.params.hals['hals_movie']=='hp_thresh':
            y_input = np.maximum(y_filt, 0).T
        elif self.params.hals['hals_movie']=='hp':
            y_input = y_filt.T
        elif self.params.hals['hals_movie']=='orig':
            y_input = -y.T
            hals_orig = True
    
            
        mask_2D = to_2D(mask, order=order)        
        if self.params.hals['use_rank_one_nmf']:
            y_seq = y_filt.copy()
            # standard deviation of average signal in each mask
            std = [np.std(y_filt[:, np.where(mask_2D[i]>0)[0]].mean(1)) for i in range(len(mask_2D))]
            seq = np.argsort(std)[::-1]
            self.seq = seq                   
            logging.info(f'sequence of rank1-nmf: {seq}')        
            W, H = nmf_sequential(y_seq, mask=mask, seq=seq, small_mask=True)
            nA = np.linalg.norm(H)
            H = H/nA
            W = W*nA
        else:
            logging.info('regressing matrices')
            nA = np.linalg.norm(mask_2D)
            H = mask_2D/nA
            W = (y_input.T@H.T)
            self.seq = np.array(range(mask_2D.shape[0]))

        if self.params.hals['do_plot_init']:
            plt.figure();plt.imshow(H.sum(axis=0).reshape(mov.shape[1:], order='F'));
            plt.colorbar();plt.title('spatial masks before HALS')
        
        logging.info('computing hals')
        output_nb = np.maximum(self.params.hals['nb'],1)
         
        A,C,b,f = hals(y_input, H.T, W.T, np.ones((y_filt.shape[1],output_nb ))/y_filt.shape[1],
                         np.random.rand(output_nb ,mov.shape[0]), bSiz=None, maxIter=3, semi_nmf=self.params.hals['semi_nmf'],
                         update_bg=self.params.hals['update_bg'], use_spikes=self.params.hals['use_spikes'],
                         hals_orig=hals_orig, fr=self.params.data['fr'])
       
        
            
        if self.params.hals['do_plot_init']:
            plt.figure();plt.imshow(A.sum(axis=1).reshape(mov.shape[1:], order='F'), vmax=np.percentile(A.sum(axis=1), 99));
            plt.colorbar();plt.title('spatial masks after hals'); plt.show()
            plt.figure(); plt.imshow(b.reshape((mov.shape[1],mov.shape[2]), order='F')); plt.title('Background components');plt.show()
        
        if self.params.hals['update_bg']:
            Ab = np.hstack((A, b))
        else:
            Ab = A.copy()                    
        Ab = Ab / norm(Ab, axis=0)
        self.Ab = Ab 
        # if self.params.spike['flip'] == True:
        #     self.Cf = HALS4activity(-y.T, Ab, C=np.zeros_like(np.vstack((C, f))))
        # else:
        #     self.Cf = HALS4activity(y.T, A, C=np.zeros_like(np.vstack((C, f))))
        return self
        
    def fit_gpu_motion_correction(self, mov, template, batch_size, min_mov):
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
        min_mov: float
            minimum of the movie will be removed. The default is 0.
        
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
        flag = 1000 # logging every 1000 frames
        index = 0
        dims = mov.shape[1:]
        mc_model = get_mc_model(template, batch_size, ms_h=self.params.mc_nnls['ms'][0], ms_w=self.params.mc_nnls['ms'][1], min_mov=min_mov,
                                use_fft=self.params.mc_nnls['use_fft'], normalize_cc=self.params.mc_nnls['normalize_cc'], 
                                center_dims=self.params.mc_nnls['center_dims'], return_shifts=True)
        mc_model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(mc_model)
        
        logging.info('beginning motion correction')
        start = timeit.default_timer()
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            out.append(i)
            times.append(timeit.default_timer()-start)
            index += 1    
            if index * batch_size >= flag:
                logging.info(f'processed {flag} frames')
                flag += 1000            
        
        logging.info('motion correction complete')
        logging.info(f'total timing: {times[-1]}')
        logging.info(f'average timing per frame: {times[-1] / len(mov)}')
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
        C_init = np.dot(Ab.T, b)
        x0 = np.array([HALS4activity(Yr=b[:,i], A=Ab, C=C_init[:, i].copy(), iters=10) for i in range(batch_size)]).T
        x, y = np.array(x0[None,:]), np.array(x0[None,:]) 
        num_components = Ab.shape[-1]
        nnls_model = get_nnls_model(dims, Ab, batch_size, self.params.mc_nnls['num_layers'], self.params.mc_nnls['n_split'])
        nnls_model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(nnls_model)
        
        logging.info('beginning source extraction')
        start = timeit.default_timer()
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            out.append(i)
            times.append(timeit.default_timer()-start)
            index += 1    
            if index * batch_size >= flag:
                logging.info(f'processed {flag} frames')
                flag += 1000            
        
        logging.info('source extraction complete')
        logging.info(f'total timing: {times[-1]}')
        logging.info(f'average timing per frame: {times[-1] / len(mov)}')
        
        trace = []; 
        for ou in out:
            keys = list(ou.keys())
            trace.append(ou[keys[0]][0])        
        trace = np.hstack(trace)
        
        return trace

    def fit_gpu_motion_correction_nnls(self, mov, template, batch_size, min_mov, Ab):
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
        min_mov: float
            minimum of the movie will be removed. The default is 0.
        
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
        C_init = np.dot(Ab.T, b)
        x0 = np.array([HALS4activity(Yr=b[:,i], A=Ab, C=C_init[:, i].copy(), iters=10) for i in range(batch_size)]).T
        x, y = np.array(x0[None,:]), np.array(x0[None,:]) 
        num_components = Ab.shape[-1]
        
        model = get_model(template, Ab, batch_size, 
                          ms_h=self.params.mc_nnls['ms'][0], ms_w=self.params.mc_nnls['ms'][1], min_mov=min_mov,
                          use_fft=self.params.mc_nnls['use_fft'], normalize_cc=self.params.mc_nnls['normalize_cc'], 
                          center_dims=self.params.mc_nnls['center_dims'], return_shifts=False, 
                          num_layers=self.params.mc_nnls['num_layers'], n_split=self.params.mc_nnls['n_split'])
        

        model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(model)
        
        logging.info('beginning motion correction and source extraction')
        start = timeit.default_timer()
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            out.append(i)
            times.append(timeit.default_timer()-start)
            index += 1    
            if index * batch_size >= flag:
                logging.info(f'processed {flag} frames')
                flag += 1000            
        
        logging.info('motion correction and source extraction complete')
        logging.info(f'total timing: {times[-1]}')
        logging.info(f'average timing per frame: {times[-1] / len(mov)}')
        
        trace = []; 
        for ou in out:
            keys = list(ou.keys())
            trace.append(ou[keys[0]][0])        
        trace = np.hstack(trace)
        
        return trace
    
    def fit_spike_extraction(self, trace):
        """
        run spike extraction on input traces (spike extraction is only available for voltage movie)

        Parameters
        ----------
        trace : ndarray
            input traces

        Returns
        -------
        saoz : instance
            object encapsulating online spike extraction

        """
        times = []
        start = timeit.default_timer()
        logging.info('beginning spike extraction')
        saoz = SignalAnalysisOnlineZ(mode=self.params.data['mode'], window=self.params.spike['window'], step=self.params.spike['step'],
                                     detrend=self.params.spike['detrend'], flip=self.params.spike['flip'],                         
                                     do_scale=self.params.spike['do_scale'], template_window=self.params.spike['template_window'], 
                                     robust_std=self.params.spike['robust_std'], adaptive_threshold = self.params.spike['adaptive_threshold'],
                                     fr=self.params.data['fr'], freq=self.params.spike['freq'],
                                     minimal_thresh=self.params.spike['minimal_thresh'], online_filter_method = self.params.spike['online_filter_method'],                                        
                                     filt_window=self.params.spike['filt_window'], do_plot=self.params.spike['do_plot'])
        saoz.fit(trace, num_frames=self.params.data['num_frames_total'])    
        times.append(timeit.default_timer()-start)
        logging.info('spike extraction complete')
        if self.params.data['mode'] == 'calcium':
            logging.info('calcium deconvolution is not currently implemented')
        else:
            logging.info(f'total timing: {times[-1]}')
            logging.info(f'average timing per neuron: {times[-1] / len(trace)}')
                  
        return saoz      
    
    def view_components(self, img, idx=None, cnm_estimates=None):
        """ View spatial and temporal components interactively
        Args:
            estimates: dict
                estimates dictionary contain results of VolPy

            idx: list
                index of selected neurons
        """
        if idx is None:
            idx = np.arange(len(self.estimates.Ab.T))
            #idx = np.arange(415)

        n = len(idx) 
        fig = plt.figure(figsize=(10, 10))
        
        dims = img.shape
        
        spatial = self.estimates.Ab.T.reshape([-1, dims[0], dims[1]], order='F')
    
        axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])
        ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
        ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
        ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])    
        s_comp = Slider(axcomp, 'Component', 0, n, valinit=0)
        vmax = np.percentile(img, 98)
        
        def arrow_key_image_control(event):
    
            if event.key == 'left':
                new_val = np.round(s_comp.val - 1)
                if new_val < 0:
                    new_val = 0
                s_comp.set_val(new_val)
    
            elif event.key == 'right':
                new_val = np.round(s_comp.val + 1)
                if new_val > n :
                    new_val = n  
                s_comp.set_val(new_val)
            
        def update(val):
            i = np.int(np.round(s_comp.val))
            print(f'component:{i}')
    
            if i < n:
                
                ax1.cla()
                imgtmp = spatial[idx][i]
                ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
                ax1.set_title(f'Spatial component {i+1}')
                ax1.axis('off')
                
                ax2.cla()
                if self.params.data['mode'] == 'calcium':
                    ax2.plot(self.estimates.trace[idx][i], alpha=0.8, label='extracted traces')
                    if cnm_estimates is not None:
                        ax2.plot(np.vstack((cnm_estimates.C, cnm_estimates.f))[idx][i], label='caiman result')                        
                        #ax2.plot((cnm_estimates.C+cnm_estimates.YrA)[idx][i], label='caiman result')                        
                    ax2.legend()
                    ax2.set_title(f'Signal {i+1}')
                else:
                    ax2.plot(normalize(self.estimates.t_s[idx][i]))            
                    spikes = np.delete(self.estimates.index[i], self.estimates.index[i]==0)
                    h_min = normalize(self.estimates.t_s[idx][i]).max()
                    ax2.vlines(spikes, h_min, h_min + 1, color='black')
                    ax2.legend(labels=['trace', 'spikes'])
                    ax2.set_title(f'Signal and spike times {i+1}')
                
                ax3.cla()
                ax3.imshow(img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
                imgtmp2 = imgtmp.copy()
                imgtmp2[imgtmp2 == 0] = np.nan
                ax3.imshow(imgtmp2, interpolation='None',
                           alpha=0.5, cmap=plt.cm.hot)
                ax3.axis('off')
                
        s_comp.on_changed(update)
        s_comp.set_val(0)
        fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
        plt.show()
        
 
