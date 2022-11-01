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
from queue import Queue
import scipy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from scipy.optimize import nnls  
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True) # limit gpu memory
import timeit
import time
#tf.get_logger().setLevel("ERROR")
#tf.autograph.set_verbosity(1)

from fiola.gpu_mc_nnls import get_mc_model, get_nnls_model, get_model, Pipeline, Pipeline_mc_nnls
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from fiola.utilities import signal_filter, to_3D, to_2D, bin_median, hals, normalize, nmf_sequential, local_correlations, quick_annotation, HALS4activity

class FIOLA(object):
    def __init__(self, fnames=None, fr=400, mode='voltage', init_method='binary_masks', num_frames_init=10000, num_frames_total=20000, 
                 ms=[10,10], offline_batch_size=200, border_to_0=0, freq_detrend = 1/3, do_plot_init=False, erosion=0, 
                 hals_movie='hp_thresh', use_rank_one_nmf=False, semi_nmf=False,
                 update_bg=True, use_spikes=False, batch_size=1, use_fft=True, normalize_cc=True,
                 center_dims=None, num_layers=10, n_split=1, initialize_with_gpu=True, 
                 window = 10000, step = 5000, flip=True, detrend=True, dc_param=0.995, do_deconvolve=True, 
                 do_scale=False, template_window=2, robust_std=False, freq=15, adaptive_threshold=True, 
                 minimal_thresh=3.0, online_filter_method = 'median_filter',
                 filt_window = 15, nb=1, lag=5, do_plot=False, params={}):
        # please check fiolaparams.py for detailed documentation of parameters in class FIOLA
        if params is None:
            logging.warning("parameters are not set from fiolaparams")
            raise Exception('parameters are not set')
        else:
            self.params = params
        
    def create_pipeline(self, mov, trace_init, template, Ab, min_mov):
          logging.info('extracting spikes for initialization')
          logging.info(trace_init.shape)
          saoz = self.fit_spike_extraction(trace_init)
          self.Ab = Ab
          logging.info('compiling new models for online analysis')
          self.pipeline = Pipeline(self.params.data['mode'], mov, template, self.params.mc_nnls['batch_size'], Ab, saoz, 
                                   ms_h=self.params.mc_nnls['ms'][0], ms_w=self.params.mc_nnls['ms'][1], min_mov=mov.min(),
                                   use_fft=self.params.mc_nnls['use_fft'], normalize_cc=self.params.mc_nnls['normalize_cc'], 
                                   center_dims=self.params.mc_nnls['center_dims'], return_shifts=False, 
                                   num_layers=self.params.mc_nnls['num_layers'], n_split=self.params.mc_nnls['n_split'], 
                                   trace_with_neg=self.params.mc_nnls['trace_with_neg'])
          
         
          return self
     
    def create_pipeline_test(self, mov, trace_init, template, Ab, min_mov=0):
        self.pipeline = Pipeline_mc_nnls(mov, template, 1, Ab)
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
        self.estimates.reconstruct_signal()
            
        try:
            del self.estimates.update_thread
            del self.estimates.update_q
        except:
            logging.warning('no queue or thread to delete')
        return self
        
    def fit_hals(self, mov, A_sparse):
        """
        optimize binary masks to be weighted masks using HALS algorithm

        Parameters
        ----------
        mov : ndarray
            input movie
        A : sparse matrix
            masks for each neuron
        """
        if self.params.spike['flip'] == True:
            logging.info('flipping movie for initialization')
            y = to_2D(-mov).copy() 
        else:
            logging.info('movie flip not requested')
            y = to_2D(mov).copy() 
            
        
        hals_orig = False        
        if self.params.hals['hals_movie']=='orig':
            y_input = -y.T
            hals_orig = True
        else: 
            y_filt = signal_filter(y.T,freq=self.params.hals['freq_detrend'], 
                                   fr=self.params.data['fr']).T        

            if self.params.hals['hals_movie']=='hp_thresh':
                y_input = np.maximum(y_filt, 0).T
            elif self.params.hals['hals_movie']=='hp':
                y_input = y_filt.T
   
        if self.params.hals['use_rank_one_nmf']:
            y_seq = y_filt.copy()
            # standard deviation of average signal in each mask
            std = [np.std(y_filt[:, np.where(A_sparse[:,i].toarray()>0)[0]].mean(1)) for i in A_sparse.shape[-1]]
            seq = np.argsort(std)[::-1]
            self.seq = seq                   
            logging.info(f'sequence of rank1-nmf: {seq}')    
            mask = np.hstack((A_sparse.toarray(), b)).reshape([mov.shape[1], mov.shape[2], -1], order='F').transpose([2, 0, 1])
            W, H = nmf_sequential(y_seq, mask=mask, seq=seq, small_mask=True)
            nA = np.linalg.norm(H)
            H = H/nA
            W = W*nA
        else:
            logging.info('regressing matrices')
            nA = scipy.sparse.linalg.norm(A_sparse)
            H = (A_sparse/nA).T
            W = (H.dot(y_input)).T
            self.seq = np.array(range(A_sparse.shape[-1]))

        
        logging.info('computing hals')
        output_nb = np.maximum(self.params.hals['nb'],1)
        A,C,b,f = hals(y_input, H.T, W.T, np.ones((y_filt.shape[1],output_nb ))/y_filt.shape[1],
                         np.random.rand(output_nb ,mov.shape[0]), bSiz=None, maxIter=3, semi_nmf=self.params.hals['semi_nmf'],
                         update_bg=self.params.hals['update_bg'], use_spikes=self.params.hals['use_spikes'],
                         hals_orig=hals_orig, fr=self.params.data['fr'])
        
        if self.params.hals['update_bg']:
            Ab = np.hstack((A, b))
        else:
            Ab = A.copy()                    
        
        Ab = Ab / norm(Ab, axis=0)
        self.Ab = Ab 

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
            # if len(mov) % batch_size != 0 :
            #     raise ValueError('batch_size needs to be a factor of frames of the movie')
            start = time.time()
            rnge = len(mov)//batch_size
            logging.info(mov[None, 0*batch_size:(0+1)*batch_size,...,None].shape)
            for idx in range(rnge):
                yield{"m":mov[None, idx*batch_size:(idx+1)*batch_size,...,None]}
                     
        def get_frs():
            dataset = tf.data.Dataset.from_generator(generator, output_types={'m':tf.float32}, 
                                                      output_shapes={"m":(1, batch_size, dims[0], dims[1], 1)})
            return dataset

        logging.info("Correct unction")
        times = [0]*(len(mov) // batch_size)
        out = [0]*(len(mov) // batch_size)
        flag = 1000 # logging every 1000 frames
        index = 0
        dims = mov.shape[1:]
        mc_model = get_mc_model(template, batch_size, ms_h=self.params.mc_nnls['ms'][0], ms_w=self.params.mc_nnls['ms'][1], min_mov=min_mov,
                                use_fft=self.params.mc_nnls['use_fft'], normalize_cc=self.params.mc_nnls['normalize_cc'], 
                                center_dims=self.params.mc_nnls['center_dims'], return_shifts=True)
        
        mc_model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(mc_model)
        # return estimator
        logging.info('beginning motion correction')
        start = timeit.default_timer()
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            out[index] = i
            times[index] = (timeit.default_timer()-start)
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
    
    def fit_gpu_nnls_test(self, mov, Ab, batch_size=1):
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
        times2 = []
        def generator():
            # if len(mov) % batch_size != 0 :
            #     raise ValueError('batch_size needs to be a factor of frames of the movie')
            for fr in mov:
                # star = timeit.default_timer()
                out = nnls_q.get()
                # out = out.astype(np.float32)
                # times2.append(timeit.default_timer() - star)
                yield {"m":fr[None,None,:,:,None], 
                        "y":out[0][None], "x":out[1][None], "k":[[0.0]]}
                
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
        
        times0 = [0]*(len(mov) // batch_size)
        times1 = [0]*(len(mov) // batch_size)
        trace = [0]*(len(mov) // batch_size)
        flag = 1000
        index = 0
        dims = mov.shape[1:]
        nnls_q = Queue()
        
        b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')       
        C_init = np.dot(Ab.T, b)
        x0 = np.array([HALS4activity(Yr=b[:,i], A=Ab, C=C_init[:, i].copy(), iters=10) for i in range(batch_size)]).T
        x0 = x0.astype(np.float32)
        x, y = np.array(x0[None]), np.array(x0[None])
        nnls_q.put(np.concatenate((x,y)))
        num_components = Ab.shape[-1]
        Ab = Ab.astype(np.float32)

        nnls_model = get_nnls_model(dims, Ab, batch_size, self.params.mc_nnls['num_layers'], 
                                    self.params.mc_nnls['n_split'], self.params.mc_nnls['trace_with_neg'])
        nnls_model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(nnls_model)
        
        logging.info('beginning source extraction')
        start = timeit.default_timer()

        count=0
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            # values = list(i.values())
            nnls_q.put(np.concatenate((i['nnls'],i['nnls_1'])))
            # trace[count] = values[-1][0]
            trace[count]=i['nnls']
            times1[count] = -start+timeit.default_timer()
            # if index * batch_size >= flag:
            #     logging.info(f'processed {flag} frames')
            #     flag += 1000 
            count +=  1

        logging.info(i.keys())
        
        logging.info('source extraction complete')
        # logging.info(f'total timing: {times[-1]}')
        # logging.info(f'average timing per frame: {times[-1] / len(mov)}')
        # 
        return trace, times1

   
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
                out = nnls_q.get()
                (y, x) = out
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
        
        times = [0]*(len(mov) // batch_size)
        out = [0]*(len(mov) // batch_size)
        trace = [0]*(len(mov) // batch_size)
        flag = 1000
        index = 0
        dims = mov.shape[1:]
        nnls_q = Queue()
        
        b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')       
        C_init = np.dot(Ab.T, b)
        x0 = np.array([HALS4activity(Yr=b[:,i], A=Ab, C=C_init[:, i].copy(), iters=10) for i in range(batch_size)]).T
        x, y = np.array(x0[None,:]), np.array(x0[None,:]) 
        nnls_q.put((y, x))
        num_components = Ab.shape[-1]
        nnls_model = get_nnls_model(dims, Ab, batch_size, self.params.mc_nnls['num_layers'], 
                                    self.params.mc_nnls['n_split'], self.params.mc_nnls['trace_with_neg'])
        nnls_model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(nnls_model)
        
        logging.info('beginning source extraction')
        start = timeit.default_timer()
        count=0
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            values = list(i.values())
            nnls_q.put((values[0], values[1]))
            trace[count] = values[-1][0]
            times[count] = timeit.default_timer()-start
            index += 1    
            if index * batch_size >= flag:
                logging.info(f'processed {flag} frames')
                flag += 1000 
            count +=  1
        trace = np.hstack(trace)
        
        logging.info('source extraction complete')
        logging.info(f'total timing: {times[-1]}')
        logging.info(f'average timing per frame: {times[-1] / len(mov)}')
        
        return trace, times

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
                out = nnls_q.get()
                (y, x) = out
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
        nnls_q = Queue()
        trace = []
        
        
        b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')    
        C_init = np.dot(Ab.T, b)
        x0 = np.array([HALS4activity(Yr=b[:,i], A=Ab, C=C_init[:, i].copy(), iters=10) for i in range(batch_size)]).T
        x, y = np.array(x0[None,:]), np.array(x0[None,:]) 
        nnls_q.put((y, x))
        num_components = Ab.shape[-1]
        
        model = get_model(template, Ab, batch_size, 
                          ms_h=self.params.mc_nnls['ms'][0], ms_w=self.params.mc_nnls['ms'][1], min_mov=min_mov,
                          use_fft=self.params.mc_nnls['use_fft'], normalize_cc=self.params.mc_nnls['normalize_cc'], 
                          center_dims=self.params.mc_nnls['center_dims'], return_shifts=False, 
                          num_layers=self.params.mc_nnls['num_layers'], n_split=self.params.mc_nnls['n_split'], 
                          trace_with_neg=self.params.mc_nnls['trace_with_neg'])

        model.compile(optimizer='rmsprop', loss='mse')   
        estimator = tf.keras.estimator.model_to_estimator(model)
        
        logging.info('beginning motion correction and source extraction')
        start = timeit.default_timer()
        for i in estimator.predict(input_fn=get_frs, yield_single_examples=False):
            values = list(i.values())
            nnls_q.put((values[0], values[1]))
            trace.append(values[-1][0])
            times.append(timeit.default_timer()-start)
            index += 1    
            if index * batch_size >= flag:
                logging.info(f'processed {flag} frames')
                flag += 1000        
        trace = np.hstack(trace)
        
        logging.info('motion correction and source extraction complete')
        logging.info(f'total timing: {times[-1]}')
        logging.info(f'average timing per frame: {times[-1] / len(mov)}')
        
        return trace,  times
    
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
                                     flip=self.params.spike['flip'], detrend=self.params.spike['detrend'], dc_param=self.params.spike['dc_param'], do_deconvolve=self.params.spike['do_deconvolve'],                         
                                     do_scale=self.params.spike['do_scale'], template_window=self.params.spike['template_window'], 
                                     robust_std=self.params.spike['robust_std'], adaptive_threshold = self.params.spike['adaptive_threshold'],
                                     fr=self.params.data['fr'], freq=self.params.spike['freq'],
                                     minimal_thresh=self.params.spike['minimal_thresh'], online_filter_method = self.params.spike['online_filter_method'],                                        
                                     filt_window=self.params.spike['filt_window'], do_plot=self.params.spike['do_plot'],
                                     p=self.params.spike['p'], nb=self.params.hals['nb'])
        saoz.fit(trace, num_frames=self.params.data['num_frames_total'])    
        times.append(timeit.default_timer()-start)
        logging.info('spike extraction complete')
        logging.info(f'total timing: {times[-1]}')
        logging.info(f'average timing per neuron: {times[-1] / len(trace)}')
        logging.info(f'times: {np.mean(times)}')
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

        n = len(idx) 
        fig = plt.figure(figsize=(30, 10))
        
        dims = img.shape
        
        spatial = self.estimates.Ab.T.reshape([-1, dims[0], dims[1]], order='F')
    
        axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])
        ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
        ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
        ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])    
        #ax2 = plt.axes([0.05, 0.3, 0.9, 0.2])    
        #ax4 = plt.axes([0.05, 0.05, 0.9, 0.2])    
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
                    
                    try: 
                        temp = self.estimates.trace.shape[0] - self.estimates.trace_deconvolved.shape[0]
                        if temp > 0:
                            ax2.plot(np.vstack((self.estimates.trace_deconvolved, 
                                                np.zeros((temp, self.estimates.trace.shape[1]))))[idx][i],
                                                alpha=0.8, label='deconv traces')
                        else:
                            ax2.plot(self.estimates.trace_deconvolved[idx][i], alpha=0.8, label='deconv traces')
                    except:               
                        pass
                    
                    if cnm_estimates is not None:
                        ax2.plot(np.vstack((cnm_estimates.C, cnm_estimates.f))[idx][i], label='caiman result')                        
                        #ax2.plot((cnm_estimates.C+cnm_estimates.YrA)[idx][i], label='caiman result')                        
                    ax2.legend()
                    ax2.set_title(f'Signal {i+1}')
                else:
                    ix = idx[i]
                    if idx[i] < self.Ab.shape[1] - self.params.hals['nb']:
                        ax2.plot(normalize(self.estimates.t_s[ix]))            
                        spikes = np.delete(self.estimates.index[ix], self.estimates.index[ix]==0)
                        h_min = normalize(self.estimates.t_s[ix]).max()
                        ax2.vlines(spikes, h_min, h_min + 1, color='black')
                        ax2.legend(labels=['trace', 'spikes'])
                        ax2.set_title(f'Signal and spike times {i+1}')
                    else:
                        ax2.plot(normalize(self.estimates.trace[ix]))            
                        ax2.legend(labels=['background trace'])
                ax2.set_xlim([0, self.params.data['num_frames_total']])
                        
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
    
 
