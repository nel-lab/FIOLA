#!/usr/bin/env python
"""
Illustration of the usage of FIOLA with calcium and voltage imaging data. 
For Calcium USE THE demo_initialize_calcium.py FILE TO GENERATE THE HDF5 files necessary for 
initialize FIOLA. 
For voltage this demo is self contained.   
copyright in license file
authors: @agiovann @changjia
"""
#%%
import caiman as cm
import logging
import matplotlib.pyplot as plt
import numpy as np
import pyximport
pyximport.install()
import scipy
from tensorflow.python.client import device_lib
from time import time
    
from fiola.demo_initialize_calcium import run_caiman_init
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, to_2D, movie_iterator

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0

#%% 
def main():
#%%
    mode = 'calcium'                    # 'voltage' or 'calcium 'fluorescence indicator
    # Parameter setting
    if mode == 'voltage':
        folder = '/home/nel/caiman_data/example_movies/volpy'
        fnames = download_demo(folder, 'demo_voltage_imaging.hdf5')
        
        # setting params
        # dataset dependent parameters
        fr = 400                        # sample rate of the movie
        
        num_frames_init =  10000        # number of frames used for initialization
        num_frames_total =  20000       # estimated total number of frames for processing, this is used for generating matrix to store data
        offline_batch_size = 200        # number of frames for one batch to perform offline motion correction
        batch_size = 1                  # number of frames processing at the same time using gpu 
        flip = True                     # whether to flip signal to find spikes   
        detrend = True                  # whether to remove the slow trend in the fluorescence data            
        do_deconvolve = True            # If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
        ms = [10, 10]                   # maximum shift in x and y axis respectively. Will not perform motion correction if None.
        update_bg = True                # update background components for spatial footprints
        filt_window = 15                # window size of median filter for removing the subthreshold activities. It can be integer or a list.
                                        # an integer means the window size of the full median filter. Suggested values range [9, 15]. It needs to be an odd number. 
                                        # a list with two values [x, y] means an antisymmetric median filter which uses x past frames and y future frames.
        minimal_thresh = 3.5            # minimal of the threshold for voltage spike detection. Suggested value range [2.8, 3.5] 
        template_window = 2             # half window size of the template; will not perform template matching if window size equals 0.  
        nb = 1                          # number of background components
        lag = 11                        # lag for retrieving the online result. 5 frames are suggested for calcium imaging. For voltage imaging, it needs to be larger than filt_window // 2 + template_window + 2. 
    
        options = {
            'fnames': fnames,
            'fr': fr,
            'mode': mode,
            'num_frames_init': num_frames_init, 
            'num_frames_total':num_frames_total,
            'offline_batch_size': offline_batch_size,
            'batch_size':batch_size,
            'flip': flip,
            'detrend': detrend,
            'do_deconvolve': do_deconvolve,
            'ms': ms,
            'update_bg': update_bg,
            'filt_window': filt_window,
            'minimal_thresh': minimal_thresh,
            'template_window':template_window, 
            'nb': nb, 
            'lag': lag}
        
        
        logging.info('Loading Movie')
        
        mov = cm.load(fnames, subindices=range(num_frames_init))
        fnames_init = fnames.split('.')[0] + '_init.tif'
        mov.save(fnames_init)
        path_ROIs = download_demo(folder, 'demo_voltage_imaging_ROIs.hdf5')
        mask = load(path_ROIs)        
        template = np.median(mov, 0)
       
    #    
    elif mode == 'calcium':
        fnames = '/home/nel/caiman_data/example_movies/demoMovie/demoMovie.tif'
        #fnames = '/home/nel/caiman_data/example_movies/Sue_2x_3000_40_-46.tif'
        fr = 30                         # sample rate of the movie
        
        mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
        num_frames_init =   1000         # number of frames used for initialization
        num_frames_total =  2000        # estimated total number of frames for processing, this is used for generating matrix to store data
        offline_batch_size = 5          # number of frames for one batch to perform offline motion correction
        batch_size= 1                   # number of frames processing at the same time using gpu 
        flip = False                    # whether to flip signal to find spikes   
        detrend = True                  # whether to remove the slow trend in the fluorescence data            
        dc_param = 0.9995               # DC blocker parameter for removing the slow trend in the fluorescence data. It is usually between
                                        # 0.99 and 1. Higher value will remove less trend. No detrending will perform if detrend=False.
        do_deconvolve = True            # If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
        ms = [5, 5]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
        center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
        hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
                                        # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
        n_split = 1                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
        nb = 1                          # number of background components
        trace_with_neg=False            # return trace with negative components (noise) if True; otherwise the trace is cutoff at 0
        lag = 5                         # lag for retrieving the online result.
                        
        options = {
            'fnames': fnames,
            'fr': fr,
            'mode': mode, 
            'num_frames_init': num_frames_init, 
            'num_frames_total':num_frames_total,
            'offline_batch_size': offline_batch_size,
            'batch_size':batch_size,
            'flip': flip,
            'detrend': detrend,
            'dc_param': dc_param,            
            'do_deconvolve': do_deconvolve,
            'ms': ms,
            'hals_movie': hals_movie,
            'center_dims':center_dims, 
            'n_split': n_split,
            'nb' : nb, 
            'trace_with_neg':trace_with_neg, 
            'lag': lag}
        
        mov = cm.load(fnames, subindices=range(num_frames_init))
        fnames_init = fnames.split('.')[0] + '_init.tif'
        mov.save(fnames_init)
        
        # run caiman initialization. User might need to change the parameters 
        # inside the file to get good initialization result
        caiman_file = run_caiman_init(fnames_init, pw_rigid=True, 
                                      max_shifts=ms, gnb=nb, K=5, gSig=[4, 4])
        
        # load results of initialization
        cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
        estimates = cnm2.estimates
        template = cnm2.estimates.template
        Cn = cnm2.estimates.Cn
        
    else: 
        raise Exception('mode must be either calcium or voltage')
          
    #%% Run FIOLA
    #example motion correction
    motion_correct = True
    #example source separation
    do_nnls = True
    #%% Mot corr only
    if motion_correct:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        # run motion correction on GPU on the initialization movie
        mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], min_mov=mov.min())             
        plt.plot(shifts_fiola)
        plt.xlabel('frames')
        plt.ylabel('pixels')                 
        plt.legend(['x shifts', 'y shifts'])
    else:    
        mc_nn_mov = mov
    
    #%% NNLS only
    if do_nnls:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        if mode == 'voltage':
            A = scipy.sparse.coo_matrix(to_2D(mask, order='F')).T
            fio.fit_hals(mc_nn_mov, A)
            Ab = fio.Ab # Ab includes spatial masks of all neurons and background
        else:
            Ab = np.hstack((estimates.A.toarray(), estimates.b))
        trace_fiola, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch_size']) 
        plt.plot(trace_fiola[:-nb].T)
        plt.xlabel('frames')
        plt.ylabel('fluorescence signal')              

    else:        
        if trace_with_neg == True:
            trace_fiola = np.vstack((estimates.C+estimates.YrA, estimates.f))
        else:
            trace_fiola = estimates.C+estimates.YrA
            trace_fiola[trace_fiola < 0] = 0
            trace_fiola = np.vstack((trace_fiola, estimates.f))
        
    #%% set up online pipeline
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    if mode == 'voltage': # not thoroughly tested and computationally intensive for large files, it will estimate the baseline
        fio.fit_hals(mc_nn_mov, A)
        Ab = fio.Ab
    else:
        Ab = np.hstack((estimates.A.toarray(), estimates.b))
    Ab = Ab.astype(np.float32)        
    fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())
    #%% run online
    time_per_step = np.zeros(num_frames_total-num_frames_init)
    online_trace = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
    online_trace_deconvolved = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1] - fio.params.hals['nb']), dtype=np.float32)
    start = time()
        
    for idx, memmap_image in movie_iterator(fnames, num_frames_init, num_frames_total):
        if idx % 1000 == 0:
                print(f'processed {idx} frames')        
        fio.fit_online_frame(memmap_image) 
        online_trace[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx-1]
        online_trace_deconvolved[idx-num_frames_init] = fio.pipeline.saoz.trace_deconvolved[:,idx-1-fio.params.retrieve['lag']]
        time_per_step[idx-num_frames_init] = (time()-start)
    
    fio.pipeline.saoz.online_trace = online_trace.T
    fio.pipeline.saoz.online_trace_deconvolved = online_trace_deconvolved.T
    logging.info(f'total time online: {time()-start}')
    logging.info(f'time per frame online: {(time()-start)/(num_frames_total-num_frames_init)}')
    plt.plot(np.diff(time_per_step),'.')
    #%% visualize result
    fio.compute_estimates()
    if 'mask' in locals():
        fio.estimates.binary_masks = mask 
    fio.view_components(template)
    
    #%% save result
    if True:
        np.save(folder + f'/fiola_result', fio.estimates)
        
#%%
if __name__ == "__main__":
    main()
    
