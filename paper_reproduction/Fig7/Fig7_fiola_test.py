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
import logging
import matplotlib.pyplot as plt
import numpy as np
import pyximport
pyximport.install()
from tensorflow.python.client import device_lib
from time import time, sleep
import scipy
import sys
sys.path.append('/home/nel/CODE/VIOLA')

import caiman as cm
from caiman.source_extraction.cnmf.utilities import get_file_size
from fiola.demo_initialize_calcium import run_caiman_init
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, play, bin_median, to_2D, local_correlations, movie_iterator, compute_residuals

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)
    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--iteration', type=int, required=True)
parser.add_argument('--init_frames', type=int, required=True)
parser.add_argument('--num_layers', type=int, required=True)
parser.add_argument('--trace_with_neg', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--center_dims', nargs='+', type=int, required=True)
#parser.add_argument('--include_bg', default=False, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()

#%% 
def main(iteration=1, init_frames=None, num_layers=None, trace_with_neg=None, center_dims=None):    
    #center_dims = tuple(args.center_dims)
#%%
    center_dims = [0, 0]
    if center_dims[0] == 0:
        center_dims = None
    
    # print input parameters
    # for name, value in locals().items():
    #     if name != 'self':
    #         logging.info(f'{name}, {value}')
    
    init_frames = 3000
    num_layers=10
    trace_with_neg=False
    center_dims = None
    
    timing = {}
    timing['start'] = time()
    mode = 'calcium'                    # 'voltage' or 'calcium 'fluorescence indicator
    # Parameter setting
    if mode == 'voltage':
        pass        
    elif mode == 'calcium':
        # fnames = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/s2p_k53/k53.tif'
        # mask = "/home/nel/caiman_data/example_movies/k53_A.npy"
        #fnames = '/media/nel/storage/fiola/R2_20190219/test/test_full/mov_scale_full.hdf5'
        fnames = '/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000.hdf5'
        fr = 15                         # sample rate of the movie
        ROIs = None                     # a 3D matrix contains all region of interests
    
        mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
        num_frames_init =  init_frames         # number of frames used for initialization
        num_frames_total = 6000#get_file_size(fnames)[-1] # estimated total number of frames for processing, this is used for generating matrix to store data
        offline_batch_size = 1          # number of frames for one batch to perform offline motion correction
        batch_size= 1                   # number of frames processing at the same time using gpu 
        flip = False                    # whether to flip signal to find spikes   
        ms = [5, 5]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
        #center_dims = None              # template dimensions for motion correction. If None, the input will be the shape of the FOV
        hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
                                        # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
        n_split = 2                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
        nb = 2                          # number of background components
        detrend = True                  # whether to remove the slow trend in the fluorescence data            
        dc_param = 0.995
        do_deconvolve = True            # If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
        
        #trace_with_neg=trace_with_neg #False     # return trace with negative components (noise) if True; otherwise the trace is cutoff at 0
        #num_layers = 10#num_layers
                        
        options = {
            'fnames': fnames,
            'fr': fr,
            'ROIs': ROIs,
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
            'num_layers': num_layers}
        print(options)
        folder = fnames.rsplit('/', 1)[0] + f'/{num_frames_init}/'
        #folder = '/media/nel/storage/fiola/R2_20190219/test/fiola_deconvolution/'
        logging.info(folder)

        #mov = cm.load(fnames, subindices=range(num_frames_init))
        fnames_init = folder + fnames.rsplit('/', 1)[1][:-5] + f'_init_{num_frames_init}.tif'
        #fnames_init = folder + fnames.rsplit('/', 1)[1][:-5] + f'_init_{num_frames_init}_els__d1_796_d2_512_d3_1_order_F_frames_{num_frames_init}_.mmap'
        
        mov = cm.load(fnames_init, in_memory=True)
        #fnames_init = '/media/nel/storage/fiola/R2_20190219/test/test_full/mov_scale_full.hdf5'
        #mov = cm.load(fnames_init, subindices=slice(0, 3000), in_memory=True)
        
        caiman_file = f'/media/nel/storage/fiola/R2_20190219/{init_frames}/memmap__d1_796_d2_512_d3_1_order_C_frames_{init_frames}__v3.7.hdf5'
        
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
    else:    
        mc_nn_mov = mov
    
    #%% NNLS only
    if do_nnls:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        if mode == 'voltage':
            pass
        else:
            Ab = np.hstack((estimates.A.toarray(), estimates.b))[:,list(range(500))+ [-2, -1]]
        trace_fiola_nnls, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab.astype(np.float32), batch_size=1) 
        #plt.plot(trace_fiola.T)
    else:
        if trace_with_neg == True:
            trace_fiola_nnls = np.vstack((estimates.C+estimates.YrA, estimates.f))
        else:
            trace_fiola_nnls = estimates.C+estimates.YrA
            trace_fiola_nnls[trace_fiola < 0] = 0
            trace_fiola_nnls = np.vstack((trace_fiola, estimates.f))
            
    
    #%%
    # params = fiolaparams(params_dict=options)
    # fio = FIOLA(params=params)
    # #mov_test = cm.load('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000.hdf5', subindices=slice(3000, 6000))
    # t3, _ = fio.fit_gpu_motion_correction_nnls(mov, template, batch_size=1, min_mov=mov.min(), Ab=Ab)
    # plt.plot(t3[22])
    # print(t3[110].max())
    
    #%%
    # mm = mov.reshape((3000, -1), order='F')
    # ttt = np.matmul(mm, Ab[:, 22])
    
    # t_c =(cnm2.estimates.C + cnm2.estimates.YrA)[22]
    
    # f1 = fio.pipeline.saoz.trace[22, :3000]
    # f2 = fio.pipeline.saoz.trace[22, 3000:]

    # np.corrcoef(f1, t_c)
    # np.corrcoef(f2, t_c)
    
    
    #%% Set up online pipeline
    logging.info('start fiola online')
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    if mode == 'voltage': # not thoroughly tested and computationally intensive for large files, it will estimate the baseline
        pass
    else:
        Ab = np.hstack((estimates.A.toarray(), estimates.b))[:,list(range(500))+ [-2, -1]]
        
    Ab = Ab.astype(np.float32)
    fio = fio.create_pipeline(mov, trace_fiola_nnls.astype(np.float32), template, Ab, min_mov=mov.min())
    
    # run online
    time_per_step = np.zeros(num_frames_total-num_frames_init)
    traces = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
    # traces_deconvolved = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1] - 2), dtype=np.float32)
    # traces_t = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
    # traces_deconvolved_t = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1] - 2), dtype=np.float32)
    
    lag=5
    #lags = [0, 1, 3, 5, 10, 20]
    #traces_deconvolved = {}
    # for lag in lags:
    #     traces_deconvolved[lag] = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1] - 2), dtype=np.float32)
    start = time()
        
    #fnames = '/media/nel/storage/fiola/R2_20190219/test/test_full/mov_scale_full.hdf5'
    for idx, memmap_image in movie_iterator(fnames, num_frames_init, num_frames_total):
        if idx % 1000 == 0:
            logging.info(f'{idx} frames processed online')    
        fio.fit_online_frame(memmap_image)   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i   
        #fio.fit_online_frame(mov[idx:idx+1])   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i   
        traces[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx-1]
        # traces_deconvolved[idx-num_frames_init] = fio.pipeline.saoz.trace_deconvolved[:,idx-lag-1]
        # traces_t[idx-num_frames_init] = fio.pipeline.saoz.trace_q.get().squeeze()
        # traces_deconvolved_t[idx-num_frames_init] = fio.pipeline.saoz.dec_trace_q.get().squeeze()
        
        
        
        # for lag in lags:
        #     traces_deconvolved[lag][idx-num_frames_init] = fio.pipeline.saoz.trace_deconvolved[:,idx-lag-1]
        time_per_step[idx-num_frames_init] = (time()-start)
    
    #traces = traces.T
    logging.info(f'total time online: {time()-start}')
    logging.info(f'time per frame online: {(time()-start)/(num_frames_total-num_frames_init)}')
    plt.plot(np.diff(time_per_step),'.')
    
    #%%
    plt.plot(traces[:, 22])
    plt.plot(fio.pipeline.saoz.trace[:,idx-1])
    
    plt.plot(traces_t[:, 22])
    plt.plot(traces_deconvolved[:, 22])
    plt.plot(traces_deconvolved_t[:, 22])
    

    #%%
    caiman_file1 = f'/media/nel/storage/fiola/R2_20190219/3000/memmap__d1_796_d2_512_d3_1_order_C_frames_3000__v3.7.hdf5'
    cc = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)

    #%% Set up online pipeline
    logging.info('start fiola online')
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    if mode == 'voltage': # not thoroughly tested and computationally intensive for large files, it will estimate the baseline
        pass
    else:
        Ab = np.hstack((estimates.A.toarray(), estimates.b))[:,list(range(500))+ [-2, -1]]
        
    Ab = Ab.astype(np.float32)
    fio = fio.create_pipeline(mov, trace_fiola_nnls.astype(np.float32), template, Ab, min_mov=mov.min())
    
    # run online
    time_per_step = np.zeros(num_frames_total-num_frames_init)
    traces = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
    #traces_deconvolved = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1] - 2), dtype=np.float32)
    #lag=5
    lags = [0, 1, 3, 5, 10, 20]
    traces_deconvolved = {}
    for lag in lags:
        traces_deconvolved[lag] = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1] - 2), dtype=np.float32)
    start = time()
        
    #fnames = '/media/nel/storage/fiola/R2_20190219/test/test_full/mov_scale_full.hdf5'
    for idx, memmap_image in movie_iterator(fnames, num_frames_init, num_frames_total):
        if idx % 1000 == 0:
            logging.info(f'{idx} frames processed online')    
        fio.fit_online_frame(memmap_image)   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i   
        #fio.fit_online_frame(mov[idx:idx+1])   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i   
        traces[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx-1]
        #traces_deconvolved[idx-num_frames_init] = fio.pipeline.saoz.trace_deconvolved[:,idx-lag-1]
        for lag in lags:
            traces_deconvolved[lag][idx-num_frames_init] = fio.pipeline.saoz.trace_deconvolved[:,idx-lag-1]
        time_per_step[idx-num_frames_init] = (time()-start)
    
    #traces = traces.T
    logging.info(f'total time online: {time()-start}')
    logging.info(f'time per frame online: {(time()-start)/(num_frames_total-num_frames_init)}')
    plt.plot(np.diff(time_per_step),'.')
    

    

        
    #%% Visualize result
    timing['online'] = time()
    timing['all_online'] = time_per_step
    fio.compute_estimates()
    fio.view_components(template)  
    fio.estimates.timing = timing
    plt.close('all')
    
    #%% save some interesting data
    if True:
        np.save(folder + f'fiola_result_init_frames_{init_frames}_iteration_{iteration}_num_layers_{num_layers}_trace_with_neg_{trace_with_neg}_center_dims_{center_dims}_nnls_True_mc_true_v3.121', fio.estimates)
        
#%%
if __name__ == "__main__":
    main(iteration=args.iteration, init_frames=args.init_frames, 
         num_layers=args.num_layers, trace_with_neg=args.trace_with_neg, center_dims=args.center_dims)