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

from fiola.demo_initialize_calcium import run_caiman_init
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from caiman.source_extraction.cnmf.utilities import get_file_size
import caiman as cm
from fiola.utilities import download_demo, load, play, bin_median, to_2D, local_correlations, movie_iterator, compute_residuals
from caiman.source_extraction.cnmf.utilities import get_file_size

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
parser.add_argument('--include_bg', default=False, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()

#%% 
def main(iteration=1, init_frames=None, num_layers=None, trace_with_neg=None, include_bg=None):    
        print(f'{iteration} !!!!!!!')    
        print(f'{init_frames} !!!!')
        print(f'{num_layers} !!!!')
        print(f'{trace_with_neg} !!!!')
        print(f'{include_bg} !!!!')
	#for iteration in range(1, 6):
        #init_frames = 3000
        #for init_frames in [1000, 1500, 3000]:
        #    for num_layers in [10]:
        timing = {}
        timing['start'] = time()
        mode = 'calcium'                    # 'voltage' or 'calcium 'fluorescence indicator
        # Parameter setting
        if mode == 'voltage':
            folder = '/home/nel/caiman_data/example_movies/volpy'
            fnames = download_demo(folder, 'demo_voltage_imaging.hdf5')
            path_ROIs = download_demo(folder, 'demo_voltage_imaging_ROIs.hdf5')
            mask = load(path_ROIs)
            #num_frames_total = get_file_size(fnames)[-1]         # number of total frames including initialization
        
            # setting params
            # dataset dependent parameters
            fr = 400                        # sample rate of the movie
            ROIs = mask                     # a 3D matrix contains all region of interests
        
            num_frames_init =  10000        # number of frames used for initialization
            num_frames_total =  20000       # estimated total number of frames for processing, this is used for generating matrix to store data
            offline_batch_size = 200        # number of frames for one batch to perform offline motion correction
            batch_size = 1                  # number of frames processing at the same time using gpu 
            flip = True                     # whether to flip signal to find spikes   
            ms = [10, 10]                   # maximum shift in x and y axis respectively. Will not perform motion correction if None.
            update_bg = True                # update background components for spatial footprints
            filt_window = 15                # window size for removing the subthreshold activities 
            minimal_thresh = 3.5            # minimal of the threshold 
            template_window = 2             # half window size of the template; will not perform template matching if window size equals 0
        
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
                'ms': ms,
                'update_bg': update_bg,
                'filt_window': filt_window,
                'minimal_thresh': minimal_thresh,
                'template_window':template_window}
            
            
            logging.info('Loading Movie')
            mov = cm.load(fnames, subindices=range(num_frames_init))
            template = np.median(mov, 0)
           
        #    
        elif mode == 'calcium':
            #fnames = '/home/nel/caiman_data/example_movies/demoMovie/demoMovie.tif'
            fnames = '/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000.hdf5'
            #fnames = '/media/nel/storage/fiola/R2_20190219/test/fiola_deconvolution/mov_3000.hdf5'
            #fnames = '/media/nel/storage/fiola/R6_20200210T2100/mov_raw.hdf5'
            fr = 15                         # sample rate of the movie
            ROIs = None                     # a 3D matrix contains all region of interests
        
            mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
            num_frames_init =  init_frames         # number of frames used for initialization
            num_frames_total = get_file_size(fnames)[-1] # estimated total number of frames for processing, this is used for generating matrix to store data
            offline_batch_size = 5          # number of frames for one batch to perform offline motion correction
            batch_size= 1                   # number of frames processing at the same time using gpu 
            flip = False                    # whether to flip signal to find spikes   
            ms = [5, 5]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
            center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
            hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
                                            # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
            n_split = 2                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
            if include_bg:
                nb = 2                          # number of background components
            else:
                nb = 0
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
                'ms': ms,
                'hals_movie': hals_movie,
                'center_dims':center_dims, 
                'n_split': n_split,
                'nb' : nb, 
                'trace_with_neg':trace_with_neg, 
                'num_layers': num_layers}
            
            folder = fnames.rsplit('/', 1)[0] + f'/{num_frames_init}/'
            #folder = '/media/nel/storage/fiola/R2_20190219/test/fiola_deconvolution/'
            print(folder)
            
            #mov = cm.load(fnames, subindices=range(num_frames_init))
            fnames_init = folder + fnames.rsplit('/', 1)[1][:-5] + f'_init_{num_frames_init}.tif'
            mov = cm.load(fnames_init)
            
            """
            mov.save(fnames_init)            
            # run caiman initialization. User might need to change the parameters 
            # inside the file to get good initialization result
            caiman_file = run_caiman_init(fnames_init)
            """
            
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
        motion_correct = False
        #example source separation
        do_nnls = False
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
                A = scipy.sparse.coo_matrix(to_2D(mask, order='F')).T
                fio.fit_hals(mc_nn_mov, A)
                Ab = fio.Ab # Ab includes spatial masks of all neurons and background
            else:
                Ab = np.hstack((estimates.A.toarray(), estimates.b))
                
            trace_fiola = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch_size']) 
            plt.plot(trace_fiola.T)
        else:
            if include_bg:
                trace_fiola = np.vstack((estimates.C, estimates.f))
            else:
                trace_fiola = estimates.C
            
        #%% Set up online pipeline
        print('start fiola online')
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        if mode == 'voltage': # not thoroughly tested and computationally intensive for large files, it will estimate the baseline
            fio.fit_hals(mc_nn_mov, A)
            Ab = fio.Ab
        else:
            if include_bg:
                Ab = np.hstack((estimates.A.toarray(), estimates.b))
            else:
                Ab = estimates.A.toarray()
        Ab = Ab.astype(np.float32)
        fio = fio.create_pipeline(mc_nn_mov, trace_fiola.astype(np.float32), template, Ab, min_mov=mov.min())
	
        
        #timing['init'] = time()
        
        #%% run online
        time_per_step = np.zeros(num_frames_total-num_frames_init)
        # traces = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
        # traces_deconvolved = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1] - 2), dtype=np.float32)
        # lag=5
        start = time()
            
        for idx, memmap_image in movie_iterator(fnames, num_frames_init, num_frames_total):
            if idx % 100 == 0:
                print(idx)    
            fio.fit_online_frame(memmap_image)   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i   
            # traces[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx-1]
            # traces_deconvolved[idx-num_frames_init-lag] = fio.pipeline.saoz.trace_deconvolved[:,idx-lag-1]
            time_per_step[idx-num_frames_init] = (time()-start)
        
        #traces = traces.T
        logging.info(f'total time online: {time()-start}')
        logging.info(f'time per frame online: {(time()-start)/(num_frames_total-num_frames_init)}')
        plt.plot(np.diff(time_per_step),'.')
        
        #%% run online
        # time_per_step = np.zeros(num_frames_total-num_frames_init)
        # traces = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
        # traces_deconvolved = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]-nb), dtype=np.float32)
        # start = time()
        
        # lag = 5
        # s_min = 0
            
        # for idx, memmap_image in movie_iterator(fnames, num_frames_init, num_frames_total):
        #     if idx % 100 == 0:
        #         print(idx)  
        #         # plt.figure()
        #         # plt.plot(fio.pipeline.saoz.t[0])
        #         # plt.show()
        #     fio.fit_online_frame(memmap_image)   
        #     sleep(0.0001)
        #     traces[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx] # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i        
        #     dec = np.zeros((fio.Ab.shape[-1]-nb))
        #     for n in range(len(dec)):
        #         tmp = np.where(fio.pipeline.saoz.t[n, :fio.pipeline.saoz._i[n]] == idx-lag)[0]
        #         if len(tmp):
        #             if (foo:=fio.pipeline.saoz.h[n, tmp[0]]) > s_min:
        #                 dec[n] = foo
        #     traces_deconvolved[idx-num_frames_init-lag] = dec
        #     time_per_step[idx-num_frames_init] = (time()-start)
        #     #plt.figure(); plt.plot(fio.pipeline.saoz.h[0])a
        
        # traces = traces.T
        # logging.info(f'total time online: {time()-start}')
        # logging.info(f'time per frame online: {(time()-start)/(num_frames_total-num_frames_init)}')
        # plt.plot(np.diff(time_per_step),'.')
            
        #%% Visualize result
        timing['online'] = time()
        timing['all_online'] = time_per_step
        fio.compute_estimates()
        fio.view_components(template)  
        fio.estimates.timing = timing
        plt.close('all')
        
        #%% save some interesting data
        if True:
            # np.savez(fnames[:-4]+'_fiola_result_new.npz', time_per_step=time_per_step, traces=np.hstack([trace_fiola, traces]), 
            #      caiman_file = caiman_file, 
            #      fnames_exp = fnames, 
            #      estimates = fio.estimates)
            
            #np.save(folder +f'fiola_result_{num_layers}_{trace_with_neg}.npy', np.hstack([trace_fiola, traces]))
            np.save(folder + f'fiola_result_init_frames_{init_frames}_iteration_{iteration}_num_layers_{num_layers}_trace_with_neg_{trace_with_neg}_include_bg_{include_bg}_no_init_v3.10', fio.estimates)
            #np.save(folder +f'fiola_timing_v3.6_num_layers_{num_layers}_trace_with_neg_{trace_with_neg}_with_dec_{iteration}_init_frames_{init_frames}_iteration_{iteration}_no_init', timing)
    
if __name__ == "__main__":
    main(iteration=args.iteration, init_frames=args.init_frames, 
         num_layers=args.num_layers, trace_with_neg=args.trace_with_neg, 
         include_bg=args.include_bg)
