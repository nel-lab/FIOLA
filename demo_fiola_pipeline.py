#!/usr/bin/env python
"""
Pipeline for online analysis of fluorescence imaging data
Voltage dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
Calcium dataset courtesy of Sue Ann Koay and David Tank (Princeton University)
@author: @agiovann, @caichangjia, @cynthia
"""
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.python.client import device_lib
from threading import Thread
from time import time

from fiola.config import load_fiola_config, load_caiman_config
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, play, bin_median, to_2D, local_correlations

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#%%
def main():
    #%% load movie and masks
    folder = '/home/nel/caiman_data/example_movies/volpy'    
    file_id = 0   
    if file_id  == 0:
        fnames = download_demo(folder, 'demo_voltage_imaging.hdf5')
        path_ROIs = download_demo(folder, 'demo_voltage_imaging_ROIs.hdf5')
        mask = load(path_ROIs)
        mode = 'voltage'    
    elif file_id == 1:
        fnames = download_demo(folder, 'k53.tif')
        path_ROIs = download_demo(folder, 'k53_ROIs.hdf5')
        mask = load(path_ROIs)
        mode = 'calcium'
    elif file_id  == 2:
        fnames = '/home/nel/caiman_data/example_movies/demoMovie/demoMovie.tif'
        path_ROIs = None
        mask = None
        mode = 'calcium'
        
    
    mov = (load(fnames)).astype(np.float32)    
    num_frames_total =  mov.shape[0]    
        
    display_images = True
    if display_images:
        plt.figure()
        plt.imshow(mov.mean(0), vmax=np.percentile(mov.mean(0), 99.9))
        plt.title('Mean img')
        if mask is not None:
            plt.figure()
            plt.imshow(mask.mean(0))
            plt.title('Masks')
    
    #%% load configuration; set up FIOLA object
    # In the first part, we will first show each part (motion correct, source separation and spike extraction) of 
    # FIOLA separately in an offline manner. 
    # Then in the second part, we will show the full pipeline and its real time frame-by-frame analysis performance
    # Note one needs to installed CaImAn beforehand to perform CaImAn initialization

    
    #%% offline GPU motion correction example
    options = load_fiola_config(fnames, num_frames_total, mode, mask) 
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    fio.dims = mov.shape[1:]
    template = bin_median(mov) 
    mc_mov, shifts, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], mov.min())
    # template = bin_median(mc_mov)
    # mc_mov, shifts_2, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], mov.min())
    if display_images:
        plt.figure()
        plt.plot(shifts)
        # plt.plot(shifts_2)
        plt.legend(['x shifts', 'y shifts'])
        plt.title('shifts')
        plt.show()
        moviehandle = mc_mov.copy().reshape((-1, template.shape[0], template.shape[1]), order='F')
        play(moviehandle, gain=1, q_min=5, q_max=99.99, fr=400)
    #%%  offline GPU source separation
    options = load_fiola_config(fnames, num_frames_total, mode, mask) 
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    estimate_neuron_baseline = False
    if mask is not None:
        pass
    else:        
        if mode == 'voltage':
            raise Exception('Automatic initialization method not implemented for Voltage. please provide a binary mask')
        elif mode == 'calcium':
            #initialize with CaImAn
            fio.params.mc_dict, fio.params.opts_dict, fio.params.quality_dict = load_caiman_config(fnames)
            _, _ , mask = fio.fit_caiman_init(mc_mov[:fio.params.data['num_frames_init']], 
                                                fio.params.mc_dict, 
                                                fio.params.opts_dict, 
                                                fio.params.quality_dict)
    
    if estimate_neuron_baseline or mode == 'voltage':
        fio.fit_hals(mc_mov, mask, order='F') # transform binary masks into weighted masks  
    else:
        mask_2D = to_2D(mask, order='F')
        Ab = mask_2D.T
        fio.Ab = Ab / norm(Ab, axis=0)
    
    #%%
    t,w,h = mc_mov.shape   
    plt.figure()
    plt.imshow(fio.Ab[:,-1].reshape((w,h), order='F'))           
    trace = fio.fit_gpu_nnls(mc_mov[:30000], fio.Ab, batch_size=fio.params.mc_nnls['offline_batch_size']) 
    plt.figure()
    plt.plot(trace[:].T) 
    #%%
    import caiman as cm
    bcg = (fio.Ab[:,:-2].dot(trace[:-2])).astype(np.float32)
    bcg = np.reshape(bcg.astype(np.float32),(h,w,t)).transpose([2,1,0])
    cm.movie(bcg).resize(1,1,.2).play(gain=1,fr=100, magnification=1)
    #%% offline spike detection (only available for voltage currently)
    fio.saoz = fio.fit_spike_extraction(trace)
    #%% put the result in fio.estimates object
    fio.compute_estimates()
    #%% show results
    # bcg = fio.Ab[:,-2:].dot(trace[-2:])
    # bcg = np.reshape(bcg.astype(np.float32),(h,w,t)).transpose([2,1,0])    
    fio.corr = local_correlations(mc_mov-bcg, swap_dim=False)
    if display_images:
        fio.view_components(fio.corr)
    
    #%% ID USING CAIMAN INSTALLATION RESTART 
    #%% Now we start the second part. It uses fit method to perform initialization 
    # which prepare parameters, spatial footprints etc for real-time analysis
    # Then we call fit_online to perform real-time analysis
    options = load_fiola_config(fnames, num_frames_total, mode, mask) 
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    if mask is None and mode == 'calcium':
        fio.params.mc_dict, fio.params.opts_dict, fio.params.quality_dict = load_caiman_config(fnames)

    scope = [fio.params.data['num_frames_init'], fio.params.data['num_frames_total']]
    fio.fit(mov[:scope[0]], mode, mask)

    #%% fit online frame by frame 
    start = time()
    for idx in range(scope[0], scope[1]):
        fio.fit_online_frame(mov[idx:idx+1])   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i
    logging.info(f'total time online: {time()-start}')
    logging.info(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')
        
    #%% fit online with a thread loading frames
    # fio.pipeline.load_frame_thread = Thread(target=fio.pipeline.load_frame, 
    #                                         daemon=True, 
    #                                         args=(mov[scope[0]:scope[1], :, :],))
    # fio.pipeline.load_frame_thread.start()
    
    # start = time()
    # fio.fit_online()
    # logging.info(f'total time online: {time()-start}')
    # logging.info(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')
    
    #%% put the result in fio.estimates object
    fio.compute_estimates()
    
    #%% visualize the result, the last component is the background
    if display_images:
        fio.view_components(fio.corr)
            
    #%% save the result
    save_name = f'{fnames.split(".")[0]}_fiola_result'
    np.save(save_name, fio.estimates)
    
    #%%
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
        
#%%
if __name__ == "__main__":
    main()