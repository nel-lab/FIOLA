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
from fiola.utilities import download_demo, load, play, bin_median, to_2D, local_correlations, play

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1

#%%
def ():
    #%% load movie and masks
    import caiman as cm
    from caiman.source_extraction.cnmf.cnmf import load_CNMF
    mode_idx = 1
    mode = ['voltage', 'calcium'][mode_idx]
    #folder = '/media/nel/DATA/fiola'
    # fnames = '/media/nel/DATA/fiola/mov_R2.hdf5'
    #fnames = '/media/nel/DATA/fiola/R2_20190219/3000/mov_R2_20190219T210000_3000.hdf5'
    #fnames = '/media/nel/DATA/fiola/R2_20190219/full_nonrigid/mov_R2_20190219T210000.hdf5'
    #fnames = '/media/nel/DATA/fiola/R2_20190219/full_nonrigid/mov_R2_20190219T210000._els__d1_796_d2_512_d3_1_order_F_frames_31933_.mmap'
    fnames = '/media/nel/DATA/fiola/R2_20190219/full_nonrigid/mov_R2_20190219T210000._els__d1_796_d2_512_d3_1_order_F_frames_31933_.mmap'
    mov = cm.load(fnames, in_memory=True)
    #fnames = '/media/nel/DATA/fiola/R2_20190219/mov_R2_20190219T210000.hdf5'
    #caiman_output_path = '/media/nel/DATA/fiola/R2_20190219/3000/memmap__d1_796_d2_512_d3_1_order_C_frames_3000_.hdf5'
    #caiman_output_path = '/media/nel/DATA/fiola/R2_20190219/3000/memmap__d1_796_d2_512_d3_1_order_C_frames_3000_all_comp_5_5_snr_1.8_K_8.hdf5'
    #caiman_output_path = '/media/nel/DATA/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8_K_8.hdf5'
    caiman_output_path = '/media/nel/DATA/fiola/R2_20190219/3000/memmap_pw_rigid_True_d1_796_d2_512_d3_1_order_C_frames_3000_non_rigid_K_5.hdf5'
    
    #mov = load(fnames)
    dims = (796, 512)
    #mask = load(path_ROIs)
    cnm = load_CNMF(caiman_output_path)
    #import caiman as cm
    #cm_mov = cm.load('/media/nel/DATA/fiola/R2_20190219/mov_R2_20190219T210000_3000._rig__d1_796_d2_512_d3_1_order_F_frames_3000_.mmap', in_memory=True)
    #c = cnm.estimates.C
    # Ab = np.hstack([cnm.estimates.A[:, cnm.estimates.idx_components].toarray(), cnm.estimates.b])
    # #Ab = np.hstack([cnm.estimates.A[:, select].toarray(), cnm.estimates.b])    
    # #Ab = np.hstack([A, cnm.estimates.b])    
    # mask = Ab.reshape([dims[0], dims[1], -1]).transpose([2, 0, 1])
    # mask_2D = mask.transpose([1,2,0]).reshape((-1, mask.shape[0]))
    # Ab = mask_2D.copy()
    # Ab = Ab / norm(Ab, axis=0)

    #mask = np.hstack([cnm.estimates.A[:, cnm.estimates.idx_components].toarray(), cnm.estimates.b])
    mask = np.hstack([cnm.estimates.A[:, select].toarray(), cnm.estimates.b])    
    mask = mask.reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1])
    Ab = to_2D(mask, order='F').T
    Ab = Ab / norm(Ab, axis=0)
    
    #fio.fit_hals(mov, mask, order='F')

    options = load_fiola_config(fnames, num_frames_total=30000, mode=mode, mask=mask) 
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    fio.Ab = Ab        
    #fio.fit_hals(mov[:1000], mask, order='F')

    min_mov = mov[:3000].min()
    template = bin_median(mov[:3000])
    plt.imshow(template)
    trace = fio.fit_gpu_motion_correction_nnls(mov[:30000], template=template, batch_size=5, 
                                min_mov=min_mov, Ab=fio.Ab)
    
    trace = fio.fit_gpu_nnls(mov[:31930], fio.Ab, batch_size=5) 
    
    #%%
    plt.plot(trace[:-2].T)
    #np.save('/media/nel/DATA/fiola/R2_20190219/3000/fiola_1285_non_rigid_init_non_rigid_movie_nnls_only.npy', trace)    
    np.save('/media/nel/storage/fiola/R6_20200210T2100/3000/fiola_non_rigid_init.npy', trace)



#%%        
    display_images = True
    if display_images:
        plt.figure()
        plt.imshow(mov.mean(0), vmax=np.percentile(mov.mean(0), 99.9))
        plt.title('Mean img')
        if mask is not None:
            plt.figure()
            plt.imshow(mask.mean(0))
            plt.title('Masks')
    #%% mean roi
    def run_meanroi(mov, A):        
        A = A.reshape([dims[0], dims[1], -1], order='F')
        A = A.transpose([2, 0, 1])
        aa = []
        for i in range(A.shape[0]):
            a = A[i].copy()
            #a[a>np.percentile(a[a>0], 30)] = 1
            a[a>np.percentile(a, 99.98)] = 1
            a[a!=1]=0
            aa.append(a)
        aa = np.array(aa)
        # plt.imshow(aa.sum(0))
        # plt.figure()
        # plt.imshow(A.sum(0))
        aa = aa.transpose([1, 2, 0])
        aa = aa.reshape([-1, aa.shape[-1]], order='F')
        
        trace = []
        for idx in range(len(aa.T)):
            nonz = np.where(aa[:, idx] >0)[0]
            t = mov[:, nonz].mean(1)
            trace.append(t)
            if idx % 50 == 0:
                print(idx)
        trace = np.array(trace)
        
        return trace
        
        
#%% load configuration; set up FIOLA object
#     # In the first part, we will first show each part (motion correct, source separation and spike extraction) of 
#     # FIOLA separately in an offline manner. 
#     # Then in the second part, we will show the full pipeline and its real time frame-by-frame analysis performance
#     # Note one needs to installed CaImAn beforehand to perform CaImAn initialization
    
#     #%% offline motion correction
#     fio.dims = mov.shape[1:]
#     template = bin_median(mov[:2000])
#     mc_mov, shifts, _ = fio.fit_gpu_motion_correction(mov[:30000], template, fio.params.mc_nnls['offline_batch_size'], mov.min())
    
#     if display_images:
#         plt.figure()
#         plt.plot(shifts)
#         plt.legend(['x shifts', 'y shifts'])
#         plt.title('shifts')
#         plt.show()
#         moviehandle = mc_mov.copy().reshape((-1, template.shape[0], template.shape[1]), order='F')
#         play(moviehandle, gain=3, q_min=5, q_max=99.99, fr=400)
    
#     #%% optimize masks using hals or initialize masks with CaImAn
#     if mode == 'voltage':
#         if fio.params.data['init_method'] == 'binary_masks':
#             fio.fit_hals(mc_mov, mask)
#     elif mode == 'calcium':
#         # we don't need to optimize masks using hals as we are using spatial footprints from CaImAn
#         if fio.params.data['init_method'] == 'weighted_masks':
#             logging.info('use weighted masks from CaImAn')     
#         elif fio.params.data['init_method'] == 'caiman':
#         # if masks are not provided, we can use caiman for initialization
#         # we need to set init_method = 'caiman' in the config.py file for caiman initialization
#             fio.params.mc_dict, fio.params.opts_dict, fio.params.quality_dict = load_caiman_config(fnames)
#             _, _, mask = fio.fit_caiman_init(mc_mov[:fio.params.data['num_frames_init']], 
#                                              fio.params.mc_dict, fio.params.opts_dict, fio.params.quality_dict)
       
#         mask_2D = to_2D(mask, order='C')
#         Ab = mask_2D.T
#         fio.Ab = Ab / norm(Ab, axis=0)
        
            
#     #%% source extraction (nnls)
#     # when FOV and number of neurons is large, use batch_size=1
#     #mc_mov = mc_mov - mc_mov.min()
#     trace = fio.fit_gpu_nnls(mc_mov, fio.Ab, batch_size=1) 
    
#     #%%
#     min_mov = mov[:2000].min()
#     template = bin_median(mov[:2000])
#     trace = fio.fit_gpu_motion_correction_nnls(mov[:4000], template=template, batch_size=1, 
#                                 min_mov=min_mov, Ab=Ab)
    
#     #%%
#     plt.plot(trace[:-2].T)
#     np.save('/media/nel/DATA/fiola/R2_20190219/fiola_30000.npy', trace)
    
#     #%% offline spike detection (only available for voltage currently)
#     fio.saoz = fio.fit_spike_extraction(trace)
    
#     #%% put the result in fio.estimates object
#     fio.compute_estimates()
    
#     #%% show results
#     fio.corr = local_correlations(mc_mov, swap_dim=False)
#     if display_images:
#         fio.view_components(fio.corr)
    
#     #%% Now we start the second part. It uses fit method to perform initialization 
#     # which prepare parameters, spatial footprints etc for real-time analysis
#     # Then we call fit_online to perform real-time analysis
#     options = load_fiola_config(fnames, mode, mask) 
#     params = fiolaparams(params_dict=options)
#     fio = FIOLA(params=params)
    
#     if fio.params.data['init_method'] == 'caiman':
#         # in caiman initialization it will save the input movie to the init_file_name from the beginning
#         fio.params.mc_dict, fio.params.opts_dict, fio.params.quality_dict = load_caiman_config(fnames)
    
#     scope = [fio.params.data['num_frames_init'], fio.params.data['num_frames_total']]
#     fio.fit(mov[:scope[0]])
    
#     #%% fit online frame by frame 
#     start = time()
#     for idx in range(scope[0], scope[1]):
#         fio.fit_online_frame(mov[idx:idx+1])   
#     logging.info(f'total time online: {time()-start}')
#     logging.info(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')
        
#     #%% fit online with a thread loading frames
#     # fio.pipeline.load_frame_thread = Thread(target=fio.pipeline.load_frame, 
#     #                                         daemon=True, 
#     #                                         args=(mov[scope[0]:scope[1], :, :],))
#     # fio.pipeline.load_frame_thread.start()
    
#     # start = time()
#     # fio.fit_online()
#     # logging.info(f'total time online: {time()-start}')
#     # logging.info(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')
    
#     #%% put the result in fio.estimates object
#     fio.compute_estimates()
    
#     #%% visualize the result, the last component is the background
#     if display_images:
#         fio.view_components(fio.corr)
            
#     #%% save the result
#     save_name = f'{fnames.split(".")[0]}_fiola_result'
#     np.save(save_name, fio.estimates)
    
#     #%%
#     log_files = glob.glob('*_LOG_*')
#     for log_file in log_files:
#         os.remove(log_file)
        
# #%%
# if __name__ == "__main__":
#     main()