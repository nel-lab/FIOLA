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
from tensorflow.python.client import device_lib
from time import time
import scipy

from fiola.config import load_fiola_config_calcium, load_fiola_config_voltage
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from caiman.source_extraction.cnmf.utilities import get_file_size
import caiman as cm
from fiola.utilities import download_demo, load, play, bin_median, to_2D, local_correlations, movie_iterator, compute_residuals

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)
    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1

#%% 
fnames_exp  = None  
mode = 'voltage'         
#%% Parameter setting
if mode == 'voltage':
    folder = '/home/nel/caiman_data/example_movies/volpy'
    fnames = [download_demo(folder, 'demo_voltage_imaging.hdf5')]
    path_ROIs = download_demo(folder, 'demo_voltage_imaging_ROIs.hdf5')
    mask = load(path_ROIs)
    mode = 'voltage'   
     
    fnames_exp  = None           
    mode = 'voltage'                                           # 'voltage' or 'calcium 'fluorescence indicator
    if fnames_exp is not None:
        num_frames_total = get_file_size(fnames_exp)[-1]
    else:
        num_frames_total = get_file_size(fnames)[-1]         # number of total frames including initialization

    # maximum shift in x and y axis respectively. Will not perform motion correction if None. 
    ms = [10, 10]   
    # number of frames used for initialization
    num_frames_init =  num_frames_total//2

    #number of neurons
    num_nr = mask.shape[0]
    
    logging.info('Loading Movie')
    mov = cm.load(fnames, subindices=range(num_frames_init))
    
    
    #estimated motion shifts from the initialization
    logging.info('Motion correcting')
    mc_mov, shift_caiman, xcorrs, template = mov.copy().motion_correct(ms[0], ms[1])              
    Cn = mc_mov.local_correlations(eight_neighbours=True)
    plt.plot(shift_caiman)
    plt.figure()
    plt.imshow(template)

    min_mov = mov.min()
    
    
elif mode == 'calcium':
    #small file used for initialization purposes (see demo_initialize calcium)
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/dandi_deconding_data/mov_R2_20190219T210000_3000.hdf5']  
    # output of the demo_initialize _calcium.py file. (alert, motion correction will be run on this whole file, although CNMF will only be run on num_frames_init frames)
    caiman_file = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/dandi_deconding_data/memmap__d1_796_d2_512_d3_1_order_C_frames_1500_.hdf5' 
    # file associated to a long experiments. If none the same file is used for both 
    fnames_exp = '/home/nel/mov_R2_20190219T210000.hdf5'

    # caiman_file = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/dandi_deconding_data/memmap__d1_100_d2_100_d3_1_order_C_frames_750_.hdf5'
    # fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/dandi_deconding_data/mov_R2_20190219T210000_3000-crop.hdf5']

    # caiman_file = '/home/nel/caiman_data/example_movies/memmap__d1_60_d2_80_d3_1_order_C_frames_1000_.hdf5'
    # fnames = ['/home/nel/caiman_data/example_movies/demoMovie.tif']  # filename to be processed

    # caiman_file = '/home/nel/caiman_data/example_movies/memmap__d1_170_d2_170_d3_1_order_C_frames_1500_.hdf5'
    # fnames = ['/home/nel/caiman_data/example_movies/Sue_2x_3000_40_-46.tif']

    # load results of initialization
    cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
    estimates = cnm2.estimates
    mode = 'calcium'                                           # 'voltage' or 'calcium 'fluorescence indicator
    if fnames_exp is not None:
        num_frames_total = get_file_size(fnames_exp)[-1]
    else:
        num_frames_total = get_file_size(fnames)[-1]         # number of total frames including initialization

    # maximum shift in x and y axis respectively. Will not perform motion correction if None. 
    ms = [10, 10]   

    #number of neurons
    num_nr = estimates.nr
    #estimated motion shifts from the initialization
    shift_caiman = cnm2.estimates.shifts
    # number of frames used for initialization
    num_frames_init =  cnm2.estimates.C.shape[1]         
    # loading initialization movie, this should not be too big
    mov = cm.load(fnames,subindices=range(num_frames_init), in_memory=True)
    min_mov = mov.min()
    # template for motion correction obtained from initialization
    template = cnm2.estimates.template

else: 
    raise Exception('mode must be either calcium or voltage')
      
#%%
#example motion correction
motion_correct = True
#example source separation
do_nnls = True
#%% Mot corr only
if motion_correct:
    plt.close('all')
    if mode == 'calcium':
       options = load_fiola_config_calcium(fnames, num_frames_total=num_frames_total,
                                       num_frames_init=num_frames_init, 
                                       batch_size=1, ms=ms)
   
    else:
       options = load_fiola_config_voltage(fnames, num_frames_total=num_frames_total,
                                       num_frames_init=num_frames_init, 
                                       batch_size=1, ms=ms)
    
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    # run motion correction on GPU on the initialization movie
    mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], min_mov=min_mov)             
    # compare shifts of CaImAn and FIOLA
    plt.plot(shift_caiman)
    plt.plot(shifts_fiola)
else:    
    mc_nn_mov = mov

#%% NNLS Only
if do_nnls:
    if mode == 'calcium':
        options = load_fiola_config_calcium(fnames, num_frames_total=num_frames_total,
                                        num_frames_init=num_frames_init, 
                                        batch_size=1, ms=ms)
    
    else:
        options = load_fiola_config_voltage(fnames, num_frames_total=num_frames_total,
                                        num_frames_init=num_frames_init, 
                                        batch_size=1, ms=ms)
    
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    if mode == 'voltage':
        A = scipy.sparse.coo_matrix(to_2D(mask, order='F')).T
        fio.fit_hals(mc_nn_mov, A)
        Ab = fio.Ab
    else:
        Ab = np.hstack((estimates.A.toarray(), estimates.b))
        
    # run NNLS and obtain nonnegative traces
    trace_fiola_no_hals = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch_size']) 
    
    if mode == 'calcium':
        
        # compute the residual to add the non-explained portion of the signal  (as YrA in CNMF)
        YrA = compute_residuals(mc_nn_mov,  estimates.A, estimates.b, trace_fiola_no_hals[num_nr:], trace_fiola_no_hals[:num_nr])
        trace_fiola_nohals_resid = np.vstack((YrA + trace_fiola_no_hals[:num_nr], trace_fiola_no_hals[num_nr:]))
        #compare with caiman traces
        trace_caiman = np.vstack((estimates.C[:,:num_frames_init] + estimates.YrA[:,:num_frames_init],estimates.f[:,:num_frames_init]))
        
        for idx in range(len(trace_caiman)//3):
            plt.cla();
            plt.plot(trace_caiman[idx]); 
            plt.plot(trace_fiola_nohals_resid[idx])
            plt.plot(trace_fiola_no_hals[idx])
            plt.ginput()
        plt.close()
        cc = [np.corrcoef(s1,s2)[0,1] for s1,s2 in zip(trace_caiman,trace_fiola_nohals_resid)]
        plt.hist(cc,30)
        plt.figure()
        plt.scatter(cnm2.estimates.SNR_comp,cc[:-1])

    else:
        plt.plot(trace_fiola_no_hals.T)
    
#%% Full Pipeline
# set up the parameters
if mode == 'calcium':
    options = load_fiola_config_calcium(fnames, num_frames_total=num_frames_total,
                                    num_frames_init=num_frames_init, 
                                    batch_size=1, ms=ms)

else:
    options = load_fiola_config_voltage(fnames, num_frames_total=num_frames_total,
                                    num_frames_init=num_frames_init, 
                                    batch_size=1, ms=ms)
    

params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
if mode == 'voltage': # not thoroughly tested and computationally intensive for large files, it will estimate the baseline
    fio.fit_hals(mc_nn_mov, A)
    Ab = fio.Ab
else:
    Ab = np.hstack((estimates.A.toarray(), estimates.b))
    
fio = fio.create_pipeline(mc_nn_mov, trace_fiola_no_hals, template, Ab, min_mov=min_mov)
#%%
time_per_step = np.zeros(num_frames_total-num_frames_init)
traces = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
start = time()
if fnames_exp is not None:
    name_movie = fnames_exp
else:
    name_movie = fnames[0]
    
for idx, memmap_image in movie_iterator(name_movie, num_frames_init, num_frames_total):
    # List all groups       

    if idx%100 == 0:
        print(idx)        
    fio.fit_online_frame(memmap_image)   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i        
    traces[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx-1]
    time_per_step[idx-num_frames_init] = (time()-start)

traces = traces.T
logging.info(f'total time online: {time()-start}')
logging.info(f'time per frame online: {(time()-start)/(num_frames_total-num_frames_init)}')
plt.plot(np.diff(time_per_step),'.')
#%%#%% add residuals 
fio.compute_estimates()
# if suing a single movie we can compute the residual (this might be very expensive for large movies)
# TODO: it should be an output of FIOLA
if fnames_exp is None and mode == 'calcium':
    mov = cm.load(cnm2.mmap_F, subindices=range(num_frames_init, num_frames_total), in_memory=True)
    YrA = compute_residuals(mov,  estimates.A, estimates.b, traces[num_nr:], traces[:num_nr])
    trace_fiola_nohals_resid = np.vstack((YrA + traces[:num_nr], traces[num_nr:]))
    fio.estimates.trace[:,num_frames_init:] = trace_fiola_nohals_resid
    fio.view_components(estimates.Cn)
else:
    fio.view_components(Cn)
#%% uncomment for demo movies to compare the resuls on the full movie
# cnm_total = cm.source_extraction.cnmf.cnmf.load_CNMF('/home/nel/caiman_data/example_movies/memmap__d1_60_d2_80_d3_1_order_C_frames_2000_.hdf5')
# cnm_total.estimates.view_components()
#%% save some interesting data
if False:
    np.savez(fnames[:-4]+'_rtx_2080ti.npz', time_per_step=time_per_step, traces=traces, 
         caiman_file = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/dandi_deconding_data/memmap__d1_796_d2_512_d3_1_order_C_frames_1500_.hdf5', 
         fnames_exp = '/home/nel/mov_R2_20190219T210000.hdf5', 
         estimates = fio.estimates)

