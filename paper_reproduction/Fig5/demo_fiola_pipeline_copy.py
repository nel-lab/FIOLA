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
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib
from time import time
import scipy

from fiola.demo_initialize_calcium import run_caiman_init
import pyximport
pyximport.install()
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
#from fiola.warmup import warmup
from caiman.source_extraction.cnmf.utilities import get_file_size
import caiman as cm
from fiola.utilities import download_demo, load, play, bin_median, to_2D, local_correlations, movie_iterator, compute_residuals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)
    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1
#%% 
def main():
#%% run s2p
    # do_warm_up = True
    # if do_warm_up:
    #     warmup(fnames='/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_512/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00001.tif', num_frames=1000)
        
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
        # fnames = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_1024/k53_1024.tif'
        fnames = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_256/k53_256.tif'
        # fnames= '/home/nel/caiman_data/example_movies/k53.tif'
        # fnames = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/s2p_k53/k53.tif'
        # fnames = sorted(glob.glob('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/YST/images_YST/*'))
        mask = "/home/nel/caiman_data/example_movies/k53_A_256.npy"
        # A = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/k53_A.npy",allow_pickle=True)[()]
        # b = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/FastResults/CMTimes/k53_b.npy",allow_pickle=True)[()]
        # Ab = np.concatenate((A.toarray(),b),axis=1)[:, :500]
        # A = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/YST_A.npy",allow_pickle=True)[()]
        # b = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/CalciumComparison/YST_b.npy",allow_pickle=True)[()]
        # Ab = np.concatenate((A.toarray(),b),axis=1)[:, :200]
        fr = 30                         # sample rate of the movie
        ROIs = mask                     # a 3D matrix contains all region of interests
    
        mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
        num_frames_init =  1500
         # number of frames used for initialization
        num_frames_total = 3000        # estimated total number of frames for processing, this is used for generating matrix to store data
        offline_batch_size = 5          # number of frames for one batch to perform offline motion correction
        batch_size= 1                   # number of frames processing at the same time using gpu 
        flip = False                    # whether to flip signal to find spikes   
        ms = [5, 5]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
        center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
        hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
                                        # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
        n_split = 1                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
        nb = 2                          # number of background components
        trace_with_neg=False             # return trace with negative components (noise) if True; otherwise the trace is cutoff at 0
                        
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
            'num_layers': 30,
            'do_deconvolve': True }
        # fnames = "/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_512/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00001.tif"
        mov = cm.load(fnames)
        # from skimage.transform import resize
        # mov = resize(mov, (3000, 256, 256))
        fnames_init = fnames.split('.')[0] + '_init.tif'
        mov.save(fnames_init)
        
        # run caiman initialization. User might need to change the parameters 
        # inside the file to get good initialization result
        caiman_file = run_caiman_init(fnames_init)
        
        # load results of initialization
        # cnm2 = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/s2p_k53/cnm2.npy", allow_pickle=True)[()]
        # cnm2 = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/suite2p_shifts/s2p_k53/cnm2_1024.npy", allow_pickle=True)[()]
        cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
        estimates = cnm2.estimates
        template = cnm2.estimates.template
        Cn = cnm2.estimates.Cn
        template = cnm2.estimates.template
        # template = np.load("/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/MotCorr/k53_20160530_RSM_125um_41mW_zoom2p2_00001_00001_template_on.npy")
        # template = np.median(mov[:1500], axis=0)
        from skimage.transform import resize
        template = resize(template,(256,256))
    else: 
        raise Exception('mode must be either calcium or voltage')
          

    #%% Run FIOLA
    #example motion correction
    motion_correct = True
    #example source separation
    do_nnls = True
#%%
import os
os.system('clear')
    #%% Mot corr only
    time_all = []
    for ii in range(1):
        if motion_correct:
            params = fiolaparams(params_dict=options)
            fio = FIOLA(params=params)
            # run motion correction on GPU on the initialization movie
            mc_nn_mov, shifts_fiola, times = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], min_mov=mov.min())             
            plt.plot(shifts_fiola)
        else:    
            mc_nn_mov = mov
        time_all.append(np.mean(np.diff(times)))
    
    #%% NNLS only
    if do_nnls:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        if mode == 'voltage':
            A = scipy.sparse.coo_matrix(to_2D(mask, order='F')).T
            fio.fit_hals(mov, A)
            Ab = fio.Ab # Ab includes spatial masks of all neurons and background
        else:
            Ab = np.hstack((estimates.A.toarray(), estimates.b))
            if Ab.shape[1] < 500:
                Ab = np.concatenate((Ab, Ab), axis=1)
            Ab = Ab[:, -500:].astype(np.float32)
            
        #trace_fiola, times_nnls0, times_nnls1,times_nnls2 = 
        t_all = []
        for ii in range(1):
            trace, tt = fio.fit_gpu_nnls_test(mov, Ab, batch_size=1)
            # tu = []
            # for jj in range(10):
            #     tu.append(fio.fit_gpu_nnls_test(mov, Ab, batch_size=1))
            # plt.plot(times_nnls0[1:])
            t_all.append(tt)
        
        ttt = [np.median(np.diff(tt[1:])) for tt in t_all]
        print(np.median(np.diff(tt[1:])))
        
# plt.plot(trace[0].reshape((100, 3000), order='F').T[:, :10])
 #%% plot
plt.plot(np.diff(times_nnls[1:-1])) 
if False:
    np.save(base_file+ movie_name+ "_nnls_" + str(options[0]['num_layers'])+ "_time.npy", np.diff(times))       
    #%% Set up online pipeline
    trace_fiola = np.array(trace)[:,0,:,0].T
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)

    if mode == 'voltage': # not thoroughly tested and computationally intensive for large files, it will estimate the baseline
        fio.fit_hals(mc_nn_mov, A)
        Ab = fio.Ab
    else:
        pass#Ab = np.hstack((estimates.A.toarray(), estimates.b))
    Ab = Ab.astype(np.float32)
    # Ab = Ab[:,:200]
        
    fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())
    #%% run online
    time_per_step = np.zeros(num_frames_total-num_frames_init)
    traces = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
    start = time()
        
    for idx, memmap_image in movie_iterator(fnames, num_frames_init, num_frames_total):
        # if idx % 100 == 0:
        #     print(idx) 

        fio.fit_online_frame(memmap_image)   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i        
        traces[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx]
        time_per_step[idx-num_frames_init] = (time()-start)
    
    traces = traces.T
    logging.info(f'total time online: {time()-start}')
    logging.info(f'time per frame online: {(time()-start)/(num_frames_total-num_frames_init)}')
    plt.plot(np.diff(time_per_step))
#%% run very  eagerly
time_per_step = np.zeros(num_frames_total-num_frames_init)
traces_out = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
start = time()
for idx  in  range(1500,2999):
    fr = mc_nn_mov[idx]
    fio.fit_online_frame(fr[None])
    traces_out[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx]
    time_per_step[idx-num_frames_init] = (time()-start)
#%% run  without estimator
from fiola.gpu_mc_nnls import get_model 
model = get_model(template, Ab, batch_size, 
                  ms_h=10, ms_w=10, min_mov=mov.min(),
                  use_fft=True, normalize_cc=True, 
                  center_dims=None, return_shifts=False, 
                  num_layers=30, n_split=1, 
                  trace_with_neg=True)
# model.compile(optimizer='rmsprop', loss='mse', run_eagerly=False) 
traces = cnm2.estimates.C
traces = np.concatenate((traces,traces))[:100] 
x,y = traces[:,0][None,:,None],traces[:,0][None,:,None]
k = np.array([[0]])
time_per_step = np.zeros(num_frames_total-num_frames_init)
traces_out = np.zeros((num_frames_total-num_frames_init,Ab.shape[-1]), dtype=np.float32)
# model.compile(optimizer='rmsprop', loss='mse')
#%% run for loop timing
start = time()
for idx,fr in enumerate(mov):
    out = model.predict([fr[None,None,:,:,None],x,y,k])
    x,y,ne = out
    time_per_step[idx] = time()-start
    traces_out[idx] = x[0,:,0]
#%% run NNLS model eagerly
from fiola.gpu_mc_nnls import get_nnls_model, get_mc_model
dims=1024
iters=30
neurons = 100
Ab = Ab[:, -neurons:]
model = get_nnls_model((dims,dims), Ab.astype(np.float32), 1, iters,1,False)
traces = cnm2.estimates.C[-neurons:]  
x,y = traces[:,0][None,:,None],traces[:,0][None,:,None]
k = np.array([[0]])
time_per_step = np.zeros(num_frames_total-num_frames_init)
traces_out = np.zeros((num_frames_total-num_frames_init,Ab.shape[-1]), dtype=np.float32)
# model.compile(optimizer='rmsprop', loss='mse')
#%% run for loop timing
import time
start = time.time()
for idx,fr in enumerate(mov):
    out = model.predict([fr[None,None,:,:,None],x,y,k])
    x,y = out
    time_per_step[idx] = time.time()-start
    # traces_out[idx] = x[0,:,0]
 plt.plot(np.diff(time_per_step))   
#%% run MC model eagerly
model = get_mc_model(template, 1)
time_per_step_mc = np.zeros(num_frames_total - num_frames_init)
start = time.time()
for idx, fr in enumerate(mov):
    out = model.predict(fr[None, None, :,:,None])
    time_per_step_mc[idx] = time.time()-start
  plt.plot(np.diff(time_per_step_mc))   

#%%SVING: 
path = fnames[0][:92]
np.save(path + "")
    #%% Visualize result
    fio.compute_estimates()
    fio.view_components(template)
    
    #%% save some interesting data
    if False:
        np.savez(fnames[:-4]+'_fiola_result.npz', time_per_step=time_per_step, traces=traces, 
             caiman_file = caiman_file, 
             fnames_exp = fnames, 
             estimates = fio.estimates)
        
    #%%
    plt.plot(fio.estimates.trace[8] / fio.estimates.trace[8].min())
    plt.plot(-tt / (-tt).min())
    plt.plot(ttt[8] / ttt[8].min())
    plt.legend(['fiola estimates trace', 'mean roi', 'traces'])
    ma = mask[0].reshape(-1, order='F')
    m = cm.load(fnames)    
    m = m.reshape([20000, -1], order='F')
    tt = m @ ma
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
