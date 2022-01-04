#!/usr/bin/env python
"""
Pipeline for online analysis of fluorescence imaging data
Voltage dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
Calcium dataset courtesy of Sue Ann Koay and David Tank (Princeton University)
@author: @agiovann, @caichangjia, @cynthia
"""
try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
except:
    pass

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


#%% load movie and masks
folder = ''    
file_id = 2   
fnames = '/Users/joe/repos/CaImAn/example_movies/demoMovie.tif'
path_ROIs = None
mask = None
mode = 'calcium'
    

mov = (load(fnames)).astype(np.float32)    
num_frames_total =  mov.shape[0]    


#%% offline GPU motion correction example
options = load_fiola_config(fnames, num_frames_total, mode, mask) 
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
fio.dims = mov.shape[1:]
template = bin_median(mov) 
mc_mov, shifts, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], mov.min())

#%%  offline GPU source separation
options = load_fiola_config(fnames, num_frames_total, mode, mask) 
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
estimate_neuron_baseline = True
if mask is not None:
    mask_C_order = True
    if mode == 'voltage': 
        mask_C_order = False
    # depending on how the matrix is reshaped we will need different order options
    if mask_C_order: 
        order = 'C'
    else:
        order = 'F'
else:
    
    if mode == 'voltage':
        raise Exception('Automatic initialization method not implemented for Voltage. please provide a binary mask')
    #initialize with CaImAn
    fio.params.mc_dict, fio.params.opts_dict, fio.params.quality_dict = load_caiman_config(fnames)
    _, _ , mask = fio.fit_caiman_init(mc_mov[:fio.params.data['num_frames_init']], 
                                        fio.params.mc_dict, 
                                        fio.params.opts_dict, 
                                        fio.params.quality_dict)
    order = 'C'

if estimate_neuron_baseline or mode == 'voltage':
    fio.fit_hals(mc_mov, mask, order=order) # transform binary masks into weighted masks  
else:
    mask_2D = to_2D(mask, order='C')
    Ab = mask_2D.T
    fio.Ab = Ab / norm(Ab, axis=0)
    
# t,w,h = mc_mov.shape   
# plt.figure()
# plt.imshow(fio.Ab[:,-1].reshape((w,h), order='F'))           
# trace = fio.fit_gpu_nnls(mc_mov, fio.Ab, batch_size=fio.params.mc_nnls['offline_batch_size']) 
# plt.figure()
# plt.plot(trace[:].T)
# #%%
# import caiman as cm
# nb = params.get('opts_dict','nb')
# bcg = (fio.Ab[:,-nb:].dot(trace[-nb:])).astype(np.float32).reshape(h,w,t).transpose([2,1,0])
# cm.movie(bcg).resize(1,1,.2).play(gain=1,fr=100, magnification=1)
# #%% offline spike detection (only available for voltage currently)
# fio.saoz = fio.fit_spike_extraction(trace)
# #%% put the result in fio.estimates object
# fio.compute_estimates()
# #%% show results
# fio.corr = local_correlations(mc_mov-bcg, swap_dim=False)
# if display_images:
#     fio.view_components(fio.corr)

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
    fio.fit_online_frame(mov[idx:idx+1])   
logging.info(f'total time online: {time()-start}')
logging.info(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')
    
print(f'total time online: {time()-start}')
print(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')
 
#%% put the result in fio.estimates object
fio.compute_estimates()
