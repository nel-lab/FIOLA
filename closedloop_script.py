#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:48:17 2020
Pipeline for online analysis of fluorescence imaging data
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
@author: @agiovann, @caichangjia, @cynthia
"""
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from skimage.io import imread
from time import time, sleep
from skimage.draw import circle
from threading import Thread

from fiola.utilities import normalize, play
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA

from slm import SLM
from NelNet import NelNet
from deepcgh_coords import DeepCGH

#%% load movie and masks
online = True
online_batch = 200
experiment_length = 500

#%% Define params
retrain = True
frame_path = 'DeepCGH_Frames/'
coordinates = True

radius = 10
n = 20
x = 1152
y = 1920
masks = np.zeros((n, x, y))
centers_x = np.random.randint(10, x-10, size=(1, n))
centers_y = np.random.randint(10, y-10, size=(1, n))
centers = np.vstack([centers_x, centers_y])

for i in range(n):
    rr, cc = circle(centers[0, i], centers[1, i], radius=radius)
    masks[i, rr, cc] = np.random.rand(1)[0]

#%
path = '\DeepCGH_Datasets\Disks'

data = {
        'path' : path,
        'shape' : (x, y, 1),
        'object_type' : 'Disk',
        'object_size' : 10,
        'object_count' : [27, 48],
        'intensity' : [0.2, 1],
        'normalize' : True,
        'centralized' : False,
        'N' : 10000,
        'train_ratio' : 9000/10000,
        'file_format' : 'tfrecords',
        'compression' : 'GZIP',
        'name' : 'target',
        }

path = '\DeepCGH_Models\Disks'
model = {
        'path' : path,
        'int_factor':16,
        'n_kernels':[ 4, 8, 16, 32, 64, 128, 256, 512],
        'plane_distance':0.005,
        'wavelength':1e-6,
        'pixel_size':0.000015,
        'input_name':'target',
        'output_name':'phi_slm',
        'lr' : 1e-3,
        'batch_size' : 16,
        'epochs' : 10,
        'token' : 'coords_small',
        'max_steps' : 400,
        'shuffle' : 16,
        'masks' : masks
        }

#%% DeepCGH
dcgh = DeepCGH(data, model)

if retrain:
    dcgh.train(data_path = 'coords',
               lr = model['lr'],
               batch_size = model['batch_size'],
               epochs = model['epochs'],
               token = model['token'],
               shuffle = model['shuffle'],
               max_steps = model['max_steps'])

#%%
if online:
    nl = NelNet()
    mov = nl.startBatch(online_batch).astype(np.float32)

else:
    movie_folder = 'C:\\Users\\nel\\Documents\\Closed Loop\\FIOLA'
    name = ['demo_voltage_imaging.hdf5', 'k53.tif'][1]
    if '.hdf5' in name:
        with h5py.File(os.path.join(movie_folder, name),'r') as h5:
            mov = np.array(h5['mov'])
    elif '.tif' in name:
        mov = imread(os.path.join(movie_folder, name))                                            
    
    mask = np.load(os.path.join(movie_folder, 'masks_caiman.npy'))
    mov = mov.astype(np.float32) - mov.min()
 
#%% setting params
# dataset dependent parameters
fnames = ''                     # name of the movie, we don't put a name here as movie is already loaded above
fr = 30                         # sample rate of the movie
ROIs = mask                     # a 3D matrix contains all region of interests
mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
num_frames_init =  1000         # number of frames used for initialization
num_frames_total =  3000        # estimated total number of frames for processing, this is used for generating matrix to store data
flip = False                     # whether to flip signal to find spikes   
ms=[5, 5]                      # maximum shift in x and y axis respectively. Will not perform motion correction if None.
offline_mc_batch_size=100
thresh_range= [2.8, 5.0]        # range of threshold factor. Real threshold is threshold factor multiply by the estimated noise level
use_rank_one_nmf=False          # whether to use rank-1 nmf, if False the algorithm will use initial masks and average signals as initialization for the HALS
hals_movie=None                 # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
                                # original movie (orig); no HALS needed if the input is from CaImAn (None)
update_bg = True                # update background components for spatial footprints
use_batch=True                  # whether to process a batch of frames (greater or equal to 1) at the same time. set use batch always as True
batch_size= 1                   # number of frames processing at the same time using gpu 
initialize_with_gpu=True        # whether to use gpu for performing nnls during initialization
do_scale = False                # whether to scale the input trace or not
adaptive_threshold=True         # whether to use adaptive threshold method for deciding threshold level
filt_window=15                  # window size for removing the subthreshold activities 
minimal_thresh=3                # minimal of the threshold 
step=2500                       # step for updating statistics
template_window=2               # half window size of the template; will not perform template matching if window size equals 0
    
options = {
    'fnames': fnames,
    'fr': fr,
    'ROIs': ROIs,
    'mode': mode, 
    'flip': flip,
    'ms': ms,
    'offline_mc_batch_size': offline_mc_batch_size,
    'num_frames_init': num_frames_init, 
    'num_frames_total':num_frames_total,
    'thresh_range': thresh_range,
    'use_rank_one_nmf': use_rank_one_nmf,
    'hals_movie': hals_movie,
    'update_bg': update_bg,
    'use_batch':use_batch,
    'batch_size':batch_size,
    'initialize_with_gpu':initialize_with_gpu,
    'do_scale': do_scale,
    'adaptive_threshold': adaptive_threshold,
    'filt_window': filt_window,
    'minimal_thresh': minimal_thresh,
    'step': step, 
    'template_window':template_window}

#%% FIOLA: process offline for initialization
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
fio.fit(mov[:num_frames_init])

#%% SLM
slm = SLM(lut_path = r"C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm4633_at1035.lut",
          verbose = False)
slm.initialize()

#%% process online
times = []
for idx in tqdm(range(experiment_length)):
    inds = np.array((np.random.rand(1, n)>0.5)*1, dtype=np.int32)
    mov[idx:idx+1]
    t0 = time()
    frame = nl.startSingle().astype(np.float32)
    fio.fit_online_frame(frame)
    phase = dcgh.get_hologram(inds)[:,:,:,0]
    slm.write(phase)
    times.append(time()-t0)
    
print(np.median(times))
nl.closeAll()
slm.terminate()

#%% compute the result in fio.estimates object
#fio.compute_estimates()

#%% visualize the result, the last component is the background
for i in range(10):
    plt.figure()
    #plt.imshow(mov[0], cmap='gray')
    plt.imshow(fio.H.reshape((mov.shape[1], mov.shape[2], fio.H.shape[1]), order='F')[:,:,i])#, alpha=0.7)
    plt.title(f'Spatial footprint of neuron {i}')
    
    if mode == 'voltage':
        plt.figure()
        plt.plot(fio.pipeline.saoz.trace[i][:scope[1]])
        plt.title(f'Temporal trace of neuron {i} before processing')
        plt.figure()
        plt.plot(normalize(fio.pipeline.saoz.t_s[i][:scope[1]]))
        spikes = np.delete(fio.pipeline.saoz.index[i], fio.pipeline.saoz.index[i]==0)
        h_min = normalize(fio.pipeline.saoz.t_s[i][:scope[1]]).max()
        plt.vlines(spikes, h_min, h_min + 1, color='black')
        plt.legend('trace', 'detected spikes')
        plt.title(f'Temporal trace of neuron {i} after processing')
    elif mode == 'calcium':
        plt.figure()
        plt.plot(fio.pipeline.saoz[i][:scope[1]])
        
#%% save the result
save_name = f'{os.path.join(movie_folder, name)[:-5]}_fiola_result'
np.save(os.path.join(movie_folder, save_name), fio.estimates)

#%%
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)