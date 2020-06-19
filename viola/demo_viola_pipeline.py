#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:48:17 2020
Pipeline for online analysis of voltage imaging data
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
@author: @agiovann, @caichangjia, @cynthia
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time, sleep
from threading import Thread

from nmf_support import normalize
from violaparams import violaparams
from viola import VIOLA

#%% load movie
movie_folder = ['/home/nellab/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data',
                    '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data'][1]
    
movie_lists = ['demo_voltage_imaging_mc.hdf5', 
               'FOV4_50um_mc.hdf5']
name = movie_lists[0]
with h5py.File(os.path.join(movie_folder, name),'r') as h5:
    mov = np.array(h5['mov'])
with h5py.File(os.path.join(movie_folder, name[:-8]+'_ROIs.hdf5'),'r') as h5:
    mask = np.array(h5['mov'])                                               
  
#%% setting params
# dataset dependent parameters
fnames = ''
fr = 400
ROIs = mask
border_to_0 = 2
num_frames_total=20000

opts_dict = {
    'fnames': fnames,
    'fr': fr,
    'ROIs': ROIs,
    'border_to_0': border_to_0, 
    'num_frames_total': num_frames_total
}
opts = violaparams(params_dict=opts_dict)

#%% process offline for init
params = violaparams(params_dict=opts_dict)
vio = VIOLA(params=params)
vio.fit(mov[:10000])
plt.plot(vio.pipeline.saoz.t_s[1])

#%% process online
scope = [10000, 20000]
vio.pipeline.load_frame_thread = Thread(target=vio.pipeline.load_frame, daemon=True, args=(mov[scope[0]:scope[1], :, :],))
vio.pipeline.load_frame_thread.start()

start = time()
vio.fit_online()
sleep(0.1) # wait finish
print(f'total time online: {time()-start}')
print(f'time per frame  online: {(time()-start)/(scope[1]-scope[0])}')

#%% resutl is stored in vio.pipeline.saoz object
plt.plot(vio.pipeline.saoz.t_s[1][:scope[1]])
