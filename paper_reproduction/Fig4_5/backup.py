#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:26:50 2022

@author: nel
"""
#%%
import os
import scipy
import caiman as cm
import h5py
from paper_reproduction.Fig4_5.utils import load_gt
import matplotlib.pyplot as plt

overlap = 'non_overlapping'
ff = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy/volpy_test/PositronSimulations/data/Viola/viola_sim7'
tf = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/simulation'

#%%
print(os.listdir(ff))
files = os.listdir(ff)
files = [f for f in files if 'raw' not in f]

#%%
for file in files:
    name = file.split('.')[0]
    try:
        os.mkdir(os.path.join(tf, overlap, name))
    except:
        print('fail')
        
#%%
for file in files:
    name = file.split('.')[0]
    result = [f for f in os.listdir(ff) if 'SimResults_'+name.split('_')[-1]+'.mat' in f][0]
    #os.rename(os.path.join(ff, file), os.path.join(tf, overlap, name, file))
    os.rename(os.path.join(ff, result), os.path.join(tf, overlap, name, result))
        
#%%
#names = [f'viola_sim7_{i}' for i in range(2, 20, 3)]
names = [f'viola_sim7_{i}' for i in range(1, 9)]

for name in names:
    fnames_mat = tf + '/'+overlap + f'/{name}/{name}.mat'
    with h5py.File(fnames_mat, 'r') as f:
        m = f['dataAll'][:].transpose([0, 2, 1])
    plt.imshow(m.mean(0))
    m = cm.movie(m)
    fnames = fnames_mat[:-4] + '.hdf5'
    m.save(fnames)
    
#%%
#names = [f'viola_sim6_{i}' for i in range(2, 20, 3)]
for name in names:
    try:
        os.mkdir(tf+'/'+overlap+ f'/{name}/viola')
        print('success!!')
    except:
        print('fail')

#%%
root = os.path.join(tf, overlap)
for name in names:
    folder = os.path.join(root, name)
    spatial, temporal, spikes = load_gt(folder)  
    #ROIs = spatial.transpose([1,2,0])
    ROIs = spatial.copy()    
    plt.figure()
    plt.imshow(ROIs.sum(0))
    plt.show()
    save_folder = os.path.join(folder, 'viola')
    cm.movie(ROIs).save(os.path.join(save_folder, 'ROIs_gt.hdf5'))

n_neurons, frames = fio.index.shape
fio.trace_deconvolved = np.zeros(fio.index.shape)
for i in range(n_neurons):
    fio.trace_deconvolved[i][fio.index[i]] = 1

#%%
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
mode = 'voltage'
num_frames_init = 10000
num_frames_total = 20000#75000
border_to_0 = 2
flip = True
use_rank_one_nmf=False
hals_movie='hp_thresh'
semi_nmf=False
update_bg = True
use_spikes= False
batch_size=1
center_dims=None
initialize_with_gpu=True
do_scale = False
adaptive_threshold=True
filt_window=15
minimal_thresh=3
step=2500
template_window=2
trace_with_neg=False
detrend=False
#num_layers = 30

#%%
trace = -fio3.trace.copy()
saoz = SignalAnalysisOnlineZ(detrend=True, nb=1)
saoz.fit(trace[:, :20000], 75000)
for ii in range(20000, 75000):
    saoz.fit_next(trace[:, ii][:, None], ii)
saoz.reconstruct_signal()

#%%    
n_idx = 4
plt.plot(saoz.t_s[n_idx], alpha=0.5)
plt.hlines(saoz.thresh[n_idx, 0], 0, 30000, color='red', linestyles='dashed')
for jj in range(saoz.thresh.shape[1]-1):
    plt.hlines(saoz.thresh[n_idx, jj+1], 30000 + jj * 5000, 30000 + (jj + 1) * 5000, 
               color='red', linestyles='dashed')
    
#%%
plt.plot(saoz.trace_deconvolved[n_idx])
plt.plot(saoz.t_rec[n_idx])
    
    
