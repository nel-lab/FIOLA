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
