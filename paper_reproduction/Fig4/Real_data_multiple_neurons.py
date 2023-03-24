#!/usr/bin/env python
import h5py
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import time, sleep
from threading import Thread

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass
#%%
from viola.metrics import metric
from viola.nmf_support import normalize
from viola.violaparams import violaparams
from viola.viola import VIOLA
import scipy.io
from viola.match_spikes import match_spikes_greedy, compute_F1
#sys.path.append('/home/nel/Code/NEL_LAB/VIOLA/use_cases')
#sys.path.append(os.path.abspath('/Users/agiovann/SOFTWARE/VIOLA'))
from use_cases.test_run_viola import run_viola # must be in use_cases folder
        
#%%
ROOT_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/multiple_neurons'
names = sorted(list(os.listdir(ROOT_FOLDER)))
frate_all = np.array([300, 400, 1000])
#freq_all = np.array([15]*16 + [5,5,15])
freq_all = np.array([15]*3)
init_frames_all = np.array([5000]+[10000]*2)
flip_all = np.array([True, True, False])
 
#%%
select = np.array(range(len(names)))[:3]
idx = 0; name = names[idx]
for idx, name in enumerate(np.array(names)[select]):
    num_frames_init = init_frames_all[select][idx]
    border_to_0 = 2
    flip = flip_all[select][idx]
    thresh_range= [3, 4]
    erosion=0 
    use_rank_one_nmf=False
    hals_movie='hp_thresh'
    semi_nmf=False
    update_bg = True
    use_spikes= False
    use_batch=True
    batch_size=100
    center_dims=None
    initialize_with_gpu=True
    do_scale = False
    adaptive_threshold=True
    filt_window=15
    freq = freq_all[select][idx]
    do_plot = True
    step = 2500
   
    options = {
        'border_to_0': border_to_0,
        'flip': flip,
        'num_frames_init': num_frames_init, 
        'thresh_range': thresh_range,
        'erosion':erosion, 
        'use_rank_one_nmf': use_rank_one_nmf,
        'hals_movie': hals_movie,
        'semi_nmf':semi_nmf,  
        'update_bg': update_bg,
        'use_spikes':use_spikes, 
        'use_batch':use_batch,
        'batch_size':batch_size,
        'initialize_with_gpu':initialize_with_gpu,
        'do_scale': do_scale,
        'adaptive_threshold': adaptive_threshold,
        'filt_window': filt_window, 
        'freq':freq,
        'do_plot':do_plot, 
        'step':step}

    
    fr = np.array(frate_all)[select][idx]
    fnames = os.path.join(ROOT_FOLDER, name, name+'_mc.tif')  # files are motion corrected before
    if not os.path.isfile(fnames):
        fnames = os.path.join(ROOT_FOLDER, name, name+'_mc.hdf5')
    path_ROIs = os.path.join(ROOT_FOLDER, name, name+'_ROI.hdf5')
    run_viola(fnames, path_ROIs, fr=fr, online_gpu=True, options=options)

#%% match spatial footprints
idx = 1; name = names[idx]
if 'FOV4' in name:
    min_spikes = 50
    t_range = [10000, 20000]    
    min_counts = 10
    idx_list = [0, 2, 4]
    scope=[18000, 20000]
elif 'Fish' in name:
    min_spikes = 30
    t_range = [5000, 10000]    
    min_counts = 3
    idx_list = [0, 1]
    scope=[8500, 10000]
elif 'IVQ' in name:
    min_spikes = 20
    t_range = [10000, 17000]    
    min_counts = 20
    idx_list = [0]
    scope=[10000, 15000]

#%%
from skimage.io import imread

v_folder = os.path.join(ROOT_FOLDER, name, 'volpy')
v_files = sorted([file for file in os.listdir(v_folder) if '.npy' in file])
print(f'files number: {len(v_files)}')
v_file = v_files[0]
v = np.load(os.path.join(v_folder, v_file), allow_pickle=True).item()
v_spatial = v['weights']
v_temporal = v['ts'].copy()
v_spikes = v['spikes'].copy()

vi_folder = os.path.join(ROOT_FOLDER, name, 'viola')
vi_files = sorted([file for file in os.listdir(vi_folder) if 'v2.1' in file and 'thresh_3.2' in file])# and '24000' in file])
if len(vi_files) != 1:
    raise Exception('file number greater than 1')
vi_file = vi_files[0]
vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
vi_spatial = vi.H.copy()
vi_spatial = vi_spatial.reshape([v_spatial.shape[1], v_spatial.shape[2], -1], order='F').transpose([2, 0, 1])
seq = vi.seq
vi_temporal = vi.t_s.copy()
vi_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vi.index])[np.argsort(vi.seq)]

fnames = [os.path.join(v_folder, file) for file in os.listdir(v_folder) if 'summary_images.tif' in file][0]
mov = imread(fnames).transpose([2, 0, 1])
path_ROIs = os.path.join(ROOT_FOLDER, name, name+'_ROI.hdf5')
with h5py.File(path_ROIs,'r') as h5:
    ROIs = np.array(h5['mov'])

#%%
mask0 = ROIs
mask1 = v['ROIs']
mask1 = mask1 * 1.0
plt.figure();plt.imshow(mask0.sum(0));plt.colorbar();plt.show()
        
vi_num = [len(vi_spike) for vi_spike in vi_spikes]
neuron_idx = np.where(np.array(vi_num)>min_spikes)[0]   # 30 , 50, 20
len(neuron_idx)

mask0 = mask0[neuron_idx]
#%%
from viola.caiman_functions import nf_match_neurons_in_binary_masks
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        mask0, mask1, thresh_cost=1, min_dist=10, print_assignment=True,
        plot_results=True, Cn=mov[0], labels=['viola', 'cm'])    

vi_t = vi_temporal[neuron_idx][tp_gt]
vi_spikes = vi_spikes[neuron_idx][tp_gt]
mask0 = mask0[tp_gt]
v_t = v_temporal[tp_comp]
v_spikes = v_spikes[tp_comp]


for idx in range(len(tp_comp)):
    plt.figure()
    plt.imshow(mask0[idx])
    plt.title(f'mask{idx}')
    
    plt.figure()
    plt.plot(normalize(vi_t[idx]))
    plt.vlines(vi_spikes[idx], 0, 10)
    plt.title(f'viola{idx}')
    
    plt.figure()
    plt.plot(normalize(v_t[idx]))
    plt.vlines(v_spikes[idx], 0, 10)
    plt.title(f'volpy{idx}')

#%%
spnr = []
from viola.metrics import compute_spnr
for idx in range(len(tp_comp)):
    spnr.append(compute_spnr(vi_t[idx], v_t[idx], vi_spikes[idx], v_spikes[idx], t_range, min_counts))
spiking_neuron_idx = np.where(np.array(spnr)[:,0] >0)[0]

save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons'
#np.save(os.path.join(save_folder, name + '_spnr.npy'), np.array(spnr))


vi_t = vi_t[spiking_neuron_idx]
vi_spikes = vi_spikes[spiking_neuron_idx]
mask0 = mask0[spiking_neuron_idx]
v_t = v_t[spiking_neuron_idx]
v_spikes = v_spikes[spiking_neuron_idx]

#%%
length = len(idx_list)
fig, ax = plt.subplots(len(idx_list),1, squeeze=True)
if len(idx_list) == 1:
    ax = [ax]
colorsets = plt.cm.tab10(np.linspace(0,1,10))
colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]

#scope=[10000, 15000]
#scope=[8500, 10000]

score_viola = []
score_caiman = []

for n, idx in enumerate(idx_list):
    ax[n].plot(normalize(v_t[idx]), 'c', linewidth=0.5, color='orange', label='volpy')
    ax[n].plot(normalize(vi_t[idx]), 'c', linewidth=0.5, color='blue', label='fiola')
    height = np.max([np.max(normalize(vi_t[idx][np.arange(scope[0], scope[1])])), 
                    np.max(normalize(v_t[idx][np.arange(scope[0], scope[1])]))])
    ax[n].vlines(v_spikes[idx], height+2.5, height+3.5, color='orange')
    ax[n].vlines(vi_spikes[idx], height+1, height+2, color='blue')

    if n<length-1:
        ax[n].get_xaxis().set_visible(False)
        ax[n].spines['right'].set_visible(False)
        ax[n].spines['top'].set_visible(False) 
        ax[n].spines['bottom'].set_visible(False) 
        ax[n].spines['left'].set_visible(False) 
        ax[n].set_yticks([])
    
    if n==length-1:
        ax[n].legend(frameon=False)
        ax[n].spines['right'].set_visible(False)
        ax[n].spines['top'].set_visible(False)  
        ax[n].spines['left'].set_visible(False) 
        ax[n].spines['bottom'].set_visible(False) 
        ax[n].get_xaxis().set_visible(False)
        ax[n].set_yticks([])
        ax[n].hlines(height-0.5, scope[0], scope[0]+(scope[1]-scope[0])/5)
        ax[n].text(scope[0],height-1.5, '1s')

    ax[n].set_ylabel('o')
    ax[n].get_yaxis().set_visible(True)
    ax[n].yaxis.label.set_color(colorsets[np.mod(n,9)])
    ax[n].set_xlim(scope)
    
    
plt.tight_layout()
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1/Fig6'
#plt.savefig(os.path.join(save_folder, name + '.pdf'))
#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/06152017Fish1-2.pdf')


    
        

