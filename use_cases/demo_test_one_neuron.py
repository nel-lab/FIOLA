#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:11:50 2020
Test on neuron
@author: @ caichangjia
"""
import h5py
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
from fiola.utilities import normalize, metric, match_spikes_greedy, compute_F1, load
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
import scipy.io
#sys.path.append('/home/nel/Code/NEL_LAB/VIOLA/use_cases')
#sys.path.append(os.path.abspath('/Users/agiovann/SOFTWARE/VIOLA'))
from use_cases.test_run_fiola import run_fiola # must be in use_cases folder
        
#%%
ROOT_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron'
names = ['454597_Cell_0_40x_patch1', '456462_Cell_3_40x_1xtube_10A2',
         '456462_Cell_3_40x_1xtube_10A3', '456462_Cell_5_40x_1xtube_10A5',
         '456462_Cell_5_40x_1xtube_10A6', '456462_Cell_5_40x_1xtube_10A7', 
         '462149_Cell_1_40x_1xtube_10A1', '462149_Cell_1_40x_1xtube_10A2',
         '456462_Cell_4_40x_1xtube_10A4', '456462_Cell_6_40x_1xtube_10A10',
         '456462_Cell_5_40x_1xtube_10A8', '456462_Cell_5_40x_1xtube_10A9', 
         '462149_Cell_3_40x_1xtube_10A3', '466769_Cell_2_40x_1xtube_10A_6',
         '466769_Cell_2_40x_1xtube_10A_4', '466769_Cell_3_40x_1xtube_10A_8', 
         '09282017Fish1-1', '10052017Fish2-2', 'Mouse_Session_1']

frate_all = np.array([400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 400.8 , 995.02, 400.8 ,
       400.8 , 400.8 , 400.8 , 400.8 , 995.02, 995.02, 995.02, 995.02,
       300.  , 300.  , 500.  ])

freq_all = np.array([15]*16 + [5,5,15])

init_frames_all = np.array([20000]*16 + [15000]*3)
 

#%%
#select = np.array([3])
#select = np.array(range(len(names)))[:16]
select = np.array(range(len(names)))

for idx, name in enumerate(np.array(names)[select]):
    num_frames_init = init_frames_all[select][idx]
    #num_frames_init = 30000
    
    border_to_0 = 0
    flip = True
    thresh_range= [2.8, 5.0]
    erosion=0 
    use_rank_one_nmf=False
    hals_movie='hp_thresh'
    semi_nmf=False
    update_bg = False
    use_spikes= False
    use_batch=True
    batch_size=100
    center_dims=None
    initialize_with_gpu=True
    do_scale = False
    adaptive_threshold=True
    filt_window=[8, 4]
    minimal_thresh=2.8
    template_window=2
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
        'minimal_thresh': minimal_thresh,
        'template_window':template_window,
        'freq':freq,
        'do_plot':do_plot, 
        'step':step}

    
    fr = np.array(frate_all)[select][idx]
    fnames = os.path.join(ROOT_FOLDER, name, name+'_mc.tif')  # files are motion corrected before
    path_ROIs = os.path.join(ROOT_FOLDER, name, name+'_ROI.hdf5')
    
    from skimage.io import imread
    if '.hdf5' in fnames:
        with h5py.File(fnames,'r') as h5:
            mov = np.array(h5['mov'])
    elif '.tif' in fnames:
        mov = imread(fnames)    
    else:
        print('do not support this movie format')
    print(mov.shape)
    #run_viola(fnames, path_ROIs, fr=fr, online_gpu=True, options=options)

#%%
for idx, name in enumerate(np.array(names)):
    try:
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'meanroi'))
        print('make folder')
    except:
        print('already exist')



#%%
modes = ['viola', 'volpy', 'meanroi', 'meanroi_online']
mode = modes[3]
result = {}
select = np.array(range(19))
#for mode in modes:
result_threshold = {}
thresh_list = np.arange(2.0, 4.1, 0.1)
#for thresh in thresh_list:    
print(f'threshold level {thresh}')
f1_scores = []                
prec = []
rec = []
thr = []
spnr_all = []
for idx, name in enumerate(np.array(names)[select]):
    gt_path = os.path.join(ROOT_FOLDER, name, name+'_output.npz')
    dict1 = np.load(gt_path, allow_pickle=True)
    length = dict1['v_sg'].shape[0] 
    print(length)
    
    if mode == 'viola':
        vi_folder = os.path.join(ROOT_FOLDER, name, 'viola')
        vi_files = sorted([file for file in os.listdir(vi_folder) if 'filt_window' not in file and 'v2.1' in file])# and '24000' in file])
        #if len(vi_files) == 0:
        #    vi_files = sorted([file for file in os.listdir(vi_folder) if 'v2.0' in file and 'thresh_factor' in file])# and '24000' in file])
        print(f'files number: {len(vi_files)}')
        if len(vi_files) != 1:
            raise Exception('file number greater than 1')
            vi_files = [file for file in vi_files if '15000' in file]
        vi_file = vi_files[0]
        vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
        
        vi_spatial = vi.H.copy()
        vi_temporal = vi.t_s.copy().flatten()
        vi_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vi.index])[np.argsort(vi.seq)][0]
        thr.append(vi.thresh_factor[0])
        
        n_cells = 1
        vi_result = {'F1':[], 'precision':[], 'recall':[]}        
        rr = {'F1':[], 'precision':[], 'recall':[]}        
        vi_spikes = np.delete(vi_spikes, np.where(vi_spikes >= dict1['v_t'].shape[0])[0])        
        dict1_v_sp_ = dict1['v_t'][vi_spikes]

    elif mode == 'volpy':
        v_folder = os.path.join(ROOT_FOLDER, name, 'volpy')
        v_files = sorted([file for file in os.listdir(v_folder)])
        print(f'files number: {len(v_files)}')
        v_file = v_files[0]
        v = np.load(os.path.join(v_folder, v_file), allow_pickle=True).item()
        
        v_spatial = v['weights'][0]
        v_temporal = v['ts'][0]
        v_spikes = v['spikes'][0]        
        v_spikes = np.delete(v_spikes, np.where(v_spikes >= dict1['v_t'].shape[0])[0])
        dict1_v_sp_ = dict1['v_t'][v_spikes]
    
    elif mode == 'meanroi':
        mov = load(os.path.join(ROOT_FOLDER, name, name+'_mc.tif'))
        spatial = load(os.path.join(ROOT_FOLDER, name, name+'_ROI.hdf5'))#.squeeze()
        spatial = spatial.squeeze()[None, :, :]
        mov = mov.reshape([mov.shape[0], -1], order='F')
        spatial_F = [np.where(sp.reshape(-1, order='F')>0) for sp in spatial]
        t_temporal = np.array([-mov[:, sp].mean((1,2)) for sp in spatial_F])
        t_spatial = spatial
        
        t_temporal_p = signal_filter(t_temporal, freq=freq_all[select][idx], fr=frate_all[select][idx])
        t_temporal_p[:, :30] = 0
        t_temporal_p[:, -30:] = 0  
        v_temporal = t_temporal_p.squeeze()              
        #thresh = 3
        t_spikes = np.array(extract_spikes(t_temporal_p, threshold=thresh)).squeeze()
        t_spikes = np.delete(t_spikes, np.where(t_spikes >= dict1['v_t'].shape[0])[0])
        dict1_v_sp_ = dict1['v_t'][t_spikes]
    
    elif mode == 'meanroi_online':
        mov = load(os.path.join(ROOT_FOLDER, name, name+'_mc.tif'))
        spatial = load(os.path.join(ROOT_FOLDER, name, name+'_ROI.hdf5'))#.squeeze()
        spatial = spatial.squeeze()[None, :, :]
        mov = mov.reshape([mov.shape[0], -1], order='F')
        spatial_F = [np.where(sp.reshape(-1, order='F')>0) for sp in spatial]
        t_temporal = np.array([-mov[:, sp].mean((1,2)) for sp in spatial_F])
        t_spatial = spatial
        from fiola.signal_analysis_online import SignalAnalysisOnlineZ
        init_frames = init_frames_all[idx]
        saoz = SignalAnalysisOnlineZ(mode='voltage', window=10000, step=5000, detrend=True, flip=False,
                                     do_scale=False, template_window=2, robust_std=False, adaptive_threshold=True, fr=frate_all[idx], freq=freq_all[idx], 
                                     minimal_thresh=3.0, online_filter_method = 'median_filter', filt_window = 15, do_plot=False)               
        saoz.fit(t_temporal[:, :init_frames], num_frames=t_temporal.shape[1])
        for n in range(init_frames, t_temporal.shape[1]):
            saoz.fit_next(t_temporal[:, n: n+1], n)
        t_spikes = np.unique(saoz.index)
        t_spikes = t_spikes[1:] # remove the first one
        t_spikes = np.delete(t_spikes, np.where(t_spikes >= dict1['v_t'].shape[0])[0])        
        dict1_v_sp_ = dict1['v_t'][t_spikes]
        v_temporal = saoz.t_s[0]
    
    if 'Cell' in name:
        for i in range(len(dict1['sweep_time']) - 1):
            dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([np.logical_and(dict1_v_sp_>dict1['sweep_time'][i][-1], dict1_v_sp_<dict1['sweep_time'][i+1][0])])[1])
        dict1_v_sp_ = np.delete(dict1_v_sp_, np.where([dict1_v_sp_>dict1['sweep_time'][i+1][-1]])[1])
    
    precision, recall, F1, sub_corr, e_match, v_match, mean_time, e_spike_aligned, v_spike_aligned, spnr\
                        = metric(name, dict1['sweep_time'], dict1['e_sg'], 
                              dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                              v_temporal, dict1_v_sp_ , 
                              dict1['v_t'], dict1['v_sub'],init_frames=init_frames_all[select][idx], save=False)
        
    p = len(e_match)/len(v_spike_aligned)
    r = len(e_match)/len(e_spike_aligned)
    f = (2 / (1 / p + 1 / r))

    f1_scores.append(f)                
    prec.append(p)
    rec.append(r)
    spnr_all.append(spnr)
#result_threshold[str(thresh)] = {'f1':f1_scores, 'precision':prec, 'recall':rec, 'spnr':spnr_all}
result[mode] = {'f1':f1_scores, 'precision':prec, 'recall':rec, 'spnr':spnr_all}

#%%
plt.plot(f1_scores);plt.plot(prec);plt.plot(rec);plt.legend(['f1','prec', 'rec','threshold'])
plt.figure();plt.plot(result['viola']['f1']);plt.plot(result['viola']['precision']);plt.plot(result['viola']['recall']);plt.legend(['f1','prec', 'rec','threshold'])

plt.plot(thr)
print(np.array(f1_scores).mean())
print(np.array(prec).mean())
print(np.array(rec).mean())
plt.bar(range(16), f1_scores)

#%%
for key in result_threshold.keys():
    print(key)
    print(np.array(result_threshold[key]['f1']).mean())

result_threshold['2.900000000000001']


#%%
SAVE_FOLDER = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron'
#np.save(os.path.join(SAVE_FOLDER, f'viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500_filt_window_15_template_window_2'), result)
#np.save(os.path.join(SAVE_FOLDER, f'viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500_filt_window_15_template_window_2'), result)
np.save(os.path.join(SAVE_FOLDER, f'mean_roi_threshold_2.9'), result_threshold['2.900000000000001'])
np.save(os.path.join(SAVE_FOLDER, f'mean_roi_online_v3.0'), result)


   
#%%
sp = spnr_all.copy()
len(sp)

np.save(os.path.join(SAVE_FOLDER, f'viola_spnr'), spnr_all)


#%%
plt.plot(dict1['e_t'], dict1['e_sg'])
plt.vlines(dict1['e_sp'], -30, 20)

#%%
plt.plot(dict1['v_t'], vi_temporal)
plt.plot(vi.thresh[0])

#%%
plt.plot(dict1['v_t'], vi_temporal)
plt.vlines(dict1['e_sp'], 0, np.max(vi_temporal))

#%%
plt.imshow(vi_spatial.reshape([30,30], order='F'))

mm = mov.reshape([50000, 900], order='F').copy()
ss = vi_spatial.reshape([30,30], order='F').copy()
tt = mm[:, vi_spatial[:,0]>0]
tt = tt.mean(1)
plt.plot(tt)

from scipy.ndimage import median_filter
plt.figure(); plt.plot(tt);
mm =  median_filter(tt, 15)
plt.plot(mm)

#%%
plt.figure(); plt.plot(dict1['v_t'], normalize(-(tt - mm)))
plt.plot(dict1['v_t'], normalize(vi_temporal))
plt.vlines(dict1['e_sp'], 0, 15)

#%%
a = np.matmul(mm, vi_spatial)
am = median_filter(a, 15)
plt.plot(dict1['v_t'], normalize(-(a-am)))
#%%
with h5py.File('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/456462_Cell_5_40x_1xtube_10A5/456462_Cell_5_40x_1xtube_10A5_ROI_2_neurons.hdf5','r') as h5:
    RR = np.array(h5['mov'])
plt.figure(); plt.imshow(RR[0])
plt.figure(); plt.imshow(RR[1])

#%%
RR = RR.reshape([2, 900], order='F')
t1 = mm[:, RR[0]>0].copy()
t2 = mm[:, RR[1]>0].copy()
t1 = t1.mean(1)
t2 = t2.mean(1)

m1 =  median_filter(t1, 15)
m2 =  median_filter(t2, 15)

#%%
plt.figure();plt.plot(dict1['v_t'], normalize(vi.t[0]))
plt.figure(); plt.plot(dict1['v_t'], normalize(-(tt - mm)))
plt.figure();plt.plot(dict1['v_t'], normalize(-(t1-m1)))
plt.figure();plt.plot(dict1['v_t'], normalize(-(t2-m2)))
plt.vlines(dict1['e_sp'], 12, 15)
#%%
step = 2500
plt.plot(vi.t_s[0])
#plt.vlines(dict1['e_sp'], 800, 1000)
for idx, tt in enumerate(vi.thresh[0]):
    if idx == 0:
        plt.hlines(tt, 0, 30000)
    else:
        plt.hlines(tt, 30000 + (idx -1) * step, 30000 + idx * step)

#%%
from viola.caiman_functions import play
play(mov, magnification=3)
#%%
for idx, name in enumerate(np.array(names)[select]):
    gt_path = os.path.join(ROOT_FOLDER, name, name+'_output.npz')
    dict1 = np.load(gt_path, allow_pickle=True)
    length = dict1['v_sg'].shape[0]    
    
    vi_folder = os.path.join(ROOT_FOLDER, name, 'viola')
    vi_files = sorted([file for file in os.listdir(vi_folder) if 'v2.0' in file and 'thresh_factor' not in file])# and '24000' in file])
    print(f'files number: {len(vi_files)}')
    if len(vi_files) > 1:
        vi_files = [file for file in vi_files if 'freq_15' in file]
    vi_file = vi_files[0]
    vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
    
    vi_spatial = vi.H.copy()
    vi_temporal = vi.t_s.copy()
    
    step = 5000
    plt.figure()
    plt.plot(vi.t_s[0])
    plt.title(idx)
    for idx, tt in enumerate(vi.thresh[0]):
        if idx == 0:
            plt.hlines(tt, 0, 30000)
        else:
            plt.hlines(tt, 30000 + (idx -1) * step, 30000 + idx * step)
            
#%%            
    
    
#%%
print(f'average_F1:{np.mean([np.nanmean(fsc) for fsc in all_f1_scores])}')
#print(f'average_sub:{np.nanmean(all_corr_subthr,axis=0)}')
print(f'F1:{np.array([np.nanmean(fsc) for fsc in all_f1_scores]).round(2)}')
print(f'prec:{np.array([np.nanmean(fsc) for fsc in all_prec]).round(2)}'); 
print(f'rec:{np.array([np.nanmean(fsc) for fsc in all_rec]).round(2)}')
print(f'average_compound_f1:{np.mean(np.array(compound_f1_scores)).round(3)}')
print(f'compound_f1:{np.array(compound_f1_scores).round(2)}')
print(f'compound_prec:{np.array(compound_prec).round(2)}')
print(f'compound_rec:{np.array(compound_rec).round(2)}')
print(f'snr:{np.array(all_snr).round(2)}')
dict2 = {}
dict2['trace'] = saoz.trace
dict2['indexes'] = sorted(indexes)
dict2['t_s'] = saoz.t_s
dict2['snr'] = saoz.SNR
dict2['sub'] = saoz.sub
dict2['template'] = saoz.PTA
dict2['thresh'] = saoz.thresh
dict2['thresh_factor'] = saoz.thresh_factor
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result'
np.save(os.path.join(save_folder, 'spike_detection_saoz_'+ name[:-7]  +'_output'), dict2)

#%%  
for idx, dist in enumerate(distance):
    vi_result_all = []
    if overlap == True:
        select = np.arange(idx * 3, (idx + 1) * 3)
    else:
        select = np.array(range(len(names)))
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
        gt_file = gt_files[0]
        gt = scipy.io.loadmat(os.path.join(folder, gt_file))
        gt = gt['simOutput'][0][0]['gt']
        spikes = gt['ST'][0][0][0]
        
        vi_folder = os.path.join(folder, 'viola')
        vi_files = sorted([file for file in os.listdir(vi_folder) if 'viola' in file and 'use_spikes_False' in file])
        vi_file = vi_files[0]
        vi = np.load(os.path.join(vi_folder, vi_file), allow_pickle=True).item()
        
        vi_spatial = vi.H.copy()
        vi_temporal = vi.t_s.copy()
        vi_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vi.index])[np.argsort(vi.seq)]
        
        n_cells = spikes.shape[0]
        vi_result = {'F1':[], 'precision':[], 'recall':[]}        
        rr = {'F1':[], 'precision':[], 'recall':[]}
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
            s2 = vi_spikes[idx]
            s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall, _, _, _ = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        vi_result_all.append(rr)

    np.save(os.path.join(SAVE_FOLDER, f'viola_result_10000_20000_{dist}'), vi_result_all)
    #np.save(os.path.join(ROOT_FOLDER, 'result_overlap', f'{dist}', f'volpy_{dist}_thresh_adaptive'), vi_save_result)

#%%
hh = vi.H.reshape((100,100,9), order='F')   
plt.imshow(hh[:,:,3]) 

plt.plot(vi_temporal[9])   



#%%
for idx, dist in enumerate(distance):  
    v_result_all = []
    if overlap == True:
        select = np.arange(idx * 3, (idx + 1) * 3)
    else:
        select = np.array(range(len(names)))
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
        gt_file = gt_files[0]
        gt = scipy.io.loadmat(os.path.join(folder, gt_file))
        gt = gt['simOutput'][0][0]['gt']
        spikes = gt['ST'][0][0][0]
        
        #summary_file = os.listdir(os.path.join(folder, 'volpy'))
        #summary_file = [file for file in summary_file if 'summary' in file][0]
        #summary = cm.load(os.path.join(folder, 'volpy', summary_file))
        
        v_folder = os.path.join(folder, 'volpy')
        v_files = sorted([file for file in os.listdir(v_folder) if 'adaptive_threshold' in file])
        v_file = v_files[0]
        v = np.load(os.path.join(v_folder, v_file), allow_pickle=True).item()
        
        v_spatial = v['weights'].copy()
        v_temporal = v['ts'].copy()
        v_ROIs = v['ROIs'].copy()
        v_ROIs = v_ROIs * 1.0
        v_templates = v['templates'].copy()
        v_spikes = v['spikes'].copy()
        
        n_cells = v_spatial.shape[0]
        v_result = {'F1':[], 'precision':[], 'recall':[]}        
        rr = {'F1':[], 'precision':[], 'recall':[]}
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s1 = s1[np.logical_and(s1>t_range[0], s1<t_range[1])]
            s2 = v_spikes[idx]
            s2 = s2[np.logical_and(s2>t_range[0], s2<t_range[1])]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall, _, _, _ = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        v_result_all.append(rr)

    np.save(os.path.join(SAVE_FOLDER, f'volpy_adaptive_threshold_10000_20000_{dist}'), v_result_all)

    
    #np.save(os.path.join(ROOT_FOLDER, 'result_overlap', f'{dist}', f'volpy_{dist}_thresh_adaptive'), v_save_result)

#%%
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping'
files = np.array(sorted(os.listdir(result_folder)))
files = np.array(['viola_result_10000_20000.npy', 'volpy_adaptive_threshold_10000_20000.npy'])
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
for idx, results in enumerate(result_all):
    try:
        if idx == 0:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15', linewidth = 3)
        else:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15')
    except:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='.', markersize='15')    

    plt.legend(['Viola', 'VolPy'])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score for non-overlapping neurons')

plt.savefig(os.path.join(SAVE_FOLDER, 'F1_Viola_vs_VolPy_10000_20000.pdf'))

#%% Overlapping neurons for volpy with different overlapping areas
area = []

for idx, dist in enumerate(distance):  
    select = np.arange(idx * 3, (idx + 1) * 3)
    v_result_all = []
    #names = [f'sim3_{i}' for i in range(10, 17)]
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
        gt_file = gt_files[0]
        gt = scipy.io.loadmat(os.path.join(folder, gt_file))
        gt = gt['simOutput'][0][0]['gt']
        spikes = gt['ST'][0][0][0]
        spatial = gt['IM2'][0][0]
        temporal = gt['trace'][0][0][:,:,0]
        spatial[spatial <= np.percentile(spatial, 99.5)] = 0 # just empirical
        spatial[spatial > 0] = 1
        spatial = spatial.transpose([2, 0, 1])
        percent = (np.where(spatial.sum(axis=0) > 1)[0].shape[0])/(np.where(spatial.sum(axis=0) > 0)[0].shape[0])
        area.append(percent)   
        
area = np.round(np.unique(area), 2)[::-1]
area = np.append(area, 0)

#%%
mode = ['viola', 'volpy'][0]
result_all = {}
distance = [f'dist_{i}' for i in [1, 3, 5, 7, 10, 15]]
x = [round(0.075 + 0.05 * i, 3) for i in range(3)] 
for dist in distance:   
    files = np.array(sorted(os.listdir(SAVE_FOLDER)))#[0]#[np.array([5, 0, 1, 2, 3, 4, 6])]
    files = [file for file in files if mode in file and dist+'.npy' in file]
    print(len(files))
    for file in files:
        result_all[file] = np.load(os.path.join(SAVE_FOLDER, file), allow_pickle=True)
    
for idx, key in enumerate(result_all.keys()):
    results = result_all[key]
    #print(results)
    if mode in key:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
                 marker='.',markersize=15, label=f'{mode}_{area[idx]:.0%}_{distance[idx]}')
    #elif 'volpy' in key:
    #    plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], 
    #             marker='^',markersize=15, label=f'volpy_{area[idx]:.0%}')


    plt.legend()
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title(f'{mode} F1 score with different overlapping areas')
    
plt.savefig(os.path.join(SAVE_FOLDER, f'F1_overlapping_{mode}.pdf'))




#%%
import scipy.io 
folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/test'
gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
gt_file = gt_files[0]
gt = scipy.io.loadmat(os.path.join(folder, gt_file))
gt = gt['simOutput'][0][0]['gt']
spikes = gt['ST'][0][0][0]
spatial = gt['IM2'][0][0]
temporal = gt['trace'][0][0][:,:,0]
spatial[spatial <= np.percentile(spatial, 99.6)] = 0
spatial[spatial > 0] = 1
spatial = spatial.transpose([2, 0, 1])
#np.save(os.path.join(folder, 'ROIs.npy'), spatial)



#%% resutl is stored in vio.pipeline.saoz object
#plt.plot(vio.pipeline.saoz.t_s[1][:scope[1]])
for i in range(vio.H.shape[1]):
    if i == 3:
        plt.figure()
        plt.imshow(mov[0], cmap='gray')
        plt.imshow(vio.H.reshape((100, 100, vio.H.shape[1]), order='F')[:,:,i], alpha=0.3)
        plt.figure()
        plt.plot(vio.pipeline.saoz.t_s[i][:scope[1]])

#%%
np.savez(os.path.join(folder, 'viola_result'), vio)        
#%%
from matplotlib.widgets import Slider
def view_components(estimates, img, idx, frame_times=None, gt_times=None):
    """ View spatial and temporal components interactively
    Args:
        estimates: dict
            estimates dictionary contain results of VolPy
            
        img: 2-D array
            summary images for detection
            
        idx: list
            index of selected neurons
    """
    n = len(idx) 
    fig = plt.figure(figsize=(10, 10))

    axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])
    ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
    ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
    ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])    
    s_comp = Slider(axcomp, 'Component', 0, n, valinit=0)
    vmax = np.percentile(img, 98)
    if frame_times is not None:
        pass
    else:
        frame_times = np.array(range(len(estimates.t0[0])))
    
    def arrow_key_image_control(event):

        if event.key == 'left':
            new_val = np.round(s_comp.val - 1)
            if new_val < 0:
                new_val = 0
            s_comp.set_val(new_val)

        elif event.key == 'right':
            new_val = np.round(s_comp.val + 1)
            if new_val > n :
                new_val = n  
            s_comp.set_val(new_val)
        
    def update(val):
        i = np.int(np.round(s_comp.val))
        print(f'Component:{i}')

        if i < n:
            
            ax1.cla()
            imgtmp = estimates.weights[idx][i]
            ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
            ax1.set_title(f'Spatial component {i+1}')
            ax1.axis('off')
            
            ax2.cla()
            ax2.plot(frame_times, estimates.t0[idx][i], alpha=0.8)
            ax2.plot(frame_times, estimates.t_sub[idx][i])            
            ax2.plot(frame_times, estimates.t_rec[idx][i], alpha = 0.4, color='red')
            ax2.plot(frame_times[estimates.spikes[idx][i]],
                     1.05 * np.max(estimates.t0[idx][i]) * np.ones(estimates.spikes[idx][i].shape),
                     color='r', marker='.', fillstyle='none', linestyle='none')
            if gt_times is not None:
                ax2.plot(gt_times,
                     1.15 * np.max(estimates['t'][idx][i]) * np.ones(gt_times.shape),
                     color='blue', marker='.', fillstyle='none', linestyle='none')
                ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes', 'gt_spikes'])
            else:
                ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes'])
            ax2.set_title(f'Signal and spike times {i+1}')
            #ax2.text(0.1, 0.1, f'snr:{round(estimates["snr"][idx][i],2)}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
            #ax2.text(0.1, 0.07, f'num_spikes: {len(estimates["spikes"][idx][i])}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            #ax2.text(0.1, 0.04, f'locality_test: {estimates["locality"][idx][i]}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            
            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None',
                       alpha=0.5, cmap=plt.cm.hot)
            ax3.axis('off')
            
    s_comp.on_changed(update)
    s_comp.set_val(0)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    plt.show()     

vio.pipeline.saoz.reconstruct_signal()
estimates = vio.pipeline.saoz
estimates.spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in vio.pipeline.saoz.index])
weights = vio.H.reshape((100, 100, vio.H.shape[1]), order='F')
weights = weights.transpose([2, 0, 1])
estimates.weights = weights
#idx = list(range(len(estimates.t_s)))
idx = np.argsort(vio.seq)


view_components(estimates, img_mean, idx = list(range(51)))
   
        
        
#%%
from match_spikes import match_spikes_greedy, compute_F1
rr = {'F1':[], 'precision':[], 'recall':[]}
for idx in range(n_cells):
    s1 = spikes[idx].flatten()
    s2 = estimates.spikes[np.argsort(vio.seq)][idx]
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
    rr['F1'].append(F1)
    rr['precision'].append(precision)
    rr['recall'].append(recall)  
    
plt.boxplot(rr['F1']); plt.title('viola')
plt.boxplot(rr['precision'])
plt.boxplot(rr['recall'])
        
        
#%%
vpy = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/simulation/test/volpy_viola_sim1_1_adaptive_threshold.npy', allow_pickle=True).item()        
rr = {'F1':[], 'precision':[], 'recall':[]}
for idx in range(n_cells):
    s1 = spikes[idx].flatten()
    s2 = vpy['spikes'][idx]
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
    rr['F1'].append(F1)
    rr['precision'].append(precision)
    rr['recall'].append(recall)  

plt.boxplot(rr['F1']); plt.title('volpy')
plt.boxplot(rr['precision'])
plt.boxplot(rr['recall'])
        
#%%
from spikepursuit import signal_filter     
for idx in [1]:
    plt.plot(normalize(vpy['t'][idx]))
    sg = estimates.trace[np.argsort(vio.seq)][idx]
    #sgg  = signal_filter(sg, 1/3, 400, order=3, mode='high')   
    plt.plot(normalize(sg))
    
#%% adaptive threshold on viola traces
from spikepursuit import denoise_spikes
viola_trace = estimates.t0[np.argsort(vio.seq)].copy()   
rr = {'F1':[], 'precision':[], 'recall':[]}

for idx in range(n_cells):
    s1 = spikes[idx].flatten()
    s2 = denoise_spikes(viola_trace[idx], window_length=8, threshold_method='adaptive_threshold')[1]   
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
    rr['F1'].append(F1)
    rr['precision'].append(precision)
    rr['recall'].append(recall)  
    
plt.boxplot(rr['F1']); plt.title('adaptive threshold on viola trace')
   
        
#%% saoz on volpy traces
from signal_analysis_online import SignalAnalysisOnlineZ
saoz = SignalAnalysisOnlineZ(window = 10000, step = 5000, detrend=False, flip=False, 
                 do_scale=False, robust_std=False, frate=400, freq=15, 
                 thresh_range=[3.2, 4], mfp=0.2, online_filter_method = 'median_filter',
                 filt_window = 9, do_plot=False)

saoz.fit(vpy['t'][:,:5000], 20000)
for n in range(5000, 20000):
    saoz.fit_next(vpy['t'][:, n:n+1], n)

viola_spikes = np.array([np.array(sorted(list(set(sp)-set([0])))) for sp in saoz.index])

rr = {'F1':[], 'precision':[], 'recall':[]}
for idx in range(n_cells):
    s1 = spikes[idx].flatten()
    s2 = viola_spikes[idx]
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)   
    rr['F1'].append(F1)
    rr['precision'].append(precision)
    rr['recall'].append(recall)  

plt.boxplot(rr['F1']); plt.title('viola spikes on volpy trace')

        
#%% Clean up folder for one neuron
import shutil
ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron'
ff = os.listdir(ROOT_FOLDER)    
ff = [f for f in ff if '_ROI.hdf5' in f]
for f in ff:
    name = f[:-9]
    folder = os.path.join(ROOT_FOLDER, name)
    try:
        os.makedirs(folder)
        print('make folder')
    except:
        print('already exist')
        
    files = [file for file in os.listdir(ROOT_FOLDER) if name in file and os.path.isfile(os.path.join(ROOT_FOLDER, file))]
    for file in files:
        shutil.move(os.path.join(ROOT_FOLDER, file), os.path.join(ROOT_FOLDER, name, file))
        
        
        
#%%
for name in names:
    try:
        #os.makedirs(os.path.join(ROOT_FOLDER, name, 'viola'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'volpy'))
        print('make folder')
    except:
        print('already exist')
        
#%%
import shutil
GT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron_result'
folders = os.listdir(ROOT_FOLDER)
folders = [folder for folder in folders if os.path.isdir(os.path.join(ROOT_FOLDER, folder))]
GT_FILES = os.listdir(GT_FOLDER)

for folder in folders:
    for file in GT_FILES:
        if folder in file:
            shutil.move(os.path.join(GT_FOLDER, file), os.path.join(ROOT_FOLDER, folder, file))
            print(f'{os.path.join(GT_FOLDER, file)}')
            print(f'{os.path.join(ROOT_FOLDER, folder, file)}')
            
#%% run volpy 
            
            
        
    
    
    
    
        
        
        
        
        
        