#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:37:36 2021

@author: nel
"""

import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
from pynwb import NWBHDF5IO
import pyximport
pyximport.install()
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from caiman.source_extraction.cnmf.utilities import fast_prct_filt

#from fiola.config import load_fiola_config, load_caiman_config
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import bin_median, to_2D
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import sys
sys.path.append('/home/nel/CODE/VIOLA/paper_reproduction')

from Fig7.Fig7_caiman_pipeline import run_caiman_fig7

#base_folder = '/media/nel/storage/fiola/F2_20190415'
#base_folder = '/media/nel/storage/fiola/R6_20200210T2100'
base_folder = '/media/nel/storage/fiola/R2_20190219'
savef = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig7/'
for file in os.listdir(base_folder):
    if '.nwb' in file:
        path = os.path.join(base_folder, file)
        print(path)

io = NWBHDF5IO(path, 'r')
nwbfile_in = io.read()
pos = nwbfile_in.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['pos'].data[:]
speed = nwbfile_in.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['speed'].data[:]

pos_n = pos.copy()
for i in range(len(pos_n)):
    if pos_n[i] < -2:
        j = i
        flag = 1
        while flag:
            j += 1
            try:
                if pos_n[j] >= -0.5:
                    pos_n[i] = pos_n[j]
                    flag = 0
            except:
                pos_n[i] = 0
                flag = 0
                
pos_s = StandardScaler().fit_transform(pos_n[:, None])[:, 0]
spd_s = StandardScaler().fit_transform(speed[:, None])[:, 0]

trace0 = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['Deconvolved'].data[:]
trace = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries'].data[:].T
                
def remove_neurons_with_sparse_activities(trace, do_remove=True, std_level=5, timepoints=10):
    from sklearn.preprocessing import StandardScaler
    print(f'original trace shape {trace.shape}')
    trace_s = StandardScaler().fit_transform(trace)
    if do_remove:
        select = []
        high_act = []
        for idx in range(len(trace_s.T)):
            t = trace_s[:, idx]
            select.append(len(t[t>t.std() * std_level]) > timepoints)
            high_act.append(len(t[t>t.std() * std_level]))
        #plt.plot(trace_s[:, ~np.array(select)][:, 15])
        trace_s = trace_s[:, select]
        high_act = np.array(high_act)[select]
        sort = np.argsort(high_act)[::-1]
        trace_s = trace_s[:, sort]
    print(f'after removing neurons trace shape {trace_s.shape}')
    return trace_s, select

def sort_neurons(trace, std_level=5, timepoints=10):
    from sklearn.preprocessing import StandardScaler
    print(f'original trace shape {trace.shape}')
    trace_s = StandardScaler().fit_transform(trace)
    high_act = []
    for idx in range(len(trace_s.T)):
        t = trace_s[:, idx]
        high_act.append(len(t[t>t.std() * std_level]))
    sort = np.argsort(high_act)[::-1]
    return sort

# mean roi
def run_meanroi(mov, cnm, select):
    A = cnm.estimates.A.toarray()
    A = A[:, select]     
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

def run_fiola(mov, cnm, select, fnames='/'):
    # load movie and masks
    mode_idx = 1
    mode = ['voltage', 'calcium'][mode_idx]
    mask = np.hstack([cnm.estimates.A[:, select].toarray(), cnm.estimates.b])    
    #mask = np.hstack([cnm.estimates.A[:, cnm.estimates.idx_components].toarray(), cnm.estimates.b])    
    #mask = mask.reshape([dims[0], dims[1], -1], order='C').transpose([2, 0, 1])
    
    mask = mask.reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1])
    Ab = to_2D(mask, order='F').T
    Ab = Ab / norm(Ab, axis=0)

    options = load_fiola_config(fnames, num_frames_total=30000, mode=mode, mask=mask) 
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    fio.Ab = Ab        

    min_mov = mov[:3000].min()
    template = bin_median(mov[:3000])
    plt.imshow(template)
    trace = fio.fit_gpu_motion_correction_nnls(mov[:30000], template=template, batch_size=5, 
                                min_mov=min_mov, Ab=fio.Ab)
    #trace = fio.fit_gpu_nnls(mov[:31930], fio.Ab, batch_size=5) 
    plt.plot(trace[:-2].T)
    return trace    

# plt.figure(); 
# plt.plot(pos)
# plt.title('pos')
# plt.plot(pos_n) 
# plt.figure()
# plt.plot(speed)
# plt.title('spd')
# 
# mov_100 = nwbfile_in.acquisition['TwoPhotonSeries'].data[:1000]
# #m1 = mov_100.mo
# mov_l_100 = nwbfile_in.acquisition['TwoPhotonSeries'].data[-5000:-4000]
# me = np.median(mov_100, axis=0)
# me1 = np.median(mov_l_100, axis=0)
# mm = cm.concatenate([cm.movie(me)[None, :, :], cm.movie(me1)[None, :, :]])
# mov = cm.movie(mm)
#mov = nwbfile_in.acquisition['TwoPhotonSeries'].data[:]
#cm.movie(mov).save('/media/nel/storage/fiola/R6_20200210T2100/mov_R6__20200210T2100.hdf5')
#mov.save('/media/nel/storage/fiola/mov_R2_20190219T210000.hdf5')
# mov[:1000].save('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_1000.hdf5')
# mov[:5000].save('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_5000.hdf5')
# mov[:7000].save('/media/nel/storage/fiola/R2_20190219/mov_R2_20190219T210000_7000.hdf5')

# mov.save('/media/nel/storage/fiola/mov_R2_20190219T210000.hdf5')

# masks = nwbfile_in.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].columns[0][:]
# centers = nwbfile_in.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].columns[1][:]
# #accepted = nwbfile_in.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].columns[2][:]
# plt.imshow(masks.sum(0))
# plt.scatter(centers[:, 0], centers[:, 1],  s=0.5, color='red')

#path = '/media/nel/storage/fiola/sub-R2_ses-20190206T210000_obj-16d9pmi_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/R2_20190219/sub-R2_ses-20190219T210000_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/sub-R6_ses-20200210T210000_obj-10qdhbr_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/sub-R6_ses-20200206T210000_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/R4_20190904/sub-R4_ses-20190904T210000_behavior+ophys.nwb'
#path = '/media/nel/storage/fiola/F2_20190415/sub-F2_ses-20190415T210000_behavior+ophys.nwb'


#%% save movie 
mov = nwbfile_in.acquisition['TwoPhotonSeries'].data[:]
mov = cm.movie(mov)
try:
    os.mkdir(os.path.join(base_folder, '3000'))
    print('make folder')
except:
    print('fail to make folder')
mov.save(os.path.join(base_folder, 'mov_raw.hdf5'))
mov[:3000].save(os.path.join(base_folder, '3000', 'mov_raw_3000.hdf5'))

#%% init
fnames_init = os.path.join(base_folder, '5000', 'mov_raw_5000.hdf5')
fnames_all = os.path.join(base_folder, 'mov_raw.hdf5')
run_caiman_fig7(fnames_init, pw_rigid=True, motion_correction_only=False, K=5)
run_caiman_fig7(fnames_all, pw_rigid=False, motion_correction_only=True)
run_caiman_fig7(fnames_all, pw_rigid=True, motion_correction_only=False, K=8)

#%%
from sklearn.preprocessing import StandardScaler
#caiman_output_path = os.path.join(base_folder, '5000', [file for file in os.listdir(base_folder+'/5000') if 'memmap_pw_rigid_True' in file and '.hdf5' in file][0])
caiman_output_path = '/media/nel/storage/fiola/R2_20190219/3000/memmap_pw_rigid_True_d1_796_d2_512_d3_1_order_C_frames_3000_non_rigid_K_5.hdf5'
cnm = load_CNMF(caiman_output_path)
dims = (796, 512)
A = cnm.estimates.A[:, cnm.estimates.idx_components].toarray()
#A = cnm.estimates.A[:, :2000].toarray()
noisy_C = cnm.estimates.C + cnm.estimates.YrA
noisy_C = noisy_C.T
noisy_C = StandardScaler().fit_transform(noisy_C)
# select = []
# high_act = []
# std_level = 5; timepoints = 10
# for idx in range(len(noisy_C.T)):
#     t = noisy_C[:, idx]
#     select.append(len(t[t>t.std() * std_level]) > timepoints)
#     high_act.append(len(t[t>t.std() * std_level]))
noisy_C_selected, select = remove_neurons_with_sparse_activities(noisy_C)
select = np.zeros(cnm.estimates.C.shape[0])
select[cnm.estimates.idx_components] = 1
select = select.astype(bool)
if np.array(select).sum()>1300:
    raise Exception('Too many components for fiola')

#%%
trace = run_fiola(mov.astype(np.float32), cnm=cnm, select=select)
np.save(os.path.join(base_folder, '3000/fiola_non_rigid_init.npy'), trace)

#%%
mmap_file = os.path.join(base_folder, [file for file in os.listdir(base_folder) if '.mmap' in file][0])
mov = cm.load(mmap_file, in_memory=True)
#mov = mov.reshape([mov.shape[0], 796, 512], order='F')
mov = mov.reshape([mov.shape[0], -1], order='F')
tracem = run_meanroi(mov, cnm, select)
np.save(os.path.join(base_folder,'3000/mean_roi_99.98_non_rigid_init.npy'), tracem.T)

#%%

plt.imshow(aaa, vmax=np.percentile(aaa, 99), vmin=np.percentile(aaa, 5), cmap='gray')
plt.savefig(savef + 'median_img_3000.pdf')
plt.imshow(cnm.estimates.Cn, vmax=np.percentile(cnm.estimates.Cn, 99), vmin=np.percentile(cnm.estimates.Cn, 5), cmap='gray')
plt.savefig(savef + 'corr_img_3000.pdf')

#%%
from caiman.base.rois import com
centers = com(cnm.estimates.A, 796, 512)

y_range = [400, 500]
x_range = [100, 200]

comp = np.where(np.array(select) == True)[0]

a = cnm.estimates.A.toarray().T.reshape([-1, 796, 512], order='F')
# plt.imshow(a.sum(0))
# plt.scatter(centers[:, 1], centers[:, 0], marker='.')

#%%
comp1 = np.where(np.logical_and(centers[:, 1] > x_range[0], centers[:, 1] <= x_range[1]))[0]
comp2 = np.where(np.logical_and(centers[:, 0] > y_range[0], centers[:, 0] <= y_range[1]))[0]
comp_all = np.intersect1d(np.intersect1d(comp,comp1), comp2)

#%%
trace1 = StandardScaler().fit_transform(trace1)


#%% Fig 7a corr img
Cn = cnm.estimates.Cn
cnm.estimates.plot_contours(img=Cn, idx=comp_all[sort], cmap='gray')
#cnm.estimates.plot_contours(img=Cn, idx=np.array(range(0)), cmap='gray')

plt.subplot(1, 2, 1)
#plt.imshow(Cn)
# plt.plot([x_range[0],x_range[0]], [y_range[0],y_range[1]], 'r-')
# plt.plot([x_range[1],x_range[1]], [y_range[0],y_range[1]], 'r-')
# plt.plot([x_range[0],x_range[1]], [y_range[0],y_range[0]], 'r-')
# plt.plot([x_range[0],x_range[1]], [y_range[1],y_range[1]], 'r-')
plt.xlim(x_range)
plt.ylim(y_range)
plt.savefig(savef+'corr_img_3000_zoom_all.pdf')

#%%
sort = sort_neurons(trace1[:, comp_all])
tt = trace1[:, comp_all][:, sort][:, :15]

#%% Fig 7b example traces

fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
for i in range(tt.shape[1]):
    xx = np.array(list(range(1000))) / 15.46
    plt.plot(xx, tt[3000:4000, i]-i*5, color='black', linewidth=0.3)
    if i % 5 == 0:
        plt.text(-3, -i*5, f'{i+1}')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.set_xlabel('time (s)')
ax1.set_yticks([])
ax1.set_ylabel('neuron index')
    
    
#plt.savefig(savef+'traces_with_neg.pdf')
#plt.plot(trace1[3000:6000, comp_all])

#%%
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_30000.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_3000_nonrigid.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_30000_K_8.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_1285_nonrigid.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_1285_non_rigid_init_non_rigid_movie.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_1285_non_rigid_init_non_rigid_movie_nnls_only.npy')
#trace1 = np.load('/media/nel/storage/fiola/R6_20200210T2100/3000/fiola_non_rigid_init.npy')
#trace1 = np.load('/media/nel/storage/fiola/F2_20190415/3000/fiola_non_rigid_init.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_do_hals.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_do_hals_new.npy')
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/mov_R2_20190219T210000._fiola_result_new.npy')


trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_v3.1.npy', allow_pickle=True).item().trace
trace2 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_result_v3.1.npy', allow_pickle=True).item().trace
trace3 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_result_v3.1.npy', allow_pickle=True).item().trace
trace4 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_result_v3.1.npy', allow_pickle=True).item().trace

# trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_v3.1.npy')
# trace1 = trace1.T
# #trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_10_False.npy')
# #trace1 = trace1.T
# trace2 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_result_v3.1.npy')
# trace2 = trace2.T
# trace3 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_result_v3.1.npy')
# trace3 = trace3.T
# trace4 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_result_v3.1.npy')
# trace4 = trace4.T

# trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_30.npy')
# trace1 = trace1.T
# trace2 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_10_False.npy')
# trace2 = trace2.T
# trace3 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_1_False.npy')
# trace3 = trace3.T
# trace4 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_1.npy')
# trace4 = trace4.T



#tracem = np.load('/media/nel/storage/fiola/R2_20190219/3000/mean_roi_99.98.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_all_masks.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/3000/mean_roi_99.98_non_rigid.npy')
#tracem = tracem.T
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks_selected_1543.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks_selected_2365.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks_selected_1285.npy')
#tracem = np.load('/media/nel/storage/fiola/R2_20190219/3000/mean_roi_99.98_non_rigid_init_non_rigid_1285.npy')
#tracem = np.load('/media/nel/storage/fiola/R6_20200210T2100/3000/mean_roi_99.98_non_rigid_init.npy')
#tracem = np.load('/media/nel/storage/fiola/F2_20190415/3000/mean_roi_99.98_non_rigid_init.npy')
#tracem = tracem.T
#%%
from sklearn.preprocessing import StandardScaler
from caiman.source_extraction.cnmf.cnmf import load_CNMF
#cnm2 = load_CNMF('/media/nel/storage/fiola/R6_20200210T2100/memmap_pw_rigid_True_d1_796_d2_512_d3_1_order_C_frames_31604_non_rigid_K_5.hdf5')
cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8_K_8.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8_cnn_True.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/full/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_.hdf5')
tracec = cnm2.estimates.C[cnm2.estimates.idx_components] + cnm2.estimates.YrA[cnm2.estimates.idx_components]
onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mov_R2_20190219T210000_caiman_online_results_v3.0_new_params.hdf5')
#onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/caiman_online_results.hdf5')
traceo = onacid.estimates.C + onacid.estimates.YrA

#%%
[print(t.shape) for t in [trace, trace1, trace2, trace3, trace4, tracec, traceo]]
trace = fast_prct_filt(trace,20).T
trace1 = fast_prct_filt(trace1,20).T
trace2 = fast_prct_filt(trace2,20).T
trace3 = fast_prct_filt(trace3,20).T
trace4 = fast_prct_filt(trace4,20).T
tracec = fast_prct_filt(tracec,20).T
traceo = fast_prct_filt(traceo,20).T

# remove atificial feature for onacid due to filtering
for idx in range(traceo.shape[1]):
    if onacid.time_neuron_added[idx, 1] > 200:
        traceo[:onacid.time_neuron_added[idx, 1]-100, idx] = np.mean(traceo[onacid.time_neuron_added[idx, 1]-100:, idx])

# trace1 = trace1.T
# traceo = traceo.T

#%%
[print(t.shape) for t in [trace, trace1, trace2, trace3, trace4, tracec, traceo]]
trace_s, _ = remove_neurons_with_sparse_activities(trace)
trace1_s, _ = remove_neurons_with_sparse_activities(trace1)
trace2_s, _ = remove_neurons_with_sparse_activities(trace2)
trace3_s, _ = remove_neurons_with_sparse_activities(trace3)
trace4_s, _ = remove_neurons_with_sparse_activities(trace4)
tracec_s, _ = remove_neurons_with_sparse_activities(tracec)
traceo_s, _ = remove_neurons_with_sparse_activities(traceo)

#%%
first_act = []
std_level = 5
for idx in range(len(traceo_s.T)):
    t = traceo_s[:, idx]
    first_act.append(np.where(t>t.std() * std_level)[0][0])

select = np.where(np.array(first_act)<10000)[0]
traceo_s = traceo_s[:, select]
#%%
t_g = trace_s.copy()
for i in range(t_g.shape[1]):
    t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)
t1_g = trace1_s.copy()
for i in range(t1_g.shape[1]):
    t1_g[:, i] = gaussian_filter1d(t1_g[:, i], sigma=5)
t2_g = trace2_s.copy()
for i in range(t2_g.shape[1]):
    t2_g[:, i] = gaussian_filter1d(t2_g[:, i], sigma=5)
t3_g = trace3_s.copy()
for i in range(t3_g.shape[1]):
    t3_g[:, i] = gaussian_filter1d(t3_g[:, i], sigma=5)    
t4_g = trace4_s.copy()
for i in range(t4_g.shape[1]):
    t4_g[:, i] = gaussian_filter1d(t4_g[:, i], sigma=5)
tc_g = tracec_s.copy()
for i in range(tc_g.shape[1]):
    tc_g[:, i] = gaussian_filter1d(tc_g[:, i], sigma=5)
to_g = traceo_s.copy()
for i in range(to_g.shape[1]):
    to_g[:, i] = gaussian_filter1d(to_g[:, i], sigma=5)
[print(t.shape) for t in [t_g, t1_g, t2_g, t3_g, t4_g, tc_g, to_g]]    
# tm_g = tracem_s.copy()
# for i in range(tm_g.shape[1]):
#     tm_g[:, i] = gaussian_filter1d(tm_g[:, i], sigma=5)

#%% Fig 7c decoding performance through cross-validation
import numpy as np
from sklearn.model_selection import KFold
def cross_validation_ridge(X, y, n_splits=5, alpha=500):
    score = []
    kf = KFold(n_splits=5)
    for cv_train, cv_test in kf.split(X):
        x_train = X[cv_train]
        y_train = y[cv_train]
        x_test = X[cv_test]
        y_test = y[cv_test]
        clf = Ridge(alpha=alpha)
        clf.fit(x_train, y_train)  

        # print(cv_train)
        # print(cv_test)
        # plt.figure()
        # plt.plot(y[cv_test])
        # plt.plot(clf.predict(X[cv_test]))
        # plt.title(f'{cv_test[0]}, {clf.score(x_test, y_test)}')
        # plt.show()
        score.append(clf.score(x_test, y_test))
    return score
# X = to_g
# idx = 2

X_list = [t_g, tc_g, to_g, t1_g, t2_g, t3_g, t4_g]
method = ['Suite2p', 'CaImAn', 'CaImAn_Online', 'Fiola_3000', 'Fiola_1500','Fiola_1000', 'Fiola_500']

t_s = 3000
t_e = 31500
dec = [pos_s, spd_s][1]

r = {}
alpha_list = [100, 500, 1000, 5000, 10000]
for idx, X in enumerate(X_list):
    print(method[idx])
    X = X[t_s:t_e]
    y = dec[t_s:t_e]
    score_all = {}
    for alpha in alpha_list:
        print(alpha)
        score = cross_validation_ridge(X, y, n_splits=5, alpha=alpha)
        score_all[alpha] = score
    #print(score_all)
    score_m = [np.mean(s) for s in score_all.values()]
    print(f'max score:{max(score_m)}')
    print(f'alpha:{alpha_list[np.argmax(score_m)]}')
    r[method[idx]] = score_all[alpha_list[np.argmax(score_m)]]

#%%
# rr = list(r.values())
# fig = plt.figure(figsize=(8, 6)) 
# ax1 = plt.subplot()
# ax1.bar(list(range(len(method))), [np.mean(x) for x in rr], yerr=[np.std(x) for x in rr])
# ax1.legend()
# #ax1.bar(0, np.mean(r), yerr=np.std(r))
# #ax1.bar(0, np.mean(r1), yerr=np.std(r1))
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# #ax1.spines['left'].set_visible(False)
# ax1.set_xlabel('Method')
# ax1.set_ylabel('Decoding R^2')

# ax1.xaxis.set_ticks_position('none') 
# #ax1.yaxis.set_ticks_position('none') 
# #ax1.set_xticks([0, 1, 2, 3])
# ax1.set_xticklabels(method)
# ax1.set_xticks(list(range(len(method))))
# #ax1.set_yticks([])
# ax1.set_ylim([0,1])
#plt.savefig(savef + 'decoding_cross_validation.pdf')

#%%
r_onacid = {}
for tt in [500, 1000, 1500, 3000, 5000, 8000, 10000, 15000, 20000, 25000, 30000]:
    traceo_s, indexes = remove_neurons_with_sparse_activities(traceo)
    first_act = []
    std_level = 5
    for idx in range(len(traceo_s.T)):
        t = traceo_s[:, idx]
        first_act.append(np.where(t>t.std() * std_level)[0][0])
        
    #plt.plot(first_act)
    select = np.where(np.array(first_act)<tt)[0]
    #select = (onacid.time_neuron_added[:, 1] < tt)[np.array(indexes)]
    traceo_s = traceo_s[:, select]
    
    to_g = traceo_s.copy()
    for i in range(to_g.shape[1]):
        to_g[:, i] = gaussian_filter1d(to_g[:, i], sigma=5)
    
    dec = [pos_s, spd_s][0]
    
    X = to_g; idx=1
    print(method[idx])
    X = X[t_s:t_e]
    y = dec[t_s:t_e]
    score_all = {}
    for alpha in alpha_list:
        print(alpha)
        score = cross_validation_ridge(X, y, n_splits=5, alpha=alpha)
        score_all[alpha] = score
    #print(score_all)
    score_m = [np.mean(s) for s in score_all.values()]
    print(f'max score:{max(score_m)}')
    print(f'alpha:{alpha_list[np.argmax(score_m)]}')
    
    r_onacid[tt] = [to_g.shape[1], score_m]

#%%
rrr = list(r_onacid.values())
fig = plt.figure() 
ax1 = plt.subplot()
xx = [rr[0] for rr in rrr]
yy = [np.max(rr[1]) for rr in rrr]
ax1.scatter(xx, yy)
for i in range(len(xx)):
    ax1.annotate(list(r_onacid.keys())[i], (xx[i] + 10, yy[i] - 0.03))

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.locator_params(axis='y', nbins=8)
ax1.locator_params(axis='x', nbins=4)
#ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_ylabel('Decoding R square')
ax1.set_xlabel('Number of neurons selected')
ax1.set_ylim([0.55, 0.95])

plt.savefig(savef + 'Fig_supp_caiman_online_pos_v3.1.pdf')
#%% Wilcoxon test
from scipy.stats import wilcoxon
wilcoxon(rr[0], rr[6], alternative='greater')

#%% number of neurons detected
method = ['Suite2p', 'CaImAn', 'CaImAn_Online', 'Fiola_3000', 'Fiola_1500','Fiola_1000', 'Fiola_500']
X_list = [t_g, tc_g, to_g,  t1_g, t2_g, t3_g, t4_g]

fig = plt.figure() 
ax1 = plt.subplot()
ax1.bar(list(range(len(method))), [x.shape[1] for x in X_list])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
#ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Method')
ax1.set_ylabel('Number of neurons detected')

ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
#ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(method)
ax1.set_xticks(list(range(len(method))))
#ax1.set_yticks([])

#%% number of neurons vs decoding performance
fig = plt.figure() 
ax1 = plt.subplot()
rr = list(r.values())
xx = [x.shape[1] for x in X_list]
yy = [np.mean(x) for x in rr]
zz = [np.std(x) for x in rr]


#ax1.scatter(xx, yy)
colors = ['C3', 'C2', 'C1', 'C0', 'C0', 'C0', 'C0']

for idx in [3, 4, 5, 6, 2, 0, 1]:
    ax1.errorbar(xx[idx], yy[idx], yerr=zz[idx], fmt='o', capsize=5, color=colors[idx], label=method[idx])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.locator_params(axis='y', nbins=8)
ax1.locator_params(axis='x', nbins=4)
ax1.set_ylabel('Decoding R square')
ax1.set_xlabel('Number of neurons')
ax1.legend(frameon=False)


#for i in range(len(method)):
#    ax1.annotate(method[i], (xx[i] + 10, yy[i] - 0.02))
    
#plt.savefig(savef + 'Fig7c_pos_v3.1.pdf')
plt.savefig(savef + 'Fig_supp_spd_v3.1.pdf')


#%% Fig 7d decoding performance across time
r_all = {}
train = [3000, 13000]
flag = 0
method = np.array(['Suite2p', 'CaImAn', 'CaImAn_Online', 'Fiola_3000', 'Fiola_1500','Fiola_1000', 'Fiola_500'])[np.array([0, 1, 2, 3])]
X_list = np.array([t_g, tc_g, to_g,  t1_g, t2_g, t3_g, t4_g])[np.array([0, 1, 2, 3])]
t_s = 3000
t_e = 31500
dec = [pos_s, spd_s][0]

#test = [10000, 12000]
alpha_list = [100, 500, 1000, 5000, 10000]
for idx, X in enumerate(X_list):
    print(method[idx])
    score_all = {}
    for alpha in alpha_list:
        print(alpha)
        r = []
        for test in [[i, i+2000] for i in range(13000, 31000, 2000)]:
            #test = [10000, 12000]
            clf = Ridge(alpha=alpha)
            #clf = LinearRegression()
            #clf = Lasso(alpha=0.000001)
            clf.fit(X[train[0]:train[1]], dec[train[0]:train[1]])  
            pos_pre_test = clf.predict(X[test[0]:test[1]])     
            pos_pre_train = clf.predict(X[train[0]:train[1]])     
            #print(f'test: {clf.score(X[test[0]:test[1]], dec[test[0]:test[1]])}')
            #print(f'train: {clf.score(X[train[0]:train[1]], dec[train[0]:train[1]])}')
            r.append(clf.score(X[test[0]:test[1]], dec[test[0]:test[1]]))
            if flag == 0:
                flag = 1
                plt.figure()
                plt.title(f'{test}')
                plt.plot(dec[train[0]:train[1]])
                plt.plot(pos_pre_train)
                plt.figure()
                plt.plot(dec[test[0]:test[1]])
                plt.plot(pos_pre_test)
        score_all[alpha] = r
        score_m = [np.mean(s) for s in score_all.values()]
    print(f'max score:{max(score_m)}')
    print(f'alpha:{alpha_list[np.argmax(score_m)]}')
    r_all[method[idx]] = score_all[alpha_list[np.argmax(score_m)]]
r_all = list(r_all.values())

#%% 
xx = list(range(13000, 31000, 2000))
xx = np.array(xx) - 13000
#fig = plt.figure(figsize=(8, 6)) 
fig = plt.figure() 
ax1 = plt.subplot()
colors = ['C3', 'C2', 'C1', 'C0']

for idx in [3, 2, 0, 1]:
    #if 'Fiola' in methods[idx]:
    ax1.plot(xx, r_all[idx], label=method[idx], color=colors[idx])

#[ax1.plot(xx, x) for x in r_all]
ax1.plot(xx, np.array(r_all[0]) - np.array(r_all[3]), label='diff between Suite2p and FIOLA', color='purple') 
ax1.legend(frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('Frame')
ax1.set_ylabel('Decoding R^2')
ax1.locator_params(axis='y', nbins=6)
ax1.locator_params(axis='x', nbins=4)
#ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
#ax1.set_xticks([])
#ax1.set_yticks([])
ax1.set_ylim([-0.1,1])
plt.savefig(savef + 'Fig7d_v3.1.pdf')

#%% Fig 7d statistical tests
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array(xx)[:, None]
y = np.array(r_all[0]) - np.array(r_all[3])
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_
y_pred = reg.predict(X)
plt.plot(xx, y)
plt.plot(xx, y_pred)

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(xx,y)
p_value
# 0.024347565665464213
slope
# 3.284592585227157e-06
intercept
#0.03982149638833927
#%% Fig 7e Decoding speed at different speed group
low_spd = np.percentile(speed, 33)
mid_spd = np.percentile(speed, 66)
t1 = np.where(speed <= low_spd)[0]
t2 = np.where(np.logical_and(speed > low_spd, speed < mid_spd))[0]
t3 = np.where(speed > mid_spd)[0]
plt.plot(speed)
plt.hlines([low_spd, mid_spd], 0, 30000, linestyles='dashed', color='black')

#%%
def cross_validation_ridge(X, y, n_splits=5, alpha=500):
    score = []
    kf = KFold(n_splits=5)
    for cv_train, cv_test in kf.split(X):
        x_train = X[cv_train]
        y_train = y[cv_train]
        x_test = X[cv_test]
        y_test = y[cv_test]
    
        clf = Ridge(alpha=alpha)
        clf.fit(x_train, y_train)  
        score.append(clf.score(x_test, y_test))
    return score

method = np.array(['Suite2p', 'CaImAn', 'CaImAn_Online', 'Fiola_3000', 'Fiola_1500','Fiola_1000', 'Fiola_500'])[np.array([0, 1, 2, 3])]
X_list = np.array([t_g, tc_g, to_g,  t1_g, t2_g, t3_g, t4_g])[np.array([0, 1, 2, 3])]
tt = [t1, t2, t3]
spd_group = ['low', 'mid', 'high']

t_s = 3000
t_e = 31500
dec = [pos_s, spd_s][1]

r = {}
alpha_list = [100, 500, 1000, 5000, 10000]
for idx, X in enumerate(X_list):
    print(method[idx])
    r[method[idx]] = {}
    for i, t in enumerate(tt):
        print(spd_group[i])
        t = t[np.logical_and(t>t_s, t<t_e)]
        xx = X[t]
        y = dec[t]
        score_all = {}
        for alpha in alpha_list:
            print(alpha)
            score = cross_validation_ridge(xx, y, n_splits=5, alpha=alpha)
            score_all[alpha] = score
        #print(score_all)
        score_m = [np.mean(s) for s in score_all.values()]
        print(f'max score:{max(score_m)}')
        print(f'alpha:{alpha_list[np.argmax(score_m)]}')
        r[method[idx]][spd_group[i]] = score_all[alpha_list[np.argmax(score_m)]]

#%%
rr = list(r.values())
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
colors = ['C3', 'C2', 'C1', 'C0']
for idx, i in enumerate([3, 2, 0, 1]):
    #i = r[methodzz][idx]
    ax1.bar(np.array(range(len(spd_group))) + (idx-1) * 0.2, [np.mean(x) for x in list(rr[i].values())], 
            yerr=[np.std(x) for x in list(rr[i].values())], width=0.2, label=method[i], color=colors[i])
ax1.legend(frameon=False)
#ax1.bar(0, np.mean(r), yerr=np.std(r))
#ax1.bar(0, np.mean(r1), yerr=np.std(r1))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Speed group')
ax1.set_ylabel('Decoding R^2')
ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
#ax1.set_xticks([0, 1, 2, 3])  
ax1.set_xticklabels(spd_group)
ax1.set_xticks(list(range(len(spd_group))))
#ax1.set_yticks([])
ax1.set_ylim([0,1])
plt.savefig(savef + 'Fig7e_speed_v3.1.pdf')

#%% Fig 7f timing for init + acquisition + online exp
data = np.array([t_3000, t_1500, t_1000, t_500])
data = np.hstack([data, np.array([3000/15.46, 1500/15.46, 1000/15.46, 500/15.46])[:, None], 
                 np.array([1800, 1800, 1800, 1800])[:, None]])
# np.array([(31932 - 3000)/15.46, (31932 - 1500)/15.46, (31932 - 1000)/15.46, (31932 - 500)/15.46])[:, None]
data = data/60
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
ax1.bar(range(4), data[:, 2], label='acquision for init')
ax1.bar(range(4), data[:, 0], bottom=data[:, 2], label='init time')
ax1.bar(range(4), data[:, 3], bottom=data[:, 2] + data[:, 0], label='online exp time')
ax1.legend(frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_xlabel('init batch size (frames)')
ax1.set_ylabel('time (mins)')

ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(['3000', '1500', '1000', '500' ])
ax1.set_yticks([0, 10, 20, 30, 40])
#ax1.set_yticks([])
#ax1.set_ylim([0,1])
#plt.show()
plt.savefig(savef + 'init_time.pdf')


#%% timing for init + online
data = np.array([t_3000, t_1500, t_1000, t_500])
data = data/60
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
ax1.bar(range(4), data[:, 0], label='init')
ax1.bar(range(4), data[:, 1], bottom=data[:, 0], label='online')
ax1.legend()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
#ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_xlabel('method')
ax1.set_ylabel('time (mins)')

#ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(['Fiola 3000', 'Fiola 1500', 'Fiola 1000', 'Fiola 500' ])
#ax1.set_yticks([])
#ax1.set_ylim([0,1])
plt.savefig(savef + 'init_time_and_online.pdf')

#%%
#%% timing
t_3000 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_timing.npy', allow_pickle=True)
t_1500 = np.load('/media/nel/storage/fiola/R2_20190219/1500/fiola_timing.npy', allow_pickle=True)
t_1000 = np.load('/media/nel/storage/fiola/R2_20190219/1000/fiola_timing.npy', allow_pickle=True)
t_500 = np.load('/media/nel/storage/fiola/R2_20190219/500/fiola_timing.npy', allow_pickle=True)

t_3000 = np.diff(list(t_3000.item().values()))
t_1500 = np.diff(list(t_1500.item().values()))
t_1000 = np.diff(list(t_1000.item().values()))
t_500 = np.diff(list(t_500.item().values()))

#%% Decoding performance with different number of neurons
import numpy as np
from sklearn.model_selection import KFold
def cross_validation_ridge(X, y, n_splits=5, alpha=500):
    score = []
    kf = KFold(n_splits=5)
    for cv_train, cv_test in kf.split(X):
        x_train = X[cv_train]
        y_train = y[cv_train]
        x_test = X[cv_test]
        y_test = y[cv_test]
    
        clf = Ridge(alpha=alpha)
        clf.fit(x_train, y_train)  
        score.append(clf.score(x_test, y_test))
    return score

method = ['Suite2p', 'CaImAn', 'Fiola_3000', 'Fiola_1500','Fiola_1000', 'Fiola_500']
X_list = [t_g, tc_g, t1_g, t2_g, t3_g, t4_g]
# method = ['Fiola_500']
# X_list = [t1_g]

t_s = 3000
t_e = 31500
r = {}
dec = [pos_s, spd_s][0]
alpha_list = [100, 500, 1000, 5000, 10000]
for idx, X in enumerate(X_list):
    r[method[idx]] = []
    print(method[idx])
    for num in range(100, 2000, 200):
        print(num)        
        xx = X[t_s:t_e]
        y = dec[t_s:t_e]
        xx = xx[:, :num]
        if num > X.shape[1] + 200:
            r[method[idx]].append(r[method[idx]][-1])
        else:
            score_all = {}
            for alpha in alpha_list:
                print(f'{alpha:} alpha')
                score = cross_validation_ridge(xx, y, n_splits=5, alpha=alpha)
                score_all[alpha] = score
            #print(score_all)
            score_m = [np.mean(s) for s in score_all.values()]
            print(f'max score:{max(score_m)}')
            print(f'alpha:{alpha_list[np.argmax(score_m)]}')
            r[method[idx]].append(np.mean(score_all[alpha_list[np.argmax(score_m)]]))
        
#%%
xx = list(range(100, 2000, 200))
# plt.figure(); plt.plot(xx, r[method[0]]); plt.plot(xx, r[method[1]]); 
# plt.plot(xx, r[method[2]]); plt.plot(xx, r[method[3]])
# #plt.plot(xx, np.array(r) - np.array(r1)); 
# plt.legend(['Suite2p', 'Fiola', 'MeanROI','CaImAn', 'Difference Suite2p FIOLA'])
# plt.title('Performance vs different number of neurons')
fig = plt.figure() 
ax1 = plt.subplot()
r_all = list(r.values())
[ax1.plot(xx, x) for x in r_all]
# ax1.plot(xx, np.array(r_all[0]) - np.array(r_all[1])); 
ax1.legend(method)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('number of neurons')
ax1.set_ylabel('Decoding R^2')
#ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
#ax1.set_xticks([])
#ax1.set_yticks([])
ax1.set_ylim([0.5,0.9])

#np.save(savef + 'decoding_vs_num_neurons.npy', r)
plt.savefig(savef + 'decoding_vs_num_neurons.pdf')

#%%
r_all = []
for k_n in list(range(0, 2100, 200)):
    r_temp = []
    for it in range(5):
        trace = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries'].data[:]
        idx = np.random.permutation(list(range(trace.shape[1])))[:k_n]
        trace[:, idx] = 1
        trace_s = StandardScaler().fit_transform(trace)
        
        t_g = trace_s.copy()
        for i in range(t_g.shape[1]):
            t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)
        
        r = []
        train = [0, 10000]
        flag = 1
        test = [10000, 12000]
        for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
            #test = [10000, 12000]
            clf = Ridge(alpha=800)
            #clf = LinearRegression()
            #clf = Lasso(alpha=0.000001)
            clf.fit(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
            pos_pre_test = clf.predict(t_g[test[0]:test[1]])     
            pos_pre_train = clf.predict(t_g[train[0]:train[1]])     
            print(f'test: {clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
            print(f'train: {clf.score(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
            r.append(clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
            if flag == 0:
                flag = 1
                plt.figure()
                plt.plot(pos_s[train[0]:train[1]])
                plt.plot(pos_pre_train)
                plt.figure()
                plt.plot(pos_s[test[0]:test[1]])
                plt.plot(pos_pre_test)
        print(f'average test: {np.mean(r)}')
        r_temp.append(np.mean(r))
    r_all.append(r_temp)
    
rr = [np.mean(r) for r in r_all]
plt.plot(list(range(50, 2250, 200)), rr[::-1])
plt.hlines(np.mean(r1), xmin=0, xmax=2000, color='orange')
plt.legend(['suite2p number of remaining', 'fiola 749 neurons with bg'])
plt.ylabel('R2')
plt.xlabel('number of neurons')


#%%
corr = []
for idx in range(len(tracem.T)):
    corr.append(np.corrcoef(tracem_s[:-1, idx], pos_n)[0, 1])



#%%
coef = clf.coef_.copy()
plt.plot(clf.coef_)
#idx = np.argsort(clf.coef_)[::-1][:40]
#idx = np.argsort(coef)[::-1][:300]
#idx1 = np.argsort(coef)[:300]
#idx = np.concatenate([idx, idx1])

trace.shape
idx = np.random.permutation(list(range(trace.shape[1])))[:1300]
trace[:, idx] = 1
plt.plot(t_g[:, idx])
plt.plot(pos_s)
plt.plot(trace_s[:, idx])
plt.plot(trace0[:, idx])


#%%
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g[train[0]:train[1]], spd_s[train[0]:train[1]])  
spd_pre_test = clf.predict(t_g[test[0]:test[1]])     
spd_pre_train = clf.predict(t_g[train[0]:train[1]])     
print(f'test: {clf.score(t_g[test[0]:test[1]], spd_s[test[0]:test[1]])}')
print(f'train: {clf.score(t_g[train[0]:train[1]], spd_s[train[0]:train[1]])}')
plt.figure()
plt.plot(spd_s[train[0]:train[1]])
plt.plot(spd_pre_train)
plt.figure()
plt.plot(spd_s[test[0]:test[1]])
plt.plot(spd_pre_test)




#%%
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t1_g[train[0]:train[1]], spd_s[train[0]:train[1]])  
spd_pre_test = clf.predict(t1_g[test[0]:test[1]])     
spd_pre_train = clf.predict(t1_g[train[0]:train[1]])     
print(f'test: {clf.score(t1_g[test[0]:test[1]], spd_s[test[0]:test[1]])}')
print(f'train: {clf.score(t1_g[train[0]:train[1]], spd_s[train[0]:train[1]])}')
plt.figure()
plt.plot(spd_s[train[0]:train[1]])
plt.plot(spd_pre_train)
plt.figure()
plt.plot(spd_s[test[0]:test[1]])
plt.plot(spd_pre_test)

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(trace_s, pos_s)  
pos_pre = clf.predict(trace_s)     
clf.score(trace_s, pos_s)

#%%
plt.figure()
plt.plot(pos_s) 
plt.plot(pos_pre, alpha=0.9)        
plt.title('pos')
plt.legend(['pos', 'fit'])

#%%
plt.plot(pos_n) 
plt.plot(spd_s)

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(trace_s, spd_s)  
spd_pre = clf.predict(trace_s)     
clf.score(trace_s, spd_s)

#%%
plt.figure()
plt.plot(spd_s) 
plt.plot(spd_pre, alpha=0.9)        
plt.title('spd')
plt.legend(['spd', 'fit'])

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(trace_s[2500:12500], pos_s[2500:12500])  
pos_pre = clf.predict(trace_s[2500:12500])     
clf.score(trace_s[2500:12500], pos_s[2500:12500])

#%%
plt.plot(pos_s[2500:12500])
plt.plot(pos_pre)

#%%
from scipy.ndimage import gaussian_filter1d
t_g = trace_s.copy()
for i in range(t_g.shape[1]):
    t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)

#%%
plt.plot(trace_s[2500:12500, 0])
plt.plot(t_g[:, 0])

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g, pos_s[2500:12500])  
pos_pre = clf.predict(t_g)     
clf.score(t_g, pos_s[2500:12500])

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g, pos_s)  
pos_pre = clf.predict(t_g)     
clf.score(t_g, pos_s)


#%%
plt.plot(pos_s)
plt.plot(pos_pre)
# #%%
# trace1_s = trace1.copy()
# for idx in range(len(trace1.T)):
#     t = trace1[:, idx]
#     window = 3000
#     weights = np.repeat(1.0, window)/window
#     t_pad = np.concatenate([t[:1500][::-1], t, t[-1499:]])
#     ma = np.convolve(t_pad, weights, 'valid')
#     # plt.plot(t)
#     # plt.plot(ma)
#     t = t - ma
#     trace1_s[:, idx] = t
#     if idx % 100 == 0:
#         print(idx)
        
# trace1_s = StandardScaler().fit_transform(trace1_s)

#%%
dims = (796, 512)
li = []
li_matrix = np.zeros((len(range(0, dims[0], 30)), len(range(0, dims[1], 30))))
for i in range(0, dims[0], 30):
    for j in range(0, dims[1], 30):
        num = 0
        for idx in range(centers.shape[0]):
            if centers[idx, 1] >= i and centers[idx, 1] < i + 30:
                if centers[idx, 0] >= j and centers[idx, 0] < j + 30:
                   num = num + 1
        li.append(num)
        li_matrix[i//30, j//30] = num
#li_matrix[li_matrix<] = 0
plt.imshow(li_matrix); plt.colorbar
#plt.hist(li_matrix.flatten())#; plt.colorbar()

#%%
from caiman.base.rois import com
centers_cm = com(cnm2.estimates.A[:, cnm2.estimates.idx_components], 796, 512)
centers_cm = centers_cm[:, np.array([1, 0])]
centers = centers_cm.copy()
#%%
dims = (796, 512)
A = cnm2.estimates.A.toarray()
A = A.reshape([dims[0], dims[1], -1], order='F')
A = A.transpose([2, 0, 1])

aa = np.load('/media/nel/storage/fiola/R2_20190219/3000/caiman_masks_3000_749.npy', allow_pickle=True)

#%%
cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/7000/memmap__d1_796_d2_512_d3_1_order_C_frames_7000_.hdf5')
A = cnm2.estimates.A.toarray()
A = A.reshape([dims[0], dims[1], -1], order='F')
A = A.transpose([2, 0, 1])

#%%
r = []
train = [0, 10000]
flag = 1
test = [10000, 30000]
#test = [10000, 12000]
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
pos_pre_test = clf.predict(t_g[test[0]:test[1]])     
pos_pre_train = clf.predict(t_g[train[0]:train[1]])     
print(f'test: {clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
print(f'train: {clf.score(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
r.append(clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
if flag == 0:
    #flag = 1
    # plt.figure()
    # plt.plot(pos_s[train[0]:train[1]])
    # plt.plot(pos_pre_train)
    plt.figure()
    plt.plot(pos_s[test[0]:test[1]])
    plt.plot(pos_pre_test)
print(f'average test: {np.mean(r)}')

r1 = []
flag = 1
#test = [10000, 12000]
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
pos_pre_test = clf.predict(t1_g[test[0]:test[1]])     
pos_pre_train = clf.predict(t1_g[train[0]:train[1]])     
print(f'test: {clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
print(f'train: {clf.score(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
r1.append(clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
if flag == 0:
    #flag = 1
    #plt.figure()
    # plt.plot(pos_s[train[0]:train[1]])
    # plt.plot(pos_pre_train)
    #plt.figure()
    #plt.plot(pos_s[test[0]:test[1]])
    plt.plot(pos_pre_test)
print(f'average test: {np.mean(r1)}')


rc = []
flag = 1
#test = [10000, 12000]
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
pos_pre_test = clf.predict(tc_g[test[0]:test[1]])     
pos_pre_train = clf.predict(tc_g[train[0]:train[1]])     
print(f'test: {clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
print(f'train: {clf.score(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
rc.append(clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
if flag == 0:
    #flag = 1
    #plt.figure()
    # plt.plot(pos_s[train[0]:train[1]])
    # plt.plot(pos_pre_train)
    #plt.figure()
    #plt.plot(pos_s[test[0]:test[1]])
    plt.plot(pos_pre_test)
print(f'average test: {np.mean(rc)}')

#%%

trace_s = StandardScaler().fit_transform(trace)
tracem_s = StandardScaler().fit_transform(tracem)
pos_s = StandardScaler().fit_transform(pos_n[:, None])[:, 0]
spd_s = StandardScaler().fit_transform(speed[:, None])[:, 0]
trace1_s = StandardScaler().fit_transform(trace1)
trace1_s = trace1_s / trace1_s.max(0)

#%%
tracec_s = StandardScaler().fit_transform(tracec)

#%%
#tracem_s = tracem_s / tracem_s.max(0)
# tracem_s= tracem_s - tracem_s.min(0)
# tracem_s = tracem_s / tracem_s.max(0)
# tracem_s = tracem_s/tracem_s.max(0)
# for i in range(tracem_s.shape[1]):
#     tracem_s[:, i] = gaussian_filter1d(tracem_s[:, i], sigma=5)
ii = 101
plt.plot(trace_s[:, ii])
print((trace_s[:3000, ii] > 5).sum())
#%%
n = []
for ii in range(trace_s.shape[1]):
    n.append((trace_s[:3000, ii] > 5).sum())
    
n = np.array(n)
(n > 5).sum()
select = np.where(n>5)[0]
#%%
# from caiman.source_extraction.cnmf.temporal import constrained_foopsi
# dec = []
# for idx in range(len(tracem_s.T)):
#     c_full, bl, c1, g, sn, s_full, lam = constrained_foopsi(tracem_s[:, idx], p=1, s_min=0.1)
#     dec.append(s_full)
#     if idx in [125, 130, 135, 140, 145, 150]:
#         plt.title(f'neuron {idx}'); plt.plot(tracem_s[:, idx]); plt.plot(c_full-3); 
#         plt.plot(s_full-6, c='black'); plt.legend(['calcium', 'fit', 'dec']); plt.show()
# trace2 = np.array(dec).T

# trace2 = np.array(dec).T
trace2 = trace1_s.copy()
trace2_s = StandardScaler().fit_transform(trace2)
t1_g = trace2_s.copy()

from caiman.source_extraction.volpy.spikepursuit import signal_filter
tt = signal_filter(trace1, freq=1, fr=15)

# plt.plot(trace1[:, 30]); plt.plot(tt[:, 30])
plt.plot(trace1[:, :].sum(1)); plt.plot(tt[:, :].sum(1))
trace1 = tt
#trace1 = trace1[1:]#%%
r = []
train = [0, 10000]
flag = 0
#test = [10000, 12000]
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    #test = [10000, 12000]
    clf = Ridge(alpha=800)
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.000001)
    clf.fit(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(t_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(t_g[train[0]:train[1]])     
    print(f'test: {clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    r.append(clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        flag = 1
        plt.figure()
        plt.title(f'{test}')
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(r)}')
    
#%%
r1 = []
flag = 0
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    clf = Ridge(alpha=800)
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.000001)
    clf.fit(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(t1_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(t1_g[train[0]:train[1]])     
    print(f'test: {clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    r1.append(clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        #flag = 1
        plt.figure()
        plt.title(f'{test}')
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(r1)}')

#%%
rm = []
flag = 0
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    clf = Ridge(alpha=800)
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.000001)
    clf.fit(tm_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(tm_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(tm_g[train[0]:train[1]])     
    print(f'test: {clf.score(tm_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(tm_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    rm.append(clf.score(tm_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        flag = 1
        plt.figure()
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(rm)}')

#%%
rc = []
flag = 0
train = [0, 10000]
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    clf = Ridge(alpha=800)    
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.05)
    clf.fit(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(tc_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(tc_g[train[0]:train[1]])     
    print(f'test: {clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    rc.append(clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        #flag = 1
        plt.figure()
        plt.title(f'{test}')
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(rc)}')

#%%
rc = []
flag = 0
sets = [[0, 10000], [10000, 20000], [20000, 30000]]
for train in sets:
    for test in [[i, i+10000] for i in range(0, 30000, 10000)]:
        clf = Ridge(alpha=800)    
        #clf = LinearRegression()
        #clf = Lasso(alpha=0.05)
        clf.fit(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
        pos_pre_test = clf.predict(tc_g[test[0]:test[1]])     
        pos_pre_train = clf.predict(tc_g[train[0]:train[1]])     
        print(f'test: {clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
        print(f'train: {clf.score(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
        rc.append(clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
        if flag == 0:
            #flag = 1
            plt.figure()
            plt.title(f'{test}')
            plt.plot(pos_s[train[0]:train[1]])
            plt.plot(pos_pre_train)
            plt.figure()
            plt.plot(pos_s[test[0]:test[1]])
            plt.plot(pos_pre_test)
    print(f'average test: {np.mean(rc)}')
    
    
#%%
#%%
import numpy as np
from sklearn.model_selection import KFold
def cross_validation_ridge(X, y, n_splits=5, alpha=500):
    score = []
    kf = KFold(n_splits=5)
    for cv_train, cv_test in kf.split(X):
        x_train = X[cv_train]
        y_train = y[cv_train]
        x_test = X[cv_test]
        y_test = y[cv_test]
    
        clf = Ridge(alpha=alpha)
        clf.fit(x_train, y_train)  
        score.append(clf.score(x_test, y_test))
    return score

#X = t_g[:30000]
method = ['Suite2p', 'Fiola', 'MeanROI', 'CaImAn'][0:2]
#X_list = [t_g, t1_g, tm_g, tc_g][0:3]
#X_list = [t_g, t1_g, tm_g][0:3]
X_list = [t_g, t1_g]
r = {}
alpha_list = [500, 1000, 5000]
for idx, X in enumerate(X_list):
    r[method[idx]] = []
    print(method[idx])
    for num in range(100, 2800, 400):
        print(num)        
        X = X[:30000]
        y = pos_s[:30000]
        xx = X[:, :num]
        score_all = {}
        for alpha in alpha_list:
            print(f'{alpha:} alpha')
            score = cross_validation_ridge(xx, y, n_splits=5, alpha=alpha)
            score_all[alpha] = score
        #print(score_all)
        score_m = [np.mean(s) for s in score_all.values()]
        print(f'max score:{max(score_m)}')
        print(f'alpha:{alpha_list[np.argmax(score_m)]}')
        r[method[idx]].append(np.mean(score_all[alpha_list[np.argmax(score_m)]]))
        
#%%
xx = list(range(100, 2200, 200))
plt.figure(); plt.plot(xx, r[method[0]]); plt.plot(xx, r[method[1]]); 
plt.plot(xx, r[method[2]]); plt.plot(xx, r[method[3]])
#plt.plot(xx, np.array(r) - np.array(r1)); 
plt.legend(['Suite2p', 'Fiola', 'MeanROI','CaImAn', 'Difference Suite2p FIOLA'])
plt.title('Performance vs different number of neurons')

        
#%%
for key in r.keys():
    print(np.mean(r[key]))

#%%
def rand_jitter(arr):
    stdev = .01 #* (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), y, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)
for i in range(4):
    jitter([i]*5, r[method[i]])
    
plt.ylim([0, 1])
plt.xticks(list(range(4)), method)
plt.title('R^2 for prediction with cross validation')

#plt.scatter([1, 1], [1, 2])#%%
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
ax1.bar([0, 1, 2, 3], [np.mean(r), np.mean(r1), np.mean(rm), np.mean(rc)], 
        yerr=[np.std(r), np.std(r1), np.std(rm), np.std(rc)], label=['Suite2p', 'Fiola', 'MeanROI', 'CaImAn'])
ax1.legend()
#ax1.bar(0, np.mean(r), yerr=np.std(r))
#ax1.bar(0, np.mean(r1), yerr=np.std(r1))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Average')
ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([])
#ax1.set_yticks([])
ax1.set_ylim([0,1])

print(np.mean(r))
print(np.mean(r1))
print(np.mean(rm))
print(np.mean(rc))# T = [trace, trace1, trace2, trace3, trace4, tracec, traceo]
# T = [fast_prct_filt(t,20).T for t in T]
# T = [remove_neurons_with_sparse_activities(t) for t in T]
# T = for i in range(t_g.shape[1]):
#     t_g[:, i] = gaussian_filter1d(t[:, i], sigma=5) for ] for t in T
