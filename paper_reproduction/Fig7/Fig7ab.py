#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:37:36 2021

@author: nel
"""
import caiman as cm
from caiman.base.rois import com
from caiman.cluster import setup_cluster
from caiman.source_extraction.cnmf.estimates import Estimates, compare_components
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.utilities import fast_prct_filt
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
from pynwb import NWBHDF5IO
import pyximport
pyximport.install()
import scipy
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from time import time

#from fiola.config import load_fiola_config, load_caiman_config
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import bin_median, to_2D
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False})

import sys
sys.path.append('/home/nel/CODE/VIOLA/paper_reproduction')
from Fig7.Fig7_caiman_pipeline import run_caiman_fig7
from Fig7.Fig7_utilities import *

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

#%% Behavior
s_t = (pos >= 0)  # remove reward time and preparation time
s_t[np.where(np.diff(pos)<-1)[0]] = False
s_t[np.where(pos==-0.5)[0]] = False
#s_t[s_t==False] = True
s_t[24710:25510] = False # remove bad trial
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
pos_ss = pos_s.copy()
pos_ss[s_t==False] = np.nan
#plt.plot(pos); plt.plot(pos_s[s_t])
print(s_t.sum())


#%% Fig 7a corr img
cnm = load_CNMF('/media/nel/storage/fiola/R2_20190219/3000/memmap__d1_796_d2_512_d3_1_order_C_frames_3000__v3.7.hdf5')
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/memmap__d1_796_d2_512_d3_1_order_C_frames_31933__v3.13.hdf5')
Cn = cnm.estimates.Cn  # use CaImAn cn 
# cnm.estimates.plot_contours(img=Cn, idx=np.array(range(cnm.estimates.A.shape[1])), cmap='gray', display_numbers=False, 
#                             thr=0.9, vmax=0.75)

#cm.utils.visualization.plot_contours(cnm.estimates.A, Cn, thr=0.9, vmax = 0.2, display_numbers=False, cmap='gray')

#%% Fig 7a left
# y_limit = [25, 780]
centers = com(sp_processed['FIOLA3000'], 796, 512)
y_range = [400, 500]
x_range = [100, 200]
#select = np.where(np.logical_and(centers[:, 0] > y_limit[0], centers[:, 0] <= y_limit[1]))[0]
# a = cnm.estimates.A.toarray().T.reshape([-1, 796, 512], order='F')[select]
comp1 = np.where(np.logical_and(centers[:, 1] > x_range[0], centers[:, 1] <= x_range[1]))[0]
comp2 = np.where(np.logical_and(centers[:, 0] > y_range[0], centers[:, 0] <= y_range[1]))[0]
comp_all = np.intersect1d(comp1,comp2)
plt.subplot(1, 2, 1)
#plt.imshow(Cn)
cm.utils.visualization.plot_contours(sp_processed['FIOLA3000'], Cn, thr=0.9, vmax = 0.2, display_numbers=False, cmap='gray', contour_args={'linewidth': 0.3})
plt.plot([x_range[0],x_range[0]], [y_range[0],y_range[1]], 'y-')
plt.plot([x_range[1],x_range[1]], [y_range[0],y_range[1]], 'y-')
plt.plot([x_range[0],x_range[1]], [y_range[0],y_range[0]], 'y-')
plt.plot([x_range[0],x_range[1]], [y_range[1],y_range[1]], 'y-')
#plt.xlim(x_range)
#plt.ylim(y_range)
plt.savefig(savef+'Fig7a_left_v3.6.pdf')

#%%
centers = com(cnm.estimates.A, 796, 512)
y_range = [400, 500]
x_range = [100, 200]
# comp = np.where(np.array(select) == True)[0]
a = cnm.estimates.A.toarray().T.reshape([-1, 796, 512], order='F')
comp1 = np.where(np.logical_and(centers[:, 1] > x_range[0], centers[:, 1] <= x_range[1]))[0]
comp2 = np.where(np.logical_and(centers[:, 0] > y_range[0], centers[:, 0] <= y_range[1]))[0]
comp_all = np.intersect1d(comp1,comp2)
plt.subplot(1, 2, 1)
#plt.imshow(Cn)
#cnm.estimates.plot_contours(img=Cn, idx=comp_all[sort], cmap='gray', thr=0.9, vmax = 0.2, display_numbers=True)
cm.utils.visualization.plot_contours(cnm.estimates.A[:, comp_all][:, sort], Cn,  cmap='gray', max_number=15, contour_args={'linewidth': 0.8})
plt.plot([x_range[0],x_range[0]], [y_range[0],y_range[1]], 'y-')
plt.plot([x_range[1],x_range[1]], [y_range[0],y_range[1]], 'y-')
plt.plot([x_range[0],x_range[1]], [y_range[0],y_range[0]], 'y-')
plt.plot([x_range[0],x_range[1]], [y_range[1],y_range[1]], 'y-')
plt.xlim(x_range)
plt.ylim(y_range)
plt.savefig(savef+'Fig7a_right.pdf')

#%%
trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_None_with_detrending_v3.12.npy', allow_pickle=True).item().trace.T#trace_deconvolved.T
sort = sort_neurons(trace1[:, comp_all])

#%%
#trace1 = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_None_with_detrending_v3.12.npy', allow_pickle=True).item().trace_deconvolved.T
tt = trace1[:, comp_all][:, sort][:, :15]
tt = tt / tt.max(0)
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
for i in range(tt.shape[1]):
    t_s = 3000
    num_frames = tt.shape[0] - t_s
    xx = np.array(list(range(num_frames))) / 15.46 / 60
    plt.plot(xx, tt[t_s:t_s+num_frames, i]-i*0.5, color='black', linewidth=0.3)
    if i % 5 == 0:
        plt.text(-int(num_frames/400/60), -i*0.5, f'{i+1}')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.set_xlabel('time (min)')
ax1.set_yticks([])
ax1.set_ylabel('neuron index')
#plt.savefig(savef+'Fig7b.pdf')
#plt.plot(trace1[3000:6000, comp_all])

#%%
c, dview, n_processes = setup_cluster(
    backend='multiprocessing', n_processes=12, single_thread=False)

#%%
cnm = load_CNMF('/media/nel/storage/fiola/R2_20190219/3000/memmap__d1_796_d2_512_d3_1_order_C_frames_3000__v3.7.hdf5')
cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/memmap__d1_796_d2_512_d3_1_order_C_frames_31933__v3.13.hdf5')

dims = [796, 512]
s2p_estimate = Estimates(A=sp_processed['Suite2p'], b=None, C=t_rm['Suite2p'], f=None, R=None, dims=dims)
cnm_estimate = Estimates(A=sp_processed['FIOLA3000'], b=None, C=t_rm['FIOLA3000'], f=None, R=None, dims=dims)

s2p_estimate.threshold_spatial_components(maxthr=0.2, dview=dview)
cnm_estimate.threshold_spatial_components(maxthr=0.2, dview=dview)

tp_gt, tp_comp, fn_gt, fp_comp, performance_suite2p = compare_components(s2p_estimate, cnm_estimate,
                                                                         Cn=cnm2.estimates.Cn, thresh_cost=.8,
                                                                         min_dist=10,
                                                                         print_assignment=False,
                                                                         labels=['Suite_2p', 'FIOLA'],
                                                                         plot_results=True)
#plt.savefig(savef + 'Fig7_supp_spatial_comparison1_v3.7.pdf')
print(performance_suite2p)
#{'recall': 0.5285234899328859, 'precision': 0.7230298393267024, 'accuracy': 0.43953488372093025, 'f1_score': 0.6106623586429726}

#%%
caiman_estimate = Estimates(A=sp_processed['CaImAn'], b=None, C=t_rm['CaImAn'], f=None, R=None, dims=dims)
caiman_estimate.threshold_spatial_components(maxthr=0.2, dview=dview)
tp_gt, tp_comp, fn_gt, fp_comp, performance_suite2p = compare_components(s2p_estimate, caiman_estimate,
                                                                         Cn=cnm2.estimates.Cn, thresh_cost=.8,
                                                                         min_dist=10,
                                                                         print_assignment=False,
                                                                         labels=['Suite_2p', 'CaImAn'],
                                                                         plot_results=True)
#plt.savefig(savef + 'Fig7_supp_spatial_comparison2_v3.7.pdf')  #1441/1789
print(performance_suite2p)
# {'recall': 0.8059284116331096, 'precision': 0.7241206030150754, 'accuracy': 0.6166024818142918, 'f1_score': 0.7628374801482266}
#%%
onacid_estimate = Estimates(A=sp_processed['CaImAn_Online'], b=None, C=t_rm['CaImAn_Online'], f=None, R=None, dims=dims)
onacid_estimate.threshold_spatial_components(maxthr=0.2, dview=dview)

tp_gt, tp_comp, fn_gt, fp_comp, performance_suite2p = compare_components(s2p_estimate, onacid_estimate,
                                                                         Cn=cnm2.estimates.Cn, thresh_cost=.8,
                                                                         min_dist=10,
                                                                         print_assignment=False,
                                                                         labels=['Suite_2p', 'CaImAn_Online'],
                                                                         plot_results=True)
plt.savefig(savef + 'Fig7_supp_spatial_comparison3_v3.9.pdf')  
print(len(tp_gt) / (len(tp_gt) + len(fn_gt)))

{'recall': 0.7768456375838926,
 'precision': 0.687964338781575,
 'accuracy': 0.5744416873449132,
 'f1_score': 0.7297084318360915}

#%%
tp_gt, tp_comp, fn_gt, fp_comp, performance_suite2p = compare_components(onacid_estimate, cnm_estimate,
                                                                         Cn=cnm2.estimates.Cn, thresh_cost=.8,
                                                                         min_dist=10,
                                                                         print_assignment=False,
                                                                         labels=['CaImAn_Online', 'FIOLA'],
                                                                         plot_results=True)
plt.savefig(savef + 'Fig7_supp_spatial_comparison4_v3.8.pdf')  
print(len(tp_gt) / (len(tp_gt) + len(fn_gt)))

{'recall': 0.5179303278688525,
 'precision': 0.7735271614384086,
 'accuracy': 0.44973309608540923,
 'f1_score': 0.6204357164774471}

#%% Fig Supp nueron detection comparison
data = np.array([[0.61, 0.72, 0.53], [0.76, 0.72, 0.81], [0.73, 0.69, 0.78]])
labels = ['FIOLA3000', 'CaImAn', 'CaImAn Online']
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()
width = 0.2  # the width of the bars
rects1 = ax.bar(x - width, data[:, 0], width, capsize=5, label=f'F1')
rects2 = ax.bar(x, data[:, 1], width, capsize=5, label=f'Precision')
rects3 = ax.bar(x + width, data[:, 2], width, capsize=5, label=f'Recall')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Methods')
ax.set_ylabel('F1 score')
ax.set_ylim([0, 1])
ax.legend()
fig.tight_layout()
plt.show()
plt.savefig(savef + 'Fig7_supp_detection_F1_v3.9.pdf') 







#%% supp raster plot
#[i for i, s in enumerate(s_t[3000:31930]) if s == True][3573]
#s_t[3000:8777].sum()
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff+'Fig7c_result_v3.6.npy', allow_pickle=True).item()
p = np.load(ff+'Fig7c_prediction_v3.6.npy', allow_pickle=True).item()
plt.figure()
plt.plot(list(p.values())[0][0]['Y'], color='black')
key_list = ['FIOLA3000', 'CaImAn_Online']
for key, pr in p.items():
    if key in key_list:
        print(key)
        plt.plot(pr[0]['Y_pr'])
        #plt.title(f'{cv_test[0]}, {clf.score(x_test, Y_test)}')
        plt.show()

#%%
#t_range = [3000, 8778] # 3573, 7145
t_range = [8778, 14450]
pos_n = pos_s.copy()
pp = pos_n[t_range[0]:t_range[1]]

# fio = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_3_num_layers_10_trace_with_neg_False_center_dims_None_with_detrending_v3.12.npy', allow_pickle=True).item()
# tt = fio.trace_deconvolved[:,t_range[0]:t_range[1]].copy()
tt = t_rm['FIOLA3000'][t_range[0]:t_range[1]].T
#cnm2 = load_CNMF('/media/nel/storage/fiola/R2_20190219/memmap__d1_796_d2_512_d3_1_order_C_frames_31933__v3.13.hdf5')
#tracec = cnm2.estimates.S
# onacid = load_CNMF('/media/nel/storage/fiola/R2_20190219/full_nonrigid/mov_R2_20190219T210000_caiman_online_results_v3.0_new_params_min_snr_2.0.hdf5')
# select= np.intersect1d(np.where(onacid.time_neuron_added[:, 1] < 10000)[0], onacid.estimates.idx_components)
# traceo = (onacid.estimates.S)[select]#(onacid.estimates.C + onacid.estimates.YrA)[select]

# tt = traceo[:, t_range[0]:t_range[1]].copy()
tt = tt[np.where((tt).sum(1) > 0)[0]]
tt = (tt - tt.min(1)[:, None]) / (tt.max(1)[:, None] - tt.min(1)[:, None])
#tt = (tt - tt.mean(1)[:, None]) / (tt.std(1)[:, None])


t_max = []
p_max = []
for i in range(len(tt)):
    t_max.append(np.argmax(tt[i]))
    p_max.append(pp[np.argmax(tt[i])])
seq = np.argsort(p_max)
tt_rank = tt[seq]

#%%
import matplotlib.gridspec as grd
gs = grd.GridSpec(2, 2, height_ratios=[1,1], width_ratios=[1, 0.05], wspace=0.1)
fig = plt.figure()
ax = plt.subplot(gs[0])
t_s = t_range[0]
t_e = t_range[1]
s_tt = s_t[t_s:t_e]
X = tt_rank.copy()[::-1][:, s_tt]

q = ax.imshow(X, cmap='hot', vmin=0, vmax=0.1, aspect='auto'); ax.set_ylabel('neuron index'); 
ax.set_xlim([0, s_tt.sum()])
ax.spines['bottom'].set_visible(False); ax.get_xaxis().set_ticks([])

ax1 = plt.subplot(gs[1])
cb = plt.colorbar(q, cax = ax1)

ax2 = plt.subplot(gs[2])
ax2.plot(list(p.values())[0][1]['Y'], color='black', linewidth=1)
key_list = ['FIOLA3000', 'Suite2p']#['FIOLA3000', 'CaImAn_Online', 'Suite2p', 'CaImAn']
for key, pr in p.items():
    if key in key_list:
        print(key)
        ax2.plot(pr[1]['Y_pr'], label=f'{key}: {np.round(r[key][1], 2)}', alpha=0.8)
ax2.set_xlim([0, s_tt.sum()]); ax2.set_ylabel('Z-scored location'); ax2.set_xlabel('frames'); ax2.legend()
#plt.savefig(savef + 'Fig7_supp_raster_plot_v3.6.pdf')  

#%%
# import matplotlib.gridspec as grd
# gs = grd.GridSpec(2, 1, height_ratios=[1,1], width_ratios=[1], wspace=0.1)
# fig = plt.figure()
# ax = plt.subplot(gs[0])
# t_s = 3000
# t_e = 8778
# s_tt = s_t[t_s:t_e]
# X = tt_rank.copy()[::-1]
# #fig, ax = plt.subplots(2, 1)

# ax.imshow(X, cmap='hot', vmin=0, vmax=0.1, aspect='auto'); ax.set_ylabel('neuron index'); 
# ax.set_xlim([0, 5778])
# ax.spines['bottom'].set_visible(False); ax.get_xaxis().set_ticks([])
# #ax[1].plot(pp[s_tt]);  
# #
# ax1 = plt.subplot(gs[1])


# y = np.zeros((5778))
# y[s_tt] = pr[0]['Y']
# y[~s_tt] = np.nan
# ax1.plot(y, color='black')
# #ax1.plot(list(p.values())[0][0]['Y'], color='black')
# key_list = ['FIOLA3000', 'CaImAn_Online']
# for key, pr in p.items():
#     if key in key_list:
#         print(key)
#         y_pred = np.zeros((5778))
#         y_pred[s_tt] = pr[0]['Y_pr']
#         y_pred[~s_tt] = np.nan
        
#         #ax1.plot(np.where(s_tt == True)[0], pr[0]['Y_pr'] )
#         ax1.plot(y_pred)
#         #plt.title(f'{cv_test[0]}, {clf.score(x_test, Y_test)}')
#         #plt.show()
# ax1.set_xlim([0, 5778]); ax1.set_ylabel('location'); ax1.set_xlabel('frames'); 

#%% supp fiola processing
fio = np.load('/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_test_v3.15.npy', allow_pickle=True).item()
select = selection['FIOLA3000']
sort = sort_neurons(fio.trace[select].T)
n_list = sort[np.array([2, 3, 4, 5, 6])]
t_range = [3000, 7000]
tt = np.array(range(0, t_range[1]-t_range[0]))
spatial = cnm3000.estimates.A.toarray()[:, select].reshape((796, 512, -1), order='F').transpose([2, 0, 1])
center = com(cnm3000.estimates.A[:, select], d1=796, d2=512)

fig, ax = plt.subplots(len(n_list), 3, figsize=(16, 8))
for i in range(len(n_list)):
    n_idx = n_list[i]
    t_raw = fio.trace[select][n_idx, t_range[0]:t_range[1]]
    scale = t_raw.max()
    t_raw = t_raw / scale
    t_d = fio.t_d[select][n_idx, t_range[0]:t_range[1]] / scale
    t_dec = fio.trace_deconvolved[select][n_idx, t_range[0]:t_range[1]] / scale
    ax[i, 0].plot(tt, t_raw)
    ax[i, 0].plot(tt, t_raw-t_d)
    ax[i, 1].plot(tt, t_d, color='green')
    ax[i, 1].plot(tt, t_dec, color='red')
    
    xx = int(center[n_idx][0])
    yy = int(center[n_idx][1])
    w = 8
    sp = spatial[n_idx][xx-w:xx+w+1, yy-w:yy+w+1]
    ax[i, 2].imshow(sp)
    
    if i == 0:
        ax[i, 0].legend(['raw trace', 'trend'])
        ax[i, 1].legend(['detrended trace', 'deconvolved trace'])
    if i < len(n_list) - 1:
        ax[i, 0].get_xaxis().set_ticks([])
        ax[i, 1].get_xaxis().set_ticks([])
        ax[i, 0].spines['bottom'].set_visible(False)
        ax[i, 1].spines['bottom'].set_visible(False)
        ax[i, 0].spines['left'].set_visible(False)
        ax[i, 1].spines['left'].set_visible(False)
        ax[i, 0].get_yaxis().set_ticks([])
        ax[i, 1].get_yaxis().set_ticks([])
    ax[i, 2].axis('off')
    ax[i, 0].set_ylim([-0.2, 1])
    ax[i, 1].set_ylim([-0.2, 1])
    ax[i, 0].text(-500, 0, f'neuron {n_list[i]}')
plt.tight_layout()
plt.savefig(savef + 'Fig7_supp_fiola_processing_v3.7.pdf')  


#%%
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False})
import matplotlib.pyplot as plt

fig, ax = plt.subplots(7, 1, figsize=(12, 6))
nid = 40
ax[0].plot(np.arange(3000-1, 6000-1), traces[:, nid]/ fio.pipeline.saoz.trace[nid].max())
ax[0].plot(fio.pipeline.saoz.trace[nid] / fio.pipeline.saoz.trace[nid].max())
ax[0].set_xlim((3000, 6000))
ax[0].legend(['retrieved online', 'retrieved afterwards'])
# ax[1].plot(np.arange(3000-lag-1, 6000-lag-1), traces_deconvolved[:, nid])
# ax[1].plot(fio.pipeline.saoz.trace_deconvolved[nid])
for ii, lag in enumerate(lags):
    scale = fio.pipeline.saoz.trace_deconvolved[nid].max()
    ax[ii+1].plot(np.arange(3000-lag-1, 6000-lag-1), traces_deconvolved[lag][:, nid]/scale)
    ax[ii+1].plot(fio.pipeline.saoz.trace_deconvolved[nid]/scale)
    ax[ii+1].set_xlim((3000, 6000))
    
    if ii < 6:
        ax[ii].get_xaxis().set_ticks([])
        ax[ii].get_yaxis().set_ticks([])
        ax[ii].spines['bottom'].set_visible(False)
        ax[ii].spines['left'].set_visible(False)
    else:
        ax[ii].set_ylabel('frames')

    corr = np.round(np.corrcoef(traces_deconvolved[lag][:, nid], fio.pipeline.saoz.trace_deconvolved[nid][3000-lag-1: 6000-lag-1])[0, 1], 3)
    ax[ii+1].text(6000, 0.4, f'lag: {lag}')
    cc = []
    for jj in range(500):
        cc.append(np.corrcoef(traces_deconvolved[lag][:, jj], fio.pipeline.saoz.trace_deconvolved[jj][3000-lag-1: 6000-lag-1])[0, 1])
    cc = np.round(np.mean(cc), 3)
    ax[ii+1].text(6000, 0.2, f'corr: {corr}')
    ax[ii+1].text(6000, 0, f'avg corr: {cc}')
plt.tight_layout()
plt.savefig(savef + 'Fig7_supp_deconvolution_v3.7.pdf')  

#%%
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False})
import matplotlib.pyplot as plt

lags = [0, 1, 3, 5, 10]
trace_deconvolved = {}
for lag in [0, 1, 3, 5, 10]:
    fio = np.load(f'/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_{lag}_test_v3.21.npy', allow_pickle=True).item()
    trace_deconvolved[lag] = fio.online_trace_deconvolved[selection['FIOLA3000']]
trace = fio.online_trace[selection['FIOLA3000']]
fio.trace_deconvolved = fio.trace_deconvolved[selection['FIOLA3000']]
#%%
fig, ax = plt.subplots(6, 1, figsize=(6, 6))
nid = 36
ax[0].plot(np.arange(3000-1, 6000-1), trace[nid, :3000]/ trace[nid].max())
ax[0].plot(fio.trace[nid] / fio.trace[nid].max())
ax[0].set_xlim((3000, 6000))
ax[0].legend(['retrieved online', 'retrieved afterwards'])
# ax[1].plot(np.arange(3000-lag-1, 6000-lag-1), traces_deconvolved[:, nid])
# ax[1].plot(fio.pipeline.saoz.trace_deconvolved[nid])
for ii, lag in enumerate(lags):
    scale = fio.trace_deconvolved[nid].max()
    ax[ii+1].plot(np.arange(3000-lag-1, 6000-lag-1), trace_deconvolved[lag][nid, :3000]/scale)
    #ax[ii+1].plot(np.arange(3000, 6000), trace_deconvolved[lag][nid, lag+1:3000+lag+1]/scale)

    ax[ii+1].plot(fio.trace_deconvolved[nid]/scale)
    ax[ii+1].set_xlim((3000, 6000))
    
    if ii < 6:
        ax[ii].get_xaxis().set_ticks([])
        ax[ii].get_yaxis().set_ticks([])
        ax[ii].spines['bottom'].set_visible(False)
        ax[ii].spines['left'].set_visible(False)
    else:
        ax[ii].set_ylabel('frames')

    corr = np.round(np.corrcoef(trace_deconvolved[lag][nid], fio.trace_deconvolved[nid][3000-lag-1: -lag-1])[0, 1], 3)
    ax[ii+1].text(6000, 0.4, f'lag: {lag}')
    cc = []
    for jj in range(1307):
        #cc.append(np.corrcoef(trace_deconvolved[lag][jj, :3000], fio.trace_deconvolved[jj][3000-lag-1: 6000-lag-1])[0, 1])
        cc.append(np.corrcoef(trace_deconvolved[lag][jj, ], fio.trace_deconvolved[jj][3000-lag-1: -lag-1])[0, 1])

    cc = np.round(np.mean(cc), 3)
    ax[ii+1].text(6000, 0.2, f'corr: {corr}')
    ax[ii+1].text(6000, 0, f'avg corr: {cc}')
plt.tight_layout()
#plt.savefig(savef + 'Fig7_supp_deconvolution_v3.8.pdf')  

#%%
lag = ['lag1', 'lag3'][1]
ff = '/media/nel/storage/fiola/R2_20190219/result/'
r = np.load(ff+f'Fig7_supp_dec_lag_v3.8.npy', allow_pickle=True).item()
#p = np.load(ff+f'Fig7c_prediction_{lag}_v3.8.npy', allow_pickle=True).item()
fig = plt.figure() 
ax1 = plt.subplot()
rr = list(r.values())
methods = list(r.keys())
#num = [tt.shape[1] for tt in t_g.values()]
#num = [1307, 1065, 907, 549, 2006, 1788, 1990]
num = [0, 1, 3, 5, 10, 20]
r_mean = [np.mean(x) for x in rr]
r_std = [np.std(x) for x in rr]

colors = ['C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C3', 'C6']

for idx in range(len(list(r.keys()))):
    if idx < 5:
        ax1.errorbar(num[idx], r_mean[idx], yerr=r_std[idx], fmt='o', capsize=5, color=colors[idx], label=methods[idx])

ax1.locator_params(axis='y', nbins=8)
ax1.locator_params(axis='x', nbins=4)
ax1.set_ylabel('Decoding R square')
ax1.set_xlabel('Lag (frame)')
ax1.legend()
ax1.set_ylim([0.85, 0.95])
plt.tight_layout()
plt.savefig(savef + 'Fig7_supp_deconvolution_b_v3.8.pdf')  



#%%
for mm in list(t_g.keys())[::-1]:
    print(mm)
    #plt.figure()
    ta = t_raw[mm][:, 120]
    plt.plot(ta / ta.max(), alpha=0.3)
    
#%%
lag=0
ttt = np.load(f'/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_{lag}_test_v3.21.npy', allow_pickle=True).item()
#ttt = np.load(f'/media/nel/storage/fiola/R2_20190219/3000/fiola_result_init_frames_3000_iteration_1_num_layers_10_trace_with_neg_False_center_dims_(398, 256)_lag_{lag}_test_v3.22.npy', allow_pickle=True).item()
corr=[]
for ii in range(1307):
    c = np.corrcoef(ttt.online_trace_deconvolved[ii, :], ttt.trace_deconvolved[ii, 3000-lag-1:-lag-1])[0, 1]
    corr.append(c)
    
print(np.mean(corr))
