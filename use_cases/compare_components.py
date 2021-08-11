#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:04:38 2021

@author: nel
"""

from caiman.source_extraction.cnmf import cnmf 
from caiman.base.rois import nf_match_neurons_in_binary_masks
from fiola.utilities import normalize
fcaiman = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/k53/memmap__d1_512_d2_512_d3_1_order_C_frames_3000_.hdf5'
cnm2 = cnmf.load_CNMF(fcaiman)

aa = cnm2.estimates.A.toarray().copy()
aa = aa.reshape([512, 512, -1]).transpose([2, 0, 1])
aa[aa > np.median(aa)] = 1
aa[aa != 1] = 0
plt.imshow(aa.sum(0))

cc = cnm2.estimates.C + cnm2.estimates.YrA

hh = fio.H.reshape([512, 512, -1]).transpose([2, 0, 1])[:-2].copy()
hh[hh > np.median(hh)] = 1
hh[hh != 1] = 0
plt.imshow(hh.sum(0))

ii = fio.estimates.copy()


idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp, performance = nf_match_neurons_in_binary_masks(aa, hh)

#%%
for idx in idx_list:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'caiman, fiola spatial footprints, {idx}')
    ax1.imshow(aa[idx_tp_gt[idx]])
    ax2.imshow(hh[idx_tp_comp[idx]])
    plt.figure()
    plt.plot(cc[idx_tp_gt[idx]])
    plt.plot(ii[idx_tp_comp[idx]])
    plt.text(0, -10, f'{snr[idx]},{corr[idx]}')
    plt.legend(['caiman', 'fiola'])

#%%
num_frames_init = 1500
corr = []
snr = []
for idx in range(len(idx_tp_gt)):
    snr.append(normalize(cc[:,num_frames_init:][idx_tp_gt[idx]]).max())
    corr.append(np.corrcoef(cc[:,num_frames_init:][idx_tp_gt[idx]], ii[:,num_frames_init:][idx_tp_comp[idx]])[0,1])
    
snr = np.array(snr); corr = np.array(corr)

idx_list = np.where(np.logical_and(snr > 15,  snr > 15))[0]
plt.scatter(snr[idx_list], corr[idx_list]); plt.xlim([0, 50])

#%%
plt.boxplot(corr[idx_list])
plt.title('Corr')
print(np.mean(corr[idx_list]))
print(np.std(corr[idx_list]))



