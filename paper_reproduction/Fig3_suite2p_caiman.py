#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:03:22 2021

@author: nel
"""

#%% load movie
m = cm.load('/home/nel/caiman_data/example_movies/demoMovie/demoMovie.tif')
dims = m.shape

#%% load suite2p result
folder = '/home/nel/caiman_data/example_movies/demoMovie'
s_folder = os.path.join(folder, 'suite2p', 'plane0')
s_C = np.load(os.path.join(s_folder, 'F.npy'), allow_pickle=True)
plt.plot(s_C.T)
stat = np.load(os.path.join(s_folder, 'stat.npy'), allow_pickle=True)
ops = np.load('/home/nel/caiman_data/example_movies/demoMovie/suite2p/plane0/ops.npy', allow_pickle=True)
iscell = np.load('/home/nel/caiman_data/example_movies/demoMovie/suite2p/plane0/iscell.npy', allow_pickle=True)
ops.item().keys()
temp = np.zeros((s_C.shape[0], dims[1], dims[2]))
for i in range(len(temp)):
    temp[i][stat[i]['ypix'], stat[i]['xpix']] = 1
s_A = temp
plt.imshow(s_A.sum(0))

#%% load CaImAn result
from caiman.source_extraction.cnmf.cnmf import load_CNMF
cnm2 = load_CNMF('/home/nel/caiman_data/example_movies/demoMovie/caiman/memmap__d1_60_d2_80_d3_1_order_C_frames_2000_.hdf5')
c_C = cnm2.estimates.C
#plt.plot(c_C.T)
c_C_noisy = cnm2.estimates.C + cnm2.estimates.YrA
#plt.plot(c_C_noisy.T)
c_A = cnm2.estimates.A.toarray().reshape((dims[1], dims[2], -1), order='F').transpose([2, 0, 1])
plt.imshow(c_A.sum(0))
c_A_binary = c_A.copy()
c_A_binary[c_A_binary>np.percentile(c_A_binary, 98)] = 1
c_A_binary[c_A_binary<1] = 0
plt.imshow(c_A_binary.sum(0))

#%% match spatial footprints
from caiman.base.rois import nf_match_neurons_in_binary_masks
idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp, performance = nf_match_neurons_in_binary_masks(s_A, c_A_binary, Cn=m.mean(0), plot_results=True)

#%% mean ROI
m_C = np.dot(s_A[idx_tp_gt].reshape([-1, dims[1]*dims[2]], order='F'), 
       m.reshape([-1, dims[1]*dims[2]], order='F').T)

m_C_c = np.dot(c_A_binary[idx_tp_comp].reshape([-1, dims[1]*dims[2]], order='F'), 
       m.reshape([-1, dims[1]*dims[2]], order='F').T)

#%% compute corr
plt.figure()
corr = []  
for i in range(len(idx_tp_gt)):
    corr.append(np.corrcoef(s_C[idx_tp_gt][i], c_C_noisy[idx_tp_comp][i])[0, 1])
plt.boxplot(corr)

#%%
#i = 8
# 0, 11, 15, 
i = 15
plt.figure()
plt.imshow(s_A[idx_tp_gt][i])
plt.figure()
plt.imshow(c_A[idx_tp_comp][i])
plt.figure()
plt.plot(nor(s_C[idx_tp_gt][i]))
plt.plot(nor(c_C_noisy[idx_tp_comp][i]))
plt.legend(['suite2p', 'caiman'])
#plt.plot(nor(m_C[i]))
#plt.plot(nor(m_C_c[i]), alpha=0.8, color='gray')
plt.figure()
plt.plot(nor(s_C[idx_tp_gt][i]))
plt.plot(nor(m_C_c[i]))
#plt.plot(nor(c_C_noisy[idx_tp_comp][i]))
#plt.plot(nor(m_C[i]))

plt.legend(['suite2p', 'meanroi_c'])
print(np.corrcoef(s_C[idx_tp_gt][i], c_C_noisy[idx_tp_comp][i])[0,1])
print(np.corrcoef(s_C[idx_tp_gt][i], m_C_c[i])[0,1])

#%% mean roi corr
plt.figure()
m_corr = []  
for i in range(len(idx_tp_gt)):
    m_corr.append(np.corrcoef(s_C[idx_tp_gt][i], m_C[i])[0, 1])
plt.boxplot(m_corr)

#%% mean roi from spatial of caiman
plt.figure()
m_corr_c = []  
for i in range(len(idx_tp_gt)):
    m_corr_c.append(np.corrcoef(s_C[idx_tp_gt][i], m_C_c[i])[0, 1])
plt.boxplot(m_corr_c)

#%%
plt.boxplot([corr, m_corr, m_corr_c])
plt.legend(['suite2p_caiman', 'suite2p_meanroi', 'suite2p_meanroi_with_caiman_spatial'])

#%%
plt.imshow(np.corrcoef(s_C[idx_tp_gt]))
mat = np.corrcoef(s_C[idx_tp_gt])
mat = (np.triu(mat,1))
np.mean(mat[mat>0])

mat = np.corrcoef(c_C_noisy[idx_tp_comp])
mat = (np.triu(mat,1))
np.mean(mat[mat>0])

mat = np.corrcoef(m_C)
mat = (np.triu(mat,1))
np.mean(mat[mat>0])


#%%
def nor(t):
    t = t - t.mean()
    t = t / t.max()
    return t








