#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:07:13 2021

@author: cxd00
"""
from viola.nnls_gpu import NNLS, compute_theta2
from viola.motion_correction_gpu import MotionCorrect
import numpy  as np
from skimage import io
import os
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#%%
base_folder = "../../../NEL-LAB Dropbox/NEL/Papers/VolPy_online/CalciumData/DATA_PAPER_ELIFE/"
dataset = "N.00.00"
slurm_data = base_folder + dataset + "/results_analysis_online_sensitive_SLURM_01.npz"
#%% get ground truth data
with np.load(slurm_data, allow_pickle=True) as ld:
    print(list(ld.keys()))
    locals().update(ld)
    Ab_gt = Ab[()].toarray()
    num_bckg = f.shape[0]
    b_gt, A_gt = Ab_gt[:,:num_bckg], Ab_gt[:,num_bckg:]
    num_comps = Ab_gt.shape[-1]
    f_gt, C_gt = Cf[:num_bckg], Cf[num_bckg:num_comps]
    noisyC = noisyC[num_bckg:num_comps]
    
#%% start comps : save the indices where each spatial footprint first appears
start_comps = []
for i in range(num_comps):
    start_comps.append(np.where(np.diff(Cf[i])>0)[0][0])

#%% initialize with i frames, calculate GT
for i in [len(noisyC[0])//4]: #50%
    included_comps = np.where(np.array(start_comps)<i)[0]
    A_start = A_gt[:,included_comps]
    C_start = C_gt[included_comps]
    noisyC_start = noisyC[included_comps] # ground truth
    b_gt = b_gt
    f_gt = f_gt 
    
#%% pick up images as a movie file and normalize
dirname = base_folder + dataset + "/images_" + dataset
a2 = []
for i in range(noisyC.shape[1]//2):
    fname = "image" + str(i).zfill(5) + ".tif"
    im = io.imread(os.path.join(dirname, fname))
    a2.append(im)

img_norm = np.std(a2, axis=0)
img_norm += np.median(img_norm)
a2 = a2/img_norm[None, :, :]

#%% one-time calculations and template
template = np.median(a2[:len(a2)//2], axis=0) # template created w/ first half
Ab_gt_start = np.concatenate([A_start, b_gt], axis=1).astype(np.float32)
AtA = Ab_gt_start.T@Ab_gt_start
Atb = Ab_gt_start.T@b_gt
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_1 = (np.eye(Ab_gt_start.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)


#Cff = np.concatenate([C_full+YrA_full,f_full], axis=0)
Cf = np.concatenate([noisyC_start,f_gt], axis=0)
x0 = Cf[:,0].copy()[:,None]

#%% create debug-able model
mc = MotionCorrect(template, a2[0].shape)
ct2 = compute_theta2(Ab_gt_start, n_AtA)
reg = NNLS(theta_1)

for i in range(20):
    mc_out, shifts = mc(a2[i, :, :, None][None, :].astype(np.float32))
    ct2_out, shifts = ct2(mc_out, shifts)
    nnls_out = reg([ x0[None, :],  x0[None, :], tf.convert_to_tensor(np.zeros_like(1), dtype=tf.int8), ct2_out, shifts])
    for k in range(1, 10):
        nnls_out = reg(nnls_out)

    y_old, x_old = nnls_out[0], nnls_out[1]
    
#%% NNLS ISSUE TESTING
from viola.nnls_gpu import NNLS, compute_theta2
from scipy.optimize import nnls

Ab = H_new.astype(np.float32)
b = mov[0].reshape(-1, order='F')
x0 = nnls(Ab,b)[0][:,None].astype(np.float32)
x_old, y_old = x0, x0
AtA = Ab.T@Ab
Atb = Ab.T@b
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
mc0 = mov[0:1,:,:, None]
        
c_th2 = compute_theta2(Ab, n_AtA)
n = NNLS(theta_1)
num_layers = 10
#%%
nnls_out = []
k = np.zeros_like(1)
for i in range(20000):
    mc = np.reshape(np.transpose(mov[i]), [-1])[None, :]
    th2 = c_th2(mc)
    x_kk = n([y_old, x_old, k, th2])

    
    for j in range(1, num_layers):
        x_kk = n(x_kk)
        
    y_old, x_old = x_kk[0], x_kk[1]
    nnls_out.append(y_old)