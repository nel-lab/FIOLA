#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:51:16 2021

@author: nel
"""
#%%
import h5py
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/simulation_non_rigid_data'
save_name = 'rotation_10_512_512.hdf5'
from fiola.utilities import play, bin_median

with h5py.File(os.path.join(save_folder, save_name), 'r') as hf:
    data = hf['mov'][:]
play(data, fr=2)    

templ = data[0].copy()


#%%
import math
import numpy as np
import os
import pylab as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import timeit
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
os.environ["TF_XLA_FLAGS"]="--tf_xla_enable_xla_devices" 
from fiola.image_warp_3d import trilinear_interpolate_tf,  dense_image_warp_3D    
from use_cases.mc_3d_testing import MotionCorrect

#%%
template = tf.convert_to_tensor(templ, dtype=tf.float32)
strides = (16, 16)#(96, 96)
overlaps = (8, 8)#(32, 32)
batch_size = 10

dims = templ.shape
patch_dim = (strides[0] + overlaps[0], strides[1] + overlaps[1])

templates = [
    it[-1] for it in sliding_window(template, overlaps=overlaps, strides=strides)]
templates = tf.convert_to_tensor(templates)
xy_grid = [(it[0], it[1]) for it in sliding_window(
    template, overlaps=overlaps, strides=strides)]
num_tiles = tf.reduce_prod(tf.add(xy_grid[-1], 1)).numpy()
dim_grid = tuple(tf.add(xy_grid[-1], 1))

hx = templates.shape[1] // 2
hy = templates.shape[2] // 2

imgs = []
for img in data:
    imgs.append([it[-1]
        for it in sliding_window(img, overlaps=overlaps, strides=strides)])
imgs = np.array(imgs)


#%%
plt.imshow(templates[0])

from tensorflow.signal import fft2d, ifft2d
imgs_fr = fft2d(tf.cast(imgs, tf.complex128))
templates_fr = fft2d(tf.cast(templates, tf.complex128))
#temp = fft2d(tf.cast(imgs[1,1], tf.complex128))
product = imgs_fr *  tf.math.conj(templates_fr)
correlation = tf.cast(tf.math.abs(ifft2d(product)), tf.float32)

#corr = tf.reshape(correlation, (batch_size*num_tiles, correlation.shape[-2], correlation.shape[-1]))
rigid_shts = np.array([0.1, 0.2] * (batch_size * num_tiles)).reshape((num_tiles * batch_size, -1))
#rigid_shts = np.array([0.1, 0.2] * batch_size).reshape(batch_size, -1)
#rigid_shts = rigid_shts + np.random.rand(rigid_shts.shape[0], rigid_shts.shape[1]) / 3 
max_deviation_rigid = [5, 5]

if max_deviation_rigid is not None:
    lb_shifts = tf.cast(tf.math.ceil(tf.subtract(
        rigid_shts, max_deviation_rigid)), tf.int64)
    ub_shifts = tf.cast(tf.math.floor(tf.add(
        rigid_shts, max_deviation_rigid)), tf.int64)
else:
    lb_shifts = None
    ub_shifts = None
    
corr = tf.roll(correlation,(hx, hy), axis=(2, 3))
corr = tf.reshape(corr, (batch_size*num_tiles, patch_dim[0], patch_dim[1]))
cc = []

for i in range(batch_size * num_tiles):
    cc.append(corr[i, (hx + lb_shifts[i, 0]) : (hx + ub_shifts[i, 0] + 1), 
                   hy + lb_shifts[i, 1] : hy + ub_shifts[i, 1] + 1])
cc = tf.stack(cc)

shifts_int = tf.reshape(argmax_2d(cc), (batch_size*num_tiles, -1))
sh_x, sh_y = shifts_int[:,0],shifts_int[:,1]
sh_x_n = tf.cast(-(sh_x + tf.cast(lb_shifts[:,0], tf.float32)), tf.float32)
sh_y_n = tf.cast(-(sh_y + tf.cast(lb_shifts[:,1], tf.float32)), tf.float32)

ncc = cc
ncc_log = tf.math.log(ncc)

ii = np.arange(ncc.shape[0])
sh_x = tf.cast(sh_x, tf.int32)
sh_y = tf.cast(sh_y, tf.int32)
idx = tf.transpose(tf.stack([ii, sh_x-1, sh_y]))
log_xm1_y = tf.gather_nd(ncc_log, idx)
idx = tf.transpose(tf.stack([ii, sh_x+1, sh_y]))
log_xp1_y = tf.gather_nd(ncc_log, idx)
idx = tf.transpose(tf.stack([ii, sh_x, sh_y-1]))
log_x_ym1 = tf.gather_nd(ncc_log, idx)
idx = tf.transpose(tf.stack([ii, sh_x, sh_y+1]))
log_x_yp1 =  tf.gather_nd(ncc_log, idx)
idx = tf.transpose(tf.stack([ii, sh_x, sh_y]))
four_log_xy = 4 * tf.gather_nd(ncc_log, idx)

sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))

#%%
import tensorflow_addons as tfa
fr_corrected = tfa.image.translate(tf.reshape(imgs, (num_tiles * batch_size, patch_dim[0], patch_dim[1], 1)), 
                                   tf.transpose(tf.stack([sh_y_n, sh_x_n]), perm=(1, 0)), interpolation="bilinear")
fr_corrected = fr_corrected[..., 0]
fr_corrected = tf.reshape(fr_corrected, (batch_size, num_tiles, patch_dim[0], patch_dim[1]))

#%%
plt.imshow(cc[0,0])

play(fr_corrected[:,13, :, :].numpy(), fr=2)

play(imgs[:,13, :, :], fr=2)

play(np.concatenate([imgs[:, 88, :, :], fr_corrected[:,88, :, :].numpy()], axis=2), fr=1)
    

#%%
#shfts = [sshh[0] for sshh in shfts_et_all]
#diffs_phase = [sshh[2] for sshh in shfts_et_all]
# create a vector field
shift_img_x = -np.reshape(sh_x_n.numpy(), (batch_size, dim_grid[0].numpy(), dim_grid[1].numpy()))
shift_img_y = -np.reshape(sh_y_n.numpy(), (batch_size, dim_grid[0].numpy(), dim_grid[1].numpy()))
#diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)


y_grid, x_grid = np.meshgrid(np.arange(0., dims[1]).astype(
    np.float32), np.arange(0., dims[0]).astype(np.float32))

data_corrected = []
data_corrected1 = []

for idx, img in enumerate(data):
    m_reg = cv2.remap(img, cv2.resize(shift_img_y[idx].astype(np.float32), dims[::-1]) + y_grid,
                      cv2.resize(shift_img_x[idx].astype(np.float32), dims[::-1]) + x_grid,
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    m_reg1 = cv2.remap(img, cv2.resize(shift_img_x[idx].astype(np.float32), dims[::-1]) + y_grid,
                      cv2.resize(shift_img_y[idx].astype(np.float32), dims[::-1]) + x_grid,
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    data_corrected.append(m_reg)
    data_corrected1.append(m_reg1)

    
data_corrected = np.array(data_corrected)
data_corrected1 = np.array(data_corrected1)
 

#%%
rr = np.concatenate([data, data_corrected], axis=2)
play(rr, fr=2)   
    
#total_shifts = [
#        (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]


#%%
imgs_test = []
imgs_test1 = []

for img in data_corrected:
    imgs_test.append([it[-1]
        for it in sliding_window(img, overlaps=overlaps, strides=strides)])
imgs_test = np.array(imgs_test)

for img in data_corrected1:
    imgs_test1.append([it[-1]
        for it in sliding_window(img, overlaps=overlaps, strides=strides)])
imgs_test1 = np.array(imgs_test1)



play(np.concatenate([imgs[:, 88, :, :], fr_corrected[:,88, :, :].numpy(), 
                     imgs_test[:, 88, :, :], imgs_test1[:, 88, :, :]], axis=2), fr=1)


#%%
def argmax_2d(tensor):
    # extract peaks from 2D tensor (takes batches as input too)
    
    # flatten the Tensor along the height and width axes
    flat_tensor = tf.reshape(tensor, (tensor.shape[0], tensor.shape[1]*tensor.shape[2]))

    argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
    # convert indexes into 2D coordinates
    argmax_x = tf.cast(argmax, tf.int32) // tf.shape(tensor)[2]
    argmax_y = tf.cast(argmax, tf.int32) % tf.shape(tensor)[2]
    # stack and return 2D coordinates
    return tf.cast(tf.stack((argmax_x, argmax_y), axis=1), tf.float32)
    








#%%
mc_layer = MotionCorrect(template[:,:,:,None,None], ms_h=ms_h, ms_w=ms_w, ms_d=ms_d)
data = Y[None, ..., None]
print(f'data shape: {data.shape}')
print(f'template shape: {template.shape}')
import time
st = time.time()
times = []
nccs = []
shifts_gpu_mc = np.zeros((Y.shape[0], 3))
for i in range(data.shape[1]):
    _, ncc = mc_layer(data[:,i:i+1])
    coords = np.array([ncc[0],ncc[1],ncc[2]]).flatten()
    shifts_gpu_mc[i] = coords
    #shifts_all.append(mc_layer.shifts)
    times.append(time.time()-st)
    
    #%%
    batch = 16
    shifts_gpu_mc = np.zeros((Y.shape[0], 3))
    for i in range(data.shape[1]//batch):
        _, ncc = mc_layer(data[:,batch*i:batch*(i+1)])
        coords = np.array([ncc[0],ncc[1],ncc[2]]).T
        shifts_gpu_mc[batch*i:batch*(i+1)] = coords
        #shifts_all.append(mc_layer.shifts)
        times.append(time.time()-st)







#%%
def sliding_window(image, overlaps, strides):
    """ efficiently and lazily slides a window across the image

    Args: 
        img:ndarray 2D
            image that needs to be slices

        windowSize: tuple
            dimension of the patch

        strides: tuple
            stride in each dimension

     Returns:
         iterator containing five items
              dim_1, dim_2 coordinates in the patch grid
              x, y: bottom border of the patch in the original matrix

              patch: the patch
     """
    windowSize = np.add(overlaps, strides)
    range_1 = list(range(
        0, image.shape[0] - windowSize[0], strides[0])) + [image.shape[0] - windowSize[0]]
    range_2 = list(range(
        0, image.shape[1] - windowSize[1], strides[1])) + [image.shape[1] - windowSize[1]]
    for dim_1, x in enumerate(range_1):
        for dim_2, y in enumerate(range_2):
            # yield the current window
            yield (dim_1, dim_2, x, y, image[x:x + windowSize[0], y:y + windowSize[1]])


#%%
    
if (lb_shifts is not None) or (ub_shifts is not None):

    if (lb_shifts[0] < 0) and (ub_shifts[0] >= 0):
        corr[ub_shifts[0]:lb_shifts[0], :, :] = 0
    else:
        corr[:lb_shifts[0], :, :] = 0
        corr[ub_shifts[0]:, :, :] = 0

    if (lb_shifts[1] < 0) and (ub_shifts[1] >= 0):
        corr[:, ub_shifts[1]:lb_shifts[1], :] = 0
    else:
        corr[:, :lb_shifts[1], :] = 0
        corr[:, ub_shifts[1]:, :] = 0

    if (lb_shifts[2] < 0) and (ub_shifts[2] >= 0):
        corr[:, :, ub_shifts[2]:lb_shifts[2]] = 0
    else:
        corr[:, :, :lb_shifts[2]] = 0
        corr[:, :, ub_shifts[2]:] = 0
else:
    corr[max_shifts[0]:-max_shifts[0], :, :] = 0
    corr[:, max_shifts[1]:-max_shifts[1], :] = 0
    corr[:, :, max_shifts[2]:-max_shifts[2]] = 0
    