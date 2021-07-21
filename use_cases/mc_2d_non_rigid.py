#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:51:16 2021

@author: nel
"""
#%%
import cv2
import h5py
import math
import numpy as np
import os
import pylab as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.signal import fft2d, ifft2d
import tensorflow_addons as tfa
import timeit
#from fiola.image_warp_3d import trilinear_interpolate_tf,  dense_image_warp_3D #will cause error when import
from fiola.utilities import play, bin_median
from use_cases.mc_3d_testing import MotionCorrect
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
os.environ["TF_XLA_FLAGS"]="--tf_xla_enable_xla_devices" 
import logging
from fiola.utilities import resize

#%%
import h5py
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/simulation_non_rigid_data'
save_name = 'rotation_10_512_512.hdf5'
with h5py.File(os.path.join(save_folder, save_name), 'r') as hf:
    data = hf['mov'][:]
data = data[:, 50:500, 100:500]    
templ = data[0]
#data = np.array([templ] * 10)
#templ = np.median(data, axis=0)

template = templ.copy()
#template = tf.convert_to_tensor(templ, dtype=tf.float32)
strides = (96, 96) #
overlaps = (32, 32) #
batch_size = 10
ms_h = 5; ms_w = 5; ms_d = 0

#%%
sig = [10, 10]
T = 20
np.random.seed(3)
shifts = np.transpose([np.convolve(np.random.randn(T-10)*4, np.ones(11)/11*s) for s in sig])
shifts = shifts[:10]
shifts.astype(np.int16)
plt.figure()
shifts = shifts.astype(np.int32)
plt.plot(shifts)
new_data = []
for idx, d in enumerate(data):
    new_data.append(np.roll(d, shifts[idx], axis=(0,1)))
data = np.array(new_data)

#data = tfa.image.translate(data[..., None].astype(np.float32), shifts.astype(np.float32), interpolation="bilinear")[:, :, :, 0].numpy()
play(data, fr=2)    

#%%
mc_layer = MotionCorrect(template[:,:,None,None,None], ms_h=5, ms_w=5, ms_d=0)
print(f'data shape: {data.shape}')
print(f'template shape: {template.shape}')
dd = data[..., None]
Y = dd
dd = dd[None, ..., None]
import time
st = time.time()
times = []
nccs = []
shifts_gpu_mc = np.zeros((Y.shape[0], 3))
for i in range(dd.shape[1]):
    _, ncc = mc_layer(dd[:,i:i+1])
    coords = np.array([ncc[0],ncc[1],ncc[2]]).flatten()
    shifts_gpu_mc[i] = coords
    #shifts_all.append(mc_layer.shifts)
    times.append(time.time()-st)
    
plt.plot(-shifts_gpu_mc)

#%%
class MotionCorrect_non_rigid_2d(keras.layers.Layer):
    def __init__(self, template, ms_h=10, ms_w=10, ms_d=10, max_deviation_rigid=[5,5], 
                 strides=(96, 96), overlaps=(32, 32), padding='SAME', epsilon=0.00000001, **kwargs):
        
        super().__init__(**kwargs)        
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)
                print(f'{name}, {value}')
        self.dims = template.shape
        print(f'template shape: {template.shape}')
        
        
        # initialize the rigid motion correction
        self.rigid_mc_layer = MotionCorrect(template[:,:,None,None,None], ms_h=ms_h, ms_w=ms_w, ms_d=ms_d)

        # divide the template into multiple patches for non-rigid motion correction        
        if max_deviation_rigid is not None:
            self.patch_dim = (self.strides[0] + self.overlaps[0], self.strides[1] + self.overlaps[1])
            templates = [
                it[-1] for it in self.sliding_window(self.template, self.overlaps, self.strides)]
            self.templates = tf.convert_to_tensor(templates)
            self.templates_fr = fft2d(tf.cast(self.templates, tf.complex128))
            self.xy_grid = [(it[0], it[1]) for it in self.sliding_window(
                self.template, self.overlaps, self.strides)]
            self.num_tiles = tf.reduce_prod(tf.add(self.xy_grid[-1], 1)).numpy()
            self.dim_grid = tuple(tf.add(self.xy_grid[-1], 1))
            self.hx = self.templates.shape[1] // 2
            self.hy = self.templates.shape[2] // 2
            
    def call(self, data):
        print(f'input data shape: {data.shape}')
        self.data = data
        self.batch_size = data.shape[0]
        # rigid motion correction
        data_corrected_rigid, rigid_shifts = self.rigid_mc_layer(data[None,..., None, None])
        data_corrected_rigid = data_corrected_rigid[..., 0]
        self.rigid_shifts = np.array([rigid_shifts[0],rigid_shifts[1],rigid_shifts[2]]).T
        
        if self.max_deviation_rigid is None:
            return data_corrected_rigid, self.rigid_shifts
        else:
            # non-rigid motion correction
            # divide the input movie into patches
            imgs = []
            for img in data:
                imgs.append([it[-1]
                    for it in self.sliding_window(img, self.overlaps, self.strides)])
            imgs = np.array(imgs)
            
            # compute correlation
            imgs_fr = fft2d(tf.cast(imgs, tf.complex128))
            product = imgs_fr *  tf.math.conj(self.templates_fr)
            correlation = tf.cast(tf.math.abs(ifft2d(product)), tf.float32)

            #import pdb
            #pdb.set_trace()
            self.rigid_shts = -self.rigid_shifts[:, :2]#[:, ::-1] 
            #self.rigid_shts = self.rigid_shifts[:, :2]#[:, ::-1] 
            self.rigid_shts = np.repeat(self.rigid_shts, self.num_tiles, axis=0)
            #rigid_shts = np.array([0.01, 0.01] * (self.batch_size * self.num_tiles)).reshape((self.num_tiles * self.batch_size, -1))
            #self.rigid_shts = rigid_shts + np.random.rand(rigid_shts.shape[0], rigid_shts.shape[1]) / 3 

            # compute the maximum shifts for each patch
            self.lb_shifts = tf.cast(tf.math.floor(tf.subtract(
                self.rigid_shts, self.max_deviation_rigid)), tf.int64)   # floor not ceil
            self.ub_shifts = tf.cast(tf.math.floor(tf.add(
                self.rigid_shts, self.max_deviation_rigid)), tf.int64)
            
            corr = tf.roll(correlation,(self.hx, self.hy), axis=(2, 3))
            corr = tf.reshape(corr, (self.batch_size * self.num_tiles, self.patch_dim[0], self.patch_dim[1]))
            ncc = []
            for i in range(self.batch_size * self.num_tiles):
                ncc.append(corr[i, (self.hx + self.lb_shifts[i, 0]) : (self.hx + self.ub_shifts[i, 0] + 1), 
                               self.hy + self.lb_shifts[i, 1] : self.hy + self.ub_shifts[i, 1] + 1])
            ncc = tf.stack(ncc)

            # extract integer shifts for each patch            
            shifts_int = tf.reshape(self.argmax_2d(ncc), (self.batch_size*self.num_tiles, -1))
            
            # extract fractional shifts for each patch
            sh_x_n, sh_y_n = self.extract_fractional_shifts(shifts_int, ncc)
            self.shift_img_x = -np.reshape(sh_x_n.numpy(), (self.batch_size, self.dim_grid[0].numpy(), self.dim_grid[1].numpy()))
            self.shift_img_y = -np.reshape(sh_y_n.numpy(), (self.batch_size, self.dim_grid[0].numpy(), self.dim_grid[1].numpy()))
            
            # estimate the shift for each pixel and apply the estimated shifts
            data_corrected_pw = self.apply_shifts_tf(data, self.shift_img_x, self.shift_img_y)
            
            return data_corrected_pw, data_corrected_rigid, self.rigid_shifts, [self.shift_img_x, self.shift_img_y]
            
    def apply_shifts_tf(self, data, shift_img_x, shift_img_y):
        y_grid, x_grid = np.meshgrid(np.arange(0., self.dims[1]).astype(
            np.float32), np.arange(0., self.dims[0]).astype(np.float32))
        warp = tf.stack([tf.image.resize(shift_img_y[..., None].astype(np.float32), self.dims)[..., 0] + y_grid, 
                         tf.image.resize(shift_img_x[..., None].astype(np.float32), self.dims)[..., 0] + x_grid], axis=3)
        data_corrected_tf = (tfa.image.resampler(tf.cast(data[..., None], tf.float32), warp))
        data_corrected_tf = data_corrected_tf[..., 0]
        return data_corrected_tf
    
    def apply_shifts_cv2(self, sh_x_n, sh_y_n):
        y_grid, x_grid = np.meshgrid(np.arange(0., self.dims[1]).astype(
            np.float32), np.arange(0., self.dims[0]).astype(np.float32))
        data_corrected = []
        for idx, img in enumerate(self.data):
            m_reg = cv2.remap(img, cv2.resize(shift_img_y[idx].astype(np.float32), dims[::-1]) + y_grid,
                              cv2.resize(shift_img_x[idx].astype(np.float32), dims[::-1]) + x_grid,
                              cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            data_corrected.append(m_reg)    
        data_corrected = np.array(data_corrected)
    
    def extract_fractional_shifts(self, shifts_int, ncc):
        sh_x, sh_y = shifts_int[:,0],shifts_int[:,1]
        sh_x_n = tf.cast(-(sh_x + tf.cast(self.lb_shifts[:,0], tf.float32)), tf.float32)
        sh_y_n = tf.cast(-(sh_y + tf.cast(self.lb_shifts[:,1], tf.float32)), tf.float32)
        ncc_log = tf.math.log(ncc)
        
        # Gaussian interpolation
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
        return sh_x_n, sh_y_n

    def argmax_2d(self, tensor):
        # extract peaks from 2D tensor (takes batches as input too)
        
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tensor.shape[0], tensor.shape[1]*tensor.shape[2]))
    
        argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
        # convert indexes into 2D coordinates
        argmax_x = tf.cast(argmax, tf.int32) // tf.shape(tensor)[2]
        argmax_y = tf.cast(argmax, tf.int32) % tf.shape(tensor)[2]
        # stack and return 2D coordinates
        return tf.cast(tf.stack((argmax_x, argmax_y), axis=1), tf.float32)
    
    def sliding_window(self, image, overlaps, strides):
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
    
    def high_pass_filter_space(self, img_orig, gSig_filt=None):
        """
        Function for high passing the image(s) with centered Gaussian if gSig_filt
        is specified or Butterworth filter if freq and order are specified
    
        Args:
            img_orig: 2-d or 3-d array
                input image/movie
    
            gSig_filt:
                size of the Gaussian filter 
    
            freq: float
                cutoff frequency of the Butterworth filter
    
            order: int
                order of the Butterworth filter
    
        Returns:
            img: 2-d array or 3-d movie
                image/movie after filtering            
        """
        ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
        ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
                
        if len(ksize) <= 2:
            ker2D = ker.dot(ker.T)
            nz = np.nonzero(ker2D >= ker2D[:, 0].max())
            zz = np.nonzero(ker2D < ker2D[:, 0].max())
            ker2D[nz] -= ker2D[nz].mean()
            ker2D[zz] = 0
            if img_orig.ndim == 2:  # image
                return cv2.filter2D(np.array(img_orig, dtype=np.float32),
                                    -1, ker2D, borderType=cv2.BORDER_REFLECT)
            else:  # movie
                mm = np.array([cv2.filter2D(np.array(img, dtype=np.float32),
                                    -1, ker2D, borderType=cv2.BORDER_REFLECT) for img in img_orig])
                return mm
            
    def tf_gaussian_kernel_1d(self, size, sigma, truncate=4.0, dtype=tf.float32):
        radius = (size - 1) // 2
        x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
        k = tf.exp(-0.5 * tf.square(x / sigma))
        k = k / tf.reduce_sum(k)
        return k
    
    def tf_high_pass_filter_space(self, img_orig, gSig_filt=None):
        ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
        ker = tf_gaussian_kernel_1d(ksize[0], gSig_filt[0])
        ker2D = tf.expand_dims(ker, 1) * ker
        nz = tf.where(ker2D >= tf.reduce_max(ker2D[:,0]))
        zz = tf.where(ker2D < tf.reduce_max(ker2D[:,0]))
        ker2D = tf.tensor_scatter_nd_sub(ker2D, nz, tf.ones(nz.shape[0], dtype=tf.float32) * tf.reduce_mean(tf.gather_nd(ker2D, nz)))
        ker2D = tf.tensor_scatter_nd_update(ker2D, zz, tf.zeros(zz.shape[0], dtype=tf.float32))
        # convolution data batch * x * y * channel; filter x * y * channel_in * channel_out 
        img_new = tf.nn.conv2d(img_orig[..., None], filters=ker2D[..., None, None], strides=1, padding='SAME',data_format='NHWC')[..., 0]  # todo reflector padding
        return img_new      
        
#%%
mc_layer = MotionCorrect_non_rigid_2d(template, ms_h=10, ms_w=10, ms_d=0, max_deviation_rigid=[5,5], 
                 strides=(96, 96), overlaps=(48,48), padding='SAME')
batch = 2
data_corrected_pw = []
data_corrected_rigid = []
rigid_shifts = []
pw_shifts = []
for i in range(data.shape[0]//batch):
    print(i)
    temp = mc_layer(data[batch*i:batch*(i+1)])
    data_corrected_pw.append(temp[0])
    data_corrected_rigid.append(temp[1])
    rigid_shifts.append(temp[2])
    pw_shifts.append(temp[3])
data_corrected_rigid = np.concatenate(data_corrected_rigid, axis=0)  
data_corrected_pw = np.concatenate(data_corrected_pw, axis=0)  
rigid_shifts = np.concatenate(rigid_shifts, axis=0)     

#%%
plt.plot(shifts);plt.plot(-rigid_shifts[..., :2])
#%%
#rr = np.concatenate([data, data_corrected_rigid[..., 0]], axis=2)
rr = np.concatenate([data, data_corrected_rigid, data_corrected_pw], axis=2)
play(rr, fr=1)     

#%%
play(data_corrected_rigid, fr=1)     
        
         #%%
#shfts = [sshh[0] for sshh in shfts_et_all]
#diffs_phase = [sshh[2] for sshh in shfts_et_all]
# create a vector field
#data = data.numpy()
#diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)

    



#%%   
#%%
img_orig = data.copy()
img_new = tf_high_pass_filter_space(img_orig, (5,5))
plt.imshow(img_new.numpy().mean(0))    
      


#%%
rr = np.concatenate([data, data_corrected, data_corrected_tf], axis=2)
     
        
        
        




#%%
plt.imshow(templates[0])

    


m = data_corrected_tf.numpy().copy()
m = data.copy()
final_size_x = 430; final_size_y = 380; swap_dim=False
pyr_scale=.5; levels=3; winsize=100; iterations=15; poly_n=5; poly_sigma=1.2 / 5; flags=0;
play_flow=True; resize_fact_flow=1; template=None;
opencv=True; resize_fact_play=1; fr_play=2; max_flow=1e-7;
gSig_filt=None
tmpl = m[0]
#%%
def compute_metrics_motion_correction(fname, final_size_x, final_size_y, swap_dim, pyr_scale=.5, levels=3,
                                      winsize=100, iterations=15, poly_n=5, poly_sigma=1.2 / 5, flags=0,
                                      play_flow=False, resize_fact_flow=.2, template=None,
                                      opencv=True, resize_fact_play=3, fr_play=2, max_flow=1,
                                      gSig_filt=None):
    #todo: todocument
    # cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    import scipy
    vmin, vmax = -max_flow, max_flow
    #m = cm.load(fname)
    if gSig_filt is not None:
        m = high_pass_filter_space(m, gSig_filt)
    mi, ma = m.min(), m.max()
    m_min = mi + (ma - mi) / 100
    m_max = mi + (ma - mi) / 4

    max_shft_x = np.int(np.ceil((np.shape(m)[1] - final_size_x) / 2))
    max_shft_y = np.int(np.ceil((np.shape(m)[2] - final_size_y) / 2))
    max_shft_x_1 = - ((np.shape(m)[1] - max_shft_x) - (final_size_x))
    max_shft_y_1 = - ((np.shape(m)[2] - max_shft_y) - (final_size_y))
    if max_shft_x_1 == 0:
        max_shft_x_1 = None

    if max_shft_y_1 == 0:
        max_shft_y_1 = None
    logging.info([max_shft_x, max_shft_x_1, max_shft_y, max_shft_y_1])
    m = m[:, max_shft_x:max_shft_x_1, max_shft_y:max_shft_y_1]
    if np.sum(np.isnan(m)) > 0:
        logging.info(m.shape)
        logging.warning('Movie contains NaN')
        raise Exception('Movie contains NaN')

    logging.debug('Local correlations..')
    img_corr = m.local_correlations(eight_neighbours=True, swap_dim=swap_dim)
    logging.debug(m.shape)
    if template is None:
        tmpl = cm.motion_correction.bin_median(m)
    else:
        tmpl = template

    logging.debug('Compute Smoothness.. ')
    smoothness = np.sqrt(
        np.sum(np.sum(np.array(np.gradient(np.mean(m, 0)))**2, 0)))
    smoothness_corr = np.sqrt(
        np.sum(np.sum(np.array(np.gradient(img_corr))**2, 0)))

    logging.debug('Compute correlations.. ')
    correlations = []
    count = 0
    for fr in m:
        if count % 100 == 0:
            logging.debug(count)

        count += 1
        correlations.append(scipy.stats.pearsonr(
            fr.flatten(), tmpl.flatten())[0])

    logging.info('Compute optical flow .. ')

    m = resize(m, 1, 1, resize_fact_flow)
    norms = []
    flows = []
    count = 0
    for fr in m:
        if count % 100 == 0:
            logging.debug(count)

        count += 1
        flow = cv2.calcOpticalFlowFarneback(
            tmpl, fr, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

        if play_flow:
            if opencv:
                dims = tuple(np.array(flow.shape[:-1]) * resize_fact_play)
                vid_frame = np.concatenate([
                    np.repeat(np.clip((cv2.resize(fr, dims)[..., None] - m_min) /
                                      (m_max - m_min), 0, 1), 3, -1),
                    np.transpose([cv2.resize(np.clip(flow[:, :, 1] / vmax, 0, 1), dims),
                                  np.zeros(dims, np.float32).T,
                                  cv2.resize(np.clip(flow[:, :, 1] / vmin, 0, 1), dims)],
                                  (1, 2, 0)),
                    np.transpose([cv2.resize(np.clip(flow[:, :, 0] / vmax, 0, 1), dims),
                                  np.zeros(dims, np.float32).T,
                                  cv2.resize(np.clip(flow[:, :, 0] / vmin, 0, 1), dims)],
                                  (1, 2, 0))], 1).astype(np.float32)
                cv2.putText(vid_frame, 'movie', (10, 20), fontFace=5, fontScale=0.8, color=(
                    0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'y_flow', (dims[0] + 10, 20), fontFace=5, fontScale=0.8, color=(
                    0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'x_flow', (2 * dims[0] + 10, 20), fontFace=5, fontScale=0.8, color=(
                    0, 255, 0), thickness=1)
                cv2.imshow('frame', vid_frame)
                pl.pause(1 / fr_play)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                pl.subplot(1, 3, 1)
                pl.cla()
                pl.imshow(fr, vmin=m_min, vmax=m_max, cmap='gray')
                pl.title('movie')
                pl.subplot(1, 3, 3)
                pl.cla()
                pl.imshow(flow[:, :, 1])
                pl.title('y_flow')
                pl.subplot(1, 3, 2)
                pl.cla()
                pl.imshow(flow[:, :, 0])
                pl.title('x_flow')
                pl.pause(1 / fr_play)

        n = np.linalg.norm(flow)
        flows.append(flow)
        norms.append(n)
    if play_flow and opencv:
        cv2.destroyAllWindows()

    np.savez(fname[:-4] + '_metrics', flows=flows, norms=norms, correlations=correlations, smoothness=smoothness,
             tmpl=tmpl, smoothness_corr=smoothness_corr, img_corr=img_corr)
    return tmpl, correlations, flows, norms, smoothness

#%%
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
data_corrected_tf = []
for idx, img in enumerate(data):
    warp = tf.stack([tf.image.resize(shift_img_y[idx][..., None].astype(np.float32), dims)[..., 0] + y_grid, 
             tf.image.resize(shift_img_x[idx][..., None].astype(np.float32), dims)[..., 0] + x_grid], axis=2)
    data_corrected_tf.append(tfa.image.resampler(img[None, ..., None].astype(np.float32), warp[None, ...]))
data_corrected_tf = np.array(data_corrected_tf)[:, 0, :, :, 0]



#%%
idx = 4
plt.imshow(cv2.resize(shift_img_y[idx].astype(np.float32), dims[::-1]))
plt.imshow(shift_img_y[idx])


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
    