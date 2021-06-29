#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:49:25 2021

@author: nel
"""
#%%
import tensorflow as tf
import numpy as np
import pylab as plt
import tensorflow.keras as keras
import tensorflow_addons as tfa
import timeit
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
os.environ["TF_XLA_FLAGS"]="--tf_xla_enable_xla_devices" 
#%%
from viola.image_warp_3d import trilinear_interpolate_tf,  dense_image_warp_3D     
#%%
class MotionCorrect(keras.layers.Layer):
    def __init__(self, template, ms_h=10, ms_w=10, ms_d=10, strides=[1,1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
        
        super().__init__(**kwargs)
        
        self.ms_h = ms_h
        self.ms_w = ms_w
        self.ms_d = ms_d
        
        self.strides = strides
        self.padding = padding
        self.epsilon =  epsilon
        
        self.template = template
        self.template_zm, self.template_var = self.normalize_template(self.template, epsilon=self.epsilon)
        # print(self.template_var.shape, self.template_zm.shape, "tempvar\n")
        # plt.imshow(self.template_zm[:,:,0, 0,0])
        
        self.shp = self.template.shape
        self.shp_prod = tf.cast(tf.reduce_prod(self.shp), tf.float32)

        self.shp_m_x, self.shp_m_y, self.shp_m_z = self.shp[0]//2, self.shp[1]//2, self.shp[2]//2
                
        self.target_freq = tf.signal.fft3d(tf.cast(self.template_zm[None, :,:,:,0,0], tf.complex128))[0,:,:,:, None]
        
        # plt.imshow(self.template_zm[:,:,5,0,0])
        plt.imshow(tf.cast(self.target_freq, dtype=tf.float32)[:,:,0])
        
        nshape = tf.cast(tf.shape(self.target_freq), dtype=tf.float32)
        self.nc = nshape[0]
        self.nr = nshape[1]
        self.nd = nshape[2]
        Nr = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(self.nr / 2.), tf.math.ceil(self.nr / 2.)))
        Nc = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(self.nc / 2.), tf.math.ceil(self.nc / 2.)))
        Nd = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(self.nd / 2.), tf.math.ceil(self.nd / 2.)))
    
        self.Nr, self.Nc, self.Nd = tf.meshgrid(Nr, Nc, Nd)            
       
    # @tf.function
    def call(self, fr):
        fr = fr[0][None]
        imgs_zm, imgs_var = self.normalize_image(fr, self.shp, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon)
        denominator = tf.sqrt(self.template_var * imgs_var)
        
        self.imgs_zm = imgs_zm
        
        fr_freq = tf.signal.fft3d(tf.cast(imgs_zm[:,:,:,:,0], tf.complex128))[0,:,:,:,None]
        
        self.fr_freq = fr_freq
        
        img_product = fr_freq *  tf.math.conj(self.target_freq)
        #return  fr_freq, self.target_freq
        plt.imshow(tf.cast(img_product, dtype=tf.float32)[:,:,0])
        
        self.img_product = img_product

        cross_correlation = tf.cast(tf.math.abs(tf.signal.ifft3d(img_product[None,:,:,:,0])), tf.float32)[:, :, :, :, None]
        
        self.corr = cross_correlation
        rolled_cc =  tf.roll(cross_correlation,(self.shp_m_x,self.shp_m_y,self.shp_m_z), axis=(1,2,3))
        
        print(rolled_cc.shape, denominator.shape,"roll\n")
        # ncc = rolled_cc[:,self.shp_m_x-self.ms_w:self.shp_m_x+self.ms_w+1, self.shp_m_y-self.ms_h:self.shp_m_y+self.ms_h+1]/denominator
        ncc = rolled_cc[:,self.shp_m_x-self.ms_w:self.shp_m_x+self.ms_w+1, self.shp_m_y-self.ms_h:self.shp_m_y+self.ms_h+1, self.shp_m_z-self.ms_d:self.shp_m_z+self.ms_d+1]/denominator
        ncc = tf.where(tf.math.is_nan(ncc), tf.zeros_like(ncc), ncc)
        # plt.imshow(ncc[0,:,:,10,0])
        # return ncc
        
        self.ncc = ncc

        sh_x, sh_y, sh_z = self.extract_fractional_peak(ncc, self.ms_h, self.ms_w, self.ms_d)
        # print(sh_x, sh_y)
        print(sh_x)
        #return sh_x, sh_y, sh_z, cross_correlation, ncc, denominator
        # print(fr_freq.shape,  fr_freq.dtype)
        fr_corrected = self.apply_shifts_dft_tf(src_freq=fr_freq, shifts=[-sh_x,-sh_y,-sh_z])
        
        # print(tf.math.reduce_sum(fr_corrected))
        # print(sh_x)
        # return tf.reshape(tf.transpose(tf.squeeze(fr_corrected)), [-1])[None, :]
        # print(sh_x,sh_y,sh_z)
        return tf.transpose(tf.squeeze(fr_corrected))
    
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1,2], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1,2], keepdims=True) + epsilon
        print(template_zm.shape, "zmt")
        # print(template.shape,tf.reduce_mean(template, axis=[0,1,2], keepdims=True).shape)
        return template_zm, template_var
        
    def normalize_image(self, imgs, shape_template, strides=[1,1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed
        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2,3], keepdims=True)
        print(imgs_zm.shape, "it")
        img_stack = tf.stack([imgs[:,:,:,:,0], tf.square(imgs)[:,:,:,:,0]], axis=4)
        # print(img_stack.shape)

        localsum_stack = tf.nn.avg_pool3d(img_stack,[1,self.template.shape[0]-2*self.ms_w, self.template.shape[1]-2*self.ms_h, self.template.shape[2]-2*self.ms_d, 1], 
                                               padding=padding, strides=strides)
        localsum_ustack = tf.unstack(localsum_stack, axis=4)
        
        localsum_sq = localsum_ustack[1][:,:,:,:,None]
        localsum = localsum_ustack[0][:,:,:,:,None]

        imgs_var = localsum_sq - tf.square(localsum)/self.shp_prod + epsilon
        # Remove small machine precision errors after subtraction
        imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
        # del localsum_sq, localsum
        return imgs_zm, imgs_var
        
        
    def extract_fractional_peak(self, ncc, ms_h, ms_w, ms_d):
        """ use gaussian interpolation to extract a fractional shift
        Args:
            tensor_ncc: tensor
                normalized cross-correlation
                ms_h: max integer shift vertical
                ms_w: max integere shift horizontal
        
        """
        # st = timeit.default_timer()
        shifts_int = self.argmax_3d(ncc)
        # tf.print(timeit.default_timer() - st, "argmax")

        # shifts_int_cast = tf.cast(shifts_int,tf.int64)
        sh_x, sh_y, sh_z = shifts_int[0],shifts_int[1],shifts_int[2]
        # print(sh_x,sh_y,sh_z,  ncc.shape)
        # tf.print(timeit.default_timer() - st, "shifts")
        
        sh_x_n = tf.cast(-(sh_x - ms_h), tf.float32)
        sh_y_n = tf.cast(-(sh_y - ms_w), tf.float32)
        sh_z_n = tf.cast(-(sh_z - ms_d), tf.float32)
        
        ncc_log = tf.math.log(ncc)
        # print(ncc_log.shape, sh_x, sh_y, sh_z, "indx\n")

        n_batches = np.arange(1)

        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x-1, axis=0), tf.squeeze(sh_y, axis=0), tf.squeeze(sh_z, axis=0)]))
        log_xm1_y_z = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x+1, axis=0), tf.squeeze(sh_y, axis=0), tf.squeeze(sh_z, axis=0)]))
        log_xp1_y_z = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y-1, axis=0), tf.squeeze(sh_z, axis=0)]))
        log_x_ym1_z = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y+1, axis=0), tf.squeeze(sh_z, axis=0)]))
        log_x_yp1_z =  tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y, axis=0), tf.squeeze(sh_z-1, axis=0)]))
        log_x_y_zm1 =  tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y, axis=0), tf.squeeze(sh_z+1, axis=0)]))
        # print(idx)
        log_x_y_zp1 =  tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y, axis=0),  tf.squeeze(sh_z, axis=0)]))
        six_log_xyz = 6 * tf.gather_nd(ncc_log, idx)
        # print()
        # print(six_log_xyz, " siz")
        # print()
        # print(np.mean(ncc_log),  "mean")
        # print()
        # print("idx", idx)
        # print()
        # print(log_x_ym1_z,log_x_yp1_z)

        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y_z - log_xp1_y_z), (2 * log_xm1_y_z - six_log_xyz + 2 * log_xp1_y_z))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1_z - log_x_yp1_z), (2 * log_x_ym1_z - six_log_xyz + 2 * log_x_yp1_z))
        sh_z_n = sh_z_n - tf.math.truediv((log_x_y_zm1 - log_x_y_zp1), (2 * log_x_y_zm1 - six_log_xyz + 2 * log_x_y_zp1))
        print(sh_x_n, "x\n")
        return tf.reshape(sh_x_n, [1, 1]), tf.reshape(sh_y_n, [1, 1]),  tf.reshape(sh_z_n, [1, 1])
    
    def argmax_3d(self, tensor):
        # extract peaks from 3D tensor (takes batches as input too)

        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (1, tensor.shape[-4]*tensor.shape[-3]*tensor.shape[-2], 1))

        argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
        # convert indexes into 3D coordinates
        shp = tf.shape(tensor)
        yz_flat = shp[1]*shp[2]

        argmax_x = argmax // yz_flat 

        argmax_y = (argmax % yz_flat) // shp[2]
        argmax_z = argmax - argmax_x*yz_flat - shp[2]*argmax_y
        # stack and return 2D coordinates
        print(argmax_x, argmax_y, argmax_z, argmax, "argmax\n")
        return (argmax_x, argmax_y, argmax_z)
    
    # @tf.function
    def apply_shifts_dft_tf(self, src_freq, shifts, diffphase=tf.cast([0],dtype=tf.complex128)):
        shifts =  (shifts[1], shifts[0], shifts[2]) #p.array(list(shifts[:-1][::-1]) + [shifts[-1]])
        # print(shifts)
        src_freq  = src_freq[:,:,:,0]
        # print(src_freq.dtype)


        sh_0 = -shifts[0] * self.Nr / self.nr
        sh_1 = -shifts[1] * self.Nc / self.nc
        sh_2 = -shifts[2] * self.Nd / self.nd
        sh_tot = (sh_0 +  sh_1 + sh_2)

        Greg = src_freq * tf.math.exp(-1j * 2 * math.pi * tf.cast(sh_tot, dtype=tf.complex128))

    
        #todo: check difphase and eventually dot product?
        Greg = Greg * tf.math.exp(1j * diffphase)
        new_img = tf.math.real(tf.signal.ifft3d(Greg))
        
    
        return new_img[None,:,:,:,None]
        
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template,"strides": self.strides,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w , "ms_d":self.ms_d}
#%%
import math
# movie = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/PanosBoyden/Organoids/Org3_BP_Fov2_run1/plane_51.hdf5'
# import h5py
# #%%
# with h5py.File(movie, "r") as f:
#     print("Keys: %s" % f.keys())
#     a_group_key = list(f.keys())[0]
#     mov = np.array(f['mov'])
# #%%    
mov = np.zeros((4, 50, 50, 50)).astype(np.float32).astype(np.float32)
#mov = mov + (np.random.rand(4, 50, 50, 50)-0.5)/3
mov = mov.astype(np.float32)
# mov = np.random.uniform(low=0.0, high=1.0, size=(4,50,50,50))

for i in range(mov.shape[0]):
    spot = 5
    mov[i][spot-2:spot+2,spot-2:spot+2,spot-2:spot+2] = 20.0
template = mov[0].copy()
mov[1] = np.roll(a=mov[1], shift=(1,3,2), axis=(0,1,2))
mov[2] = np.roll(a=mov[2], shift=(3,3), axis=(0,1))
mov[3] = np.roll(a=mov[3], shift=(-2,-3), axis=(0,1))
data = np.expand_dims(mov, axis=4)[None, :]#.astype(np.double)  # batch size, time, x, y, z, channel

#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

i = 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = np.where(mov[i]>0)
c = np.ones(x.shape)
img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
ax.set_xlim([0, 50])
ax.set_ylim([0, 50])
ax.set_zlim([0, 50])
#fig.colorbar(img)
plt.show()#%%

#%%
mc_layer = MotionCorrect(template[:,:,:,None,None])
import time
st = time.time()
times = []
nccs = []
for i in range(mov.shape[0]):
    ncc = mc_layer(data[:,3])
    times.append(time.time()-st)
    # nccs.append(np.array(ncc).squeeze())
    break
# print(time.time()-st)
#fr_corrected = ncc
#coords = np.array([ncc[0],ncc[1],ncc[2]]).flatten()
#tensor = np.squeeze(ncc[-1])  # template frequency
#cc = np.squeeze(ncc[-2])    # frame frequency
#print(coords)
#xx = fr_corrected.numpy()
#np.where(xx == xx.max())
np.where(mc_layer.ncc.numpy()==mc_layer.ncc.numpy().max())
#np.where(ncc == ncc.max())
ii = 6
plt.imshow(template[:,:,ii])
plt.imshow(mov[3][:,:,ii])
plt.imshow(ncc.numpy()[:,:,ii])

#%%
from numpy.fft import fftn, ifftn
mc_layer.template_zm.shape
mc_layer.target_freq.shape
mc_layer.imgs_zm.shape
#self.target_freq = tf.signal.fft3d(tf.cast(self.template_zm[:,:,:,:,0], tf.complex128))

fft_np = fftn(mc_layer.template_zm[:,:,:,:,0].numpy().copy())
print(fft_np.mean())

imgs_zm = mc_layer.imgs_zm.numpy().copy()
fft_np1 = fftn(mc_layer.imgs_zm[:,:,:,:,0].numpy().copy())[0,:,:,:,None]
print(fft_np1.mean())

product_np = np.conj(fft_np) * fft_np1
corr_np = np.abs(ifftn(product_np))
print(np.where(corr_np ==corr_np.max()))

#%%
fft_tf = tf.signal.fft3d(tf.cast(mc_layer.template_zm[None,:,:,:,0,0], tf.complex128))
tf.shape(fft_tf)
print(fft_tf.numpy().mean())

fr_freq = tf.signal.fft3d(tf.cast(imgs_zm[:,:,:,:, 0], tf.complex128))
tf.shape(fr_freq)
print(fr_freq.numpy().mean())

img_product = fr_freq *  tf.math.conj(fft_tf)

corr_tf = tf.cast(tf.math.abs(tf.signal.ifft3d(img_product)), tf.float32)[None].numpy()
print(np.where(corr_tf ==corr_tf.max()))

#%%
cc = mc_layer.corr.numpy().copy()
print(np.where(cc ==cc.max()))

#%%
mm = ncc[-2].numpy()
np.where(mm == mm.max())

#%%
mc_layer.template_zm.numpy().mean()
mc_layer.imgs_zm.numpy().mean()



img_product = fr_freq *  tf.math.conj(self.target_freq)
#return  fr_freq, self.target_freq
plt.imshow(tf.cast(img_product, dtype=tf.float32)[:,:,0])

cross_correlation = tf.cast(tf.math.abs(tf.signal.ifft3d(img_product)), tf.float32)[None]

fr_freq = tf.signal.fft3d(tf.cast(imgs_zm[:,:,:,:,0], tf.complex128))[0,:,:,:,None]

#%%
img_stack = tf.stack([data[:,:,:,:,0], tf.square(data)[:,:,:,:,0]], axis=4)
print(img_stack.shape)

localsum_stack = tf.nn.avg_pool3d(img_stack,[1,template.shape[0]-20, template.shape[1]-2*10, template.shape[2]-4, 1], 
                                  padding="VALID", strides=[1,1,1,1,1])
#%%
tensor  = mov[0,:50,:28,:30,None]
        # extract peaks from 3D tensor (takes batches as input too)

        # flatten the Tensor along the height and width axes
flat_tensor = tf.reshape(tensor, (1, tensor.shape[-4]*tensor.shape[-3]*tensor.shape[-2], 1))

argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
# convert indexes into 3D coordinates
shp = tf.shape(tensor)
yz_flat = shp[1]*shp[2]

argmax_x = argmax // yz_flat 
print(yz_flat)
argmax_y = (argmax % yz_flat) // shp[2]
argmax_z = argmax - argmax_x*yz_flat - shp[2]*argmax_y
# stack and return 2D coordinates
print(argmax_x, argmax_y, argmax_z, argmax, "argmax\n")
# return (argmax_x, argmax_y, argmax_z)

