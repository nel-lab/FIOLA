#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:51:50 2021

@author: nel
"""

#%%
import tensorflow as tf
import numpy as np
import pylab as plt
import tensorflow.keras as keras
import tensorflow_addons as tfa
import timeit
#%%
class MotionCorrectBatch(keras.layers.Layer):
    def __init__(self, template, batch_size, ms_h=10, ms_w=10, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
        
        super().__init__(**kwargs)
        
        self.ms_h = ms_h
        self.ms_w = ms_w
        self.batch_size = batch_size
        
        self.strides = strides
        self.padding = padding
        self.epsilon =  epsilon
        
        self.template = template
        self.template_zm, self.template_var = self.normalize_template(self.template[:,:,None,None], epsilon=self.epsilon)
        
        self.shp = self.template.shape
        self.shp_prod = tf.cast(tf.reduce_prod(self.shp), tf.float32)

        self.shp_m_x, self.shp_m_y = self.shp[0]//2, self.shp[1]//2
        self.target_freq = tf.signal.fft3d(tf.cast(self.template_zm[:,:,:,0], tf.complex128))

        self.target_freq = tf.repeat(self.target_freq[None,:,:,0], repeats=[self.batch_size], axis=0)
            
       
    # @tf.function
    def call(self, fr):

        # fr = tf.cast(fr[None, :, :, None], tf.float32)
        # fr =  fr[0:1,:,:,None]
        fr = fr[0]

        imgs_zm, imgs_var = self.normalize_image(fr, self.shp, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon)
        denominator = tf.sqrt(self.template_var * imgs_var)
        # print(imgs_zm.shape, imgs_var.shape)

        fr_freq = tf.signal.fft3d(tf.cast(imgs_zm[:,:,:,0], tf.complex128))
        img_product = fr_freq *  tf.math.conj(self.target_freq)

        cross_correlation = tf.cast(tf.math.abs(tf.signal.ifft3d(img_product)), tf.float32)

        rolled_cc =  tf.roll(cross_correlation,(self.batch_size, self.shp_m_x,self.shp_m_y), axis=(0,1,2))

        ncc = rolled_cc[:,self.shp_m_x-self.ms_w:self.shp_m_x+self.ms_w+1, self.shp_m_y-self.ms_h:self.shp_m_y+self.ms_h+1, None]/denominator

        ncc = tf.where(tf.math.is_nan(ncc), tf.zeros_like(ncc), ncc)

        
        sh_x, sh_y = self.extract_fractional_peak(ncc, self.ms_h, self.ms_w)
        fr_corrected = tfa.image.translate(fr, (tf.squeeze(tf.stack([sh_x, sh_y], axis=1))), 
                                            interpolation="bilinear")

        return tf.reshape(tf.transpose(tf.squeeze(fr_corrected, axis=3), perm=[0,2,1]), (self.batch_size, self.shp[0]*self.shp[1]))

    
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
        return template_zm, template_var
        
    def normalize_image(self, imgs, shape_template, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed
        # print(imgs.shape)
        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2], keepdims=True)
        img_stack = tf.stack([imgs[:,:,:,0], tf.square(imgs)[:,:,:,0]], axis=3)
        print(img_stack.shape)
        
        localsum_stack = tf.nn.avg_pool2d(img_stack,[1,self.template.shape[0]-2*self.ms_w, self.template.shape[1]-2*self.ms_h, 1], 
                                               padding=padding, strides=strides)
        
        localsum_ustack = tf.unstack(localsum_stack, axis=3)
        localsum_sq = localsum_ustack[1][:,:,:,None]
        localsum = localsum_ustack[0][:,:,:,None]
        
        imgs_var = localsum_sq - tf.square(localsum)/np.prod(shape_template) + epsilon
        # Remove small machine precision errors after subtraction
        imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
        return imgs_zm, imgs_var
        
        
    def extract_fractional_peak(self, ncc, ms_h, ms_w):
        """ use gaussian interpolation to extract a fractional shift
        Args:
            tensor_ncc: tensor
                normalized cross-correlation
                ms_h: max integer shift vertical
                ms_w: max integere shift horizontal
        
        """
        # st = timeit.default_timer()
        shifts_int = self.argmax_2d(ncc) 
        # tf.print(timeit.default_timer() - st, "argmax")

        shifts_int_cast = tf.cast(shifts_int,tf.int32)
        sh_x, sh_y = shifts_int_cast[:,0],shifts_int_cast[:,1]
        # tf.print(timeit.default_timer() - st, "shifts")
        
        sh_x_n = tf.cast(-(sh_x - ms_h), tf.float32)
        sh_y_n = tf.cast(-(sh_y - ms_w), tf.float32)
        
        ncc_log = tf.math.log(ncc)

        n_batches = np.arange(self.batch_size)

        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x-1,axis=1), tf.squeeze(sh_y,axis=1)]))
        log_xm1_y = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x+1,axis=1), tf.squeeze(sh_y,axis=1)]))
        log_xp1_y = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=1), tf.squeeze(sh_y-1, axis=1)]))
        log_x_ym1 = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=1), tf.squeeze(sh_y+1, axis=1)]))
        log_x_yp1 =  tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=1), tf.squeeze(sh_y, axis=1)]))
        four_log_xy = 4 * tf.gather_nd(ncc_log, idx)

        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))

        return tf.reshape(sh_x_n, [self.batch_size, 1]), tf.reshape(sh_y_n, [self.batch_size, 1])
    
    def argmax_2d(self, tensor):
        # extract peaks from 2D tensor (takes batches as input too)
        
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))

        argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
        # convert indexes into 2D coordinates
        argmax_x = tf.cast(argmax, tf.int32) // tf.shape(tensor)[2]
        argmax_y = tf.cast(argmax, tf.int32) % tf.shape(tensor)[2]
        # stack and return 2D coordinates
        return tf.cast(tf.stack((argmax_x, argmax_y), axis=1), tf.float32)
        
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template,"strides": self.strides, "batch_size":self.batch_size,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w }