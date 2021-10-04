#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:27:21 2021

@author: nel
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from threading import Thread
import tensorflow_addons as tfa
import numpy as np
from queue import Queue
import timeit
from time import time

class MotionCorrectBatch(keras.layers.Layer):
    def __init__(self, template, batch_size=1, ms_h=5, ms_w=5, 
                 strides=[1,1,1,1], padding='VALID',center_dims=None, use_fft=True, 
                 normalize_cc=True, epsilon=0.00000001, **kwargs):
        """
        Class for GPU motion correction        

        Parameters
        ----------
        template : ndarray
            The template used for motion correction
        batch_size : int
            number of frames used for motion correction each time. The default is 1.
        ms_h : int
            maximum shift horizontal. The default is 5.
        ms_w : int
            maximum shift vertical. The default is 5.
        strides : list
            stride for convolution. The default is [1,1,1,1].
        padding : str
            padding for convolution. The default is 'VALID'.
        center_dims : tuple
            size of center crops of template for motion correction. If None, it will not crop. The default is None.
        use_fft : bool
            use FFT for convolution or not. Will use tf.nn.conv2D if False. The default is True.
        normalize_cc : bool
            whether to normalize the cross correlations coefficients or not. The default is True.
        epsilon : float
            epsilon to avoid deviding by zero. The default is 0.00000001.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        motion corrected frames
        """        
        
        super().__init__(**kwargs)        
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)
                print(f'{name}, {value}')
        
        self.shp_0  = template.shape   
        self.template_0 = template.copy()
        self.xmin, self.ymin = self.shp_0[0]-2*ms_w, self.shp_0[1]-2*ms_h
        if self.xmin < 10 or self.ymin < 10:
            raise ValueError("The frame dimensions you entered are too small. Please provide a larger field of view or resize your movie.") 
        
        if self.center_dims is not None:
            self.shp_c_x, self.shp_c_y = (self.shp_0[0] - center_dims[0])//2, (self.shp_0[1] - center_dims[1])//2
            self.template = self.template_0[self.shp_c_x:-self.shp_c_x, self.shp_c_y:-self.shp_c_y]
        else:
            self.shp_c_x, self.shp_c_y = (0, 0)
 
        if self.use_fft == False:
            self.template = self.template_0[(ms_w+self.shp_c_x):-(ms_w+self.shp_c_x),(ms_h+self.shp_c_y):-(ms_h+self.shp_c_y)]
        # else:
        #     self.template[:ms_w] = 0
        #     self.template[:, :ms_h] = 0
        #     self.template[-ms_w:] = 0
        #     self.template[:, -ms_h:] = 0
            
        self.template_zm, self.template_var = self.normalize_template(self.template[:,:,None,None], epsilon=self.epsilon)
        self.shp = self.template.shape
        self.shp_m_x, self.shp_m_y = self.shp[0]//2, self.shp[1]//2
        self.target_freq = tf.signal.fft3d(tf.cast(self.template_zm[:,:,:,0], tf.complex128))
        self.target_freq = tf.repeat(self.target_freq[None,:,:,0], repeats=[self.batch_size], axis=0)
      
    @tf.function
    def call(self, fr):
        if self.center_dims is None:
            fr_center = fr[0]
        else:      
            fr_center = fr[0,:, self.shp_c_x:(self.shp_0[0]-self.shp_c_x), self.shp_c_y:(self.shp_0[1]-self.shp_c_y)]

        imgs_zm, imgs_var = self.normalize_image(fr_center, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon)
        denominator = tf.sqrt(self.template_var * imgs_var)

        if self.use_fft:
            fr_freq = tf.signal.fft3d(tf.cast(imgs_zm[:,:,:,0], tf.complex128))
            img_product = fr_freq *  tf.math.conj(self.target_freq)
            cross_correlation = tf.cast(tf.math.abs(tf.signal.ifft3d(img_product)), tf.float32)
            rolled_cc =  tf.roll(cross_correlation,(self.batch_size, self.shp_m_x,self.shp_m_y), axis=(0,1,2))
            nominator = rolled_cc[:,self.shp_m_x-self.ms_w:self.shp_m_x+self.ms_w+1, self.shp_m_y-self.ms_h:self.shp_m_y+self.ms_h+1, None] 
        else:
            nominator = tf.nn.conv2d(imgs_zm, self.template_zm, padding=self.padding, 
                                     strides=self.strides)
           
        if self.normalize_cc:    
            ncc = tf.truediv(nominator, denominator)        
        else:
            ncc = nominator    
        
        ncc = tf.where(tf.math.is_nan(ncc), tf.zeros_like(ncc), ncc)
        sh_x, sh_y = self.extract_fractional_peak(ncc, self.ms_h, self.ms_w)
        #self.shifts = [sh_x, sh_y]
        fr_corrected = tfa.image.translate(fr[0], (tf.squeeze(tf.stack([sh_y, sh_x], axis=1))), 
                                            interpolation="bilinear")
        return tf.reshape(tf.transpose(tf.squeeze(fr_corrected, axis=3), perm=[0,2,1]), (self.batch_size, self.shp_0[0]*self.shp_0[1]))#, self.shifts
    
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
        return template_zm, template_var
        
    def normalize_image(self, imgs, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed
        if self.use_fft:
            shape = [self.template.shape[0]-2*self.ms_w, self.template.shape[1]-2*self.ms_h]            
        else:
            shape = [self.template.shape[0], self.template.shape[1]]
        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2], keepdims=True)
        img_stack = tf.stack([imgs[:,:,:,0], tf.square(imgs)[:,:,:,0]], axis=3)
        localsum_stack = tf.nn.avg_pool2d(img_stack,[1, shape[0], shape[1], 1], 
                                               padding=padding, strides=strides)
        localsum_ustack = tf.unstack(localsum_stack, axis=3)
        localsum_sq = localsum_ustack[1][:,:,:,None]
        localsum = localsum_ustack[0][:,:,:,None]      
        imgs_var = localsum_sq - tf.square(localsum)/np.prod(shape) + epsilon
        # Remove small machine precision errors after subtraction
        imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
        return imgs_zm, imgs_var        
        
    def extract_fractional_peak(self, ncc, ms_h, ms_w):
        # use gaussian interpolation to extract a fractional shift
        shifts_int = self.argmax_2d(ncc) 
        shifts_int_cast = tf.cast(shifts_int,tf.int32)
        sh_x, sh_y = shifts_int_cast[:,0],shifts_int_cast[:,1]
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