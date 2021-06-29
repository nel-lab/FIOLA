#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:48:46 2019

@author: agiovann
"""
#%%
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from past.utils import old_div
from skimage import io
import numpy as np
import pylab as plt
import cv2
import timeit
#%% 
class MotionCorrect(keras.layers.Layer):
    def __init__(self, template, ms_h=10, ms_w=10, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
        """
        Tenforflow layer which perform motion correction on batches of frames. Notice that the input must be a 
        tensorflow tensor with dimension batch x width x height x channel
        Args:
           template: ndarray
               template against which to register
           ms_h: int
               estimated minimum value of vertical shift
           ms_w: int
               estimated minimum value of horizontal shift    
           strides: list
               convolutional strides of tf.conv2D              
           padding: str
               "VALID" or "SAME": convolutional padding of tf.conv2D
           epsilon':float
               small value added to variances to prevent division by zero
        
        
        Returns:
           X_corrected batch corrected when called upon a batch of inputs
        
        """
        super().__init__(**kwargs)
        self.template = tf.convert_to_tensor(template[ms_h:-ms_h, ms_w:-ms_w, None, None])
        self.ms_h = ms_h
        self.ms_w = ms_w
        self.strides = strides
        self.padding = padding
        self.epsilon = epsilon
        self.template_numel = np.prod(self.template.shape)
        ## normalize template
        self.template_zm, self.template_var = self.normalize_template(self.template, epsilon=self.epsilon)

    def build(self, batch_input_shape):
        # weights here represent the template, so that we can update in case we 
        # want to use the online algorithm
        self.kernel = self.add_weight(
            name="kernel", shape=[*self.template_zm.shape.as_list()],
            initializer=tf.constant_initializer(self.template_zm.numpy()))
        # the normalizer also needs to be updated if the template is updated
        self.normalizer = self.add_weight(
            name="normalizer", shape=[*self.template_var.shape.as_list()],
            initializer=tf.constant_initializer(self.template_var.numpy()))   
        super().build(batch_input_shape) # must be at the end

    def call(self, X):
        # takes as input a tensorflow batch tensor (batch x width x height x channel)
        # normalize images
        imgs_zm, imgs_var = self.normalize_image(X, self.template.shape, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon)        
        
        denominator = tf.sqrt(self.normalizer * imgs_var)
        numerator = tf.nn.conv2d(imgs_zm, self.kernel, padding=self.padding, 
                                 strides=self.strides)
        
        tensor_ncc = tf.truediv(numerator, denominator)
       
        # Remove any NaN in final output
        tensor_ncc = tf.where(tf.math.is_nan(tensor_ncc), tf.zeros_like(tensor_ncc), tensor_ncc)
        
        
        xs, ys = self.extract_fractional_peak(tensor_ncc, ms_h=self.ms_h, ms_w=self.ms_w)
        
        X_corrected = tfa.image.translate(X, tf.squeeze(tf.stack([ys,xs], axis=1)),
                                                             interpolation="BILINEAR") 
        return X_corrected


    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "strides": self.strides,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w }
        
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
        return template_zm, template_var
        
    def normalize_image(self, imgs, shape_template, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed
        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2], keepdims=True)
        localsum_sq = tf.nn.conv2d(tf.square(imgs), tf.ones(shape_template), 
                                               padding=padding, strides=strides)
        localsum = tf.nn.conv2d(imgs,tf.ones(shape_template), 
                                               padding=padding, strides=strides)
        
        
        imgs_var = localsum_sq - tf.square(localsum)/np.prod(shape_template) + epsilon
        # Remove small machine precision errors after subtraction
        imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
        return imgs_zm, imgs_var
    
    def argmax_2d(self, tensor):
        # extract peaks from 2D tensor (takes batches as input too)
        
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))
          
        # argmax of the flat tensor
        argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
          
        # convert indexes into 2D coordinates
        argmax_x = argmax // tf.shape(tensor)[2]
        argmax_y = argmax % tf.shape(tensor)[2]
        
        # stack and return 2D coordinates
        return tf.cast(tf.stack((argmax_x, argmax_y), axis=1), tf.float32)
    
    def extract_fractional_peak(self, tensor_ncc, ms_h, ms_w):
        """ use gaussian interpolation to extract a fractional shift
        Args:
            tensor_ncc: tensor
                normalized cross-correlation
                ms_h: max integer shift vertical
                ms_w: max integere shift horizontal
        
        """
        shifts_int = self.argmax_2d(tensor_ncc)
        shifts_int_cast = tf.cast(shifts_int,tf.int32)
        sh_x, sh_y = shifts_int_cast[:,0],shifts_int_cast[:,1]
        
        sh_x_n = tf.cast(-(sh_x - ms_h), tf.float32)
        sh_y_n = tf.cast(-(sh_y - ms_w), tf.float32)
        
        tensor_ncc_log = tf.math.log(tensor_ncc)      
        
        n_batches = np.arange(tensor_ncc_log.shape[0])
        
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x-1), tf.squeeze(sh_y)]))
        log_xm1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x+1), tf.squeeze(sh_y)]))
        log_xp1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x), tf.squeeze(sh_y-1)]))
        log_x_ym1 = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x), tf.squeeze(sh_y+1)]))
        log_x_yp1 =  tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x), tf.squeeze(sh_y)]))
        four_log_xy = 4 * tf.gather_nd(tensor_ncc_log, idx)
        
        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
        
        return sh_x_n, sh_y_n
    
#%% load batch, initialize template and transform to tensor
num_frames = 300
a = io.imread('Sue_2x_3000_40_-46.tif')
template = np.median(a,axis=0)
batch = tf.convert_to_tensor(a[:num_frames,:,:,None])
#%% run motion correction on a batch
mod = MotionCorrect(template)
#%%
start = timeit.default_timer()
mov_corr = mod(batch)
print(float(timeit.default_timer() - start)/num_frames)
#%% visualie movie
min_, max_ = np.min(batch), np.max(batch)
for fr, fr_raw in zip(mov_corr, batch):
    # Our operations on the frame come here
    gray = np.concatenate((fr.numpy().squeeze(), fr_raw.numpy().squeeze()))
    # Display the resulting frame
    cv2.imshow('frame', (gray-min_)/(max_-min_)*10)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
    
    
