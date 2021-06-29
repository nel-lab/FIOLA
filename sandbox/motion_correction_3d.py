#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:42:46 2021

@author: nellab
"""
#%%
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import timeit
#%%
class MotionCorrect3D(keras.layers.Layer):
    def __init__(self, template, center_dims, ms_h=5, ms_w=5, ms_d=5, strides=[1,1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
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

        self.shp_x, self.shp_y, self.shp_z = template.shape[0], template.shape[1], template.shape[2]
        self.center_dims = center_dims

        self.c_shp_x, self.c_shp_y, self.c_shp_z = (self.shp_x - center_dims[0])//2, (self.shp_y - center_dims[1])//2, (self.shp_z - center_dims[2])//2
    
        
        self.template_0 = template
        self.template=self.template_0[(ms_w+self.c_shp_x):-(ms_w+self.c_shp_x),(ms_h+self.c_shp_y):-(ms_h+self.c_shp_y), None, None]
        
        # if self.template.shape[0] < 10 or self.template.shape[1] < 10:
        #     print(self.template.shape)
        #     raise ValueError("The frame dimensions you entered are too small. Please provide a larger field of view or resize your movie.")

        self.ms_h = ms_h
        self.ms_w = ms_w
        self.ms_d = ms_d
        self.strides = strides
        self.padding = padding
        self.epsilon = epsilon

        ## normalize template
        self.template_zm, self.template_var = self.normalize_template(self.template, epsilon=self.epsilon)
        
        ## assign to kernel, normalizer
        self.kernel = self.template_zm
        self.normalizer = self.template_var


    # @tf.function
    def call(self, X):
        # takes as input a tensorflow batch tensor (batch x width x height x depth x channel)
        #print(X.shape)
        X_center = X[:, self.c_shp_x:(self.shp_x-self.c_shp_x), self.c_shp_y:(self.shp_y-self.c_shp_y), self.c_shp_z:(self.shp_z-self.c_shp_z)]

        # X_center = X[:,:,:]

        # pass in center for normalization
        #print(X_center.shape, self.template.shape)
        imgs_zm, imgs_var = self.normalize_image(X_center, self.template.shape, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon)
        denominator = tf.sqrt(self.normalizer * imgs_var)
        # tf.print(timeit.default_timer()-st, "normalize")
        numerator = tf.nn.conv3d(imgs_zm, self.kernel, padding=self.padding, 
                                 strides=self.strides)
       
        tensor_ncc = tf.truediv(numerator, denominator)
        # tf.print(timeit.default_timer()-st, "conv2d 1")
       
        # Remove any NaN in final output
        tensor_ncc = tf.where(tf.math.is_nan(tensor_ncc), tf.zeros_like(tensor_ncc), tensor_ncc)
        
        xs, ys = self.extract_fractional_peak(tensor_ncc, ms_h=self.ms_h, ms_w=self.ms_w)
        # tf.print(timeit.default_timer()-st, "extract")
        
        X_corrected = tfa.image.translate(X, (tf.squeeze(tf.stack([ys, xs], axis=1))), 
                                            interpolation="BILINEAR")
        # tf.print(timeit.default_timer()-st, "translate")
        # return (tf.reshape(tf.transpose(tf.squeeze(X_corrected)), [-1])[None, :], [xs, ys])
        #return (tf.squeeze(X_corrected), [xs, ys])
        return (tf.reshape(tf.transpose(tf.squeeze(X_corrected)), [-1])[None, :])


    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template_0,"strides": self.strides, "center_dims": self.center_dims,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w }  
        
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1,2], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1,2], keepdims=True) + epsilon
        return tf.cast(template_zm, tf.float32), tf.cast(template_var, tf.float32)
        
    def normalize_image(self, imgs, shape_template, strides=[1,1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed
        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2,3], keepdims=True)
        import pdb; pdb.set_trace()
        localsum_sq = tf.nn.conv3d(tf.square(imgs[None, :]), tf.ones(shape_template), 
                                               padding=padding, strides=strides)
        localsum = tf.nn.conv3d(imgs,tf.ones(shape_template), 
                                               padding=padding, strides=strides)
        
        
        imgs_var = localsum_sq - tf.square(localsum)/np.prod(shape_template) + epsilon
        # Remove small machine precision errors after subtraction
        imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
        # del localsum_sq, localsum
        return imgs_zm, imgs_var
    
    def argmax_3d(self, tensor):
        # extract peaks from 3D tensor (takes batches as input too)
        
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (1, tensor.shape[-4]*tensor.shape[-3]*tensor.shape[-2], 1))

        argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
        # convert indexes into 3D coordinates
        shp = tf.shape(tensor)
        z = shp[3]*shp[2]
        argmax_z = argmax // z
        z_coord = argmax % z
        argmax_x = z_coord // shp[2]
        argmax_y = argmax % shp[2]
        # stack and return 2D coordinates
        return (argmax_x, argmax_y, argmax_z)
        
    def extract_fractional_peak(self, tensor_ncc, ms_h, ms_w):
        """ use gaussian interpolation to extract a fractional shift
        Args:
            tensor_ncc: tensor
                normalized cross-correlation
                ms_h: max integer shift vertical
                ms_w: max integere shift horizontal
        
        """
        # st = timeit.default_timer()
        shifts_int = self.argmax_3d(tensor_ncc)
        # tf.print(timeit.default_timer() - st, "argmax")

        # shifts_int_cast = tf.cast(shifts_int,tf.int64)
        sh_x, sh_y, sh_z = shifts_int[0],shifts_int[1],shifts_int[2]
        # tf.print(timeit.default_timer() - st, "shifts")
        
        sh_x_n = tf.cast(-(sh_x - ms_h), tf.float32)
        sh_y_n = tf.cast(-(sh_y - ms_w), tf.float32)
        sh_z_n = tf.cast(-(sh_z - ms_d), tf.float32)
        
        tensor_ncc_log = tf.math.log(tensor_ncc)

        n_batches = np.arange(1)

        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x-1, axis=0), tf.squeeze(sh_y, axis=0)]))
        log_xm1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x+1, axis=0), tf.squeeze(sh_y, axis=0)]))
        log_xp1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y-1, axis=0)]))
        #tf.print(idx, sh_y)
        log_x_ym1 = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y+1, axis=0)]))
        log_x_yp1 =  tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y, axis=0)]))
        four_log_xy = 4 * tf.gather_nd(tensor_ncc_log, idx)
        # tf.print(timeit.default_timer() - st, "four")

        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))

        return tf.reshape(sh_x_n, [1, 1]), tf.reshape(sh_y_n, [1, 1])