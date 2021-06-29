#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:10:10 2020

@author: nellab
"""
#%%
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import numpy as np
import timeit
import pylab as plt
#%%
class MotionCorrect(keras.layers.Layer):
    def __init__(self, template, center_dims, ms_h=10, ms_w=10, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
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
        self.shp_x, self.shp_y = template.shape[0], template.shape[1]
        self.center_dims = center_dims
        self.c_shp_x, self.c_shp_y = (self.shp_x - center_dims[0])//2, (self.shp_y - center_dims[1])//2
    
        
        self.template_0 = template
        self.xmin, self.ymin = self.shp_x-2*ms_w, self.shp_y-2*ms_h

        if ms_h==0 or ms_w==0:
            self.template = self.template_0[:,:,None,None]
        else:
            self.template=self.template_0[(ms_w+self.c_shp_x):-(ms_w+self.c_shp_x),(ms_h+self.c_shp_y):-(ms_h+self.c_shp_y), None, None]    
            
        if self.xmin < 10 or self.ymin < 10:  #for small movies, change default min shift
            if ms_h==0 or ms_w==0:
                raise ValueError("The frame dimensions you entered are too small. Please provide a larger field of view or resize your movie.") 
            else:    
                ms_h = 5
                ms_w = 5
                self.template=self.template_0[(ms_w+self.c_shp_x):-(ms_w+self.c_shp_x),(ms_h+self.c_shp_y):-(ms_h+self.c_shp_y), None, None]
                
            if self.template.shape[0] < 5 or self.template.shape[1] < 5:
                raise ValueError("The frame dimensions you entered are too small. Please provide a larger field of view or resize your movie.") 
                
        self.ms_h = ms_h
        self.ms_w = ms_w
        self.strides = strides
        self.padding = padding
        self.epsilon = epsilon

        ## normalize template
        # print(self.template.min(), self.template.max())
        self.template_zm, self.template_var = self.normalize_template(self.template, epsilon=self.epsilon)
        ## assign to kernel, normalizer
        self.kernel = self.template_zm
        self.normalizer = self.template_var
        # print(self.normalizer)
        # plt.imshow(tf.squeeze(self.normalizer))
        self.template_shape_prod = tf.cast(1/tf.math.reduce_prod(self.template.shape), tf.float32)


    # @tf.function
    def call(self, X):
        # takes as input a tensorflow batch tensor (batch x width x height x channel)
        # print(X).shape
        X_center = X[:, self.c_shp_x:(self.shp_x-self.c_shp_x), self.c_shp_y:(self.shp_y-self.c_shp_y)]
        # X_center = X[:,:,:]
        # print(X.shape, X_center.shape)
        # import pdb; pdb.set_trace()
        # pass in center for normalization
        imgs_zm, imgs_var = self.normalize_image(X_center, self.template.shape, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon) 

        denominator = tf.sqrt(self.normalizer * imgs_var)
        
        # tf.print(timeit.default_timer()-st, "normalize")
        numerator = tf.nn.conv2d(imgs_zm, self.kernel, padding=self.padding, 
                                  strides=self.strides)
       
        tensor_ncc = tf.truediv(numerator, denominator)
        # tf.print(timeit.default_timer()-st, "conv2d 1")
       
        # Remove any NaN in final output
        tensor_ncc = tf.where(tf.math.is_nan(tensor_ncc), tf.zeros_like(tensor_ncc), tensor_ncc)
        # plt.imshow(tf.squeeze(tensor_ncc))
        # plt.imshow(tf.squeeze(tensor_ncc))
        
        xs, ys = self.extract_fractional_peak(tensor_ncc, ms_h=self.ms_h, ms_w=self.ms_w)
        # tf.print(timeit.default_timer()-st, "extract")
        
        X_corrected = tfa.image.translate(X, (tf.squeeze(tf.stack([ys, xs], axis=1))), 
                                            interpolation="bilinear")
        # print(tf.math.reduce_sum(X_corrected))
        # tf.print(timeit.default_timer()-st, "translate")
        return (tf.reshape(tf.transpose(tf.squeeze(X_corrected)), [-1])[None, :], [xs, ys])
        # return (tf.reshape(tf.transpose(tf.squeeze(X_corrected)), [-1])[None, :])
        # return (X_corrected, [xs,ys])


    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template_0,"strides": self.strides, "center_dims": self.center_dims,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w }  
        
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        # print(template_zm.shape, template.shape)
        # plt.imshow(tf.squeeze(template_zm))
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
        return tf.cast(template_zm, tf.float32), tf.cast(template_var, tf.float32)
        
    def normalize_image(self, imgs, shape_template, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed
        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2], keepdims=True)
        # print(shape_template)
        # localsum_sq = tf.nn.conv2d(tf.square(imgs), tf.ones(shape_template), 
        #                                        padding=padding, strides=strides)
        # localsum = tf.nn.conv2d(imgs,tf.ones(shape_template), 
        #                                        padding=padding, strides=strides)
        # print(imgs.dtype, imgs.shape,  self.template.shape)
        localsum_sq = tf.nn.avg_pool2d(tf.square(imgs), [1,self.template.shape[0], self.template.shape[1], 1], 
                                               padding=padding, strides=strides)*self.template.shape[1]*self.template.shape[2]
        localsum = tf.nn.avg_pool2d(imgs,[1,self.template.shape[0], self.template.shape[1], 1], 
                                               padding=padding, strides=strides)*self.template.shape[1]*self.template.shape[2]
        # print(timeit.default_timer()-st, "ni2", localsum.dtype, self.template_prod.dtype)
        # print(localsum_sq.shape, localsum.shape)
        # plt.imshow(tf.squeeze(imgs_zm))
        
        
        imgs_var = localsum_sq - tf.square(localsum)*self.template_shape_prod + epsilon
        # imgs_var = tf.ones((1, 21, 21, 1))
        # Remove small machine precision errors after subtraction
        imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
        return imgs_zm, imgs_var
    
    def argmax_2d(self, tensor):
        # extract peaks from 2D tensor (takes batches as input too)
        
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (1, tensor.shape[-3]*tensor.shape[-2], 1))

        argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
        # convert indexes into 2D coordinates
        argmax_x = tf.cast(argmax, tf.int32) // tf.shape(tensor)[2]
        argmax_y = tf.cast(argmax, tf.int32) % tf.shape(tensor)[2]
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
        # st = timeit.default_timer()
        shifts_int = self.argmax_2d(tensor_ncc)
        # tf.print(timeit.default_timer() - st, "argmax")

        shifts_int_cast = tf.cast(shifts_int,tf.int64)
        sh_x, sh_y = shifts_int_cast[:,0],shifts_int_cast[:,1]
        # tf.print(timeit.default_timer() - st, "shifts")
        
        sh_x_n = tf.cast(-(sh_x - ms_h), tf.float32)
        sh_y_n = tf.cast(-(sh_y - ms_w), tf.float32)
        
        tensor_ncc_log = tf.math.log(tensor_ncc)

        n_batches = np.arange(1)

        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x-1, axis=0), tf.squeeze(sh_y, axis=0)]))
        log_xm1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x+1, axis=0), tf.squeeze(sh_y, axis=0)]))
        log_xp1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y-1, axis=0)]))
        log_x_ym1 = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y+1, axis=0)]))
        log_x_yp1 =  tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y, axis=0)]))
        four_log_xy = 4 * tf.gather_nd(tensor_ncc_log, idx)
        # tf.print(timeit.default_timer() - st, "four")
        # return tf.reshape(ms_h-1.001, [1,1]), tf.reshape(ms_w-1.001, [1,1])

        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))

        return tf.reshape(sh_x_n, [1, 1]), tf.reshape(sh_y_n, [1, 1])
#%%
class MotionCorrectTest(keras.layers.Layer):
    def __init__(self, template, center_dims, ms_h=10, ms_w=10, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
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
        self.shp_x, self.shp_y = template.shape[0], template.shape[1]
        self.center_dims = center_dims
        self.c_shp_x, self.c_shp_y = (self.shp_x - center_dims[0])//2, (self.shp_y - center_dims[1])//2
    
        
        self.template_0 = template
        self.xmin, self.ymin = self.shp_x-2*ms_w, self.shp_y-2*ms_h

        if ms_h==0 or ms_w==0:
            self.template = self.template_0[:,:,None,None]
        else:
            self.template=self.template_0[(ms_w+self.c_shp_x):-(ms_w+self.c_shp_x),(ms_h+self.c_shp_y):-(ms_h+self.c_shp_y), None, None]    
            
        if self.xmin < 10 or self.ymin < 10:  #for small movies, change default min shift
            if ms_h==0 or ms_w==0:
                raise ValueError("The frame dimensions you entered are too small. Please provide a larger field of view or resize your movie.") 
            else:    
                ms_h = 5
                ms_w = 5
                self.template=self.template_0[(ms_w+self.c_shp_x):-(ms_w+self.c_shp_x),(ms_h+self.c_shp_y):-(ms_h+self.c_shp_y), None, None]
                
            if self.template.shape[0] < 5 or self.template.shape[1] < 5:
                raise ValueError("The frame dimensions you entered are too small. Please provide a larger field of view or resize your movie.") 
                
                
        self.template_prod = tf.cast(1/tf.math.reduce_prod(self.template.shape), tf.float16)
        self.ms_h = ms_h
        self.ms_w = ms_w
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
        # takes as input a tensorflow batch tensor (batch x width x height x channel)
        # print(X).shape
        X = tf.cast(X, tf.float16)
        st = timeit.default_timer()
        X_center = X[:, self.c_shp_x:(self.shp_x-self.c_shp_x), self.c_shp_y:(self.shp_y-self.c_shp_y)]
        # X_center = X[:,:,:]
        print(timeit.default_timer()-st, X_center.dtype, X.dtype)
        # pass in center for normalization
        imgs_zm, imgs_var = self.normalize_image(X_center, self.template.shape, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon) 
        tf.print(timeit.default_timer()-st, "normalize")
        denominator = tf.sqrt(self.normalizer * imgs_var)
        
        numerator = tf.nn.conv2d(imgs_zm, self.kernel, padding=self.padding, 
                                 strides=self.strides)
        print(numerator.shape, denominator.shape)
        tensor_ncc = tf.truediv(numerator, denominator)
        plt.imshow(tensor_ncc)
        plt.colorbar()
        print(timeit.default_timer()-st, "conv2d 1")
       
        # Remove any NaN in final output
        tensor_ncc = tf.where(tf.math.is_nan(tensor_ncc), tf.zeros_like(tensor_ncc), tensor_ncc)
        
        # print(tensor_ncc)
        # return
        
        xs, ys = tf.cast(self.extract_fractional_peak(tensor_ncc, ms_h=self.ms_h, ms_w=self.ms_w), dtype=tf.float32)
        print(timeit.default_timer()-st, "extract", X.dtype, xs.dtype)
        
        X_corrected = tfa.image.translate(X, (tf.squeeze(tf.stack([ys, xs], axis=1))), 
                                            interpolation="bilinear")
        print(timeit.default_timer()-st, "translate")
        return (tf.reshape(tf.transpose(tf.squeeze(X_corrected)), [-1])[None, :], [xs, ys])
        # return (tf.reshape(tf.transpose(tf.squeeze(X_corrected)), [-1])[None, :])
        # return (X_corrected, [xs,ys])


    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template_0,"strides": self.strides, "center_dims": self.center_dims,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w }  
        
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
        return tf.cast(template_zm, tf.float16), tf.cast(template_var, tf.float16)
        
    def normalize_image(self, imgs, shape_template, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed
        st = timeit.default_timer()
        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2], keepdims=True)
        # print(imgs_zm.dtype, imgs.dtype)
        localsum_sq = tf.nn.avg_pool2d(tf.square(imgs), [1,self.template.shape[1], self.template.shape[2], 1], 
                                               padding=padding, strides=strides)*self.template.shape[1]*self.template.shape[2]
        localsum = tf.nn.avg_pool2d(imgs,[1,self.template.shape[1], self.template.shape[2], 1], 
                                               padding=padding, strides=strides)*self.template.shape[1]*self.template.shape[2]
        # print(timeit.default_timer()-st, "ni2", localsum.dtype, self.template_prod.dtype)
        
        imgs_var = localsum_sq - tf.square(localsum)*self.template_prod + epsilon

        print(timeit.default_timer()-st, "ni3", imgs_var.shape)
        # Remove small machine precision errors after subtraction
        imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
        return imgs_zm, imgs_var
    
    def argmax_2d(self, tensor):
        print(tf.executing_eagerly())
        sta = timeit.default_timer()
        # extract peaks from 2D tensor (takes batches as input too)
        
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (1, tensor.shape[-3]*tensor.shape[-2], 1))

        argmax = tf.cast(tf.argmax(flat_tensor, axis=1), dtype=tf.int32)
        shp = tf.shape(tensor)[2]
        print(timeit.default_timer() - sta, "argmax1")
        # convert indexes into 2D coordinates
        argmax_x = argmax / shp
        print(timeit.default_timer() - sta, "argmax1.5")
        argmax_x = tf.cast(tf.floor(argmax_x), dtype=tf.int32)
        print(timeit.default_timer() - sta, "argmax2")
        argmax_y = argmax % shp
        # stack and return 2D coordinates
        return tf.stack((argmax_x, argmax_y), axis=1)
        
    def extract_fractional_peak(self, tensor_ncc, ms_h, ms_w):
        """ use gaussian interpolation to extract a fractional shift
        Args:
            tensor_ncc: tensor
                normalized cross-correlation
                ms_h: max integer shift vertical
                ms_w: max integere shift horizontal
        
        """
        st = timeit.default_timer()
        shifts_int = self.argmax_2d(tensor_ncc)
        print(timeit.default_timer() - st, shifts_int, shifts_int.shape)

        shifts_int_cast = tf.cast(shifts_int,tf.int64)
        sh_x, sh_y = shifts_int_cast[:,0],shifts_int_cast[:,1]
        tf.print(timeit.default_timer() - st, "shifts")
        
        sh_x_n = tf.cast(-(sh_x - ms_h), tf.float16)
        sh_y_n = tf.cast(-(sh_y - ms_w), tf.float16)
        
        tensor_ncc_log = tf.math.log(tensor_ncc)

        n_batches = np.arange(1)

        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x-1, axis=0), tf.squeeze(sh_y, axis=0)]))
        log_xm1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x+1, axis=0), tf.squeeze(sh_y, axis=0)]))
        log_xp1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y-1, axis=0)]))
        log_x_ym1 = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y+1, axis=0)]))
        log_x_yp1 =  tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y, axis=0)]))
        four_log_xy = 4 * tf.gather_nd(tensor_ncc_log, idx)
        tf.print(timeit.default_timer() - st, "four")

        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
        return tf.reshape(sh_x_n, [1, 1]), tf.reshape(sh_y_n, [1, 1])