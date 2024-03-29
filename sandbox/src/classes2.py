#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:41:18 2020

@author: nellab
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from queue import Queue
#from threading import Thread
#from past.utils import old_div
#from skimage import io
import numpy as np
import timeit
#import time as time
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
        #self.template = tf.convert_to_tensor(template[ms_h:-ms_h, ms_w:-ms_w, None, None])
        self.shp = 128
#        self.template = template[self.shp+ms_h:-(ms_h+self.shp),self.shp+ms_w:-(self.shp+ms_w), None, None]
        self.template=template
        self.ms_h = ms_h
        self.ms_w = ms_w
        self.strides = strides
        self.padding = padding
        self.epsilon = epsilon
        self.q = Queue()
#        self.A_ = Ab
#        self.n_AtA_ = n_AtA
        ## normalize template
        self.template_zm, self.template_var = self.normalize_template(self.template, epsilon=self.epsilon)
#        self.template_zm = tf.Variable(template_zm)
#        self.template_var = tf.Variable(template_var)
        self.kernel = self.template_zm
        self.normalizer = self.template_var

#    def build(self, batch_input_shape):
#        # weights here represent the template, so that we can update in case we 
#        # want to use the online algorithm
#        #print(self.template_zm)
##        temp_zm = tf.py_function(func=self.to_numpy, inp=[self.template_zm], Tout=float).numpy()
##        tf.print(temp_zm.dtype)
#        self.kernel = self.add_weight(
#            name="kernel", shape=[*list(self.template_zm.shape)],
#            initializer=tf.constant_initializer(self.template_zm))
#
#        # the normalizer also needs to be updated if the template is updated
#        self.normalizer = self.add_weight(
#            name="normalizer", shape=[*list(self.template_var.shape)],
#            initializer=tf.constant_initializer(self.template_var))
#
#        super().build(batch_input_shape) # must be at the end


    @tf.function
    def call(self, X):
        # takes as input a tensorflow batch tensor (batch x width x height x channel)
#        start = timeit.default_timer()
        X_center = X[:, self.shp:-self.shp, self.shp:-self.shp]
#        X_center = X
        # pass in center (128x128)
        imgs_zm, imgs_var = self.normalize_image(X_center, self.template.shape, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon) 
        denominator = tf.sqrt(self.normalizer * imgs_var)
        numerator = tf.nn.conv2d(imgs_zm, self.kernel, padding=self.padding, 
                                 strides=self.strides)
        tensor_ncc = tf.truediv(numerator, denominator)
        self.kernel = self.kernel*1
        self.normalizer = self.normalizer*1
       
        # Remove any NaN in final output
        tensor_ncc = tf.where(tf.math.is_nan(tensor_ncc), tf.zeros_like(tensor_ncc), tensor_ncc)

#        if tensor_ncc.shape[0] == None:
#            shape = tensor_ncc.shape
#            tensor_ncc = tf.reshape(tensor_ncc, [1, shape[2], shape[2], 1])
        X = tf.reshape(X, [1, X.shape[2], X.shape[2], 1])

        xs, ys = self.extract_fractional_peak(tensor_ncc, ms_h=self.ms_h, ms_w=self.ms_w)

        X_corrected = tfa.image.translate(X, tf.squeeze(tf.stack([ys, xs], axis=1)), 
                                            interpolation="BILINEAR")
#        tf.print(timeit.default_timer()-start)
        return tf.reshape(tf.squeeze(X_corrected), [1, 512**2])
#        return X_corrected


    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template,"strides": self.strides,
#                "template_zm":self.kernel, "template_var": self.normalizer,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w }  
        
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
        return tf.cast(template_zm, tf.float32), tf.cast(template_var, tf.float32)
        
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
        del localsum_sq, localsum
        return imgs_zm, imgs_var
    
    def argmax_2d(self, tensor):
        # extract peaks from 2D tensor (takes batches as input too)
        
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))
        # argmax of the flat tensor
        #argmax = tf.cast(self.softargmax(flat_tensor), tf.int32)
        argmax = self.softargmax(flat_tensor)
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
#        shifts_int = self.argmax_2d(tensor_ncc)

        x_range = tf.range(tensor_ncc.shape[-1])
        x_range = tf.cast(x_range, dtype=tf.float32)
        beta = tensor_ncc*10000000000.0
        shifts_int = tf.reduce_sum(tf.nn.softmax(beta) * x_range, axis=1)

        shifts_int_cast = tf.cast(shifts_int,tf.int64)
        sh_x, sh_y = shifts_int_cast[:,0],shifts_int_cast[:,1]
        
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


        
        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
#        tf.print(float(timeit.default_timer())-start, "3 fractional peak")
        return tf.reshape(sh_x_n, [1, 1]), tf.reshape(sh_y_n, [1, 1])
  
    def generator(self):
        while True:
            try:
                yield self.q.get_nowait()
            except:
                break
        return
    
    def enqueue(self, q, batch):
        for fr in batch:
            q.put(fr)
        return

#%% 
class NNLS(keras.layers.Layer):
    def __init__(self, theta_1, theta_2, **kwargs):
        """
        Tensforflow layer which perform Non Negative Least Squares. Using  https://angms.science/doc/NMF/nnls_pgd.pdf
            arg min f(x) = 1/2 || Ax − b ||_2^2
             {x≥0}
             
        Notice that the input must be a tensorflow tensor with dimension batch 
        x width x height x channel
        Args:
           theta_1: ndarray
               theta_1 = (np.eye(A.shape[-1]) - AtA/n_AtA)
           theta_2: ndarray
               theta_2 = (Atb/n_AtA)[:,None]  
          
        
        Returns:
           x regressed values
        
        """
        super().__init__(**kwargs)
#        self.th1 = theta_1
        self.th1 = theta_1.astype(np.float32)
        self.theta_2 = theta_2
#        self.wt = weight
        

    def build(self, batch_input_shape):
        # theta_1 param in https://angms.science/doc/NMF/nnls_pgd.pdf
#        self.th1 = self.add_weight(
#            name="kernel", shape=[*self.theta_1.shape],
#            initializer=tf.constant_initializer(self.theta_1))
        self.th2 = self.add_weight(
            name="normalizer", shape=[*self.theta_2.shape],
            initializer=tf.constant_initializer(self.theta_2)) 
        super().build(batch_input_shape) # must be at the end
    
#    @tf.function
    def call(self, X):
        """
        pass as inputs the new Y, and the old X. see  https://angms.science/doc/NMF/nnls_pgd.pdf
        """
        (Y,X_old) = X
#        mm = tf.matmul(self.th1.astype(np.float32), Y)
        mm = tf.matmul(self.th1, Y)
        new_X = tf.nn.relu(mm + self.th2)

        Y_new = new_X + (-1)/(2)*(new_X-X_old)  
        return (Y_new, new_X)
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "theta_2": self.theta_2, "theta_1": self.th1}  
#%%
class NNLS1(keras.layers.Layer):
    def __init__(self, theta_1, **kwargs):
        """
        Tensforflow layer which perform Non Negative Least Squares. Using  https://angms.science/doc/NMF/nnls_pgd.pdf
            arg min f(x) = 1/2 || Ax − b ||_2^2
             {x≥0}
             
        Notice that the input must be a tensorflow tensor with dimension batch 
        x width x height x channel
        Args:
           theta_1: ndarray
               theta_1 = (np.eye(A.shape[-1]) - AtA/n_AtA)
           theta_2: ndarray
               theta_2 = (Atb/n_AtA)[:,None]  
          
        
        Returns:
           x regressed values
        
        """
        super().__init__(**kwargs)
#        self.th1 = theta_1
        self.th1 = theta_1.astype(np.float32)
#        self.theta_2 = theta_2
#        self.wt = weight
        

#    def build(self, batch_input_shape):
#        # theta_1 param in https://angms.science/doc/NMF/nnls_pgd.pdf
##        self.th1 = self.add_weight(
##            name="kernel", shape=[*self.theta_1.shape],
##            initializer=tf.constant_initializer(self.theta_1))
#        self.th2 = self.add_weight(
#            name="normalizer", shape=[*self.theta_2.shape],
#            initializer=tf.constant_initializer(self.theta_2)) 
#        super().build(batch_input_shape) # must be at the end
    
#    @tf.function
    def call(self, X):
        """
        pass as inputs the new Y, and the old X. see  https://angms.science/doc/NMF/nnls_pgd.pdf
        """
        (Y,X_old,weight) = X
#        mm = tf.matmul(self.th1.astype(np.float32), Y)
        mm = tf.matmul(self.th1, Y)
        new_X = tf.nn.relu(mm + weight)

        Y_new = new_X + (-1)/(2)*(new_X-X_old)  
        return (Y_new, new_X, weight)
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "theta_1": self.th1} 
#%%
class compute_theta2_AG(keras.layers.Layer):
    def __init__(self, A, n_AtA, **kwargs):
        # tf.keras.backend.clear_session() 
        super().__init__(**kwargs)
        self.A = A
        self.n_AtA = n_AtA
#        self.a = 0
        
#    @tf.function
    def call(self, X):
#        self.A = tf.Variable(self.A)
#        self.a = tf.Variable(self.A, name="BAD", trainable=True)
        Y = tf.matmul(X, self.A)
#        self.A = tf.Variable(self.A, name="BAD", trainable=True)
        Y = tf.divide(Y, self.n_AtA)
        Y = tf.transpose(Y)
        return Y    
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "A":self.A, "n_AtA":self.n_AtA}
#%%
class compute_theta2(keras.layers.Layer):
    def __init__(self, A, n_AtA,t1, **kwargs):
        # tf.keras.backend.clear_session()
       
        super().__init__(**kwargs)
        self.A_ = A
        self.n_AtA_ = n_AtA
        self.t1 = t1
        

    def build(self, batch_input_shape):
        # theta_1 param in https://angms.science/doc/NMF/nnls_pgd.pdf
        self.A = self.add_weight(
            name="A", shape=[*self.A_.shape],
            initializer=tf.constant_initializer(self.A_))
        # theta_2 param in https://angms.science/doc/NMF/nnls_pgd.pdf
        self.n_AtA = self.add_weight(
            name="n_AtA", shape=[*self.n_AtA_.shape],
            initializer=tf.constant_initializer(self.n_AtA_))
        super().build(batch_input_shape) # must be at the end
    
    @tf.function
    def call(self, X):
        """
        pass as inputs the new Y, and the old X. see  https://angms.science/doc/NMF/nnls_pgd.pdf
        """
        self.t1=self.t1*1
        return self.temp(X)
    
    def temp(self, X):
#        tf.compat.v1.disable_v2_behavior()
        self.A = self.A*1
        self.n_AtA = self.n_AtA*1
        Y = tf.matmul(X, self.A) 
        Y = tf.divide(Y,self.n_AtA)
        Y = tf.transpose(Y)
        return Y
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "A":self.A, "n_AtA":self.n_AtA, "t1":self.t1}