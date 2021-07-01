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

#%% Extra models for timings and figure generation
def get_mc_model(template, center_dims, ms_h=10, ms_w=10):
    """
    takes as input a template (median) of the movie, A_sp object, and b object from caiman.
    outputs the model: {Motion_Correct layer => Compute_Theta2 layer => NNLS * numlayer}
    """
    shp_x, shp_y = template.shape[0], template.shape[1] #dimensions of the movie
    template = template.astype(np.float32)

    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 

    #Initialization of the motion correction layer, initialized with the template
    mc_layer = MotionCorrect(template, center_dims, ms_h=ms_h, ms_w=ms_w)   
    mc, shifts = mc_layer(fr_in)

   
    #create final model, returns it and the first weight
    model = keras.Model(inputs=[fr_in], outputs=[mc, shifts])   
    return model
#%%
from fiola.nnls_gpu import NNLS, compute_theta2
def get_nnls_model(template, Ab, num_layers=30, ms_h=10, ms_w=10):
    """
    takes as input a template (median) of the movie, A_sp object, and b object from caiman.
    outputs the model: {Motion_Correct layer => Compute_Theta2 layer => NNLS * numlayer}
    """
    shp_x, shp_y = template.shape[0], template.shape[1] #dimensions of the movie
    Ab = Ab.astype(np.float32)
    template = template.astype(np.float32)
    num_components = Ab.shape[-1]

    y_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, 1]), name="y") # Input Layer for components
    x_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, 1]), name="x") # Input layer for components
    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 
    k_in = tf.keras.layers.Input(shape=(1,), name="k") #Input layer for the counter within the NNLS layers
 
    #Calculations to initialize Motion Correction
    AtA = Ab.T@Ab
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)

    #Initialization of the motion correction layer, initialized with the template
    mc_layer = Empty()
    mc = mc_layer(fr_in)
    shifts = [0.0, 0.0]
    #Chains motion correction layer to weight-calculation layer
    c_th2 = compute_theta2(Ab, n_AtA)
    (th2, shifts) = c_th2(mc, shifts)
    #Connects weights, calculated from the motion correction, to the NNLS layer
    nnls = NNLS(theta_1)
    x_kk = nnls([y_in, x_in, k_in, th2, shifts])
    #stacks NNLS 9 times
    for j in range(1, num_layers):
        x_kk = nnls(x_kk)
   
    #create final model, returns it and the first weight
    model = keras.Model(inputs=[fr_in, y_in, x_in, k_in], outputs=[x_kk])   
    return model  
#%%
# max_shifts = [10,10]
# vect = a2[10].copy()
# templ = np.roll(a2.mean(0), (0,0), axis=(0,1))
# # plt.imshow(vect)
# # plt.pause(1)
# # plt.imshow(templ)
# mask = np.ones_like(template).astype(np.float64)
# mask[max_shifts[0]:-max_shifts[0], :] = 0
# mask[:, max_shifts[1]:-max_shifts[1]] = 0
# mask = tf.convert_to_tensor(mask)
# target_image = tf.convert_to_tensor(templ[None,:,:,None].astype(np.complex128))
# target_freq = tf.signal.fft2d(target_image[0,:,:,0])
# shp = vect.shape
#%%
# def mot_cor(vect,templ, target_freq):
#     src_image = tf.convert_to_tensor(vect[None,:,:,None].astype(np.complex128))
#     target_image = tf.convert_to_tensor(templ[None,:,:,None].astype(np.complex128))
#     src_freq = tf.signal.fft2d(src_image[0,:,:,0])
#     print(np.mean(target_image))
#     shape = src_freq.shape
#     image_product = src_freq * tf.math.conj(target_freq)
#     cross_correlation = tf.signal.ifft2d(image_product)
#     new_cross_corr = tf.math.abs(cross_correlation)
#     # plt.imshow(new_cross_corr)
#     plt.imshow(np.roll(new_cross_corr,(shp[0]//2,shp[1]//2), axis=(0,1))[245:267, 245:267])
#     plt.colorbar()
#     print(np.roll(new_cross_corr,(shp[0]//2,shp[1]//2), axis=(0,1)).argmax())
#     #
#     shifts = np.array(np.unravel_index(np.roll(new_cross_corr,(shp[0]//2,shp[1]//2), axis=(0,1)).argmax(), templ.shape))
#     print(shifts)
#     shifts -=  np.array([shp[0]//2, shp[1]//2])
#     return shifts
#     # print(shifts)
# mot_cor(mc, template, target_freq)
#%%
# maxima = np.unravel_index(np.argmax(new_cross_corr),cross_correlation.shape)
# midpoints = np.array([np.fix(axis_size//2) for axis_size in shape])
# shifts = np.array(maxima, dtype=np.float64)
# shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints] 
# print(shifts)
class Empty(keras.layers.Layer):
    def call(self,  fr):
        fr = fr[0][None]
        return tf.reshape(tf.transpose(tf.squeeze(fr)), [-1])[None, :]
        