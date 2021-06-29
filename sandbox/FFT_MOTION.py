#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:15:25 2021

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
        
#%%
class MotionCorrect(keras.layers.Layer):
    def __init__(self, template, center_dims, ms_h=10, ms_w=10, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
        
        super().__init__(**kwargs)
        
        self.ms_h = ms_h
        self.ms_w = ms_w
        
        self.strides = strides
        self.padding = padding
        self.epsilon =  epsilon
        
        self.template_0 = template
        self.shp_0 = self.template_0.shape
        self.center_dims = center_dims

        self.shp_c_x, self.shp_c_y = (self.shp_0[0] - center_dims[0])//2, (self.shp_0[1] - center_dims[1])//2
        
        self.xmin, self.ymin = self.shp_0[0]-2*ms_w, self.shp_0[1]-2*ms_h
        if self.xmin < 5 or self.ymin < 5:
            raise ValueError("The frame dimensions you entered are too small. Please provide a larger field of view or resize your movie.") 
        
        self.template = template
        if self.shp_0[0] != center_dims[0]:
            self.template = self.template[self.shp_c_x:-self.shp_c_x, :]
        if self.shp_0[1] != center_dims[1]:
            self.template = self.template[:, self.shp_c_y:-self.shp_c_y]
        
        self.shp = self.template.shape
        self.shp_prod = tf.cast(tf.reduce_prod(self.shp), tf.float32)

        self.shp_m_x, self.shp_m_y = self.shp[0]//2, self.shp[1]//2
        
        self.template_zm, self.template_var = self.normalize_template(self.template, epsilon=self.epsilon)
        
        self.target_freq = tf.signal.fft3d(tf.cast(self.template_zm[:,:,:,0], tf.complex128))
        plt.imshow(tf.cast(self.target_freq, tf.float32));plt.colorbar()
            
       
    @tf.function
    def call(self, fr):
        # print(fr.shape)
        # fr = tf.cast(fr[None, :, :, None], tf.float32)
        # fr =  fr[0:1,:,:,None]
        fr_center = fr[0, self.shp_c_x:(self.shp_0[0]-self.shp_c_x), self.shp_c_y:(self.shp_0[1]-self.shp_c_y)][None]
        # fr = fr[0,128:-128,128:-128][None]
        # fr = fr[0][None]
        # print(tf.math.reduce_mean(fr))
        # fr = fr[0][None]
        # print(fr.shape, self.template_var.shape)
        imgs_zm, imgs_var = self.normalize_image(fr_center, self.shp, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon)
        denominator = tf.sqrt(self.template_var * imgs_var)
        # print(imgs_zm.shape, imgs_var.shape)

        fr_freq = tf.signal.fft3d(tf.cast(imgs_zm[0], tf.complex128))
        img_product = fr_freq *  tf.math.conj(self.target_freq)

        cross_correlation = tf.cast(tf.math.abs(tf.signal.ifft3d(img_product)), tf.float32)[None,:]
        # print(cross_correlation.shape, img_product.shape, self.shp_m_x)
        # print(self.shp_m_x, self.shp_m_y)
        rolled_cc =  tf.roll(cross_correlation,(self.shp_m_x,self.shp_m_y), axis=(1,2))
        # print(rolled_cc.shape)
        # ncc = rolled_cc[:,self.shp_m_x-self.ms_w:self.shp_m_x+self.ms_w+1, self.shp_m_y-self.ms_h:self.shp_m_y+self.ms_h+1]/denominator
        ncc = rolled_cc[:,self.shp_m_x-self.ms_w:self.shp_m_x+self.ms_w+1, self.shp_m_y-self.ms_h:self.shp_m_y+self.ms_h+1]/denominator
        ncc = tf.where(tf.math.is_nan(ncc), tf.zeros_like(ncc), ncc)

        # plt.imshow(tf.squeeze(ncc))
        # print(tf.math.reduce_sum(ncc))
        
        sh_x, sh_y = self.extract_fractional_peak(ncc, self.ms_h, self.ms_w)

        fr_corrected = tfa.image.translate(fr, (tf.squeeze(tf.stack([sh_x, sh_y], axis=1))), 
                                            interpolation="bilinear")
        # print(tf.math.reduce_sum(fr_corrected))
        # print(sh_x)
        return (tf.reshape(tf.transpose(tf.squeeze(fr_corrected)), [-1])[None, :], [sh_x, sh_y])
        # return(tf.squeeze(fr), [sh_x,sh_y])
    
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
        return template_zm, template_var
        
    def normalize_image(self, imgs, shape_template, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed

        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2], keepdims=True)
        img_stack = tf.stack([imgs[:,:,:,0], tf.square(imgs)[:,:,:,0]], axis=3)
        # print(img_stack.shape, imgs_zm.shape, imgs.shape, shape_template)

        localsum_stack = tf.nn.avg_pool2d(img_stack,[1,self.template.shape[0]-2*self.ms_w, self.template.shape[1]-2*self.ms_h, 1], 
                                               padding=padding, strides=strides)
        localsum_ustack = tf.unstack(localsum_stack, axis=3)

        localsum_sq = localsum_ustack[1][:,:,:,None]
        localsum = localsum_ustack[0][:,:,:,None]

        imgs_var = localsum_sq - tf.square(localsum)/self.shp_prod + epsilon
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
        # print(ncc_log.shape, np.mean(ncc_log), np.min(ncc))

        n_batches = np.arange(1)

        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x-1, axis=0), tf.squeeze(sh_y, axis=0)]))
        log_xm1_y = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x+1, axis=0), tf.squeeze(sh_y, axis=0)]))
        log_xp1_y = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y-1, axis=0)]))
        log_x_ym1 = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y+1, axis=0)]))
        log_x_yp1 =  tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=0), tf.squeeze(sh_y, axis=0)]))
        four_log_xy = 4 * tf.gather_nd(ncc_log, idx)

        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))

        return tf.reshape(sh_x_n, [1, 1]), tf.reshape(sh_y_n, [1, 1])
    
    def argmax_2d(self, tensor):
        # extract peaks from 2D tensor (takes batches as input too)
        
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (1, tensor.shape[-3]*tensor.shape[-2], 1))

        argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
        # print(argmax, flat_tensor.shape, tensor.shape, "amax")
        # convert indexes into 2D coordinates
        argmax_x = tf.cast(argmax, tf.int32) // tf.shape(tensor)[2]
        argmax_y = tf.cast(argmax, tf.int32) % tf.shape(tensor)[2]
        # print(argmax_x)
        # stack and return 2D coordinates
        return tf.cast(tf.stack((argmax_x, argmax_y), axis=1), tf.float32)
        
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template_0,"strides": self.strides, "center_dims": self.center_dims,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w }
#%%
def get_mc_model(template, center_dims, ms_h=10, ms_w=10):
    """
    takes as input a template (median) of the movie, A_sp object, and b object from caiman.
    outputs the model: {Motion_Correct layer => Compute_Theta2 layer => NNLS * numlayer}
    """
    shp_x, shp_y = template.shape[0], template.shape[1] #dimensions of the movie
    # print(shp_x,shp_y,center_dims)
    template = template.astype(np.float32)

    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 

    #Initialization of the motion correction layer, initialized with the template
    #import pdb; pdb.set_trace();
    mc_layer = MotionCorrect(template, center_dims, ms_h=ms_h, ms_w=ms_w)   
    mc, shifts = mc_layer(fr_in)

   
    #create final model, returns it and the first weight
    model = keras.Model(inputs=[fr_in], outputs=[mc, shifts])   
    return model
#%%
from viola.nnls_gpu import NNLS, compute_theta2
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
#    theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)

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