#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:13:01 2021

@author: nel
"""
import os
import logging
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from threading import Thread
#mport tensorflow_addons as tfa
import numpy as np
from queue import Queue
import timeit
from time import time
import tensorflow.keras as keras
from scipy.ndimage.filters import gaussian_filter
from fiola.utilities import apply_shifts_dft, local_correlations_movie, play, bin_median_3d, bin_median
from tifffile.tifffile import imsave
from skimage.io import imread
import matplotlib.pyplot as plt
import math

#%%
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
     
    #@tf.function
    def call(self, fr):
        print(self.center_dims)
        if self.center_dims is None:
            fr_center = fr[0]
        else:      
            fr_center = fr[0,:, self.shp_c_x:(self.shp_0[0]-self.shp_c_x), self.shp_c_y:(self.shp_0[1]-self.shp_c_y)]

        print(fr_center.shape)
        imgs_zm, imgs_var = self.normalize_image(fr_center, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon)
        denominator = tf.sqrt(self.template_var * imgs_var)

        if self.use_fft:
            fr_freq = tf.signal.fft3d(tf.cast(imgs_zm[:,:,:,0], tf.complex128))
            
            print(fr_freq.shape)
            print(self.target_freq.shape)
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
        shifts = [sh_x, sh_y]
        print(f'frame shape: {fr[0].shape}')
        #fr_corrected = tfa.image.translate(fr[0], (tf.squeeze(tf.stack([sh_y, sh_x], axis=1))), 
        #                                    interpolation="bilinear")
        fr_corrected = self.apply_shifts_dft_tf(fr[0], [-sh_x, -sh_y])
        return tf.reshape(tf.transpose(tf.squeeze(fr_corrected, axis=3), perm=[0,2,1]), (self.batch_size, self.shp_0[0]*self.shp_0[1])), shifts
    
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
    
    def apply_shifts_dft_tf(self, img, shifts, diffphase=tf.cast([0],dtype=tf.complex64)):
        img = tf.cast(img, dtype=tf.complex64)
        if len(shifts) == 3:
            shifts =  (shifts[1], shifts[0], shifts[2]) 
        elif len(shifts) == 2:
            shifts = (shifts[1], shifts[0], tf.zeros(shifts[1].shape))
        src_freq = tf.signal.fft3d(img)
        nshape = tf.cast(tf.shape(src_freq), dtype=tf.float32)[1:]
        nc = nshape[0]
        nr = nshape[1]
        nd = nshape[2]
        Nr = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(nr / 2.), tf.math.ceil(nr / 2.)))
        Nc = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(nc / 2.), tf.math.ceil(nc / 2.)))
        Nd = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(nd / 2.), tf.math.ceil(nd / 2.)))
        Nr, Nc, Nd = tf.meshgrid(Nr, Nc, Nd)
        sh_0 = tf.tensordot(-shifts[0], Nr[None], axes=[[1], [0]]) / nr
        sh_1 = tf.tensordot(-shifts[1], Nc[None], axes=[[1], [0]]) / nc
        sh_2 = tf.tensordot(-shifts[2], Nd[None], axes=[[1], [0]]) / nd
        sh_tot = (sh_0 +  sh_1 + sh_2)
        Greg = src_freq * tf.math.exp(-1j * 2 * math.pi * tf.cast(sh_tot, dtype=tf.complex64))
        
        #todo: check difphase and eventually dot product?
        Greg = Greg * tf.math.exp(1j * diffphase)
        new_img = tf.math.real(tf.signal.ifft3d(Greg)) 
        return new_img
        
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template,"strides": self.strides, "batch_size":self.batch_size,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w }
    
    #%%
    ms_h = 5; ms_w = 5
    center_dims = None#(256, 256)
    batch = 1   
    #mov = imread('/media/nel/DATA/Panos/fun_x_4/movie_100_1_100/movie_730.tiff')
    mov = imread('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/demo_K53/k53.tif')
    #with h5py.File('/home/nel/caiman_data/example_movies/volpy/demo_voltage_imaging.hdf5','r') as h5:
    #    mov = np.array(h5['mov'])
    
    mov = mov.astype(np.float32)[:3000]
    template = bin_median(mov)
        
    #%%
    mc_mov = []
    for normalize_cc in [True]:#, False]:
        for use_fft in [True]:
            #for center_dims in [None]:
            Y = mov[:].copy()
            mc_layer = MotionCorrectBatch(template, batch_size=batch, center_dims=center_dims, 
                                          ms_h=ms_h, ms_w=ms_w, normalize_cc=normalize_cc, use_fft=use_fft,
                                          epsilon=0.00001)
            data = Y[None, ..., None]
            shifts_all = []
            shifts_gpu_mc = np.zeros((Y.shape[0], 3))
            for i in range(data.shape[1]//batch):
                out, shifts = (mc_layer(data[:,batch*i:batch*(i+1)]))
                shifts_all.append(shifts)
                mc_mov.append(out)
            sh_x_all = []
            sh_y_all = []
            #with tf.compat.v1.Session() as sess:  
                #print(shifts_all[0][0].eval())
            for shifts in shifts_all:
                sh_x_all.append(shifts[0])
                sh_y_all.append(shifts[1])
            sh_x_all = tf.concat(sh_x_all, axis=0)
            sh_y_all = tf.concat(sh_y_all, axis=0)
        
            #tf.compat.v1.enable_eager_execution()
            with tf.compat.v1.Session() as sess:  
                sh_x_all = sh_x_all.numpy()
                sh_y_all = sh_y_all.numpy()
                
        
            #plt.figure()
            plt.plot(sh_x_all, label=f'normalize:{normalize_cc}, fft:{use_fft}, center_dims:{center_dims}'); plt.plot(sh_y_all)
            #plt.title(f'center dims {center_dims}')
            plt.legend()
    #%%
    mc_mov = np.array(mc_mov)
    mc_mov.shape    
    mm = mc_mov.reshape([-1, 512,512], order='F')   
    mmm = np.concatenate((mov, mm), axis=2)
    play(mmm, q_max=99.9)
    #%%
    
    
    
    
    mov = np.zeros((300, 70, 50)).astype(np.float32).astype(np.float32)
    #mov = mov + np.random.rand(mov.shape[0], mov.shape[1], mov.shape[2], mov.shape[3])/1.5
    mov = mov.astype(np.float32)
    ms_w = 5; ms_h = 5

    spots = [5,10,15, 20,25, 30,35, 40, 45]
    
    for i in range(mov.shape[0]):
        for spot in spots:
            mov[i][spot-2:spot+2,spot-2:spot+2] = 5
    template = bin_median(mov)
    plt.imshow(template)
    sig = (5,5)
    T = mov.shape[0]
    shifts_gt = np.transpose([np.convolve(np.random.randn(T-10), np.ones(11)/11*s) for s in sig])
    shifts_gt = np.hstack((shifts_gt, np.zeros(shifts_gt.shape[0])[:, None]))
    shifts_gt[0] = 0
    mov = mov[..., None]
    mov = np.array([apply_shifts_dft(img, tuple(sh), 0, is_freq=False, border_nan='copy') for img, sh in zip(mov, shifts_gt)])[..., 0]
    #mov[1] = np.roll(a=mov[0], shift=(1,3), axis=(0,1))
    #mov[2] = np.roll(a=mov[2], shift=(3,3), axis=(0,1))
    #mov[3] = np.roll(a=mov[3], shift=(-2,-3), axis=(0,1))
    data = mov[None, ..., None]#.astype(np.double)  # batch size, time, x, y, z, channel
   
    gt_x = shifts_gt[:,0]
    gt_y = shifts_gt[:,1]
    print(f'data: {data.shape}')
    print(f'template: {template.shape}')
    
    #%%
    #plt.figure(); plt.imshow(template)
    batch = 20; normalize_cc = True; use_fft = True; center_dims = None
    
    for normalize_cc in [True]:
        for use_fft in [True, False]:
            mc_layer = MotionCorrectBatch(template, batch_size=batch, 
                                          ms_h=ms_h, ms_w=ms_w, normalize_cc=normalize_cc, use_fft=use_fft, epsilon=0.0001)
            
            #data = Y[None, ..., ]
            print(f'data shape: {data.shape}')
            print(f'template shape: {template.shape}')
            sh_x_all = [] 
            sh_y_all = [] 
            shifts_all = []
            
            for i in range(data.shape[1]//batch):
                out, shifts = (mc_layer(data[:,batch*i:batch*(i+1)]))
                shifts_all.append(shifts)
            
            for shifts in shifts_all:
                sh_x_all.append(shifts[0])
                sh_y_all.append(shifts[1])
            sh_x_all = tf.concat(sh_x_all, axis=0)
            sh_y_all = tf.concat(sh_y_all, axis=0)
        
            #tf.compat.v1.enable_eager_execution()
            with tf.compat.v1.Session() as sess:  
                sh_x_all = sh_x_all.numpy()
                sh_y_all = sh_y_all.numpy()
            
            plt.figure()
            plt.plot(sh_y_all, label='y infer')
            plt.plot(np.array(gt_y), label='y gt')
            plt.legend()
            plt.title(f'normalize:{normalize_cc}, fft:{use_fft}')
            
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(mc_layer.ncc[1, :, :, 0])
            ax[1].imshow(mc_layer.nominator[1, :, :, 0])
            ax[2].imshow(mc_layer.denominator[1, :, :, 0])
            plt.title(f'normalize:{normalize_cc}, fft:{use_fft}')
                        
            print(mc_layer.template_var)
            
#%%
fig, ax = plt.subplots(1, 3)
ax[0].imshow(mc1.ncc[1, :, :, 0])
ax[1].imshow(mc1.nominator[1, :, :, 0] / mc2.denominator[1, :, :, 0])
ax[2].imshow(mc1.denominator[1, :, :, 0])

#%%
fig, ax = plt.subplots(1, 3)
ax[0].imshow(mc2.ncc[1, :, :, 0])
ax[1].imshow(mc2.nominator[1, :, :, 0]/mc2.denominator[1, :, :, 0])
ax[2].imshow(mc2.denominator[1, :, :, 0])

            
    #%%
    batch = 20; normalize_cc = True; use_fft = True; center_dims = None
    
    for normalize_cc in [True]:#[True, False]:
        for use_fft in [True, False]:
            mc_layer = MotionCorrectBatch(template, batch_size=batch, 
                                          ms_h=ms_h, ms_w=ms_w, normalize_cc=normalize_cc, use_fft=use_fft)
            
            #data = Y[None, ..., ]
            print(f'data shape: {data.shape}')
            print(f'template shape: {template.shape}')
            sh_x_all = [] 
            sh_y_all = [] 
            shifts_all = []
            
            for i in range(data.shape[1]//batch):
                out, shifts = (mc_layer(data[:,batch*i:batch*(i+1)]))
                shifts_all.append(shifts)
            
            for shifts in shifts_all:
                sh_x_all.append(shifts[0])
                sh_y_all.append(shifts[1])
            sh_x_all = tf.concat(sh_x_all, axis=0)
            sh_y_all = tf.concat(sh_y_all, axis=0)
        
            #tf.compat.v1.enable_eager_execution()
            with tf.compat.v1.Session() as sess:  
                sh_x_all = sh_x_all.numpy()
                sh_y_all = sh_y_all.numpy()
    
            
            plt.figure()
            plt.plot(sh_y_all, label='y infer')
            plt.plot(np.array(gt_y), label='y gt')
            plt.legend()
            plt.title(f'normalize:{normalize_cc}, fft:{use_fft}')
            
            
    #%%
    plt.figure();plt.imshow(mc_layer_fft.denominator[1,:,:,0]); plt.colorbar()
    tf.reduce_min(mc_layer_fft.denominator[1,:,:,0])
    
    
    plt.figure();plt.imshow(mc_layer_nofft.denominator[1,:,:,0]); plt.colorbar()
    tf.reduce_min(mc_layer_nofft.denominator[1,:,:,0])
    
    plt.figure();plt.imshow(mc_layer_fft.nominator[1,:,:,0]- mc_layer_nofft.nominator[1,:,:,0]); plt.colorbar()
    tf.reduce_mean(mc_layer_nofft.nominator[1,:,:,0])
    tf.reduce_mean(mc_layer_fft.nominator[1,:,:,0])
    
    
    #%%
    plt.figure()
    plt.plot(sh_x_all, label='x infer'); 
    plt.plot(np.array(gt_x), label='x gt'); 
    plt.legend()
#%%
    from fiola.utilities import bin_median
    Y = mov[:300].copy()
    template = bin_median(mov)
    
   

#%%
    sh_x_all = []
    sh_y_all = []
    batch = 100
    mc_layer = MotionCorrectBatch(template, batch_size=batch, ms_h=ms_h, ms_w=ms_w)
    data = Y[None, ..., None]
    shifts_all = []
    shifts_gpu_mc = np.zeros((Y.shape[0], 3))
    for i in range(data.shape[1]//batch):
        out, shifts = (mc_layer(data[:,batch*i:batch*(i+1)]))
        with tf.compat.v1.Session() as sess:  
            sh_x_all.append(shifts[0].eval())
#            shifts_all.append(shifts)
        
    
#%%
    output = []        
    batch_size = 100; shp_x = 512; shp_y = 512
    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([batch_size, shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 
    mc_layer = MotionCorrectBatch(template, batch_size, ms_h=ms_h, ms_w=ms_w)   
    mc = mc_layer(fr_in)
    mod = keras.Model(inputs=[fr_in], outputs=[mc])  
    mod.compile(optimizer='rmsprop', loss='mse')
    estimator = tf.keras.estimator.model_to_estimator(keras_model = mod)
    
    #def gen(data, batch):
    #    for i in range(data.shape[1]//batch):
    #        yield {'m':data[:,batch*i:batch*(i+1)]}
        
    #dataset = tf.data.Dataset.from_generator(gen(data, batch=100),
    #                                         output_types={"m": tf.float32}, 
    #                                         output_shapes={"m":(1, batch_size, shp_x, shp_y, 1)})
        
    out = estimator.predict(input_fn=data[:,batch*i:batch*(i+1)], yield_single_examples=False)
        

#%%
    sh_x_all = []
    sh_y_all = []
    #with tf.compat.v1.Session() as sess:  
        #print(shifts_all[0][0].eval())
    for shifts in shifts_all:
        sh_x_all.append(shifts[0])
        sh_y_all.append(shifts[1])
    sh_x_all = tf.concat(sh_x_all, axis=0)
    sh_y_all = tf.concat(sh_y_all, axis=0)

    #tf.compat.v1.enable_eager_execution()
    with tf.compat.v1.Session() as sess:  
        sh_x_all = sh_x_all.numpy()
        sh_y_all = sh_y_all.numpy()
        
#%%
    plt.plot(np.array(sh_x_all).reshape(-1))
    
        
#%%
        sh_x_all.append(sh_x)
        sh_y_all.append(sh_y)
        
    sh_x_all = np.concatenate(sh_x_all, axis=0)
    sh_y_all = np.concatenate(sh_y_all, axis=0)   


    #%% Generate toy datasets
    D = 2
    
    if D == 3:
        fname = os.path.join('/home/nel/caiman_data', 'example_movies', 'demoMovie3D.tif')
        Y, truth, trueSpikes, centers, dims, shifts = gen_data(D=3, p=2)
        imsave(fname, Y)
        print(fname)#%%
        dims = (70, 50, 10)
        
        Y = imread(fname)   
        Cn = local_correlations_movie(Y, swap_dim=False)
        d1, d2, d3 = dims
        x, y = (int(1.2 * (d1 + d3)), int(1.2 * (d2 + d3)))
        scale = 6/x
        fig = plt.figure(figsize=(scale*x, scale*y))
        axz = fig.add_axes([1-d1/x, 1-d2/y, d1/x, d2/y])
        plt.imshow(Cn.max(2).T, cmap='gray')
        plt.title('Max.proj. z')
        plt.xlabel('x')
        plt.ylabel('y')
        axy = fig.add_axes([0, 1-d2/y, d3/x, d2/y])
        plt.imshow(Cn.max(0), cmap='gray')
        plt.title('Max.proj. x')
        plt.xlabel('z')
        plt.ylabel('y')
        axx = fig.add_axes([1-d1/x, 0, d1/x, d3/y])
        plt.imshow(Cn.max(1).T, cmap='gray')
        plt.title('Max.proj. y')
        plt.xlabel('x')
        plt.ylabel('z');
        plt.show()
        
        play(Y[...,5], magnification=2)
        
    elif D == 2:
        fname = os.path.join('/home/nel/caiman_data', 'example_movies', 'demoMovie2D.tif')
        Y, truth, trueSpikes, centers, dims, shifts = gen_data(D=2, T=256)
        imsave(fname, Y)
        print(fname)#%%
    
    #%%
    if D == 3:
        ms_h = 4; ms_w = 4; ms_d = 2
    elif D == 2:
        if len(Y.shape) != 4:
            Y = Y[..., None]
        ms_h = 5; ms_w = 5; ms_d = 0
    Y = Y.astype(np.float32)
    template = bin_median_3d(Y, 30)
    
    #%%
    mc_layer = MotionCorrectBatch(template[:,:,0], batch_size=1, ms_h=ms_h, ms_w=ms_w)
    data = Y[None, ..., ]
    print(f'data shape: {data.shape}')
    print(f'template shape: {template.shape}')
    sh_x_all = [] 
    sh_y_all = [] 
    
    for i in range(data.shape[1]):
        sh_x, sh_y = mc_layer(data[:,i:i+1])
        sh_x_all.append(sh_x)
        sh_y_all.append(sh_y)
        
    sh_x_all = np.concatenate(sh_x_all, axis=0)
    sh_y_all = np.concatenate(sh_y_all, axis=0) 
        
    #%%
    mc_layer = MotionCorrectBatch(template[:,:,0], batch_size=16, ms_h=ms_h, ms_w=ms_w)
    data = Y[None, ...]
    batch = 16
    sh_x_all = [] 
    sh_y_all = [] 
    shifts_gpu_mc = np.zeros((Y.shape[0], 3))
    for i in range(data.shape[1]//batch):
        sh_x, sh_y = (mc_layer(data[:,batch*i:batch*(i+1)]))
        sh_x_all.append(sh_x)
        sh_y_all.append(sh_y)
        
    sh_x_all = np.concatenate(sh_x_all, axis=0)
    sh_y_all = np.concatenate(sh_y_all, axis=0)        
        #coords = np.array([ncc[0],ncc[1],ncc[2]]).T
        #shifts_gpu_mc[batch*i:batch*(i+1)] = coords
        #shifts_all.append(mc_layer.shifts)
        #times.append(time.time()-st)
    #%%
    plt.plot(-sh_x_all)
    plt.plot(-sh_y_all)
    plt.plot(shifts)
    plt.legend(['fiola inferred x', 'fiola inferred y', 'gt x', 'gt y'])
    
    print(f'correlation x: {np.corrcoef(-sh_x_all.flatten(), shifts[:,0].flatten())[0,1]}')
    print(f'correlation y: {np.corrcoef(-sh_y_all.flatten(), shifts[:,1].flatten())[0,1]}')





#%%    
   
    for i in range(data.shape[1]):
        sh_x, sh_y = mc_layer(data[:,i:i+1])
        sh_x_all.append(sh_x)
        sh_y_all.append(sh_y)
        
    sh_x_all = np.concatenate(sh_x_all, axis=0)
    sh_y_all = np.concatenate(sh_y_all, axis=0) 
    