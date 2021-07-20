#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:49:25 2021

@author: nel
"""
#%%
import math
import numpy as np
import os
import pylab as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import timeit
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
os.environ["TF_XLA_FLAGS"]="--tf_xla_enable_xla_devices" 
#from fiola.image_warp_3d import trilinear_interpolate_tf,  dense_image_warp_3D     

from scipy.ndimage.filters import gaussian_filter
from fiola.utilities import apply_shifts_dft, local_correlations_movie, play, bin_median_3d
from tifffile.tifffile import imsave
from skimage.io import imread

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
        self.shp = self.template.shape
        self.shp_prod = tf.cast(tf.reduce_prod(self.shp), tf.float32)

        self.shp_m_x, self.shp_m_y, self.shp_m_z = self.shp[0]//2, self.shp[1]//2, self.shp[2]//2
                
        self.target_freq = tf.signal.fft3d(tf.cast(self.template_zm[None, :,:,:,0,0], tf.complex128))   # 1*x*y*z
        nshape = tf.cast(tf.shape(self.target_freq[1:]), dtype=tf.float32)
        self.nc = nshape[1]
        self.nr = nshape[2]
        self.nd = nshape[3]
        Nr = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(self.nr / 2.), tf.math.ceil(self.nr / 2.)))
        Nc = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(self.nc / 2.), tf.math.ceil(self.nc / 2.)))
        Nd = tf.signal.ifftshift(tf.range(-tf.experimental.numpy.fix(self.nd / 2.), tf.math.ceil(self.nd / 2.)))
    
        self.Nr, self.Nc, self.Nd = tf.meshgrid(Nr, Nc, Nd)            
       
    # @tf.function
    def call(self, fr):
        fr = fr[0]
        print(f'fr: {fr.shape}')
        imgs_zm, imgs_var = self.normalize_image(fr, self.shp, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon)
        denominator = tf.sqrt(tf.cast(self.template_var, tf.float32) * imgs_var)[...,0]
        self.imgs_zm = imgs_zm
        self.denominator = denominator
        fr_freq = tf.signal.fft3d(tf.cast(imgs_zm[:,:,:,:,0], tf.complex128)) # batch *x*y*z
        self.fr_freq = fr_freq
        img_product = fr_freq *  tf.math.conj(self.target_freq)
        self.img_product = img_product
        #import pdb
        #pdb.set_trace()
        cross_correlation = tf.cast(tf.math.abs(tf.signal.ifft3d(img_product)), tf.float32)
        
        self.corr = cross_correlation
        self.denominator = denominator
        rolled_cc =  tf.roll(cross_correlation,(self.shp_m_x,self.shp_m_y,self.shp_m_z), axis=(1,2,3))
        self.cc = rolled_cc[:,self.shp_m_x-self.ms_h:self.shp_m_x+self.ms_h+1, self.shp_m_y-self.ms_w:self.shp_m_y+self.ms_w+1, self.shp_m_z-self.ms_d:self.shp_m_z+self.ms_d+1]
        ncc = rolled_cc[:,self.shp_m_x-self.ms_h:self.shp_m_x+self.ms_h+1, self.shp_m_y-self.ms_w:self.shp_m_y+self.ms_w+1, self.shp_m_z-self.ms_d:self.shp_m_z+self.ms_d+1]#/denominator
        # denominator seems not useful as we are doing fft/ifft
        
        ncc = tf.where(tf.math.is_nan(ncc), tf.zeros_like(ncc), ncc)
        self.ncc = ncc
        sh_x, sh_y, sh_z = self.extract_fractional_peak(ncc, self.ms_h, self.ms_w, self.ms_d)
        self.shifts = [-sh_x,-sh_y,-sh_z]
        fr_corrected = self.apply_shifts_dft_tf(src_freq=fr_freq, shifts=[-sh_x,-sh_y,-sh_z])

        return fr_corrected, [sh_x, sh_y, sh_z]
    
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
        print(f'img_stack: {img_stack.shape}')

        localsum_stack = tf.nn.avg_pool3d(img_stack,[1,self.template.shape[0]-2*self.ms_h, self.template.shape[1]-2*self.ms_w, self.template.shape[2]-2*self.ms_d, 1], 
                                               padding=padding, strides=strides)
        print(f'localsum_stack: {localsum_stack.shape}')
        localsum_ustack = tf.unstack(localsum_stack, axis=4)
        
        localsum_sq = localsum_ustack[1][:,:,:,:,None]
        localsum = localsum_ustack[0][:,:,:,:,None]

        imgs_var = localsum_sq - tf.square(localsum) + epsilon #/self.shp_prod + epsilon
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
        # tf.print(timeit.default_timer() - st, "shifts")
        
        sh_x_n = tf.cast(-(sh_x - ms_h), tf.float32)
        sh_y_n = tf.cast(-(sh_y - ms_w), tf.float32)
        sh_z_n = tf.cast(-(sh_z - ms_d), tf.float32)
        
        ncc_log = tf.math.log(ncc)
        # print(ncc_log.shape, sh_x, sh_y, sh_z, "indx\n")

        n_batches = np.arange(ncc.shape[0])
        
        idx = tf.transpose(tf.stack([n_batches, sh_x-1, sh_y, sh_z]))
        log_xm1_y_z = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x+1, sh_y, sh_z]))
        log_xp1_y_z = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x, sh_y-1, sh_z]))
        log_x_ym1_z = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x, sh_y+1, sh_z]))
        log_x_yp1_z =  tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x, sh_y, sh_z-1]))
        log_x_y_zm1 =  tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x, sh_y, sh_z+1]))
        log_x_y_zp1 =  tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x, sh_y, sh_z]))
        six_log_xyz = 6 * tf.gather_nd(ncc_log, idx)    # 4 or 6 need check

        # print()
        # print(six_log_xyz, " siz")
        # print()
        # print(np.mean(ncc_log),  "mean")
        # print()
        # print("idx", idx)
        # print()
        # print(log_x_ym1_z,log_x_yp1_z)

        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y_z - log_xp1_y_z), (3 * log_xm1_y_z - six_log_xyz + 3 * log_xp1_y_z))   
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1_z - log_x_yp1_z), (3 * log_x_ym1_z - six_log_xyz + 3 * log_x_yp1_z))
        sh_z_n = sh_z_n - tf.math.truediv((log_x_y_zm1 - log_x_y_zp1), (3 * log_x_y_zm1 - six_log_xyz + 3 * log_x_y_zp1))
        print(sh_x_n, "x\n")
        return sh_x_n, sh_y_n, sh_z_n
    
    def argmax_3d(self, tensor):
        # extract peaks from 3D tensor (takes batches as input too)

        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tensor.shape[0], -1))

        argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
        # convert indexes into 3D coordinates
        shp = tf.shape(tensor)        
        yz_flat = shp[2]*shp[3]
        argmax_x = argmax // yz_flat 
        argmax_y = (argmax % yz_flat) // shp[3]
        argmax_z = argmax - argmax_x*yz_flat - shp[3]*argmax_y
        
        # stack and return 2D coordinates
        print(argmax_x, argmax_y, argmax_z, argmax, "!!!!argmax\n")
        return (argmax_x, argmax_y, argmax_z)
    
    # @tf.function
    def apply_shifts_dft_tf(self, src_freq, shifts, diffphase=tf.cast([0],dtype=tf.complex128)):
        shifts =  (shifts[1], shifts[0], shifts[2]) #p.array(list(shifts[:-1][::-1]) + [shifts[-1]])
        # print(shifts)
        #src_freq  = src_freq
        # print(src_freq.dtype)
        #import pdb
        #pdb.set_trace()
        #sh_0 = -shifts[0] * self.Nr / self.nr
        #sh_1 = -shifts[1] * self.Nc / self.nc
        #sh_2 = -shifts[2] * self.Nd / self.nd
        sh_0 = tf.tensordot(-shifts[0], self.Nr / self.nr, axes=0)
        sh_1 = tf.tensordot(-shifts[1], self.Nc / self.nc, axes=0)
        sh_2 = tf.tensordot(-shifts[2], self.Nd / self.nd, axes=0)


        sh_tot = (sh_0 +  sh_1 + sh_2)
        Greg = src_freq * tf.math.exp(-1j * 2 * math.pi * tf.cast(sh_tot, dtype=tf.complex128))

        #todo: check difphase and eventually dot product?
        Greg = Greg * tf.math.exp(1j * diffphase)
        new_img = tf.math.real(tf.signal.ifft3d(Greg))
        
    
        return new_img
        
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template,"strides": self.strides,
                "padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w , "ms_d":self.ms_d}



#%% A test dataset
if __name__ == "__main__":
    # movie = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/PanosBoyden/Organoids/Org3_BP_Fov2_run1/plane_51.hdf5'
    # import h5py
    # #%%
    # with h5py.File(movie, "r") as f:
    #     print("Keys: %s" % f.keys())
    #     a_group_key = list(f.keys())[0]
    #     mov = np.array(f['mov'])
    # #%%    
    mov = np.zeros((4, 70, 50, 20)).astype(np.float32).astype(np.float32)
    #mov = mov + np.random.rand(mov.shape[0], mov.shape[1], mov.shape[2], mov.shape[3])/1.5
    mov = mov.astype(np.float32)
    
    for i in range(mov.shape[0]):
        spot = 5
        mov[i][spot-2:spot+2,spot-2:spot+2,spot-2:spot+2] = 20.0
    template = mov[0].copy()
    mov[1] = np.roll(a=mov[1], shift=(1,3,2), axis=(0,1,2))
    mov[2] = np.roll(a=mov[2], shift=(3,3), axis=(0,1))
    mov[3] = np.roll(a=mov[3], shift=(-2,-3), axis=(0,1))
    data = mov[None, ..., None]#.astype(np.double)  # batch size, time, x, y, z, channel
    
    print(f'data: {data.shape}')
    print(f'template: {template.shape}')
    
    #%%
    mc_layer = MotionCorrect(template[:,:,:,None,None], ms_h=6, ms_w=5, ms_d=4)
    import time
    st = time.time()
    times = []
    nccs = []
    for i in range(mov.shape[0]):
        corrected = mc_layer(data[:,3:4])
        times.append(time.time()-st)
        # nccs.append(np.array(ncc).squeeze())
        break
    # print(time.time()-st)
    #fr_corrected = ncc
    #coords = np.array([ncc[0],ncc[1],ncc[2]]).flatten()
    #tensor = np.squeeze(ncc[-1])  # template frequency
    #cc = np.squeeze(ncc[-2])    # frame frequency
    self = mc_layer
    ncc = self.ncc.numpy()
    np.where(ncc==ncc.max())
    
    corr = self.corr.numpy()
    np.where(corr==corr.max())
    
    cc = self.cc.numpy()
    np.where(cc==cc.max())
    plt.imshow(cc[0, :, :, 6]) 
    
    
    dd = self.denominator.numpy()
    np.where(dd==dd.max())
    dd[0, 7, 8, 6]
    dd[0, 8, 8, 6]
    plt.imshow(dd[0, :, :, 6]) 
    
    print(self.shifts)
    #xx = fr_corrected.numpy()
    #np.where(xx == xx.max())
    #np.where(mc_layer.ncc.numpy()==mc_layer.ncc.numpy().max())
    #np.where(ncc == ncc.max())
    #%%
    ii = 5
    plt.imshow(template[:,:,ii])
    plt.imshow(mov[3][:,:,ii])
    plt.imshow(corrected[0].numpy()[0, :,:,ii])#; plt.colorbar()
    #plt.imshow(self.denominator.numpy()[0, ii, :, :])
    
    
    
    #%%
    def gen_data(p=1, D=3, dims=(70,50,10), sig=(4,4,2), bkgrd=10, N=20, noise=.5, T=256, 
                 framerate=30, firerate=2., motion=True, plot=False):
        if p == 2:
            gamma = np.array([1.5, -.55])
        elif p == 1:
            gamma = np.array([.9])
        else:
            raise
        dims = dims[:D]
        sig = sig[:D]
        
        np.random.seed(0)#7)
        centers = np.asarray([[np.random.randint(s, x - s)
                               for x, s in zip(dims, sig)] for i in range(N)])
        if motion:
            centers += np.array(sig) * 2
            Y = np.zeros((T,) + tuple(np.array(dims) + np.array(sig) * 4), dtype=np.float32)      
        else:
            Y = np.zeros((T,) + dims, dtype=np.float32)
        trueSpikes = np.random.rand(N, T) < firerate / float(framerate)
        trueSpikes[:, 0] = 0
        truth = trueSpikes.astype(np.float32)
        for i in range(2, T):
            if p == 2:
                truth[:, i] += gamma[0] * truth[:, i - 1] + gamma[1] * truth[:, i - 2]
            else:
                truth[:, i] += gamma[0] * truth[:, i - 1]
        for i in range(N):
            Y[(slice(None),) + tuple(centers[i, :])] = truth[i]
            #Y[:, centers[i, 0], centers[i, 1], centers[i, 2]] = truth[i]
        tmp = np.zeros(dims)
        tmp[tuple(np.array(dims)//2)] = 1.
        z = np.linalg.norm(gaussian_filter(tmp, sig).ravel())
        Y = bkgrd + noise * np.random.randn(*Y.shape) + 10 * gaussian_filter(Y, (0,) + sig) / z
        if motion:
            shifts = np.transpose([np.convolve(np.random.randn(T-10), np.ones(11)/11*s) for s in sig])
            
            if D == 2:
                shifts = np.hstack((shifts, np.zeros(shifts.shape[0])[:, None]))
                Y = Y[..., None]
            
            Y = np.array([apply_shifts_dft(img, tuple(sh), 0,
                                                                         is_freq=False, border_nan='copy')
                                   for img, sh in zip(Y, shifts)])
            #Y = Y[:, 2*sig[0]:-2*sig[0], 2*sig[1]:-2*sig[1], 2*sig[2]:-2*sig[2]]
            Y = Y[(slice(None),) + tuple([(slice(2*si, -2*si, 1)) for si in sig])]
            
            if D == 2:
                #shifts = shifts[..., 0]
                Y = Y[..., 0]
        else:
            shifts = None
            
        print(Y.shape)
        #T, d1, d2, d3 = Y.shape
    
        if plot:
            Cn = local_correlations_movie(Y, swap_dim=False)
            plt.figure(figsize=(15, 3))
            plt.plot(truth.T)
            plt.figure(figsize=(15, 3))
            for c in centers:
                plt.plot(Y[c[0], c[1], c[2]])
    
            d1, d2, d3 = dims
            x, y = (int(1.2 * (d1 + d3)), int(1.2 * (d2 + d3)))
            scale = 6/x
            fig = plt.figure(figsize=(scale*x, scale*y))
            axz = fig.add_axes([1-d1/x, 1-d2/y, d1/x, d2/y])
            plt.imshow(Cn.max(2).T, cmap='gray')
            plt.scatter(*centers.T[:2], c='r')
            plt.title('Max.proj. z')
            plt.xlabel('x')
            plt.ylabel('y')
            axy = fig.add_axes([0, 1-d2/y, d3/x, d2/y])
            plt.imshow(Cn.max(0), cmap='gray')
            plt.scatter(*centers.T[:0:-1], c='r')
            plt.title('Max.proj. x')
            plt.xlabel('z')
            plt.ylabel('y')
            axx = fig.add_axes([1-d1/x, 0, d1/x, d3/y])
            plt.imshow(Cn.max(1).T, cmap='gray')
            plt.scatter(*centers.T[np.array([0,2])], c='r')
            plt.title('Max.proj. y')
            plt.xlabel('x')
            plt.ylabel('z');
            plt.show()
    
        return Y, truth, trueSpikes, centers, dims, -shifts
    
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
    mc_layer = MotionCorrect(template[:,:,:,None,None], ms_h=ms_h, ms_w=ms_w, ms_d=ms_d)
    data = Y[None, ..., None]
    print(f'data shape: {data.shape}')
    print(f'template shape: {template.shape}')
    import time
    st = time.time()
    times = []
    nccs = []
    shifts_gpu_mc = np.zeros((Y.shape[0], 3))
    for i in range(data.shape[1]):
        _, ncc = mc_layer(data[:,i:i+1])
        coords = np.array([ncc[0],ncc[1],ncc[2]]).flatten()
        shifts_gpu_mc[i] = coords
        #shifts_all.append(mc_layer.shifts)
        times.append(time.time()-st)
        
    #%%
    batch = 16
    shifts_gpu_mc = np.zeros((Y.shape[0], 3))
    for i in range(data.shape[1]//batch):
        _, ncc = mc_layer(data[:,batch*i:batch*(i+1)])
        coords = np.array([ncc[0],ncc[1],ncc[2]]).T
        shifts_gpu_mc[batch*i:batch*(i+1)] = coords
        #shifts_all.append(mc_layer.shifts)
        times.append(time.time()-st)
    
    
    
    #%%
    plt.figure(figsize=(12,3))
    for i, s in enumerate((-shifts_gpu_mc +shifts_gpu_mc.mean(0), shifts - shifts.mean(0))):
        plt.subplot(1,2,i+1)
        for k in (0,1,2):
            plt.plot(np.array(s)[:,k], label=('x','y','z')[k])
        plt.legend()
        plt.title(('inferred shifts', 'true shifts')[i])
        plt.xlabel('frames')
        plt.ylabel('pixels')
        plt.ylim(-4, 4)
        plt.xlim(0, 300)
    
    #print(f'shifts: {shifts[1]}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #%%
    for i in range(3):
        print(f'{["x","y","z"][i]}: {np.corrcoef(-shifts_gpu_mc[:, i], shifts[:, i])}')
    
    #%%
    tensor = self.ncc.numpy()
    
    flat_tensor = np.reshape(tensor, (1, tensor.shape[-4]*tensor.shape[-3]*tensor.shape[-2], 1))
    
    argmax= np.argmax(flat_tensor, axis=1)
    # convert indexes into 3D coordinates
    shp = tensor.shape
    
    yz_flat = shp[2]*shp[3]
    argmax_z = argmax // xy_flat 
    argmax_y = (argmax % xy_flat) // shp[1]
    argmax_x = argmax - argmax_z*xy_flat - shp[1]*argmax_y
    
    
    #%%
    def gen_data(D=3, noise=.5, T=300, framerate=30, firerate=2.):
        N = 4                                                              # number of neurons
        dims = [(20, 30), (12, 14, 16)][D - 2]                             # size of image
        sig = (2, 2, 2)[:D]                                                # neurons size
        bkgrd = 10                                                         # fluorescence baseline
        gamma = .9                                                         # calcium decay time constant
        np.random.seed(5)
        centers = np.asarray([[np.random.randint(4, x - 4) for x in dims] for i in range(N)])
        trueA = np.zeros(dims + (N,), dtype=np.float32)
        trueS = np.random.rand(N, T) < firerate / float(framerate)
        trueS[:, 0] = 0
        trueC = trueS.astype(np.float32)
        for i in range(1, T):
            trueC[:, i] += gamma * trueC[:, i - 1]
        for i in range(N):
            trueA[tuple(centers[i]) + (i,)] = 1.
        tmp = np.zeros(dims)
        tmp[tuple(d // 2 for d in dims)] = 1.
        z = np.linalg.norm(gaussian_filter(tmp, sig).ravel())
        trueA = 10 * gaussian_filter(trueA, sig + (0,)) / z
        Yr = bkgrd + noise * np.random.randn(*(np.prod(dims), T)) + \
            trueA.reshape((-1, 4), order='F').dot(trueC)
        return Yr, trueC, trueS, trueA, centers, dims
    
    #%%
    import numpy as np
    mc_layer.template_zm.shape
    mc_layer.target_freq.shape
    mc_layer.imgs_zm.shape
    #self.target_freq = tf.signal.fft3d(tf.cast(self.template_zm[:,:,:,:,0], tf.complex128))
    
    fft_np = np.fft.fftn(mc_layer.template_zm[:,:,:,:,0].numpy().copy())
    print(fft_np.mean())
    
    imgs_zm = mc_layer.imgs_zm.numpy().copy()
    fft_np1 = np.fft.fftn(mc_layer.imgs_zm[:,:,:,:,0].numpy().copy())[0,:,:,:,None]
    print(fft_np1.mean())
    
    product_np = np.conj(fft_np) * fft_np1
    corr_np = np.abs(np.fft.ifftn(product_np))
    print(np.where(corr_np ==corr_np.max()))
    
    #%%
    mc_layer.template_zm.shape
    mc_layer.target_freq.shape
    mc_layer.imgs_zm.shape
    #self.target_freq = tf.signal.fft3d(tf.cast(self.template_zm[:,:,:,:,0], tf.complex128))
    
    fft_np = np.fft.fftn(mc_layer.template_zm[:,:,:,0,0].numpy().copy())
    print(fft_np.mean())
    print(fft_np.shape)
    
    imgs_zm = mc_layer.imgs_zm.numpy().copy()
    fft_np1 = np.fft.fftn(mc_layer.imgs_zm[0,:,:,:,0].numpy().copy())
    print(fft_np1.mean())
    print(fft_np1.shape)
    
    product_np = np.conj(fft_np) * fft_np1
    corr_np = np.abs(np.fft.ifftn(product_np))
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
    fft_tf = tf.signal.fft3d(tf.cast(mc_layer.template_zm[:,:,:,0], tf.complex128))
    tf.shape(fft_tf)
    print(fft_tf.numpy().mean())
    
    fr_freq = tf.signal.fft3d(tf.cast(imgs_zm[0,:,:,:], tf.complex128))
    tf.shape(fr_freq)
    print(fr_freq.numpy().mean())
    
    img_product = fr_freq *  tf.math.conj(fft_tf)
    
    corr_tf = tf.cast(tf.math.abs(tf.signal.ifft3d(img_product)), tf.float32)[...,0].numpy()
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