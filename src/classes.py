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
from threading import Thread
from past.utils import old_div
from skimage import io
import numpy as np
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
        self.q = Queue()
#        self.A_ = Ab
#        self.n_AtA_ = n_AtA
        ## normalize template
        self.template_zm, self.template_var = self.normalize_template(self.template, epsilon=self.epsilon)

    def build(self, batch_input_shape):
        # weights here represent the template, so that we can update in case we 
        # want to use the online algorithm
        self.kernel = self.add_weight(
            name="kernel", shape=[*self.template_zm.shape.as_list()],
#            initializer=tf.constant_initializer(np.ndarray(self.template_zm)))
            initializer=tf.constant_initializer(self.template_zm.numpy()))

        # the normalizer also needs to be updated if the template is updated
        self.normalizer = self.add_weight(
            name="normalizer", shape=[*self.template_var.shape.as_list()],
            #initializer=tf.constant_initializer(np.ndarray(self.template_var) 
            initializer=tf.constant_initializer(self.template_var.numpy())) 
#        self.A = self.add_weight(
#            name="A", shape=[*self.A_.shape],
#            initializer=tf.constant_initializer(self.A_))
#        # theta_2 param in https://angms.science/doc/NMF/nnls_pgd.pdf
#        self.n_AtA = self.add_weight(
#            name="n_AtA", shape=[*self.n_AtA_.shape],
#            initializer=tf.constant_initializer(self.n_AtA_)) 
        super().build(batch_input_shape) # must be at the end

    @tf.function
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
        try:
            X_corrected = tfa.image.translate(X, tf.squeeze(tf.stack([ys, xs], axis=1)), 
                                            interpolation="BILINEAR")
        except:
            n_shape = tensor_ncc.shape
            xs, ys = self.extract_fractional_peak(tf.reshape(tensor_ncc, [1, n_shape[1], n_shape[2], n_shape[3]]), ms_h=self.ms_h, ms_w=self.ms_w)
            X_corrected = tfa.image.translate(tf.reshape(X, [1, 512, 512, 1]), tf.squeeze(tf.stack([ys, xs], axis=1)), 
                                            interpolation="BILINEAR")
        # print(timeit.default_timer()-start)
        return tf.reshape(tf.squeeze(X_corrected), [1, X_corrected.shape[-2]**2])
    
#        Y = tf.matmul(X_corr_translated, self.A) 
#        Y = tf.divide(Y,self.n_AtA)
#        Y = tf.transpose(Y)
#        return Y


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
        del localsum_sq, localsum
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

        try:
            n_batches = np.arange(tensor_ncc_log.shape[0])
        except:
            n_batches = np.arange(tensor_ncc_log.shape[1])

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
        
        return sh_x_n, sh_y_n
    
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
def update_order_greedy(A, flag_AA=True):
    """Determines the update order of the temporal components

    this, given the spatial components using a greedy method
    Basically we can update the components that are not overlapping, in parallel

    Args:
        A:  sparse crc matrix
            matrix of spatial components (d x K)
        OR:
            A.T.dot(A) matrix (d x d) if flag_AA = true

        flag_AA: boolean (default true)

     Returns:
         parllcomp:   list of sets
             list of subsets of components. The components of each subset can be updated in parallel

         len_parrllcomp:  list
             length of each subset

    Author:
        Eftychios A. Pnevmatikakis, Simons Foundation, 2017
    """
    K = np.shape(A)[-1]
    parllcomp = []
    for i in range(K):
        new_list = True
        for ls in parllcomp:
            if flag_AA:
                if A[i, ls].nnz == 0:
                    ls.append(i)
                    new_list = False
                    break
            else:
                if (A[:, i].T.dot(A[:, ls])).nnz == 0:
                    ls.append(i)
                    new_list = False
                    break

        if new_list:
            parllcomp.append([i])
    len_parrllcomp = [len(ls) for ls in parllcomp]
    return parllcomp, len_parrllcomp
#%%
from math import sqrt
def HALS4activity(Yr, A, noisyC, AtA=None, iters=5, tol=1e-3, groups=None,
                  order=None):
    """Solves C = argmin_C ||Yr-AC|| using block-coordinate decent. Can use
    groups to update non-overlapping components in parallel or a specified
    order.

    Args:
        Yr : np.array (possibly memory mapped, (x,y,[,z]) x t)
            Imaging data reshaped in matrix format

        A : scipy.sparse.csc_matrix (or np.array) (x,y,[,z]) x # of components)
            Spatial components and background

        noisyC : np.array  (# of components x t)
            Temporal traces (including residuals plus background)

        AtA : np.array, optional (# of components x # of components)
            A.T.dot(A) Overlap matrix of shapes A.

        iters : int, optional
            Maximum number of iterations.

        tol : float, optional
            Change tolerance level

        groups : list of sets
            grouped components to be updated simultaneously

        order : list
            Update components in that order (used if nonempty and groups=None)

    Returns:
        C : np.array (# of components x t)
            solution of HALS

        noisyC : np.array (# of components x t)
            solution of HALS + residuals, i.e, (C + YrA)
    """

    AtY = A.T.dot(Yr)
    num_iters = 0
    C_old = np.zeros_like(noisyC)
    C = noisyC.copy()
    if AtA is None:
        AtA = A.T.dot(A)
    AtAd = AtA.diagonal() + np.finfo(np.float32).eps

    # faster than np.linalg.norm
    def norm(c): return sqrt(c.ravel().dot(c.ravel()))
    while (norm(C_old - C) >= tol * norm(C_old)) and (num_iters < iters):
        C_old[:] = C
        if groups is None:
            if order is None:
                order = list(range(AtY.shape[0]))
            for m in order:
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C)) / AtAd[m]
                C[m] = np.maximum(noisyC[m], 0)
        else:
            for m in groups:
                noisyC[m] = C[m] + ((AtY[m] - AtA[m].dot(C)).T/AtAd[m]).T
                C[m] = np.maximum(noisyC[m], 0)
        num_iters += 1
    return C, noisyC
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
        self.th1 = tf.convert_to_tensor(theta_1, dtype=np.float32)
        self.theta_2 = theta_2
        

    def build(self, batch_input_shape):
        # theta_1 param in https://angms.science/doc/NMF/nnls_pgd.pdf
#        self.th1 = self.add_weight(
#            name="kernel", shape=[*self.theta_1.shape],
#            initializer=tf.constant_initializer(self.theta_1))
        # theta_2 param in https://angms.science/doc/NMF/nnls_pgd.pdf
        self.th2 = self.add_weight(
            name="normalizer", shape=[*self.theta_2.shape],
            initializer=tf.constant_initializer(self.theta_2))   
        super().build(batch_input_shape) # must be at the end
    
    @tf.function
    def call(self, X):
        """
        pass as inputs the new Y, and the old X. see  https://angms.science/doc/NMF/nnls_pgd.pdf
        """
        (Y,X_old,k) = X        
        new_X = tf.nn.relu(tf.matmul(self.th1, Y) + self.th2)
        Y_new = new_X + (k - 1)/(k + 2)*(new_X-X_old)  
        k += 1
        return (Y_new, new_X, k)
#%%
class compute_theta2(keras.layers.Layer):
    def __init__(self, A, n_AtA, **kwargs):
        # tf.keras.backend.clear_session()
       
        super().__init__(**kwargs)
        self.A_ = A
        self.n_AtA_ = n_AtA
        

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
        Y = tf.matmul(X, self.A) 
        Y = tf.divide(Y,self.n_AtA)
        Y = tf.transpose(Y)
        return Y