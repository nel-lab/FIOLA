#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:18:35 2020

@author: nellab
"""
#%%
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
#%%
class NNLS(keras.layers.Layer):
    def __init__(self, theta_1, name="NNLS",**kwargs):
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
        self.th1 = theta_1.astype(np.float32)

    def call(self, X):
        """
        pass as inputs the new Y, and the old X. see  https://angms.science/doc/NMF/nnls_pgd.pdf
        """
        (Y,X_old,k,weight,shifts) = X
        mm = tf.matmul(self.th1, Y)
        new_X = tf.nn.relu(mm + weight)

        Y_new = new_X + tf.cast(tf.divide(k - 1, k + 2), tf.float32)*(new_X - X_old)  
        k += 1
        return (Y_new, new_X, k, weight, shifts)
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "theta_1": self.th1} 
#%%
class compute_theta2(keras.layers.Layer):
    def __init__(self, A, n_AtA, **kwargs): 
        super().__init__(**kwargs)
        self.A = A
        self.n_AtA = n_AtA
        
    def call(self, X, shifts):
        Y = tf.matmul(X, self.A)
        Y = tf.divide(Y, self.n_AtA)
        Y = tf.transpose(Y)
        shifts = tf.squeeze(shifts)
        return (Y, shifts)   
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "A":self.A, "n_AtA":self.n_AtA}