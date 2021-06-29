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
from time import time
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
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        

    def build(self, batch_input_shape):
        # theta_1 param in https://angms.science/doc/NMF/nnls_pgd.pdf
        self.th1 = self.add_weight(
            name="kernel", shape=[*self.theta_1.shape],
            initializer=tf.constant_initializer(self.theta_1))
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
        return (Y_new, new_X)
#%% 
class NNLS(keras.layers.Layer):
    def __init__(self, theta_1, theta_2, k, **kwargs):
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
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.k = k
        

    def build(self, batch_input_shape):
        # theta_1 param in https://angms.science/doc/NMF/nnls_pgd.pdf
        self.th1 = self.add_weight(
            name="kernel", shape=[*self.theta_1.shape],
            initializer=tf.constant_initializer(self.theta_1))
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
        (Y,X_old) = X    
        k = self.k
        new_X = tf.nn.relu(tf.matmul(self.th1, Y) + self.th2)
        Y_new = new_X + (k - 1)/(k + 2)*(new_X-X_old)       
        return (Y_new, new_X)


    # def get_config(self):
    #     base_config = super().get_config()
    #     return {**base_config, "strides": self.strides,
    #             "padding": self.padding, "epsilon": self.epsilon, 
    #                                     "ms_h": self.ms_h,"ms_w": self.ms_w }
 #%%
from scipy.optimize import nnls
a = np.arange(0,3, .00001).reshape(-1,60)
x = np.arange(0,.6,0.01)
y = np.dot(a,x)
x, res = nnls(a,y)
print(np.linalg.norm(np.dot(a,x)-y)/np.linalg.norm(y))
assert(res < 1e-6)
assert(np.linalg.norm(np.dot(a,x)-y) < 1e-6)
#%%
A = a
b = y
AtA = A.T@A
Atb = A.T@b
n_AtA = np.linalg.norm(AtA, ord='fro')

# theta_1 = tf.convert_to_tensor(np.ones_like(AtA) - AtA/n_AtA, dtype=tf.float32)
# theta_2 = tf.convert_to_tensor(Atb/n_AtA, dtype=tf.float32)

theta_1 = (np.eye(A.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:,None]      
#%%
# x_old = tf.nn.relu(tf.random.uniform((len(x),1))).numpy()
x_old = x.copy()[:,None] + np.random.random(x.shape)[:,None]*10
y_k = x_old.copy()
for k in range(1,30):
    print(np.linalg.norm(np.dot(A,np.squeeze(x_old))-b))
    if False:
        x_new = np.maximum(theta_1@x_old + theta_2,0)
    else:
        x_new = np.maximum(theta_1@y_k + theta_2,0)
        y_k = x_new + (k-1)/(k+2)*(x_new-x_old)
    
    x_old = x_new
        
#%%
# x_old = tf.nn.relu(tf.random.uniform((6,1)))
x_old = tf.convert_to_tensor(x.copy()[:,None] + np.random.random(x.shape)[:,None]*10+10, dtype=np.float32)
y_old = x_old
mod = NNLS(theta_1, theta_2)
mod((y_old, x_old,tf.convert_to_tensor(i, dtype=tf.float32)))
t_0  = time()
for i in range(1,30):    
    # print(np.linalg.norm(np.dot(A,np.squeeze(x_old))-b))
    y_new, x_new = mod((y_old, x_old,tf.convert_to_tensor(i, dtype=tf.float32)))
    y_old, x_old = y_new, x_new

print(time()-t_0)
    
#%% EXAMMPLE NEURONS    
with np.load('regression_demo_movie.npz') as ld:
    Y = ld['Y']
    Ab = ld['Ab']
    Cf = ld['Cf']

Cf_est = np.array([nnls(Ab,y)[0] for y in Y.T[:]]).T
#%%
print(np.linalg.norm(Y-Ab@Cf)/np.linalg.norm(Y))
#%%
A = Ab
b = Y[:,1]
AtA = A.T@A
Atb = A.T@b
n_AtA = np.linalg.norm(AtA, ord='fro')

# theta_1 = tf.convert_to_tensor(np.ones_like(AtA) - AtA/n_AtA, dtype=tf.float32)
# theta_2 = tf.convert_to_tensor(Atb/n_AtA, dtype=tf.float32)

theta_1 = (np.eye(A.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:,None]  
#%%
newy_loop = [Cf[:,0][:,None]]
x_old = Cf[:,0].copy()[:,None] 
y_k = x_old.copy()
for k in range(1,15):
    
    if False:
        x_new = np.maximum(theta_1@x_old + theta_2,0)
    else:
        x_new = np.maximum(theta_1@y_k + theta_2,0)
        y_k = x_new + (k-1)/(k+2)*(x_new-x_old)
    
    x_old = x_new 
    newy_loop.append(x_old)

print(np.linalg.norm(np.dot(A,np.squeeze(x_old))-b)/np.linalg.norm(b))
print('**' + str(np.linalg.norm(np.dot(A,np.squeeze(Cf[:,1]))-b)/np.linalg.norm(b)))
#%%
newy = [Cf[:,0][:,None] ]
x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
y_old = tf.identity(x_old)
mod = NNLS(theta_1, theta_2)
t_0 = time()
for i in range(1,Y.shape[-1]):
    b = Y[:,i]  
    # print('**' + str(np.linalg.norm(np.dot(A,np.squeeze(Cf[:,i]))-b)/np.linalg.norm(b) 
    #                   - np.linalg.norm(np.dot(A,np.squeeze(x_old))-b)/np.linalg.norm(b)))
    for k in range(1,5):    
        y_new, x_new = mod((y_old, x_old,tf.convert_to_tensor(k, dtype=tf.float32)))
        y_old, x_old = y_new, x_new
    newy.append(x_old.numpy())
    Atb = A.T@b
    theta_2 = (Atb/n_AtA)[:,None]   
    mod.set_weights([theta_1, theta_2])
    # print('****' + str(np.linalg.norm(np.dot(A,np.squeeze(Cf[:,i]))-b)/np.linalg.norm(b) 
    #                   - np.linalg.norm(np.dot(A,np.squeeze(x_old))-b)/np.linalg.norm(b)))
print(time()-t_0)    
#%%
newy = [Cf[:,0][:,None] ]
x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
y_old = tf.identity(x_old)
y_in = tf.keras.layers.Input(shape=y_old.shape)
x_in = tf.keras.layers.Input(shape=x_old.shape)

nnls = NNLS(theta_1, theta_2, k=tf.convert_to_tensor(0, dtype=tf.float32))
x_kk = nnls([y_in, x_in])
for k in range(1,5):    
    x_kk = NNLS(theta_1, theta_2, k=tf.convert_to_tensor(k, dtype=tf.float32))(x_kk)

mod = keras.Model(inputs=[y_in, x_in], outputs=x_kk)
y_new, x_new = mod((y_old, x_old))    
        
t_0 = time()    
for i in range(1,Y.shape[-1]):
    b = Y[:,i]  
    # print('**' + str(np.linalg.norm(np.dot(A,np.squeeze(Cf[:,i]))-b)/np.linalg.norm(b) 
    #                   - np.linalg.norm(np.dot(A,np.squeeze(x_old))-b)/np.linalg.norm(b)))
    #mod = NNLS(theta_1, theta_2)
    y_new, x_new = mod((y_old, x_old))
    y_old, x_old = y_new, x_new
    newy.append(x_old.numpy())
    Atb = A.T@b
    theta_2 = (Atb/n_AtA)[:,None]   
    for lyr in mod.layers[2:]: 
        lyr.th2.assign(theta_2)
        
    # print('****' + str(np.linalg.norm(np.dot(A,np.squeeze(Cf[:,i]))-b)/np.linalg.norm(b) 
    #                   - np.linalg.norm(np.dot(A,np.squeeze(x_old))-b)/np.linalg.norm(b)))
print(time()-t_0)    
    
#%%
Cf_nn = np.array(newy).squeeze().T
print(np.linalg.norm(Y-Ab@Cf)/np.linalg.norm(Y))
print(np.linalg.norm(Y-Ab@Cf_est)/np.linalg.norm(Y))
print(np.linalg.norm(Y-Ab@Cf_nn)/np.linalg.norm(Y))
#%%
for i in range(50):
    pl.plot(Cf_nn[i]);pl.plot(Cf[i])
    pl.pause(1)
    pl.cla()