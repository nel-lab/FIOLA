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
from skimage import io
import numpy as np
import pylab as plt
import cv2
from time import time
#%% 
class NNLS(keras.layers.Layer):
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
        self.th1 = tf.convert_to_tensor(theta_1, dtype=np.float32)
        
        

    def build(self, batch_input_shape):        
        super().build(batch_input_shape) # must be at the end
    
    @tf.function
    def call(self, X):
        """
        pass as inputs the new Y, and the old X. see  https://angms.science/doc/NMF/nnls_pgd.pdf
        """
        (Y,X_old,k, theta_2) = X        
        new_X = tf.nn.relu(tf.matmul(self.th1, Y) + theta_2)
        Y_new = new_X + (k - 1)/(k + 2)*(new_X-X_old)  
        k += 1
        return (Y_new, new_X, k, theta_2)
#%%
class compute_theta2(keras.layers.Layer):
    def __init__(self, A, n_AtA, **kwargs):
       
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
#%% EXAMMPLE NEURONS    
with np.load('/home/nellab/SOFTWARE/SANDBOX/src/regression_n.01.01_less_neurons.npz', allow_pickle=True) as ld:
    print(list(ld.keys()))
    Y_tot = ld['Y']
import h5py
import scipy
with h5py.File('/home/nellab/caiman_data/example_movies/memmap__d1_512_d2_512_d3_1_order_C_frames_1825_.hdf5','r') as f:
    for k,i in f['estimates'].items():
        print((k, i))
    
    data = np.array(f['estimates']['A']['data'])
    indices = np.array(f['estimates']['A']['indices'])
    indptr = np.array(f['estimates']['A']['indptr'])
    shape = np.array(f['estimates']['A']['shape'])
    dims = np.array(f['estimates']['dims'])
    idx_components = f['estimates']['idx_components']
    A_sp = scipy.sparse.csc_matrix((data[:], indices[:], indptr[:]), shape[:])
    YrA = np.array(f['estimates']['YrA'])
    S = np.array(f['estimates']['S'])
    C = np.array(f['estimates']['C']) 
    b = np.array(f['estimates']['b']) 
    f = np.array(f['estimates']['f'])
    A_sp = A_sp[:,idx_components ]
    C = C[idx_components]
    YrA = YrA[idx_components]
    S = S[idx_components]
   


#%%
n_frames = 1800
Ab = np.concatenate([A_sp.toarray()[:],b], axis=1)
Cf = np.concatenate([C[:,:n_frames]+YrA[:,:n_frames],f[:,:n_frames]], axis=0)
Y = Y_tot[:,:n_frames]
S = S[:,:n_frames]
#%%
cms = [np.array(scipy.ndimage.center_of_mass(aa.reshape(dims, order='F').toarray()), dtype=np.int) 
                    for aa in A_sp[:,:10].T]

cms = np.array(cms)
cms[0,:] = np.minimum(cms[0,:]-20,dims[0]) 
cms[1,:] = np.minimum(cms[1,:],dims[1])
cms[0,:] = np.maximum(cms[0,:] -20,0)
cms[1,:] = np.maximum(cms[1,:],1)

#%%
idx_act = np.argsort(S, axis=-1)[:,:20]
for act,cm in zip(idx_act[:10], cms):
    img = Y[:,act].mean(axis=-1).reshape(dims, order='F')    
    plt.cla()
    img = img[cm[0]:cm[0]+40, cm[1]:cm[1]+40]
    plt.imshow(img)
    plt.pause(1)
#%%
print(np.linalg.norm(Y-Ab@Cf)/np.linalg.norm(Y))
#%%
A = Ab
b = Y[:,0]
AtA = A.T@A
Atb = A.T@b
n_AtA = np.linalg.norm(AtA, ord='fro')
# theta_1 = tf.convert_to_tensor(np.ones_like(AtA) - AtA/n_AtA, dtype=tf.float32)
# theta_2 = tf.convert_to_tensor(Atb/n_AtA, dtype=tf.float32)
theta_1 = (np.eye(A.shape[-1]) - AtA/n_AtA)
theta_2 = (Atb/n_AtA)[:,None]  

#%% BATCH
x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
y_old = tf.identity(x_old)
fr =  tf.convert_to_tensor(Y[:,0][None,:], dtype=tf.float32)
y_in = tf.keras.layers.Input(shape=y_old.shape)
x_in = tf.keras.layers.Input(shape=x_old.shape)
k_in = tf.keras.layers.Input(shape=(1,))
b_in = tf.keras.layers.Input(shape=fr.shape)

nnls = NNLS(theta_1)
c_th2 = compute_theta2(A, n_AtA)

th2 = c_th2(b_in) 
x_kk = nnls([y_in, x_in, k_in, th2])
for k in range(1,20):    
    x_kk = nnls(x_kk)    
   
mod = keras.Model(inputs=[b_in, y_in, x_in, k_in], outputs=x_kk)

#%%
newy = []  
batch_size = 100
t_0 = time()
x_old = tf.convert_to_tensor(Cf[:,:batch_size].copy(), dtype=np.float32)
y_old = tf.identity(x_old)
for i in range(0,Y.shape[-1], batch_size):
    b = tf.convert_to_tensor(Y[:,i:i+batch_size].T, dtype=tf.float32)      
    (y_new, x_new, kkk, tht2) = mod((b, y_old, x_old, tf.convert_to_tensor(np.zeros_like(batch_size), dtype=tf.int8)))
    y_old, x_old = y_new, x_new
    newy.append(x_old.numpy())
print(time()-t_0) 
Cf_nn = np.concatenate(newy,axis=1).squeeze()
 
print(np.linalg.norm(Y-Ab@Cf_nn)/np.linalg.norm(Y))
print(np.linalg.norm(Y-Ab@Cf)/np.linalg.norm(Y))
#%%
for i in range(20):
    plt.plot(Cf[i,:])
    plt.plot(Cf_nn[i,1:])
    plt.pause(1)
    plt.cla()  

#%% ONLINE 
x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
y_old = tf.identity(x_old)
fr =  tf.convert_to_tensor(Y[:,0][None,:], dtype=tf.float32)
y_in = tf.keras.layers.Input(shape=y_old.shape)
x_in = tf.keras.layers.Input(shape=x_old.shape)
k_in = tf.keras.layers.Input(shape=(1,))
b_in = tf.keras.layers.Input(shape=fr.shape)


nnls = NNLS(theta_1)
c_th2 = compute_theta2(A, n_AtA)

th2 = c_th2(b_in) 
x_kk = nnls([y_in, x_in, k_in, th2])
for k in range(1,10):    
    x_kk = nnls(x_kk)    
   
mod = keras.Model(inputs=[b_in, y_in, x_in, k_in], outputs=x_kk)
#%%
newy = []  
t_0 = time()
for i in range(0,Y.shape[-1]):
    b = tf.convert_to_tensor(Y[:,i][None,:], dtype=tf.float32)      
    (y_new, x_new, kkk, tht2) = mod((b, y_old, x_old, tf.convert_to_tensor(0, dtype=tf.int8)))
    y_old, x_old = y_new, x_new
    newy.append(x_old.numpy())
print(time()-t_0) 
Cf_nn = np.array(newy[:]).squeeze().T
 
print(np.linalg.norm(Y-Ab@Cf_nn)/np.linalg.norm(Y))
print(np.linalg.norm(Y-Ab@Cf_bc)/np.linalg.norm(Y))
