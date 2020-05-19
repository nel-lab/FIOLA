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
    idx_components = f['estimates']['idx_components']
    A_sp = scipy.sparse.csc_matrix((data[:], indices[:], indptr[:]), shape[:])
    YrA = np.array(f['estimates']['YrA'])
    C = np.array(f['estimates']['C']) 
    b = np.array(f['estimates']['b']) 
    f = np.array(f['estimates']['f'])
    A_sp = A_sp[:,idx_components ]
    C = C[idx_components]
    YrA = YrA[idx_components]


#%%
n_frames = 1
Ab = np.concatenate([A_sp.toarray()[:],b], axis=1)
Cf = np.concatenate([C[:,:n_frames]+YrA[:,:n_frames],f[:,:n_frames]], axis=0)
Y = np.expand_dims(Y_tot[:,n_frames-1], axis=1)
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
#%%
t_0 = time() 
groups = update_order_greedy(A_sp,flag_AA=False)[0]
Cf_bc = [Cf[:,0].copy()]
# count = 0
#for frame in Y.T[0]:
Cf_bc = HALS4activity(Y.T[0], A, noisyC = Cf_bc[-1], AtA=AtA, iters=5, groups=None)[0]
print(len(Cf_bc))
print(time()-t_0) 
Cf_bc = np.expand_dims(np.array(Cf_bc).T, axis=1)
print(np.linalg.norm(Y-Ab@Cf_bc)/np.linalg.norm(Y))
#%%
for i in range(50):
    plt.plot(Cf_bc[i]); plt.plot(Cf[i])
    plt.pause(1)
    plt.cla()
    break
#%%
#newy_loop = [Cf[:,0][:,None]]
#x_old = Cf[:,0].copy()[:,None] 
#y_k = x_old.copy()
#for k in range(1,n_frames):
#    
#    if False:
#        x_new = np.maximum(theta_1@x_old + theta_2,0)
#    else:
#        x_new = np.maximum(theta_1@y_k + theta_2,0)
#        y_k = x_new + (k-1)/(k+2)*(x_new-x_old)
#    
#    x_old = x_new 
#    newy_loop.append(x_old)
#
#print(np.linalg.norm(np.dot(A,np.squeeze(x_old))-b)/np.linalg.norm(b))
#print('**' + str(np.linalg.norm(np.dot(A,np.squeeze(Cf[:,1]))-b)/np.linalg.norm(b)))
#%%

#%%
x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
y_old = tf.identity(x_old)
fr =  tf.convert_to_tensor(Y[:,0][None,:], dtype=tf.float32)
#mod = NNLS(theta_1, theta_2)

y_in = tf.keras.layers.Input(shape=y_old.shape)
x_in = tf.keras.layers.Input(shape=x_old.shape)
k_in = tf.keras.layers.Input(shape=(1,))
b_in = tf.keras.layers.Input(shape=fr.shape)

nnls = NNLS(theta_1, theta_2)
x_kk = nnls([y_in, x_in, k_in])
for k in range(1,10):    
    x_kk = nnls(x_kk)    

c_th2 = compute_theta2(A, n_AtA)
th2 = c_th2(b_in)    
mod = keras.Model(inputs=[b_in, y_in, x_in, k_in], outputs=[x_kk, th2])
newy = []
b = tf.convert_to_tensor(Y[:,0][None,:], dtype=tf.float32) 
(y_new, x_new, kkk), tht2 = mod((b, y_old, x_old, tf.convert_to_tensor(0, dtype=tf.int8)))   

t_0 = time()
for i in range(0,Y.shape[-1]):
    b = tf.convert_to_tensor(Y[:,i][None,:], dtype=tf.float32)      
    mod.layers[3].set_weights([tht2])
    (y_new, x_new, kkk), tht2 = mod((b, y_old, x_old, tf.convert_to_tensor(0, dtype=tf.int8)))
    y_old, x_old = y_new, x_new
    newy.append(x_old.numpy())
print(time()-t_0) 
Cf_nn = np.expand_dims(np.array(newy[:]).squeeze().T, axis=1)
 
print(np.linalg.norm(Y-Ab@Cf_nn)/np.linalg.norm(Y))
print(np.linalg.norm(Y-Ab@Cf_bc)/np.linalg.norm(Y))
#%% BATCH
x_old = tf.convert_to_tensor(Cf[:,:10].copy()[:,None], dtype=np.float32)
y_old = tf.identity(x_old)
fr =  tf.convert_to_tensor(Y[:,:10][None,:], dtype=tf.float32)
#mod = NNLS(theta_1, theta_2)

y_in = tf.keras.layers.Input(shape=y_old.shape)
x_in = tf.keras.layers.Input(shape=x_old.shape)
k_in = tf.keras.layers.Input(shape=(1,))
b_in = tf.keras.layers.Input(shape=fr.shape)

nnls = NNLS(theta_1, theta_2)
x_kk = nnls([y_in, x_in, k_in])
for k in range(1,10):    
    x_kk = nnls(x_kk)    

c_th2 = compute_theta2(A, n_AtA)
th2 = c_th2(b_in)    
mod = keras.Model(inputs=[b_in, y_in, x_in, k_in], outputs=[x_kk, th2])

newy = []
b = tf.convert_to_tensor(Y[:,0][None,:], dtype=tf.float32) 
(y_new, x_new, kkk), tht2 = mod((b, y_old, x_old, tf.convert_to_tensor(0, dtype=tf.int8)))   

t_0 = time()
for i in range(0,Y.shape[-1]):
    b = tf.convert_to_tensor(Y[:,i][None,:], dtype=tf.float32)      
    mod.layers[3].set_weights([tht2])
    (y_new, x_new, kkk), tht2 = mod((b, y_old, x_old, tf.convert_to_tensor(0, dtype=tf.int8)))
    y_old, x_old = y_new, x_new
    newy.append(x_old.numpy())
print(time()-t_0) 
Cf_nn = np.array(newy[:]).squeeze().T
 
print(np.linalg.norm(Y-Ab@Cf_nn)/np.linalg.norm(Y))
print(np.linalg.norm(Y-Ab@Cf_bc)/np.linalg.norm(Y))
#%%
for i in range(20):
    plt.plot(Cf_nn[i,1:]); plt.plot(Cf[i,:])
    plt.pause(1)
    plt.cla() 
#%%
newy = [Cf[:,0][:,None] ]
x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
y_old = tf.identity(x_old)
y_in = tf.keras.layers.Input(shape=y_old.shape)
x_in = tf.keras.layers.Input(shape=x_old.shape)

nnls = NNLS(theta_1, theta_2, k=tf.convert_to_tensor(0, dtype=tf.float32))
x_kk = nnls([y_in, x_in])
for k in range(1,20):    
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
Cf_nn = np.array(newy).squeeze().T
print(np.linalg.norm(Y-Ab@Cf_nn)/np.linalg.norm(Y))
#%%

for i in range(50):
    plt.plot(Cf_nn[i]); plt.plot(Cf[i])
    plt.pause(1)
    plt.cla()
#%%
print(np.linalg.norm(Y-Ab@Cf)/np.linalg.norm(Y))
# print(np.linalg.norm(Y-Ab@C_in)/np.linalg.norm(Y))
print(np.linalg.norm(Y-Ab@Cf_nn)/np.linalg.norm(Y))
#%%

for i in range(50):
    plt.plot(Cf_nn[i]);plt.plot(C_in[i]); plt.plot(Cf[i])
    plt.pause(1)
    plt.cla()