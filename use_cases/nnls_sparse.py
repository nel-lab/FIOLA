#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:19:02 2021

@author: nel
"""
#%%
from __future__ import print_function
import tensorflow as tf

x = tf.constant([[1.0,2.0],
                 [3.0,4.0]])
y = tf.SparseTensor(indices=[[0,0],[1,1]], values=[1.0,1.0], dense_shape=[2,2])
z = tf.sparse.sparse_dense_matmul(y, x)


#%%
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# tf.config.run_functions_eagerly(True)
import tensorflow.keras as keras
import numpy as np
from fiola.utilities import HALS4activity

mov = np.load('/media/nel/storage/fiola/test/mov.npy')
Ab = np.load('/media/nel/storage/fiola/test/Ab.npy')

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
        NNLS for each iteration. see https://angms.science/doc/NMF/nnls_pgd.pdf
        
        Parameters
        ----------
        X : tuple
            output of previous iteration

        Returns
        -------
        Y_new : ndarray
            auxilary variables
        new_X : ndarray
            new extracted traces
        k : int
            number of iterations
        weight : ndarray
            equals to theta_2

        """        
        (Y,X_old,k,weight) = X
        mm = tf.matmul(self.th1, Y)
        new_X = tf.nn.relu(mm + weight)

        Y_new = new_X + (k - 1)/(k + 2)*(new_X - X_old)  
        k += 1
        return (Y_new, new_X, k, weight)
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "theta_1": self.th1} 

class compute_theta2(keras.layers.Layer):
    def __init__(self, A, n_AtA, **kwargs): 
        super().__init__(**kwargs)
        self.A = A
        self.n_AtA = n_AtA
        
    def call(self, X):
        Y = tf.matmul(X, self.A)
        Y = tf.divide(Y, self.n_AtA)
        Y = tf.transpose(Y)
        return Y    
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "A":self.A, "n_AtA":self.n_AtA}
    
class compute_trace_with_noise(keras.layers.Layer):
    def __init__(self, AtA, n_AtA,  **kwargs): 
        super().__init__(**kwargs)
        self.n_AtA = n_AtA
        self.AtA = AtA
                
    def call(self, th2, trace):
        YA = tf.multiply(th2, self.n_AtA)
        YrA = YA - tf.matmul(self.AtA.T, trace[0])
        trace_with_noise = YrA + trace
        return trace_with_noise    
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "n_AtA":self.n_AtA, "AtA":self.AtA}
    
class compute_theta2_split(keras.layers.Layer):
    def __init__(self, A, n_AtA, split=1, **kwargs): 
        super().__init__(**kwargs)
        self.A = A
        self.n_AtA = n_AtA
        self.split = split
        
    def call(self, X):
        if self.split == 1:
            Y = tf.matmul(X, self.A)
        else:
            size = np.ceil(self.A.shape[1] / self.split).astype(np.int32)
            aa = [self.A[:, n * size : (n + 1) * size] for n in range(self.split)]
            Y = tf.concat([tf.matmul(X, a) for a in aa], axis=1)
        Y = tf.transpose(Y)
        Y = tf.divide(Y, self.n_AtA)
        return Y
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "A":self.A, "n_AtA":self.n_AtA, "split": self.split}
    
class compute_theta2_sparse(keras.layers.Layer):
    def __init__(self, A, b, n_AtA, **kwargs): 
        super().__init__(**kwargs)
        self.A = A
        self.b = b
        self.n_AtA = n_AtA
        
    def call(self, X):
        Y = tf.sparse.sparse_dense_matmul(self.A, X, adjoint_a=True, adjoint_b=True)
        
        if self.b is not None:
            Y1 = tf.matmul(self.b, X, adjoint_a=True, adjoint_b=True)
            Y = tf.concat([Y, Y1], axis=0)
        Y = tf.divide(Y, self.n_AtA)
        return Y    
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "A":self.A, "b":self.b, "n_AtA":self.n_AtA}
    
class Empty(keras.layers.Layer):
    def call(self,  fr):
        fr = fr[0, ..., 0]
        return tf.reshape(tf.transpose(fr, perm=[0, 2, 1]), (fr.shape[0], -1))
    
#%%
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = coo_matrix(X)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)
from scipy.sparse import coo_matrix
A_sparse = convert_sparse_matrix_to_sparse_tensor(Ab[:, :-1])
bg = Ab[:, -1:] 
#%%
times = []
out = []
flag = 1000
index = 0
dims = mov.shape[1:]
batch_size=1
num_layers=10

#%%
b = mov[0:batch_size].T.reshape((-1, batch_size), order='F')       
C_init = np.dot(Ab.T, b)
x0 = np.array([HALS4activity(Yr=b[:,i], A=Ab, C=C_init[:, i].copy(), iters=10) for i in range(batch_size)]).T
x, y = np.array(x0[None,:]), np.array(x0[None,:]) 
num_components = Ab.shape[-1]

shp_x, shp_y = dims[0], dims[1] 
Ab = Ab.astype(np.float32)
num_components = Ab.shape[-1]

y_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, batch_size]), name="y") # Input Layer for components
x_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, batch_size]), name="x") # Input layer for components
fr_in = tf.keras.layers.Input(shape=tf.TensorShape([batch_size, shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 
k_in = tf.keras.layers.Input(shape=(1,), name="k") #Input layer for the counter within the NNLS layers
 
# calculations to initialize NNLS
AtA = Ab.T@Ab
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)

# empty motion correction layer
mc_layer = Empty()
mc = mc_layer(fr_in)
#mc_layer(mov[:batch_size, :, :])

# chains motion correction layer to weight-calculation layer
#c_th2 = compute_theta2(Ab, n_AtA)
#c_th2 = compute_theta2_sparse(A_sparse, bg, n_AtA)
c_th2 = compute_theta2_split(Ab, n_AtA, split=2)
th2 = c_th2(mc)

# connects weights, to the NNLS layer
nnls = NNLS(theta_1)
x_kk = nnls([y_in, x_in, k_in, th2])

# stacks NNLS 
for j in range(1, num_layers):
    x_kk = nnls(x_kk)
    
c_trace_with_noise = compute_trace_with_noise(AtA=AtA, n_AtA=n_AtA)
tt = c_trace_with_noise(th2, x_kk[0])
   
#create final model
model = keras.Model(inputs=[fr_in, y_in, x_in, k_in], outputs=[tt])   
model.compile(optimizer='rmsprop', loss='mse')   
estimator = tf.keras.estimator.model_to_estimator(model)


#%%
from time import time
st = time()
nnls_out = []
y_old = y.copy()
x_old = x.copy()
num_layers = 10
k = np.zeros_like(1).astype(np.float32)

# def convert_sparse_matrix_to_sparse_tensor(X):
#     coo = coo_matrix(X)
#     indices = np.mat([coo.row, coo.col]).transpose()
#     return tf.SparseTensor(indices, coo.data, coo.shape)

AtA = Ab.T@Ab
n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)

from scipy.sparse import coo_matrix
A_sparse = convert_sparse_matrix_to_sparse_tensor(Ab[:, :-1])
bg = Ab[:, -1:] 

c_th2 = compute_theta2_split(Ab, n_AtA, split=2)
#c_th2 = compute_theta2(Ab, n_AtA)

nnls = NNLS(theta_1)
c_trace_with_noise = compute_trace_with_noise(AtA, n_AtA)

for i in range(2000):
    mc = np.reshape(np.transpose(mov[i]), [-1])[None, :]
    th2 = c_th2(mc)
    x_kk = nnls([y_old, x_old, k, th2])
    
    for j in range(1, num_layers):
        x_kk = nnls(x_kk)
        
    y_old, x_old = x_kk[0], x_kk[1]
    tt = c_trace_with_noise(th2, y_old)
    
    nnls_out.append(tt)

import matplotlib.pyplot as plt
plt.plot(np.array(nnls_out)[:, 0, :, 0])
ed = time()
print(ed-st)



#%%
n_bg = 2
A = Ab[:, :-n_bg]
nA = A.power(2).sum(0)

#%%
nA2 = np.power(Ab, 2).sum(axis=0)
nA2_inv_mat = scipy.sparse.spdiags(
    1. / (nA2 + np.finfo(np.float32).eps), 0, nA2.shape[0], nA2.shape[0])




#%%
trace_fiola_no_hals = fio.fit_gpu_nnls(mc_nn_mov, np.hstack((estimates.A.toarray(), estimates.b)), batch_size=fio.params.mc_nnls['offline_batch_size']) 
Ain = estimates.A
b_in = estimates.b
f_in = trace_fiola_no_hals[-1:]
Cin = trace_fiola_no_hals[:-1]
Yr = np.reshape(mov.transpose([1,2,0]), (Ain.shape[0], mov.shape[0]), order='F')
nA = (Ain.power(2).sum(axis=0))
nr = nA.size

YA = spdiags(old_div(1., nA), 0, nr, nr) * \
    (Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in))
AA = spdiags(old_div(1., nA), 0, nr, nr) * (Ain.T.dot(Ain))
YrA = YA - AA.T.dot(Cin)
trace_fiola_nohals_resid = np.vstack((YrA + trace_fiola_no_hals[:-1], trace_fiola_no_hals[-1:]))
trace_caiman = np.vstack((estimates.C[:,:num_frames_init] + estimates.YrA[:,:num_frames_init],estimates.f[:,:num_frames_init]))
idx = 0 

#%%
#Ab = scipy.sparse.hstack((self.estimates.A, self.estimates.b)).tocsc()
nA2 = np.ravel(Ab.power(2).sum(axis=0))
nA2_inv_mat = scipy.sparse.spdiags(
    1. / (nA2 + np.finfo(np.float32).eps), 0, nA2.shape[0], nA2.shape[0])
Cf = np.vstack((self.estimates.C, self.estimates.f))
if 'numpy.ndarray' in str(type(Yr)):
    YA = (Ab.T.dot(Yr)).T * nA2_inv_mat
else:
    YA = mmapping.parallel_dot_product(Yr, Ab, dview=self.dview, block_size=block_size,
                                   transpose=True, num_blocks_per_run=num_blocks_per_run) * nA2_inv_mat

AA = Ab.T.dot(Ab) * nA2_inv_mat
self.estimates.YrA = (YA - (AA.T.dot(Cf)).T)[:, :self.estimates.A.shape[-1]].T
self.estimates.R = self.estimates.YrA

#%%
trace = np.array(nnls_out)[:, 0, :, 0].T


Yr = np.reshape(mov.transpose([1,2,0]), (4800, 2000), order='F')



YA = (Ab.T.dot(Yr)).T
YrA = YA - AtA.T.dot(trace).T 

plt.plot(YrA)
plt.plot(trace.T)
plt.plot(YrA+trace.T)

#%%
tf.sparse.sparse_dense_matmul(Ab_sparse, mc, adjoint_a=True, adjoint_b=True)




#%%
num_layers = 10

nnls_out = []
k = np.zeros_like(1)
shifts = [0.0, 0.0]
for i in range(500):
    mc = np.reshape(np.transpose(mov[i]), [-1])[None, :]
    (th2, shifts) = c_th2(mc, shifts)
    x_kk = n([y_old, x_old, k, th2, shifts])

    
    for j in range(1, num_layers):
        x_kk = n(x_kk)
        
    y_old, x_old = x_kk[0], x_kk[1]
    nnls_out.append(y_old)
nnls_out = np.array(nnls_out).squeeze().T
#%%


#%%
np.save('/media/nel/storage/fiola/test/mov.npy', mov)
np.save('/media/nel/storage/fiola/test/Ab.npy', Ab)

mov = np.load('/media/nel/storage/fiola/test/mov.npy')
Ab = np.load('/media/nel/storage/fiola/test/Ab.npy')
#Ab = Ab[:, :-2]

#%%
i=0
mc = np.reshape(np.transpose(mov[i]), [-1])[None, :]
th2 = c_th2(mc)
x_kk = n([y_old, x_old, k, th2])


for j in range(1, num_layers):
    x_kk = n(x_kk)
    
y_old, x_old = x_kk[0], x_kk[1]
nnls_out.append(y_old)












