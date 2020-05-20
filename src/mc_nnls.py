# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from queue import Queue
from threading import Thread
from past.utils import old_div
from skimage import io
import numpy as np
import pylab as plt
import cv2
import timeit
from classes import MotionCorrect, compute_theta2, NNLS
import caiman as cm
import multiprocessing as mp

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  print("pass")
  pass
# from NNLS_keras_fast_online import compute_theta2, NNLS

#%% HELPER FUNCTIONS
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
                noisyC[m] = C[m][:, None] + (AtY[:, None][m] - AtA[m].dot(C)) / AtAd[:, None][m]
                C[m] = np.maximum(noisyC[m], 0)
        else:
            for m in groups:
                noisyC[m] = C[m] + ((AtY[m] - AtA[m].dot(C)).T/AtAd[m]).T
                C[m] = np.maximum(noisyC[m], 0)
        num_iters += 1
    return C, noisyC
#%%
    
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
    A_sp_full = scipy.sparse.csc_matrix((data[:], indices[:], indptr[:]), shape[:])
    YrA_full = np.array(f['estimates']['YrA'])
    C_full = np.array(f['estimates']['C']) 
    b_full = np.array(f['estimates']['b']) 
    f_full = np.array(f['estimates']['f'])
    A_sp_full = A_sp_full[:,idx_components ]
    C_full = C_full[idx_components]
    YrA_full = YrA_full[idx_components]
#%%
a = cm.load('/home/nellab/caiman_data/example_movies/n.01.01._rig__d1_512_d2_512_d3_1_order_F_frames_1825_.mmap', in_memory=True)


#%%
q = mp.Queue()
def generator():
    while True:
        try:
            yield q.get_nowait()
        except:
            break
    return
    
def enqueue(q, batch):
    for fr in batch:
        q.put(tf.expand_dims(fr, 0))
    return

#%%
# tf.compat.v1.disable_eager_execution()
if __name__ == "__main__":

    num_frames = 5
    #a = io.imread('Sue_2x_3000_40_-46.tif')
#    a = cm.load('/home/nellab/caiman_data/example_movies/n.01.01._rig__d1_512_d2_512_d3_1_order_F_frames_1825_.mmap', in_memory=True)
    shape = a.shape
    template = np.median(a,axis=0)
    batch_init = tf.convert_to_tensor(a[:num_frames,:,:,None])

    # min_, max_ = -296.0, 1425.0
    y_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1])) #num comps x 1fr
    x_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1])) # num components x fr
    k_in = tf.keras.layers.Input(shape=(1,))
    b_in = tf.keras.layers.Input(shape=tf.TensorShape([1, shape[-1]**2])) # num pixels => 1x512**2
    mc_0 = tf.reshape(a[0, :, :], [1, 512, 512, 1]) # initial input tensor for the motion correction layer
    #mc_in = tf.keras.layers.Input(tensor=tf.convert_to_tensor(tf.reshape(a[0, :, :], [512, 512, 1])))
    mc_in = tf.keras.layers.Input(shape=tf.TensorShape([512, 512, 1]))
    
    
    Ab = np.concatenate([A_sp_full.toarray()[:], b_full], axis=1)
    b = Y_tot[:, 0]
    AtA = Ab.T@Ab
    Atb = Ab.T@b
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
    theta_2 = (Atb/n_AtA)[:, None]
    
    mod_mc = MotionCorrect(template) #initialize motion correction layer

    load_thread = Thread(target=mod_mc.enqueue, args=(mod_mc.q, batch_init), daemon=True)
    load_thread.start() #thread puts frames on queue
    
    dataset = tf.data.Dataset.from_generator(mod_mc.generator, output_types=tf.float32)
#    th2 = mod_mc(mc_in) #first call
    #import pdb; pdb.set_trace()
    mc = mod_mc(mc_in)
    
    
    c_th2 = compute_theta2(Ab, n_AtA)
    th2 = c_th2(mc)
    nnls = NNLS(theta_1, theta_2)
    x_kk = nnls([y_in, x_in, k_in])
    for k in range(1, 10):
        x_kk = nnls(x_kk)
    
    mod_nnls = keras.Model(inputs=[mc_in, y_in, x_in, k_in], outputs=[x_kk, th2])
    newy = []
    tht2 = 0
    
    groups = update_order_greedy(A_sp_full,flag_AA=False)[0]
#    
    f, Y =  f_full[:, 0][:, None], Y_tot[:, 0][:, None]
    YrA = YrA_full[:, 0][:, None]
    C = C_full[:, 0][:, None]

    Cf = np.concatenate([C+YrA,f], axis=0)
    Cf_bc = Cf.copy()
    Cf_bc = HALS4activity(Y_tot[:, 0][:, None].T[0], Ab, noisyC = Cf_bc, AtA=AtA, iters=5, groups=None)[0]
    
    x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
    y_old = tf.identity(x_old)
    #print(x_old, y_old)
    
    
    i=0
    mov_corr = []
    cfnn = []
    
    # tf.compat.v1.get_default_graph().finalize()
    for elt in dataset:
        if len(elt) == 0:
            break
        start = float(timeit.default_timer())
        mc = tf.reshape(tf.squeeze(elt), [shape[1], shape[1], 1])
        
        if i == 0:
            #b = tf.convert_to_tensor(mc, dtype=tf.float32)  # tensor of the frame in question => CHANGE TO IMAGE
            (y_new, x_new, kkk), tht2 = mod_nnls((mc[None, :], y_old[None, :], x_old[None, :], tf.expand_dims(tf.convert_to_tensor(0, dtype=tf.int8), axis=0)[None, :]))
            tht2 = tf.squeeze(tht2)[:, None]
    
        #b = tf.convert_to_tensor(mc, dtype=tf.float32)
        mod_nnls.layers[3].set_weights([tht2]) 
        (y_new, x_new, kkk), tht2 = mod_nnls((mc[None, :], y_old[None, :], x_old[None, :], tf.expand_dims(tf.convert_to_tensor(0, dtype=tf.int8), axis=0)[None, :]))   
        
        x_old, y_old = tf.squeeze(x_new, 0), tf.squeeze(y_new, 0)

        
        print(float(timeit.default_timer())-start, "finish nnls")

        cfnn.append(x_old) # or whatever display/saving mechanism
        
        i+= 1
    load_thread.join()
