# -*- coding: utf-8 -*-

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
def create_estimates(counter):
    return A_sp_full, YrA_full[:, counter][:, None], C_full[:, counter][:, None], b_full, f_full[:, counter][:, None], Y_tot[:, counter][:, None]
    
#%%
if __name__ == "__main__":
    num_frames = 300
    a = io.imread('Sue_2x_3000_40_-46.tif')
    template = np.median(a,axis=0)
    batch_init = tf.convert_to_tensor(a[:num_frames,:,:,None])
    min_, max_ = -296.0, 1425.0
    
    mod = MotionCorrect(template)
    
    load_thread = Thread(target=mod.enqueue, args=(mod.q, batch_init), daemon=True)
    load_thread.start()
    
    dataset = tf.data.Dataset.from_generator(mod.generator, output_types=tf.float32)
    
    i=0
    time_arr = []
    mov_corr = []
    
    
    
    for elt in dataset:
        if i >= num_frames:
            break
        start = float(timeit.default_timer())
        # motion correction
        mov_corr.append(mod(elt))
        # get frame-by-frame estimates
        A_sp, YrA, C, b, f, Y = create_estimates(i)
        # start prepping for NNLS
        Ab = np.concatenate([A_sp.toarray()[:],b], axis=1)
        Cf = np.concatenate([C+YrA,f], axis=0)
        print(np.linalg.norm(Y-Ab@Cf)/np.linalg.norm(Y))
        A = Ab
        b = Y
        AtA = A.T@A
        Atb = A.T@b
        n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
        
        theta_1 = (np.eye(A.shape[-1]) - AtA/n_AtA)
        theta_2 = (Atb/n_AtA) 
        
        groups = update_order_greedy(A_sp,flag_AA=False)[0]
        Cf_bc = Cf.copy()
        Cf_bc = HALS4activity(Y.T[0], A, noisyC = Cf_bc, AtA=AtA, iters=5, groups=None)[0]
        print(np.linalg.norm(Y-Ab@Cf_bc)/np.linalg.norm(Y))
        
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
        
        from time import time
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
        
        time_arr.append(float(timeit.default_timer())-start)
    
        i += 1
            
    
    for fr, fr_raw in zip(mov_corr, batch_init):
            # Our operations on the frame come here
            gray = np.concatenate((fr.numpy().squeeze(), fr_raw.numpy().squeeze()))
            # Display the resulting frame
            cv2.imshow('frame', (gray-min_)/(max_-min_)*10)
            if cv2.waitKey(1) == ord('q'):
                break
    load_thread.join()
    print(np.mean(time_arr[30:]))
    cv2.destroyAllWindows()
        

