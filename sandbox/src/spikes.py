#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:36:11 2020

@author: cxd00
"""

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
from mc_nnls import HALS4activity
import caiman as cm
import multiprocessing as mp
from tensorflow.python.keras import backend as K
tb_path = "logs/"
#%%
with np.load('/home/nellab/SOFTWARE/SANDBOX/src/regression_n.01.01_less_neurons.npz', allow_pickle=True) as ld:
    Y_tot = ld['Y']
import h5py
import scipy
with h5py.File('/home/nellab/caiman_data/example_movies/memmap__d1_512_d2_512_d3_1_order_C_frames_1825_.hdf5','r') as f:
        
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
def get_model():
#    num_frames = 200
    #a = io.imread('Sue_2x_3000_40_-46.tif')
#    a = cm.load('/home/nellab/caiman_data/example_movies/n.01.01._rig__d1_512_d2_512_d3_1_order_F_frames_1825_.mmap', in_memory=True)
    template = np.median(a,axis=0)
    epsilon=0.00000001
    #import pdb; pdb.set_trace()
    shp = int(template.shape[1]/4)
    template = template[shp+10:-(10+shp),shp+10:-(shp+10), None, None]
    #temp_in = tf.reshape(tf.keras.layers.Input(shape=tf.TensorShape([512, 512])), [512, 512])
    template_zm = (template - tf.reduce_mean(template, axis=[0,1], keepdims=True)).numpy()
    template_var = (tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon).numpy()
    
    # min_, max_ = -296.0, 1425.0
#    y_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1])) #num comps x 1fr
#    x_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1])) # num components x fr
#    k_in = tf.keras.layers.Input(shape=(1,))
    #b_in = tf.keras.layers.Input(shape=tf.TensorShape([1, shape[-1]**2])) # num pixels => 1x512**2
    mc_0 = tf.reshape(a[0, :, :], [1, 512, 512, 1]) # initial input tensor for the motion correction layer
    #mc_in = tf.keras.layers.Input(tensor=tf.convert_to_tensor(tf.reshape(a[0, :, :], [512, 512, 1])))
    mc_in = tf.keras.layers.Input(shape=tf.TensorShape([512, 512, 1]))
    
    
#    Ab = np.concatenate([A_sp_full.toarray()[:], b_full], axis=1)
#    b = Y_tot[:, 0]
#    AtA = Ab.T@Ab
#    Atb = Ab.T@b
#    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
#    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
#    theta_2 = (Atb/n_AtA)[:, None]
    
    mc_layer = MotionCorrect(template, template_zm, template_var)   
    mc = mc_layer(mc_in)
    mod = keras.Model(inputs=[mc_in], outputs=[mc])
    
#    c_th2 = compute_theta2(Ab, n_AtA)
#    th2 = c_th2(mc)
#    nnls = NNLS(theta_1, theta_2)
#    x_kk = nnls([y_in, x_in, k_in])
#    for k in range(1, 10):
#        x_kk = nnls(x_kk)
#    
#    mod_nnls = keras.Model(inputs=[mc_in, y_in, x_in, k_in], outputs=[x_kk, th2])
#    
#    f, Y =  f_full[:, 0][:, None], Y_tot[:, 0][:, None]
#    YrA = YrA_full[:, 0][:, None]
#    C = C_full[:, 0][:, None]
#
#    Cf = np.concatenate([C+YrA,f], axis=0)
#    Cf_bc = Cf.copy()
#    Cf_bc = HALS4activity(Y_tot[:, 0][:, None].T[0], Ab, noisyC = Cf_bc, AtA=AtA, iters=5, groups=None)[0]
#    
#    x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
#    y_old = tf.identity(x_old)

#    (y_new, x_new, kkk), tht2 = mod_nnls((mc_0, y_old[None, :], x_old[None, :], tf.convert_to_tensor([[0]], dtype=tf.int8)))
#    tht2 = tf.squeeze(tht2)[:, None]
    
#    return mod_nnls, y_old, x_old, mc_0
    return mod, mc_0
#%%
class Spikes(object):
    
    def __init__(self, model, mc0):
#        self.model, self.y0, self.x0, self.mc0 = get_model()
        self.model, self.mc0 = model, mc0
        #self.frame_input_q = tf.queue.FIFOQueue(capacity=4, dtypes=tf.float32)
#        self.spike_input_q =  tf.queue.FIFOQueue(capacity=4, dtypes=[tf.float32, tf.float32, tf.float32, tf.float32])
        self.frame_input_q = mp.Queue()
#        self.spike_input_q = mp.Queue()
        self.output_q = mp.Queue()
        self.zero_tensor = tf.convert_to_tensor([[0]], dtype=tf.float32)
        self.estimator = self.load_estimator()
        
        self.frame_input_q.put(self.mc0)
#        self.spike_input_q.enqueue([y, x, k, tht2])
        #self.dataset = self.get_inputs()

        self.extraction_thread = Thread(target=self.extract, daemon=True)
        self.extraction_thread.start()
        
    def extract(self):
        for i in self.estimator.predict(input_fn=self.get_dataset, yield_single_examples=False):
            self.output_q.put(i['motion_correct'])
#        while True:
#            y_, x_, k_, tht2_ = self.spike_input_q.dequeue()
#            self.model.layers[3].set_weights([tht2_])
#        #        import pdb; pdb.set_trace()
#            (y, x, k), tht2 = self.model([self.frame_input_q.dequeue(), y_, x_, self.zero_tensor])
#            self.spike_input_q.enqueue([y, x, k, tht2])
#            self.output_q.enqueue(y)
    
    def load_estimator(self):

        self.model.compile(optimizer='rmsprop', loss='mse')
        return tf.keras.estimator.model_to_estimator(keras_model = self.model)

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generator, output_types=tf.float32, output_shapes=tf.TensorShape([1, 512, 512, 1]))
        return dataset
    
    def generator(self):
        while True:
            #(y, x, k), tht2 = self.spike_input_q.dequeue()
            fr = self.frame_input_q.get()
            #self.model.layers[3].set_weight([tht2])
#            yield [self.frame_input_q.dequeue(), y, x, k], tht2
            yield fr

    def get_spikes(self, idx):
#        for i in range(1, idx+1):
        t = tf.convert_to_tensor(a[idx, :, :, None])
        self.frame_input_q.put(t[None, :])
        
        out = self.output_q.get()
#        print(out.shape)
        return out
            
#%%
#tf.compat.v1.enable_eager_execution()
#from classes import MotionCorrect, compute_theta2, NNLS
model, mc_0 = get_model()
#%%    
spike_extractor = Spikes(model, mc_0)
print()
print()
print("out of init")
cfnn = []
start = float(timeit.default_timer())
#cfnn.append(spike_extracter.get_spikes(50))
for i in range(1, 6):
    cfnn.append(tf.squeeze(spike_extractor.get_spikes(i)))
print((float(timeit.default_timer())-start)/5)
#%%
cfnn=np.array(cfnn)
for i in range(500):
    plt.plot(cfnn[:, i])
    plt.pause(1)
    plt.cla()

    