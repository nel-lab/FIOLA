#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:36:11 2020

@author: cxd00
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
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
from classes2 import MotionCorrect, compute_theta2_AG, NNLS
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
a = np.array(a[:, :, :])
#%%
def get_model():
#    num_frames = 200
    #a = io.imread('Sue_2x_3000_40_-46.tif')
#    a = cm.load('/home/nellab/caiman_data/example_movies/n.01.01._rig__d1_512_d2_512_d3_1_order_F_frames_1825_.mmap', in_memory=True)
    template = np.median(a,axis=0)
#    epsilon=0.00000001
#    #import pdb; pdb.set_trace()
    shp = int(template.shape[1]/4)
    template = template[shp+10:-(10+shp),shp+10:-(shp+10), None, None]
    #temp_in = tf.reshape(tf.keras.layers.Input(shape=tf.TensorShape([512, 512])), [512, 512])
#    template_zm = (template - tf.reduce_mean(template, axis=[0,1], keepdims=True))
#    template_var = (tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon)
    
    # min_, max_ = -296.0, 1425.0
    y_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1]), name="y") #num comps x 1fr
    x_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1]), name="x") # num components x fr
    k_in = tf.keras.layers.Input(shape=(1,), name="k")
#    b_in = tf.keras.layers.Input(shape=tf.TensorShape([1, 512**2]), name="B") # num pixels => 1x512**2
    mc_0 = tf.reshape(a[0, :, :], [1, 512, 512, 1]) # initial input tensor for the motion correction layer
#    #mc_in = tf.keras.layers.Input(tensor=tf.convert_to_tensor(tf.reshape(a[0, :, :], [512, 512, 1])))
    mc_in = tf.keras.layers.Input(shape=tf.TensorShape([512, 512, 1]), name="m")
#    mc_in = tf.reshape(mc_in, [1, 512, 512, 1])
    
    
    Ab = np.concatenate([A_sp_full.toarray()[:], b_full], axis=1).astype(np.float32)
    b = Y_tot[:, 0]
    AtA = Ab.T@Ab
    Atb = Ab.T@b
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
    theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)

###########This is for the MOTION CORRECTION LAYER    
    mc_layer = MotionCorrect(template)   
    mc = mc_layer(mc_in)
#    mod = keras.Model(inputs=[mc_in], outputs=[mc])
 #########This is for the COMPUTE_THETA2 LAYER   
    c_th2 = compute_theta2_AG(Ab, n_AtA)
    th2 = c_th2(mc)
#    mod = keras.Model(inputs=[b_in], outputs=[th2])
########THIS IS FOR THE NNLS LAYER ONLY####### => Note: I haven't reconfigured the Spikes object for the other layers yet
    nnls = NNLS(theta_1, theta_2)
    x_kk = nnls([y_in, x_in, k_in])
    for k in range(1, 10):
        x_kk = nnls(x_kk)
#    mod_nnls = keras.Model(inputs=[y_in, x_in, k_in], outputs=[x_kk])
#    
    mod_nnls = keras.Model(inputs=[mc_in, y_in, x_in, k_in], outputs=[x_kk, th2])

#    
    f, Y =  f_full[:, 0][:, None], Y_tot[:, 0][:, None]
    YrA = YrA_full[:, 0][:, None]
    C = C_full[:, 0][:, None]

    Cf = np.concatenate([C+YrA,f], axis=0)
    Cf_bc = Cf.copy()
    Cf_bc = HALS4activity(Y_tot[:, 0][:, None].T[0], Ab, noisyC = Cf_bc, AtA=AtA, iters=5, groups=None)[0]
    
    x0 = Cf[:,0].copy()[:,None]
    x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
    y_old = tf.identity(x_old)

    (y_new, x_new, kkk), tht2 = mod_nnls((mc_0, y_old[None, :], x_old[None, :], tf.convert_to_tensor([[0]], dtype=tf.int8)))
    tht2 = tf.squeeze(tht2)[:, None]
    
    return mod_nnls, x0[None, :], x0[None, :], mc_0, th2
#    return mod, mc_0
#%%
class Spikes(object):
    
    def __init__(self, model, y_0, x_0, mc_0, tht2):
#        self.model = model
        self.model, self.mc0, self.y0, self.x0 = model, mc_0, y_0, x_0
        #self.frame_input_q = tf.queue.FIFOQueue(capacity=4, dtypes=tf.float32)
#        self.spike_input_q =  tf.queue.FIFOQueue(capacity=4, dtypes=[tf.float32, tf.float32, tf.float32, tf.float32])
        self.frame_input_q = Queue()
        self.spike_input_q = Queue()
        self.output_q = Queue()
        self.zero_tensor = tf.convert_to_tensor([[0]], dtype=tf.float32)
        self.estimator = self.load_estimator()
        
        self.frame_input_q.put(a[0, :, :, None][None, :])
        self.spike_input_q.put((y_0, x_0, tht2))
#        self.frame_input_q.put(self.mc0)
#        import pdb; pdb.set_trace()

        #self.dataset = self.get_inputs()

        self.extraction_thread = Thread(target=self.extract, daemon=True)
        self.extraction_thread.start()
        
    def extract(self):
        for i in self.estimator.predict(input_fn=self.get_dataset, yield_single_examples=False):
#            import pdb; pdb.set_trace()
            self.output_q.put(i)
#        while True:
#            y_, x_, k_, tht2_ = self.spike_input_q.dequeue()
#            self.model.layers[3].set_weights([tht2_])
#        #        import pdb; pdb.set_trace()
#            (y, x, k), tht2 = self.model([self.frame_input_q.dequeue(), y_, x_, self.zero_tensor])
#            self.spike_input_q.enqueue([y, x, k, tht2])
#            self.output_q.enqueue(y)
    
    def load_estimator(self):

        self.model.compile(optimizer='rmsprop', loss='mse')
#        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="/tmp",
#                       vars_to_warm_start=".*")
        
        return tf.keras.estimator.model_to_estimator(keras_model = self.model)

    def get_dataset(self):
#        import pdb; pdb.set_trace()
        dataset = tf.data.Dataset.from_generator(self.generator, 
                                                 output_types={"m": tf.float32,
                                                               "y": tf.float32,
                                                               "x": tf.float32,
                                                               "k": tf.float32}, 
                                                 output_shapes={"m":(1, 512, 512, 1),
                                                                "y":(1, 572, 1),
                                                                "x":(1, 572, 1),
                                                                "k":(1, 1)})
        return dataset
    
    def generator(self):
        while True:
#            import pdb; pdb.set_trace()
#            print("WAITING")
#            print()
            out = self.spike_input_q.get()
            (y, x, tht2) = out
#            tf.print(y)
#            fr = self.frame_input_q.get()
            fr = self.frame_input_q.get()
#            self.model.layers[3].set_weights([tht2])
#            self.mc0=tf.reshape(self.mc0, [1, 512, 512, 1])
#            print(y.shape, x.shape)
            
            yield {"m":fr, "y":y, "x":x, "k":[[0]]}
#            yield fr

    def get_spikes(self):
        for idx in range(1, 101):
#        t = tf.convert_to_tensor(a[idx, :, :, None])
#        print(t[None,:].shape)
            self.frame_input_q.put(a[idx, :, :, None][None, :])
        
            out = self.output_q.get()
#        print(y.shape, x.shape, k.shape, tht2.shape)
            self.spike_input_q.put((out["nnls"], out["nnls_1"], out["compute_theta2_ag"]))
#        print(out.shape)
        return out['nnls']
            

#%%
#tf.compat.v1.enable_eager_execution()
model, y_0, x_0, mc_0, tht2= get_model()
print(y_0.shape, x_0.shape, tht2.shape, mc_0.shape)

#%%
spike_extractor = Spikes(model, y_0, x_0, mc_0, tht2)
print()
print("out of init")
cfnn = []
start = float(timeit.default_timer())
#cfnn.append(spike_extracter.get_spikes(50))
#for i in range(1, 101):
cfnn.append(spike_extractor.get_spikes())
print((float(timeit.default_timer())-start)/100)
#%%
cfnn=np.array(cfnn)
for i in range(5):
    plt.plot(cfnn[:, i])
    plt.pause(1)
    plt.cla()

    