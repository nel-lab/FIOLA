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
from classes2 import MotionCorrect, compute_theta2_AG, NNLS, NNLS1
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
#a = a.resize(0.5, 0.5)
a = np.array(a[:, :, :])
#%%
def get_model():
#    num_frames = 200
    #a = io.imread('Sue_2x_3000_40_-46.tif')
#    a = cm.load('/home/nellab/caiman_data/example_movies/n.01.01._rig__d1_512_d2_512_d3_1_order_F_frames_1825_.mmap', in_memory=True)
    template = np.median(a,axis=0) # sets up a template, consisting of the median of the imported movie.
#    epsilon=0.00000001
#    #import pdb; pdb.set_trace()
    shp = int(template.shape[1]) #determines shape of the tensor
    c_shp = int(shp/4) #for cropping the center
    template = template[c_shp+10:-(c_shp+10),c_shp+10:-(c_shp+10), None, None]# cr
    #temp_in = tf.reshape(tf.keras.layers.Input(shape=tf.TensorShape([512, 512])), [512, 512])
#    template_zm = (template - tf.reduce_mean(template, axis=[0,1], keepdims=True))
#    template_var = (tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon)
    # min_, max_ = -296.0, 1425.0
    y_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1]), name="y") #num comps x 1fr
    x_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1]), name="x") # num components x fr
#    k_in = tf.keras.layers.Input(shape=(1,), name="k")
#    b_in = tf.keras.layers.Input(shape=tf.TensorShape([1, 512**2]), name="B") # num pixels => 1x512**2
    mc_0 = tf.reshape(a[0, :, :], [1, shp, shp, 1]) # initial input tensor for the motion correction layer
#    #mc_in = tf.keras.layers.Input(tensor=tf.convert_to_tensor(tf.reshape(a[0, :, :], [512, 512, 1])))
    mc_in = tf.keras.layers.Input(shape=tf.TensorShape([shp, shp, 1]), name="m")
#    t2_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1]), name="th2")
#    mc_in = tf.reshape(mc_in, [1, 512, 512, 1])
    
    
    Ab = np.concatenate([A_sp_full.toarray()[:], b_full], axis=1).astype(np.float32)
    b = Y_tot[:, 0]
    AtA = Ab.T@Ab
    Atb = Ab.T@b
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
    theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)
    th2_2 = a[0,:,:].flatten()[None, :]@Ab
    th2_2 = th2_2 / n_AtA
    th2_2 = th2_2.T

###########This is for the MOTION CORRECTION LAYER    
    mc_layer = MotionCorrect(template)   
    mc = mc_layer(mc_in)
#    mod = keras.Model(inputs=[mc_in], outputs=[mc])
 #########This is for the COMPUTE_THETA2 LAYER   
    c_th2 = compute_theta2_AG(Ab, n_AtA)
    th2 = c_th2(mc)
#    mod = keras.Model(inputs=[b_in], outputs=[th2])
########THIS IS FOR THE NNLS LAYER ONLY####### => Note: I haven't reconfigured the Spikes object for the other layers yet
    nnls = NNLS1(theta_1)
    x_kk = nnls([y_in, x_in, th2])
#    nnls2 = NNLS(theta_1, theta_2)
#    x_kk = nnls2([x_kk[0], x_kk[1]])
#    nnls = NNLS(theta_1, theta_2)
#    x_kk = nnls(x_kk1)
    for k in range(1, 10):
        x_kk = nnls(x_kk)
#    mod_nnls = keras.Model(inputs=[y_in, x_in, k_in], outputs=[x_kk])
#    
    mod_nnls = keras.Model(inputs=[mc_in, y_in, x_in], outputs=[x_kk])

#    
    f, Y =  f_full[:, 0][:, None], Y_tot[:, 0][:, None]
    YrA = YrA_full[:, 0][:, None]
    C = C_full[:, 0][:, None]

    Cf = np.concatenate([C+YrA,f], axis=0)
    Cf_bc = Cf.copy()
    Cf_bc = HALS4activity(Y_tot[:, 0][:, None].T[0], Ab, noisyC = Cf_bc, AtA=AtA, iters=5, groups=None)[0]
    
    x0 = Cf[:,0].copy()[:,None]
#    x_old = tf.convert_to_tensor(Cf[:,0].copy()[:,None], dtype=np.float32)
#    y_old = tf.identity(x_old)
#    import pdb; pdb.set_trace()
#    output = mod_nnls((mc_0, y_old[None, :], x_old[None, :]))
#    tht2 = tf.squeeze(tht2)[:, None]
#
#    
    return mod_nnls, x0[None, :], x0[None, :], mc_0, th2_2
#    return mod, mc_0
#%%
class Spikes(object):
    
    def __init__(self, model, y_0, x_0, mc_0, tht2):

        self.model, self.mc0, self.y0, self.x0, self.tht2 = model, mc_0, y_0, x_0, tht2
        self.dim = self.mc0.shape[1]
        self.frame_input_q = Queue()
        self.spike_input_q = Queue()
        self.output_q = Queue()
        self.estimator = self.load_estimator()
        
        self.frame_input_q.put(a[0, :, :, None][None, :])
        self.spike_input_q.put((y_0, x_0))

        self.extraction_thread = Thread(target=self.extract, daemon=True)
        self.extraction_thread.start()
        
    def extract(self):
        for i in self.estimator.predict(input_fn=self.get_dataset, yield_single_examples=False):
#            print(i.keys())
            self.output_q.put(i)
    
    def load_estimator(self):
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./summaries")
        return tf.keras.estimator.model_to_estimator(keras_model = self.model, model_dir="./summaries")

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generator, 
                                                 output_types={"m": tf.float32,
                                                               "y": tf.float32,
                                                               "x": tf.float32}, 
                                                 output_shapes={"m":(1, self.dim, self.dim, 1),
                                                                "y":(1, 572, 1),
                                                                "x":(1, 572, 1)})
        return dataset
    
    def generator(self):
        while True:
            out = self.spike_input_q.get()
            (y, x) = out
            fr = self.frame_input_q.get()
            
            yield {"m":fr, "y":y, "x":x}

    def get_spikes(self, bound, output):
        start = timeit.default_timer()
        for idx in range(1, bound):

            self.frame_input_q.put(a[idx, :, :, None][None, :])
        
            out = self.output_q.get()

            self.spike_input_q.put((out["nnl_s1_1"], out["nnl_s1_1_1"]))
            output.append(out["nnls_s1_1"])

        print(timeit.default_timer() - start)
        return output
            

#%%
#tf.compat.v1.enable_eager_execution()
from classes2 import MotionCorrect, compute_theta2_AG, NNLS1
model, y_0, x_0, mc_0, tht2= get_model()
#model.layers[3].set_weights([tht2])
model.compile(optimizer='rmsprop', loss='mse')
print(y_0.shape, x_0.shape, tht2.shape, mc_0.shape)

#%%
spike_extractor = Spikes(model, y_0.astype(np.float32), x_0, mc_0, tht2)
print()
print("out of init")
cfnn = []

cfnn = spike_extractor.get_spikes(1800, cfnn)

#%%
#cfnn=np.array(cfnn)
#for i in range(5):
#    plt.plot(cfnn[:, i])
#    plt.pause(1)
#    plt.cla()
#%%
#%%
class SpikesMC(object):
    
    def __init__(self, model, mc0):
#        self.model, self.y0, self.x0, self.mc0 = get_model()
        self.model, self.mc0 = model, mc0
        #self.frame_input_q = tf.queue.FIFOQueue(capacity=4, dtypes=tf.float32)
#        self.spike_input_q =  tf.queue.FIFOQueue(capacity=4, dtypes=[tf.float32, tf.float32, tf.float32, tf.float32])
        self.frame_input_q = mp.Queue()
#        self.spike_input_q = mp.Queue()
        self.output_q = mp.Queue()
#        self.zero_tensor = tf.convert_to_tensor([[0]], dtype=tf.float32)
        self.estimator = self.load_estimator()
        
        self.frame_input_q.put(self.mc0)
#        self.spike_input_q.enqueue([y, x, k, tht2])
        #self.dataset = self.get_inputs()
        self.data = []
        self.times = []

        self.extraction_thread = Thread(target=self.extract, daemon=True)
        self.extraction_thread.start()
        
#        self.output_thread = Thread(target=self.get_output, daemon=True)
#        self.output_thread.start()
        
    def extract(self):
        start = timeit.default_timer()
        for i in self.estimator.predict(input_fn=self.get_dataset, yield_single_examples=False):
#            print(i)
            self.output_q.put(i["motion_correct_7"])
            print(timeit.default_timer()-start)
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
        dataset = tf.data.Dataset.from_generator(self.generator, output_types=tf.float32, output_shapes=(1, 512, 512, 1))
        return dataset
    
    def generator(self):
        while True:
            #(y, x, k), tht2 = self.spike_input_q.dequeue()
            fr = self.frame_input_q.get()
            #self.model.layers[3].set_weight([tht2])
#            yield [self.frame_input_q.dequeue(), y, x, k], tht2
            yield fr

    def get_spikes(self, idx):
        for i in range(1, idx):
            start = timeit.default_timer()
            t = a[i, :, :, None]
            self.frame_input_q.put(t[None, :])
            
            self.data.append(self.output_q.get())
            self.times.append(timeit.default_timer()-start)
        return self.times
    
    def get_output(self):
        for i in range(999):
            start = timeit.default_timer()
#            self.data.append(self.output_q.get())
            self.times.append(timeit.default_timer()-start)
    
    def out(self):
        return self.data, self.times

        
from classes2 import MotionCorrect
model, mc_0 = get_model()
spike_extractor = SpikesMC(model, mc_0)
spikes = []
times = spike_extractor.get_spikes(1000)
#data, time = spike_extractor.out()
    