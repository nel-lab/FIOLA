#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:20:47 2020

@author: nellab
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from threading import Thread
import numpy as np
from motion_correction_gpu import MotionCorrect
from nnls_gpu import NNLS, compute_theta2
from queue import Queue
import timeit

#%%
def get_model(template, A_sp, b_full):
    """
    takes as input a template (median) of the movie, A_sp object, and b object from caiman.
    """
    shp_x, shp_y = template.shape[0], template.shape[1] #dimensions of the movie
    c_shp_x, c_shp_y = shp_x//4, shp_y//4

    template = template[c_shp_x+10:-(c_shp_x+10),c_shp_y+10:-(c_shp_y+10), None, None]

    y_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1]), name="y") # Input Layer for components
    x_in = tf.keras.layers.Input(shape=tf.TensorShape([572, 1]), name="x") # Input layer for components
    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 
    
    #Calculations to initialize Motion Correction
    Ab = np.concatenate([A_sp.toarray()[:], b_full], axis=1).astype(np.float32)
    AtA = Ab.T@Ab
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
#    theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)

    #Initialization of the motion correction layer, initialized with the template   
    mc_layer = MotionCorrect(template)   
    mc = mc_layer(fr_in)
    #Chains motion correction layer to weight-calculation layer
    c_th2 = compute_theta2(Ab, n_AtA)
    th2 = c_th2(mc)
    #Connects weights, calculated from the motion correction, to the NNLS layer
    nnls = NNLS(theta_1)
    x_kk = nnls([y_in, x_in, th2])
    #stacks NNLS 9 times
    for k in range(1, 10):
        x_kk = nnls(x_kk)
   
    #create final model, returns it and the first weight
    model = keras.Model(inputs=[fr_in, y_in, x_in], outputs=[x_kk])   
    return model

#%%
    
class Pipeline(object):
    
    def __init__(self, model, y_0, x_0, mc_0, tht2, tot):
        """
        Inputs: the model from get_model, and the initial input values as numpy arrays (y_0, x_0, mc_0, tht2)
        To run, after initializing, run self.get_spikes()
        """
        self.model, self.mc0, self.y0, self.x0, self.tht2 = model, mc_0, y_0, x_0, tht2
        self.tot = tot
        self.dim_x, self.dim_y = self.mc0.shape[1], self.mc0.shape[2]
        
        self.frame_input_q = Queue()
        self.spike_input_q = Queue()
        self.output_q = Queue()
        
        self.estimator = self.load_estimator()
        
        self.frame_input_q.put(self.mc0)
        self.spike_input_q.put((y_0, x_0))

        self.extraction_thread = Thread(target=self.extract, daemon=True)
        self.extraction_thread.start()
        
    def extract(self):
        for i in self.estimator.predict(input_fn=self.get_dataset, yield_single_examples=False):
#            print(i.keys())
            self.output_q.put(i)
    
    def load_estimator(self):
        return tf.keras.estimator.model_to_estimator(keras_model = self.model)

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generator, 
                                                 output_types={"m": tf.float32,
                                                               "y": tf.float32,
                                                               "x": tf.float32}, 
                                                 output_shapes={"m":(1, self.dim_x, self.dim_y, 1),
                                                                "y":(1, 572, 1),
                                                                "x":(1, 572, 1)})
        return dataset
    
    def generator(self):
        while True:
            out = self.spike_input_q.get()
            (y, x) = out
            fr = self.frame_input_q.get()
            
            yield {"m":fr, "y":y, "x":x}

    def get_spikes(self, bound):
        output = []
        for idx in range(1, bound):

            self.frame_input_q.put(self.tot[idx, :, :, None][None, :]) #here, a represents a numpy array defined outside of the class.
        
            out = self.output_q.get()

            self.spike_input_q.put((out["nnls"], out["nnls_1"]))
            output.append(out["nnls"])

        return output
    