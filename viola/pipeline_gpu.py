#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:20:47 2020

@author: nellab
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from threading import Thread
import numpy as np
from motion_correction_gpu import MotionCorrect
from nnls_gpu import NNLS, compute_theta2
from queue import Queue
import timeit
#%%
def get_model(template, center_dims, Ab, num_layers=5):
    """
    takes as input a template (median) of the movie, A_sp object, and b object from caiman.
    """
    shp_x, shp_y = template.shape[0], template.shape[1] #dimensions of the movie
    Ab = Ab.astype(np.float32)
    template = template.astype(np.float32)
    num_components = Ab.shape[-1]  
#    c_shp_x, c_shp_y = shp_x//4, shp_y//4
#    template = template[c_shp_x+10:-(c_shp_x+10),c_shp_y+10:-(c_shp_y+10), None, None]

    y_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, 1]), name="y") # Input Layer for components
    x_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, 1]), name="x") # Input layer for components
    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 
    k_in = tf.keras.layers.Input(shape=(1,), name="k")
 
    #Calculations to initialize Motion Correction
    AtA = Ab.T@Ab
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
#    theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)

    #Initialization of the motion correction layer, initialized with the template   
    mc_layer = MotionCorrect(template, center_dims)   
    mc = mc_layer(fr_in)
    #Chains motion correction layer to weight-calculation layer
    c_th2 = compute_theta2(Ab, n_AtA)
    th2 = c_th2(mc)
    #Connects weights, calculated from the motion correction, to the NNLS layer
    nnls = NNLS(theta_1)
    x_kk = nnls([y_in, x_in, k_in, th2])
    #stacks NNLS 9 times
    for j in range(1, num_layers):
        x_kk = nnls(x_kk)
   
    #create final model, returns it and the first weight
    model = keras.Model(inputs=[fr_in, y_in, x_in, k_in], outputs=[x_kk])   
    return model

#%%    
class Pipeline(object):    
    def __init__(self, model, y_0, x_0, mc_0, tht2, tot):
        """
        Inputs: the model from get_model, and the initial input values as numpy arrays (y_0, x_0, mc_0, tht2)
        To run, after initializing, run self.get_spikes()
        @todo: check if nAtA is computed at every iteration!
        """
        self.model, self.mc0, self.y_0, self.x_0, self.tht2 = model, mc_0, y_0, x_0, tht2
        self.tot = tot
        self.num_neurons = tht2.shape[0]
        self.dim_x, self.dim_y = self.mc0.shape[1], self.mc0.shape[2]
        self.zero_tensor = [[0.0]]
        
        self.frame_input_q = Queue()
        self.spike_input_q = Queue()
        self.output_q = Queue()
        
        #load estimator from the keras model
        self.estimator = self.load_estimator()
        
        #seed the queues
        self.frame_input_q.put(self.mc0)
        self.spike_input_q.put((self.y_0, self.x_0))

        #start extracting frames: extract calls the estimator to predict using the outputs from the dataset, which
            #pull from the generator.
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
                                                               "x": tf.float32,
                                                               "k": tf.float32}, 
                                                 output_shapes={"m":(1, self.dim_x, self.dim_y, 1),
                                                                "y":(1, self.num_neurons, 1),
                                                                "x":(1, self.num_neurons, 1),
                                                                "k":(1, 1)})
        return dataset
    
    def generator(self):
        #generator waits until data has been enqueued, then yields the data to the dataset
        while True:
            out = self.spike_input_q.get()
            (y, x) = out
            fr = self.frame_input_q.get()
            
            yield {"m":fr, "y":y, "x":x, "k":self.zero_tensor}

    def get_spikes(self, bound):
        #to be called separately. Input "bound" represents the number of frames. Starts at one because of initial values put on queue.
        output = [0]*bound
        # times = [0]*bound
        start = timeit.default_timer()
        for idx in range(1, bound):
            self.frame_input_q.put(self.tot[idx:idx+1,:,:, None])
            out = self.output_q.get()
            self.spike_input_q.put((out["nnls"], out["nnls_1"]))
            output[idx-1] = out["nnls_1"]
            # times[idx-1]= timeit.default_timer()-start
            
        output[-1] = (self.output_q.get()["nnls_1"])
        print(timeit.default_timer()-start)
        self.frame_input_q.put(self.mc0)
        self.spike_input_q.put((self.y_0, self.x_0))
        # return (output, times)
        return output
   
#%%
class Pipeline_overall(object):    
    def __init__(self, model, y_0, x_0, mc_0, tht2, saoz, n):
        """
        Inputs: the model from get_model, and the initial input values as numpy arrays (y_0, x_0, mc_0, tht2)
        To run, after initializing, run self.get_spikes()
        @todo: check if nAtA is computed at every iteration!
        """
        self.model, self.mc0, self.y_0, self.x_0, self.tht2, self.saoz, self.num_frames_init = model, mc_0, y_0, x_0, tht2, saoz, n
        self.n = self.num_frames_init - 1
        self.num_neurons = tht2.shape[0]
        self.dim_x, self.dim_y = self.mc0.shape[1], self.mc0.shape[2]
        self.zero_tensor = [[0.0]]
        
        self.frame_input_q = Queue()
        self.spike_input_q = Queue()
        self.signal_q = Queue()
        self.sao_input_q = Queue()
        
        #load estimator from the keras model
        self.estimator = self.load_estimator()
        
        #seed the queues
        self.frame_input_q.put(self.mc0)
        self.spike_input_q.put((self.y_0, self.x_0))
        
        #start extracting frames: extract calls the estimator to predict using the outputs from the dataset, which
            #pull from the generator.
        self.extraction_thread = Thread(target=self.extract, daemon=True)
        self.extraction_thread.start()
        self.detection_thread = Thread(target=self.detect, daemon=True)
        self.detection_thread.start()

    def load_frame(self, mov_online):
        for i in range(len(mov_online)):
            self.frame_input_q.put(mov_online[i:i + 1, :, :, None])

    def detect(self):
        while True:
            traces_input = self.sao_input_q.get()
            if self.n >= self.num_frames_init:  # neglect the first frame collected, it is just for initialization
                self.saoz.fit_next(traces_input[:, None], self.n)
            self.n += 1
            if self.n % 100 == 0:
                print(f'{self.n} frames processed ####DETECT##### ')
        
    def extract(self):
        for i in self.estimator.predict(input_fn=self.get_dataset, yield_single_examples=False):
            self.signal_q.put(i)
    
    def load_estimator(self):
        return tf.keras.estimator.model_to_estimator(keras_model = self.model)

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generator, 
                                                 output_types={"m": tf.float32,
                                                               "y": tf.float32,
                                                               "x": tf.float32,
                                                               "k": tf.float32}, 
                                                 output_shapes={"m":(1, self.dim_x, self.dim_y, 1),
                                                                "y":(1, self.num_neurons, 1),
                                                                "x":(1, self.num_neurons, 1),
                                                                "k":(1, 1)})
        return dataset
    
    def generator(self):
        #generator waits until data has been enqueued, then yields the data to the dataset
        while True:
            out = self.spike_input_q.get()
            (y, x) = out
            fr = self.frame_input_q.get()
            yield {"m":fr, "y":y, "x":x, "k":self.zero_tensor}

    def get_spikes(self):
        #to be called separately. Input "bound" represents the number of frames. Starts at one because of initial values put on queue.
        #self.frame_input_q.put(new_frame[:, :, :, None])
        while self.frame_input_q.qsize() > 0:
            out = self.signal_q.get().copy()       
            self.spike_input_q.put((out["nnls"], out["nnls_1"]))
            traces_input = np.array(out["nnls_1"]).squeeze().T
            self.sao_input_q.put(traces_input)
                
        