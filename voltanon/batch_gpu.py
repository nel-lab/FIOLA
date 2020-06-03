#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:26:40 2020

@author: nellab
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from threading import Thread
import tensorflow_addons as tfa
import numpy as np
from queue import Queue
import timeit

#%%
def get_model(template, center_dims, Ab, num_components, batch_size):
    """
    takes as input a template (median) of the movie, A_sp object, and b object from caiman.
    """
    shp_x, shp_y = template.shape[0], template.shape[1] #dimensions of the movie
#    c_shp_x, c_shp_y = shp_x//4, shp_y//4

#    template = template[c_shp_x+10:-(c_shp_x+10),c_shp_y+10:-(c_shp_y+10), None, None]

    y_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, batch_size]), name="y") # Input Layer for components
    x_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, batch_size]), name="x") # Input layer for components
    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([batch_size, shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 
    k_in = tf.keras.layers.Input(shape=(1,), name="k")
    #Calculations to initialize Motion Correction
#    Ab = np.concatenate([A_sp.toarray()[:], b_full], axis=1).astype(np.float32)
    AtA = Ab.T@Ab
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)
#    theta_2 = (Atb/n_AtA)[:, None].astype(np.float32)

    #Initialization of the motion correction layer, initialized with the template   
    mc_layer = MotionCorrect(template, center_dims, batch_size)   
    mc = mc_layer(fr_in)
    #Chains motion correction layer to weight-calculation layer
    c_th2 = compute_theta2(Ab, n_AtA)
    th2 = c_th2(mc)
    #Connects weights, calculated from the motion correction, to the NNLS layer
    nnls = NNLS(theta_1)
    x_kk = nnls([y_in, x_in, k_in, th2])
    #stacks NNLS 9 times
    for j in range(1, 10):
        x_kk = nnls(x_kk)
   
    #create final model, returns it and the first weight
    mod = keras.Model(inputs=[fr_in, y_in, x_in, k_in], outputs=[x_kk])   
    return mod

#%%
    
class Pipeline(object):
    
    def __init__(self, model, y_0, x_0, mc_0, tht2, tot, num_components, batch_size):
        """
        Inputs: the model from get_model, and the initial input values as numpy arrays (y_0, x_0, mc_0, tht2)
        To run, after initializing, run self.get_spikes()
        """
        self.model, self.mc0, self.y0, self.x0, self.tht2 = model, mc_0, y_0, x_0, tht2
        self.batch_size = batch_size
        self.num_components = num_components
        self.tot = tot
        self.dim_x, self.dim_y = self.mc0.shape[2], self.mc0.shape[3]
        self.zero_tensor = [[0.0]]
        
        self.frame_input_q = Queue()
        self.spike_input_q = Queue()
        self.output_q = Queue()
        
        #load estimator from the keras model
        self.estimator = self.load_estimator()
        
        #seed the queues
        self.frame_input_q.put(mc_0)
        self.spike_input_q.put((y_0, x_0))

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
                                                 output_shapes={"m":(1, self.batch_size, self.dim_x, self.dim_y, 1),
                                                                "y":(1, self.num_components, self.batch_size),
                                                                "x":(1, self.num_components, self.batch_size),
                                                                "k":(1, 1)})
        return dataset
    
    def generator(self):
        #generator waits until data has been enqueued, then yields the data to the dataset
        while True:
            fr = self.frame_input_q.get()
            out = self.spike_input_q.get()
            (y, x) = out
            yield {"m":fr, "y":y, "x":x, "k":self.zero_tensor}

    def get_spikes(self, bound):
        #to be called separately. Input "bound" represents the number of frames. Starts at one because of initial values put on queue.
        output = [0]*bound
        start = timeit.default_timer()
        for idx in range(self.batch_size, bound, self.batch_size):
#            st = timeit.default_timer()

            out = self.output_q.get()
            output[idx-1] = out["nnls_1"]
            self.frame_input_q.put(self.tot[idx:idx+self.batch_size, :, :, None][None, :])
            self.spike_input_q.put((out["nnls"], out["nnls_1"]))
#            output.append(timeit.default_timer()-st)
        output[-1] = self.output_q.get()["nnls_1"]
        print(timeit.default_timer()-start)
        return output
#%%
class MotionCorrect(keras.layers.Layer):
    def __init__(self, template, center_dims, batch_size, ms_h=10, ms_w=10, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
        """
        Tenforflow layer which perform motion correction on batches of frames. Notice that the input must be a 
        tensorflow tensor with dimension batch x width x height x channel
        Args:
           template: ndarray
               template against which to register
            center_dims: tuple of ints
                dimensions after cropping movie and template
           ms_h: int
               estimated minimum value of vertical shift
           ms_w: int
               estimated minimum value of horizontal shift    
           strides: list
               convolutional strides of tf.conv2D              
           padding: str
               "VALID" or "SAME": convolutional padding of tf.conv2D
           epsilon':float
               small value added to variances to prevent division by zero
        
        
        Returns:
           X_corrected batch corrected when called upon a batch of inputs
        
        """
        super().__init__(**kwargs)
        self.shp_x, self.shp_y = template.shape[0], template.shape[1]
        self.center_dims = center_dims
        self.c_shp_x, self.c_shp_y = (self.shp_x - center_dims[0])//2, (self.shp_y - center_dims[1])//2

        self.template_0 = template
        self.template=self.template_0[(ms_w+self.c_shp_x):-(ms_w+self.c_shp_x),(ms_h+self.c_shp_y):-(ms_h+self.c_shp_y), None, None]
        
        if self.template.shape[0] < 10 or self.template.shape[1] < 10:
            raise ValueError("The vertical or horizontal shift you entered is too large for the given video dimensions. Enter a smaller shift.")
        self.batch_size = batch_size

        self.ms_h = ms_h
        self.ms_w = ms_w
        self.strides = strides
        self.padding = padding
        self.epsilon = epsilon

        ## normalize template
        self.template_zm, self.template_var = self.normalize_template(self.template, epsilon=self.epsilon)
        
        ## assign to kernel, normalizer
        self.kernel = self.template_zm
        self.normalizer = self.template_var


    @tf.function
    def call(self, X):
        # takes as input a tensorflow batch tensor (batch x width x height x channel)
        X = X[0]
        X_center = X[:, self.c_shp_x:(self.shp_x-self.c_shp_x), self.c_shp_y:(self.shp_y-self.c_shp_y), :]

        # pass in center for normalization
        imgs_zm, imgs_var = self.normalize_image(X_center, self.template.shape, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon) 
        denominator = tf.sqrt(self.normalizer * imgs_var)
        numerator = tf.nn.conv2d(imgs_zm, self.kernel, padding=self.padding, 
                                 strides=self.strides)
#        
        tensor_ncc = tf.truediv(numerator, denominator)
##        self.kernel = self.kernel*1
##        self.normalizer = self.normalizer*1
       
        # Remove any NaN in final output
        tensor_ncc = tf.where(tf.math.is_nan(tensor_ncc), tf.zeros_like(tensor_ncc), tensor_ncc)
        xs, ys = self.extract_fractional_peak(tensor_ncc, ms_h=self.ms_h, ms_w=self.ms_w)
        X_corrected = tfa.image.translate(X, tf.squeeze(tf.stack([ys, xs], axis=1)), 
                                            interpolation="BILINEAR")

        return tf.reshape(tf.transpose(tf.squeeze(X_corrected, axis=3), perm=[0,2,1]), (self.batch_size, self.shp_x*self.shp_y))


    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template_0,"strides": self.strides, "batch_size":self.batch_size,
                "center_dims":self.center_dims,"padding": self.padding, "epsilon": self.epsilon, 
                                        "ms_h": self.ms_h,"ms_w": self.ms_w }  
        
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
        return tf.cast(template_zm, tf.float32), tf.cast(template_var, tf.float32)
        
    def normalize_image(self, imgs, shape_template, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed
        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2], keepdims=True)
        localsum_sq = tf.nn.conv2d(tf.square(imgs), tf.ones(shape_template), 
                                               padding=padding, strides=strides)
        localsum = tf.nn.conv2d(imgs,tf.ones(shape_template), 
                                               padding=padding, strides=strides)
        
        
        imgs_var = localsum_sq - tf.square(localsum)/np.prod(shape_template) + epsilon
        # Remove small machine precision errors after subtraction
        imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
        del localsum_sq, localsum
        return imgs_zm, imgs_var
    
    def argmax_2d(self, tensor):
        # extract peaks from 2D tensor (takes batches as input too)
        
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))

        argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
        # convert indexes into 2D coordinates
        argmax_x = tf.cast(argmax, tf.int32) // tf.shape(tensor)[2]
        argmax_y = tf.cast(argmax, tf.int32) % tf.shape(tensor)[2]
        # stack and return 2D coordinates
        return tf.cast(tf.stack((argmax_x, argmax_y), axis=1), tf.float32)
        
    def extract_fractional_peak(self, tensor_ncc, ms_h, ms_w):
        """ use gaussian interpolation to extract a fractional shift
        Args:
            tensor_ncc: tensor
                normalized cross-correlation
                ms_h: max integer shift vertical
                ms_w: max integere shift horizontal
        
        """
        shifts_int = self.argmax_2d(tensor_ncc)


        shifts_int_cast = tf.cast(shifts_int,tf.int64)
        sh_x, sh_y = shifts_int_cast[:,0],shifts_int_cast[:,1]
        
        sh_x_n = tf.cast(-(sh_x - ms_h), tf.float32)
        sh_y_n = tf.cast(-(sh_y - ms_w), tf.float32)
        
        tensor_ncc_log = tf.math.log(tensor_ncc)

        n_batches = np.arange(self.batch_size)

        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x-1,axis=1), tf.squeeze(sh_y,axis=1)]))
        log_xm1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x+1,axis=1), tf.squeeze(sh_y,axis=1)]))
        log_xp1_y = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=1), tf.squeeze(sh_y-1, axis=1)]))
        log_x_ym1 = tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=1), tf.squeeze(sh_y+1, axis=1)]))
        log_x_yp1 =  tf.gather_nd(tensor_ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x, axis=1), tf.squeeze(sh_y, axis=1)]))
        four_log_xy = 4 * tf.gather_nd(tensor_ncc_log, idx)

        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))

        return tf.reshape(sh_x_n, [self.batch_size, 1]), tf.reshape(sh_y_n, [self.batch_size, 1])
#%%
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
        pass as inputs the new Y, and the old X. see  https://angms.science/doc/NMF/nnls_pgd.pdf
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
#%%
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