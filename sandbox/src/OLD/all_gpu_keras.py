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

from past.utils import old_div
from skimage import io
import numpy as np
import pylab as plt
import cv2
#%%
num_frames = 300
a = io.imread('Sue_2x_3000_40_-46.tif')
template = np.median(a,axis=0)
# plt.imshow(template)
aa = tf.convert_to_tensor(a[0][None,:,:,None])
aa_batch = tf.convert_to_tensor(a[:num_frames,:,:,None])
temp_batch = tf.convert_to_tensor(np.repeat(template[None,20:-20,20:-20,None], num_frames, axis=0))
temp = tf.convert_to_tensor(template[20:-20,20:-20,None,None])
#%%
class TemplMatch(keras.layers.Layer):
    def __init__(self, template, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
        super().__init__(**kwargs)
        self.strides = strides
        self.padding = padding
        self.epsilon = epsilon
        self.template_numel = np.prod(template.shape)
        self.template_zm = temp - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        self.template_var = tf.reduce_sum(tf.square(self.template_zm), axis=[0,1], keepdims=True) + epsilon 
        self.conv = lambda x, y: tf.nn.conv2d(x, y, padding=padding, strides=strides)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[*self.template_zm.shape.as_list()],
            initializer=tf.constant_initializer(self.template_zm.numpy()))
        self.normalizer = self.add_weight(
            name="normalizer", shape=[*self.template_var.shape.as_list()],
            initializer=tf.constant_initializer(self.template_var.numpy()))   
        super().build(batch_input_shape) # must be at the end

    def call(self, X):
        img_zm = X - tf.reduce_mean(X, axis=[1,2], keepdims=True)
        localsum_sq = self.conv(tf.square(X), tf.ones_like(self.template_zm))
        localsum = self.conv(X, tf.ones_like(self.template_zm))
        img_var = localsum_sq - tf.square(localsum)/self.template_numel + self.epsilon
        # Remove small machine precision errors after subtraction
        img_var = tf.where(img_var<0, tf.zeros_like(img_var), img_var)
        # compute 2D NCC
        denominator = tf.sqrt(self.normalizer * img_var)
        numerator = self.conv(img_zm, self.kernel)
        out = tf.truediv(numerator, denominator)
        # Remove any NaN in final output
        out = tf.where(tf.math.is_nan(out), tf.zeros_like(out), out)
        
        return out


   
    # def compute_output_shape(self, batch_input_shape):
    #     return tf.TensorShape(batch_input_shape.as_list()[:-1] + [1])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "strides": self.strides,
                "padding": self.padding, "epsilon": self.epsilon }
#%%
mod = TemplMatch(temp)
ncc = mod(aa_batch)
#%%
def normalize_template(template, epsilon=0.00000001):
    template_zm = temp - tf.reduce_mean(template, axis=[0,1], keepdims=True)
    template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
    return template_zm, template_var
    
def normalize_image(imgs, shape_template, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
    imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2], keepdims=True)
    localsum_sq = tf.nn.conv2d(tf.square(imgs), tf.ones(shape_template), 
                                           padding=padding, strides=strides)
    localsum = tf.nn.conv2d(imgs,tf.ones(shape_template), 
                                           padding=padding, strides=strides)
    
    
    imgs_var = localsum_sq - tf.square(localsum)/np.prod(shape_template) + epsilon
    # Remove small machine precision errors after subtraction
    imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
    return imgs_zm, imgs_var

def compute_ncc(imgs_zm, imgs_var, template_zm, template_var, strides=[1,1,1,1], padding='VALID'):
    denominator = tf.sqrt(tf.multiply(template_var,imgs_var))
    numerator = tf.nn.conv2d(imgs_zm, template_zm, 
                                           padding=padding, strides=strides)
    ncc = tf.truediv(numerator, denominator)
    # Remove any NaN in final output
    ncc = tf.where(tf.math.is_nan(ncc), tf.zeros_like(ncc), ncc)    
    return ncc
    
def argmax_2d(tensor):

  # flatten the Tensor along the height and width axes
  flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))

  # argmax of the flat tensor
  argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

  # convert indexes into 2D coordinates
  argmax_x = argmax // tf.shape(tensor)[2]
  argmax_y = argmax % tf.shape(tensor)[2]
  
  # stack and return 2D coordinates
  return tf.cast(tf.stack((argmax_x, argmax_y), axis=1), tf.float32)

def interpolate_peak(tensor_ncc, ms_h=20, ms_w=20):
    shifts_int = argmax_2d(tensor_ncc)
    shifts_int_cast = tf.cast(shifts_int,tf.int32)
    sh_x, sh_y = shifts_int_cast[:,0],shifts_int_cast[:,1]
    h_i, w_i,_,_ = temp.shape
    
    sh_x_n = tf.cast(-(sh_x - ms_h), tf.float32)
    sh_y_n = tf.cast(-(sh_y - ms_w), tf.float32)
    
    tensor_ncc_log = tf.math.log(tensor_ncc)      
    
    n_batches = np.arange(tensor_ncc_log.shape[0])
    
    idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x-1), tf.squeeze(sh_y)]))
    log_xm1_y = tf.gather_nd(tensor_ncc_log, idx)
    idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x+1), tf.squeeze(sh_y)]))
    log_xp1_y = tf.gather_nd(tensor_ncc_log, idx)
    idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x), tf.squeeze(sh_y-1)]))
    log_x_ym1 = tf.gather_nd(tensor_ncc_log, idx)
    idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x), tf.squeeze(sh_y+1)]))
    log_x_yp1 =  tf.gather_nd(tensor_ncc_log, idx)
    idx = tf.transpose(tf.stack([n_batches, tf.squeeze(sh_x), tf.squeeze(sh_y)]))
    four_log_xy = 4 * tf.gather_nd(tensor_ncc_log, idx)
    
    sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
    sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
    
    return sh_x_n, sh_y_n
#%%
t_zm, t_var = normalize_template(temp)
i_zm, i_var = normalize_image(aa_batch, temp.shape)
cc_img = compute_ncc(i_zm, i_var, t_zm, t_var)
xs, ys = interpolate_peak(cc_img)
aa_batch_shift = tfa.image.translate(aa_batch, tf.squeeze(tf.stack([ys,xs], axis=1)) , interpolation="BILINEAR")     

#%%
min_, max_ = np.min(aa_batch), np.max(aa_batch)
for fr, fr_raw in zip(aa_batch_shift, aa_batch):
    # Our operations on the frame come here
    gray = np.concatenate((fr.numpy().squeeze(), fr_raw.numpy().squeeze()))
    # Display the resulting frame
    cv2.imshow('frame', (gray-min_)/(max_-min_)*10)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
#%%
model = keras.models.Sequential([
    keras.layers.conv2d(1, kernel=temp.shape[:2] , activation="elu", kernel_initializer=tf.constant_initializer(temp)),
    keras.layers.Dense(1, kernel_regularizer=l2_reg)
])

#%%
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]
#%%        
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]

#%%
for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train_scaled, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)
        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()
    
    
