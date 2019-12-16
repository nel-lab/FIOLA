#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:48:46 2019

@author: agiovann
"""
#%%
from past.utils import old_div
from skimage import io
import tensorflow as tf
import numpy as np
import pylab as plt
import tensorflow_addons as tfa
import cv2
#%%
num_frames =100
a = io.imread('Sue_2x_3000_40_-46.tif')
template = np.median(a,axis=0)
# plt.imshow(template)
aa = tf.convert_to_tensor(a[0][None,:,:,None])
aa_batch = tf.convert_to_tensor(a[:num_frames,:,:,None])
temp_batch = tf.convert_to_tensor(np.repeat(template[None,20:-20,20:-20,None], num_frames, axis=0))
temp = tf.convert_to_tensor(template[20:-20,20:-20,None,None])
#%%
epsilon=0.00000001
# subtract template and image mean across [height, width] dimension
template_zm = temp - tf.reduce_mean(temp, axis=[0,1], keepdims=True)
# compute template variance   
template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon 
#%%
class TemplMatch(keras.layers.Layer):
    def __init__(self, template, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001, **kwargs):
        super().__init__(**kwargs)
        self.template_zm = temp - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        self.template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon 
        self.conv = lambda x, y: tf.nn.conv2d(x, y, padding=padding, strides=strides)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[self.template_zm.shape],
            initializer= tf.constant_initializer(self.template_zm))        
        super().build(batch_input_shape) # must be at the end

    def call(self, X):
        img_zm = X - tf.reduce_mean(X, axis=[1,2], keepdims=True)
        localsum_sq = conv(tf.square(X), tf.ones_like(self.template_zm))
        localsum = conv(img, tf.ones_like(self.template_zm))
        img_var = localsum_sq - tf.square(localsum)/template_numel + epsilon
        # Remove small machine precision errors after subtraction
        img_var = tf.where(img_var<0, tf.zeros_like(img_var), img_var)
    
        # compute 2D NCC
        denominator = tf.sqrt(self.template_var * img_var)
        numerator = self.conv(img_zm, template_zm)
        out = tf.truediv(numerator, denominator)
        # Remove any NaN in final output
        out = tf.where(tf.math.is_nan(out), tf.zeros_like(out), out)
        return out

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}
#%%
def my_ncc_normalizer(z): # 
    img_zm = img - tf.reduce_mean(img, axis=[1,2], keepdims=True)
    ones = tf.ones_like(template)
    localsum_sq = conv(tf.square(img), ones)
    localsum = conv(img, ones)
    img_var = localsum_sq - tf.square(localsum)/template_numel + epsilon
    
    denominator = tf.sqrt(template_var * img_var)
    numerator = conv(img_zm, template_zm)
    out = tf.truediv(numerator, denominator)
    # Remove any NaN in final output
    out = tf.where(tf.math.is_nan(out), tf.zeros_like(out), out)
    return tf.math.log(tf.exp(z) + 1.0)
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
    
    
