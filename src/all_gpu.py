#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 03:56:46 2019

@author: agiovann
"""
from past.utils import old_div
from skimage import io
import tensorflow as tf
import numpy as np
import pylab as plt
import tensorflow_addons as tfa
import cv2


def ncc2d(img, template, template_numel, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
    """
    perform 2D NCC
    Inputs: img: larger image of shape [batch=1, height, width, channel=1]
            template: template of shape [height, width, in_channel=1, out_channel=1]
    		template_numel: height * width
    """
    # define conv function
    conv = lambda x, y: tf.nn.conv2d(x, y, padding=padding, strides=strides)
    
    # subtract template and image mean across [height, width] dimension
    template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
    # compute template variance   
    template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
    
    # compute local image variance
    img_zm = img - tf.reduce_mean(img, axis=[1,2], keepdims=True)
    ones = tf.ones_like(template)
    localsum_sq = conv(tf.square(img), ones)
    localsum = conv(img, ones)
    img_var = localsum_sq - tf.square(localsum)/template_numel + epsilon

    # Remove small machine precision errors after subtraction
    img_var = tf.where(img_var<0, tf.zeros_like(img_var), img_var)

    # compute 2D NCC
    denominator = tf.sqrt(template_var * img_var)
    numerator = conv(img_zm, template_zm)
    out = tf.truediv(numerator, denominator)
    # Remove any NaN in final output
    out = tf.where(tf.math.is_nan(out), tf.zeros_like(out), out)

    return out


# ### test function 'ncc2d' ###
# img = tf.random_normal([1,22,22,1])
# temp = tf.random_normal([6,6,1,1])
# ncc_out = ncc2d(img, temp)

# with tf.Session() as sess:
# 	xcorr_out = sess.run(ncc_out)
# 	print(xcorr_out.shape)


def batch_ncc(img, template):
	"""
	NCC of 4D tensor per batch and channel dimension
	Inputs: img: larger image of shape [batch, height, width, channel]
	        template: template of shape [batch, height, width, channel]
	NOTE: img and template have the same batch and channel size
	"""

	i_shape = img.get_shape().as_list()
	t_shape = template.get_shape().as_list()
	B = i_shape[0] # number of batches
	# param. used for slicing img and template into batch*channel 2D slices
	num_slices_template = int(t_shape[0]*t_shape[-1])
	num_slices_img = int(i_shape[0]*i_shape[-1])
	assert(num_slices_template == num_slices_img)
	# transpose and shape
	template = tf.transpose(template, perm=[1,2,0,3]) # [H,W,B,C]
	img = tf.transpose(img, perm=[0,3,1,2]) # [B,C,H,W]
	Ht, Wt, Bt, Ct = tf.unstack(tf.shape(template))
	Bi, Ci, Hi, Wi = tf.unstack(tf.shape(img))
	template = tf.reshape(template, [Ht, Wt, 1, Bt*Ct])
	img = tf.reshape(img, [Bi*Ci, Hi, Wi, 1])
	# get slices per channel per batch
	template_slices = tf.split(value=template, num_or_size_splits=num_slices_template, axis=3)
	img_slices = tf.split(value=img, num_or_size_splits=num_slices_img, axis=0)
	# slice-wise NCC
	nt = int(t_shape[1]*t_shape[2])
	ncc_slices = [ncc2d(x, y, template_numel=nt) for x, y in zip(img_slices, template_slices)]
	# adjust final dimension
	ncc_out = tf.concat(values=ncc_slices, axis=3) # [1,Hout,Wout,B*C]
	ncc_out = tf.concat(tf.split(ncc_out, B, axis=3), axis=0) # [B,Hout,Wout,C]
	ncc_out = tf.expand_dims(tf.reduce_mean(ncc_out, axis=3), axis=3) # [B,Hout,Wout,1]

	return ncc_out

def get_local_maxima(in_tensor):
  max_pooled_in_tensor = tf.nn.pool(in_tensor, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
  maxima = tf.where(tf.equal(in_tensor, max_pooled_in_tensor), in_tensor, tf.zeros_like(in_tensor))
  return maxima

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
#%%
num_frames =3000
a = io.imread('Sue_2x_3000_40_-46.tif')
template = np.median(a,axis=0)
# plt.imshow(template)
aa = tf.convert_to_tensor(a[0][None,:,:,None])
aa_batch = tf.convert_to_tensor(a[:num_frames,:,:,None])
temp_batch = tf.convert_to_tensor(np.repeat(template[None,20:-20,20:-20,None], num_frames, axis=0))
temp = tf.convert_to_tensor(template[20:-20,20:-20,None,None])
#%%
# cv = ncc2d(aa, temp, np.prod(temp.shape))
# plt.imshow(cv.numpy().squeeze())
#%%
cvv = batch_ncc(aa_batch , temp_batch)
shifts = tf.squeeze(argmax_2d(cvv))
#%%
plt.plot(shifts )
#%%
# for i in range(num_frames):
#     # plt.imshow(a[i])
#     plt.imshow(tf.squeeze(cvv[i]))
#     plt.pause(.03)
#     plt.cla()
#%%
oc_sh = []
for fr in a:
    res = cv2.matchTemplate(fr, np.array(tf.squeeze(temp)), cv2.TM_CCORR_NORMED)
    top_left = cv2.minMaxLoc(res)[3]
    sh_y, sh_x = top_left
    h_i, w_i,_,_ = temp.shape
    ms_h = 20
    ms_w = 20
    if (0 < top_left[1] < 2 * ms_h - 1) & (0 < top_left[0] < 2 * ms_w - 1):
        # if max is internal, check for subpixel shift using gaussian
        # peak registration
        log_xm1_y = np.log(res[sh_x - 1, sh_y])
        log_xp1_y = np.log(res[sh_x + 1, sh_y])
        log_x_ym1 = np.log(res[sh_x, sh_y - 1])
        log_x_yp1 = np.log(res[sh_x, sh_y + 1])
        four_log_xy = 4 * np.log(res[sh_x, sh_y])

        sh_x_n = -(sh_x - ms_h + old_div((log_xm1_y - log_xp1_y),
                                         (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y)))
        sh_y_n = -(sh_y - ms_w + old_div((log_x_ym1 - log_x_yp1),
                                         (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1)))
    else:
        sh_x_n = -(sh_x - ms_h)
        sh_y_n = -(sh_y - ms_w)
        
    oc_sh.append([sh_x_n,sh_y_n])
    
#%%
aa_batch_shift = tfa.image.translate(aa_batch, -(shifts-20), interpolation="BILINEAR")   
for i in range(6):
    plt.subplot(121)
    plt.cla()
    plt.imshow(a[i])
    plt.subplot(122)
    plt.imshow(tf.squeeze(aa_batch_shift[i]))
    plt.pause(.03)
    plt.cla()
