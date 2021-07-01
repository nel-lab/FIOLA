# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Image warping using per-pixel flow vectors."""

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils import types
from typing import Optional
import tensorflow_probability as tfp
import pylab as plt

#%%
def _get_dim(x, idx):
    if x.shape.ndims is None:
        return tf.shape(x)[idx]
    return x.shape[idx] or tf.shape(x)[idx]

@tf.function
def dense_image_warp_3D(
    image: types.TensorLike, flow: types.TensorLike, name: Optional[str] = None
, dense=False) -> tf.Tensor:
    """Image warping using per-pixel flow vectors.
    Apply a non-linear warp to the image, where the warp is specified by a
    dense flow field of offset vectors that define the correspondences of
    pixel values in the output image back to locations in the source image.
    Specifically, the pixel value at `output[b, j, i, c]` is
    `images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c]`.
    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    `(b, j - flow[b, j, i, 0], i - flow[b, j, i, 1])`. For locations outside
    of the image, we use the nearest pixel values at the image boundary.
    NOTE: The definition of the flow field above is different from that
    of optical flow. This function expects the negative forward flow from
    output image to source image. Given two images `I_1` and `I_2` and the
    optical flow `F_12` from `I_1` to `I_2`, the image `I_1` can be
    reconstructed by `I_1_rec = dense_image_warp(I_2, -F_12)`.
    Args:
      image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
      flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
      name: A name for the operation (optional).
      Note that image and flow can be of type `tf.half`, `tf.float32`, or
      `tf.float64`, and do not necessarily have to be the same type.
    Returns:
      A 4-D float `Tensor` with shape`[batch, height, width, channels]`
        and same type as input image.
    Raises:
      ValueError: if `height < 2` or `width < 2` or the inputs have the wrong
        number of dimensions.
    """
    
    batch_size, height, width, depth, channels = (
        _get_dim(image, 0),
        _get_dim(image, 1),
        _get_dim(image, 2),
        _get_dim(image, 3),
        _get_dim(image, 4),
    )

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y, grid_z = tf.meshgrid( tf.range(width), tf.range(height) ,tf.range(depth))
    stacked_grid = tf.cast(tf.stack([grid_y, grid_x, grid_z], axis=3), flow.dtype)
    batched_grid = tf.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = tf.reshape(
        query_points_on_grid, [batch_size, height * width * depth, 3])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    if dense:
        interpolated = tfp.math.batch_interp_regular_nd_grid(query_points_flattened, [0,0,0], [height-1,width-1,depth-1], image, axis=-4)
    else:
        interpolated = trilinear_interpolate_tf(query_points_flattened, image)

    interpolated = tf.reshape(interpolated, [batch_size, height, width, depth, channels])
    return interpolated   


#%%
@tf.function
def trilinear_interpolate_tf(query_points_flattened, image):
    '''
    def trilinear(xyz, data):
    xyz: array with coordinates inside data
    data: 3d volume
    returns: interpolated data values at coordinates
    ijk = xyz.astype(np.int32)
    i, j, k = ijk[:,0], ijk[:,1], ijk[:,2]
    V000 = data[ i   , j   ,  k   ].astype(np.int32)
    V100 = data[(i+1), j   ,  k   ].astype(np.int32)
    V010 = data[ i   ,(j+1),  k   ].astype(np.int32)
    V001 = data[ i   , j   , (k+1)].astype(np.int32)
    V101 = data[(i+1), j   , (k+1)].astype(np.int32)
    V011 = data[ i   ,(j+1), (k+1)].astype(np.int32)
    V110 = data[(i+1),(j+1),  k   ].astype(np.int32)
    V111 = data[(i+1),(j+1), (k+1)].astype(np.int32)
    xyz = xyz - ijk
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    Vxyz = (V000 * (1 - x)*(1 - y)*(1 - z)
            + V100 * x * (1 - y) * (1 - z) +
            + V010 * (1 - x) * y * (1 - z) +
            + V001 * (1 - x) * (1 - y) * z +
            + V101 * x * (1 - y) * z +
            + V011 * (1 - x) * y * z +
            + V110 * x * y * (1 - z) +
            + V111 * x * y * z)
    return Vxyz  
    '''
    im = image[0,:,:,:,0]

    ijk = query_points_flattened[0]
    ijk0 = tf.cast(tf.floor(ijk), dtype=tf.int32)
    ijk1 = ijk0 + 1
    
    i0 = tf.clip_by_value(ijk0[:,0], 0, im.shape[0]-1)
    i1 = tf.clip_by_value(ijk1[:,0], 0, im.shape[0]-1)
    j0 = tf.clip_by_value(ijk0[:,1], 0, im.shape[1]-1)
    j1 = tf.clip_by_value(ijk1[:,1], 0, im.shape[1]-1)
    k0 = tf.clip_by_value(ijk0[:,2], 0, im.shape[2]-1)
    k1 = tf.clip_by_value(ijk1[:,2], 0, im.shape[2]-1)
        
    V000 = tf.gather_nd(im, tf.stack([i0,j0,k0], axis=1))
    V100 = tf.gather_nd(im, tf.stack([i1,j0,k0], axis=1))
    V010 = tf.gather_nd(im, tf.stack([i0,j1,k0], axis=1))
    V001 = tf.gather_nd(im, tf.stack([i0,j0,k1], axis=1))
    V101 = tf.gather_nd(im, tf.stack([i1,j0,k1], axis=1))
    V011 = tf.gather_nd(im, tf.stack([i0,j1,k1], axis=1))
    V110 = tf.gather_nd(im, tf.stack([i1,j1,k0], axis=1))
    V111 = tf.gather_nd(im, tf.stack([i1,j1,k1], axis=1))
    
    i0, j0, k0 =  tf.cast(i0, dtype=tf.float32), tf.cast(j0, dtype=tf.float32), tf.cast(k0, dtype=tf.float32)
    i1, j1, k1 =  tf.cast(i1, dtype=tf.float32), tf.cast(j1, dtype=tf.float32), tf.cast(k1, dtype=tf.float32)
    i, j, k = ijk[:,0], ijk[:,1], ijk[:,2]
    
    w000 = (i1-i) * (j1-j) * (k1-k)
    w100 = (i-i0) * (j1-j) * (k1-k)
    w010 = (i1-i) * (j-j0) * (k1-k)
    w001 = (i1-i) * (j1-j) * (k-k0)
    w101 = (i-i0) * (j1-j) * (k-k0)
    w011 = (i1-i) * (j-j0) * (k-k0)
    w110 = (i-i0) * (j-j0) * (k1-k)
    w111 = (i-i0) * (j-j0) * (k-k0)
   
    out = tf.transpose(tf.transpose(V000)*w000) \
          + tf.transpose(tf.transpose(V100)*w100) \
          + tf.transpose(tf.transpose(V010)*w010) \
          + tf.transpose(tf.transpose(V001)*w001) \
          + tf.transpose(tf.transpose(V101)*w101) \
          + tf.transpose(tf.transpose(V011)*w011) \
          + tf.transpose(tf.transpose(V110)*w110) \
          + tf.transpose(tf.transpose(V111)*w111)
          
          
    return out

#%% 3D
shape = np.array([500,300,100])
a = np.zeros(shape);
a[shape[0]//2,shape[1]//2,shape[2]//2] = 1
at = tf.convert_to_tensor(a[None,:,:,:,None], dtype=tf.float32)
image = at
flow = np.concatenate([2.2*np.ones((1,shape[0],shape[1],shape[2],1)), -3.3*np.ones((1,shape[0],shape[1],shape[2],1)),4.6*np.ones((1,shape[0],shape[1],shape[2],1))], axis=-1)
flow = tf.convert_to_tensor(flow, tf.float32)

at_t_3D= dense_image_warp_3D(at,flow, dense=False)
print(np.unravel_index(np.argmax(a), shape=shape))
print(np.unravel_index(np.argmax(at_t_3D), shape=shape))
i,j,k = np.unravel_index(np.argmax(at_t_3D), shape=shape)
print(np.max(at))
print(np.max(at_t_3D))
plt.figure()
plt.subplot(3,1,1)
plt.imshow(at_t_3D[0,i,:,:,0])    
plt.subplot(3,1,2)
plt.imshow(at_t_3D[0,:,j,:,0])
plt.subplot(3,1,3)
plt.imshow(at_t_3D[0,:,:,k,0])

# at_t_3D= dense_image_warp_3D(at,flow, dense=True)
# print(np.unravel_index(np.argmax(a), shape=shape))
# print(np.unravel_index(np.argmax(at_t_3D), shape=shape))
# i,j,k = np.unravel_index(np.argmax(at_t_3D), shape=shape)
# print(np.max(at))
# print(np.max(at_t_3D))
# plt.figure()
# plt.subplot(3,1,1)
# plt.imshow(at_t_3D[0,i,:,:,0])    
# plt.subplot(3,1,2)
# plt.imshow(at_t_3D[0,:,j,:,0])
# plt.subplot(3,1,3)
# plt.imshow(at_t_3D[0,:,:,k,0])