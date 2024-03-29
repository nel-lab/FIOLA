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
# # @tf.function
# def interpolate_bilinear(
#     grid: types.TensorLike,
#     query_points: types.TensorLike,
#     indexing: str = "ij",
#     name: Optional[str] = None,
# ) -> tf.Tensor:
#     """Similar to Matlab's interp2 function.
#     Finds values for query points on a grid using bilinear interpolation.
#     Args:
#       grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
#       query_points: a 3-D float `Tensor` of N points with shape
#         `[batch, N, 2]`.
#       indexing: whether the query points are specified as row and column (ij),
#         or Cartesian coordinates (xy).
#       name: a name for the operation (optional).
#     Returns:
#       values: a 3-D `Tensor` with shape `[batch, N, channels]`
#     Raises:
#       ValueError: if the indexing mode is invalid, or if the shape of the
#         inputs invalid.
#     """
#     if indexing != "ij" and indexing != "xy":
#         raise ValueError("Indexing mode must be 'ij' or 'xy'")

#     with tf.name_scope(name or "interpolate_bilinear"):
#         grid = tf.convert_to_tensor(grid)
#         query_points = tf.convert_to_tensor(query_points)

#         # grid shape checks
#         grid_static_shape = grid.shape
#         grid_shape = tf.shape(grid)
#         if grid_static_shape.dims is not None:
#             if len(grid_static_shape) != 4:
#                 raise ValueError("Grid must be 4D Tensor")
#             if grid_static_shape[1] is not None and grid_static_shape[1] < 2:
#                 raise ValueError("Grid height must be at least 2.")
#             if grid_static_shape[2] is not None and grid_static_shape[2] < 2:
#                 raise ValueError("Grid width must be at least 2.")
#         else:
#             with tf.control_dependencies(
#                 [
#                     tf.debugging.assert_greater_equal(
#                         grid_shape[1], 2, message="Grid height must be at least 2."
#                     ),
#                     tf.debugging.assert_greater_equal(
#                         grid_shape[2], 2, message="Grid width must be at least 2."
#                     ),
#                     tf.debugging.assert_less_equal(
#                         tf.cast(
#                             grid_shape[0] * grid_shape[1] * grid_shape[2],
#                             dtype=tf.dtypes.float32,
#                         ),
#                         np.iinfo(np.int32).max / 8.0,
#                         message="The image size or batch size is sufficiently "
#                         "large that the linearized addresses used by "
#                         "tf.gather may exceed the int32 limit.",
#                     ),
#                 ]
#             ):
#                 pass

#         # query_points shape checks
#         query_static_shape = query_points.shape
#         query_shape = tf.shape(query_points)
#         if query_static_shape.dims is not None:
#             if len(query_static_shape) != 3:
#                 raise ValueError("Query points must be 3 dimensional.")
#             query_hw = query_static_shape[2]
#             if query_hw is not None and query_hw != 2:
#                 raise ValueError("Query points last dimension must be 2.")
#         else:
#             with tf.control_dependencies(
#                 [
#                     tf.debugging.assert_equal(
#                         query_shape[2],
#                         2,
#                         message="Query points last dimension must be 2.",
#                     )
#                 ]
#             ):
#                 pass

#         batch_size, height, width, channels = (
#             grid_shape[0],
#             grid_shape[1],
#             grid_shape[2],
#             grid_shape[3],
#         )

#         num_queries = query_shape[1]

#         query_type = query_points.dtype
#         grid_type = grid.dtype

#         alphas = []
#         floors = []
#         ceils = []
#         index_order = [0, 1] if indexing == "ij" else [1, 0]
#         unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

#         for i, dim in enumerate(index_order):
#             with tf.name_scope("dim-" + str(dim)):
#                 queries = unstacked_query_points[dim]

#                 size_in_indexing_dimension = grid_shape[i + 1]

#                 # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
#                 # is still a valid index into the grid.
#                 max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
#                 min_floor = tf.constant(0.0, dtype=query_type)
#                 floor = tf.math.minimum(
#                     tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
#                 )
#                 int_floor = tf.cast(floor, tf.dtypes.int32)
#                 floors.append(int_floor)
#                 ceil = int_floor + 1
#                 ceils.append(ceil)

#                 # alpha has the same type as the grid, as we will directly use alpha
#                 # when taking linear combinations of pixel values from the image.
#                 alpha = tf.cast(queries - floor, grid_type)
#                 min_alpha = tf.constant(0.0, dtype=grid_type)
#                 max_alpha = tf.constant(1.0, dtype=grid_type)
#                 alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

#                 # Expand alpha to [b, n, 1] so we can use broadcasting
#                 # (since the alpha values don't depend on the channel).
#                 alpha = tf.expand_dims(alpha, 2)
#                 alphas.append(alpha)

#             flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
#             batch_offsets = tf.reshape(
#                 tf.range(batch_size) * height * width, [batch_size, 1]
#             )

#         # This wraps tf.gather. We reshape the image data such that the
#         # batch, y, and x coordinates are pulled into the first dimension.
#         # Then we gather. Finally, we reshape the output back. It's possible this
#         # code would be made simpler by using tf.gather_nd.
#         def gather(y_coords, x_coords, name):
#             with tf.name_scope("gather-" + name):
#                 linear_coordinates = batch_offsets + y_coords * width + x_coords
#                 gathered_values = tf.gather(flattened_grid, linear_coordinates)
#                 return tf.reshape(gathered_values, [batch_size, num_queries, channels])

#         # grab the pixel values in the 4 corners around each query point
#         top_left = gather(floors[0], floors[1], "top_left")
#         top_right = gather(floors[0], ceils[1], "top_right")
#         bottom_left = gather(ceils[0], floors[1], "bottom_left")
#         bottom_right = gather(ceils[0], ceils[1], "bottom_right")

#         # now, do the actual interpolation
#         with tf.name_scope("interpolate"):
#             interp_top = alphas[1] * (top_right - top_left) + top_left
#             interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
#             interp = alphas[0] * (interp_bottom - interp_top) + interp_top

#         return interp


# def _get_dim(x, idx):
#     if x.shape.ndims is None:
#         return tf.shape(x)[idx]
#     return x.shape[idx] or tf.shape(x)[idx]


# @tf.function
# def dense_image_warp(
#     image: types.TensorLike, flow: types.TensorLike, name: Optional[str] = None
# ) -> tf.Tensor:
#     """Image warping using per-pixel flow vectors.
#     Apply a non-linear warp to the image, where the warp is specified by a
#     dense flow field of offset vectors that define the correspondences of
#     pixel values in the output image back to locations in the source image.
#     Specifically, the pixel value at `output[b, j, i, c]` is
#     `images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c]`.
#     The locations specified by this formula do not necessarily map to an int
#     index. Therefore, the pixel value is obtained by bilinear
#     interpolation of the 4 nearest pixels around
#     `(b, j - flow[b, j, i, 0], i - flow[b, j, i, 1])`. For locations outside
#     of the image, we use the nearest pixel values at the image boundary.
#     NOTE: The definition of the flow field above is different from that
#     of optical flow. This function expects the negative forward flow from
#     output image to source image. Given two images `I_1` and `I_2` and the
#     optical flow `F_12` from `I_1` to `I_2`, the image `I_1` can be
#     reconstructed by `I_1_rec = dense_image_warp(I_2, -F_12)`.
#     Args:
#       image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
#       flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
#       name: A name for the operation (optional).
#       Note that image and flow can be of type `tf.half`, `tf.float32`, or
#       `tf.float64`, and do not necessarily have to be the same type.
#     Returns:
#       A 4-D float `Tensor` with shape`[batch, height, width, channels]`
#         and same type as input image.
#     Raises:
#       ValueError: if `height < 2` or `width < 2` or the inputs have the wrong
#         number of dimensions.
#     """
#     with tf.name_scope(name or "dense_image_warp"):
#         image = tf.convert_to_tensor(image)
#         flow = tf.convert_to_tensor(flow)
#         batch_size, height, width, channels = (
#             _get_dim(image, 0),
#             _get_dim(image, 1),
#             _get_dim(image, 2),
#             _get_dim(image, 3),
#         )

#         # The flow is defined on the image grid. Turn the flow into a list of query
#         # points in the grid space.
#         grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
#         stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
#         batched_grid = tf.expand_dims(stacked_grid, axis=0)
#         query_points_on_grid = batched_grid - flow
#         query_points_flattened = tf.reshape(
#             query_points_on_grid, [batch_size, height * width, 2]
#         )
#         # Compute values at the query points, then reshape the result back to the
#         # image grid.
#         interpolated = interpolate_bilinear(image, query_points_flattened)
#         interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])
#         return interpolated
#%%
@tf.function
def bilinear_interpolate_tf(querypoints, image):
    '''
    def bilinear_interpolate(im, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
    
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1
    
        x0 = np.clip(x0, 0, im.shape[1]-1);
        x1 = np.clip(x1, 0, im.shape[1]-1);
        y0 = np.clip(y0, 0, im.shape[0]-1);
        y1 = np.clip(y1, 0, im.shape[0]-1);
    
        Ia = im[ y0, x0 ]
        Ib = im[ y1, x0 ]
        Ic = im[ y0, x1 ]
        Id = im[ y1, x1 ]
    
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)
        
        return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T
    '''
    x = querypoints[0,:,1]
    y = querypoints[0,:,0]
    im = image[0,:,:,0]
    x0 = tf.cast(tf.floor(x),dtype=tf.int32)
    x1 = x0 + 1
    
    y0 = tf.cast(tf.floor(y),dtype=tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, im.shape[1]-1)
    x1 = tf.clip_by_value(x1, 0, im.shape[1]-1)
    y0 = tf.clip_by_value(y0, 0, im.shape[0]-1)
    y1 = tf.clip_by_value(y1, 0, im.shape[0]-1)
    
    # Ia = im[ y0, x0 ][0]
    Ia = tf.gather_nd(im, tf.stack([y0,x0], axis=1))
    # Ib = im[ y1, x0 ][0]
    Ib = tf.gather_nd(im, tf.stack([y1,x0], axis=1))
    # Ic = im[ y0, x1 ][0]
    Ic = tf.gather_nd(im, tf.stack([y0,x1], axis=1))
    # Id = im[ y1, x1 ][0]
    Id= tf.gather_nd(im, tf.stack([y1,x1], axis=1))
    
    
    x0 = tf.cast(x0, dtype=tf.float32)
    x1 = tf.cast(x1, dtype=tf.float32)
    y0 = tf.cast(y0, dtype=tf.float32)
    y1 = tf.cast(y1, dtype=tf.float32)
    
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    print([wa, wb, wc, wd])

    return tf.transpose(tf.transpose(Ia)*wa) + tf.transpose(tf.transpose(Ib)*wb) + tf.transpose(tf.transpose(Ic)*wc) + tf.transpose(tf.transpose(Id)*wd)
   
#%% my 2D FUNCTION 
def _get_dim(x, idx):
    if x.shape.ndims is None:
        return tf.shape(x)[idx]
    return x.shape[idx] or tf.shape(x)[idx]

@tf.function
def dense_image_warp_2D(
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
    # with tf.name_scope(name or "dense_image_warp_2D"):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    flow = tf.convert_to_tensor(flow, dtype=tf.float32)
    batch_size, height, width, channels = (
        _get_dim(image, 0),
        _get_dim(image, 1),
        _get_dim(image, 2),
        _get_dim(image, 3),
    )

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = tf.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = tf.reshape(
        query_points_on_grid, [batch_size, height * width, 2]
    )
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    # interpolated = interpolate_bilinear(image, query_points_flattened)
    if dense:
        interpolated = tfp.math.batch_interp_regular_nd_grid(query_points_flattened, [0,0], [height-1,width-1], image, axis=-3)         
    else:
        interpolated = bilinear_interpolate_tf(query_points_flattened, image)
    # 
    interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])
    return interpolated
#%% 2D
shape = np.array([512,256])
a = np.zeros(shape);
a[shape[0]//2,shape[1]//2] = 1
at = tf.convert_to_tensor(a[None,:,:,None], dtype=tf.float32)
image = at
flow = np.concatenate([2.24*np.ones((1,shape[0],shape[1],1)),-3.45*np.ones((1,shape[0],shape[1],1))], axis=-1)
flow = tf.convert_to_tensor(flow, dtype=tf.float32)
# at_t= dense_image_warp(at,flow)
at_t_ag= dense_image_warp_2D(at,flow)

# plt.figure()
# plt.imshow(tf.squeeze(at_t-at_t_ag))
# plt.colorbar()
plt.figure()
plt.imshow(tf.squeeze(at_t_ag))

