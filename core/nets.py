from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

from .utils import flow_inverse_warp

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

def resize_like(inputs, ref, type='nearest'):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    if type == 'nearest':
        return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])
    elif type == 'bilinear':
        return tf.image.resize_bilinear(inputs, [rH.value, rW.value])

# Reference: https://github.com/sampepose/flownet2-tf/blob/master/src/utils.py
def pad(tensor, num=1):
    """
    Pads the given tensor along the height and width dimensions with `num` 0s on each side
    """
    return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")

def pad_4(tensor, u, b, l, r):
    return tf.pad(tensor, [[0, 0], [u, b], [l, r], [0, 0]], 'CONSTANT')

def antipad(tensor, num=1):
    """
    Performs a crop. "padding" for a deconvolutional layer (conv2d tranpose) removes
    padding from the output rather than adding it to the input.
    """
    batch, h, w, c = tensor.shape.as_list()
    return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])

def antipad_4(tensor, u, b, l, r):
    batch, h, w, c = tensor.shape.as_list()
    return tf.slice(tensor, begin=[0, u, l, 0], size=[batch, h - u - b, w - l - r, c])

# Reference: https://github.com/scaelles/OSVOS-TensorFlow/blob/master/osvos.py
def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])

# Reference: https://github.com/tensorflow/tensorflow/issues/4079
def LeakyReLU(x, leak=0.1, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + leak)
        f2 = 0.5 * (1.0 - leak)
        return f1 * x + f2 * abs(x)

# Both target->source and source->target
def pose_net_fb(tgt_image, src_image_stack, is_training=True, reuse=False):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_net') as sc:
        if reuse:
            sc.reuse_variables()
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
            cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
            # Double the number of channels
            pose_pred = slim.conv2d(cnv7, 6*num_source*2, [1, 1], scope='pred', 
                stride=1, normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            # Empirically we found that scaling by a small constant 
            # facilitates training.
            # 1st half: target->source, 2nd half: source->target
            pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6*2])
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, end_points

# helper functions
# Credit: https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
def resconv(x, num_layers, stride):
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    conv1 = slim.conv2d(x, num_layers, [1, 1], stride=1, activation_fn=tf.nn.elu)
    conv2 = slim.conv2d(conv1, num_layers, [3, 3], stride=stride, activation_fn=tf.nn.elu)
    conv3 = slim.conv2d(conv2, 4 * num_layers, [1, 1], stride=1, activation_fn=None)
    if do_proj:
        shortcut = slim.conv2d(x, 4* num_layers, [1, 1], stride=stride, activation_fn=None)
    else:
        shortcut = x
    return tf.nn.elu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out

def upsample_nn(x, ratio):
    s = tf.shape(x)
    h = s[1]
    w = s[2]
    return tf.image.resize_nearest_neighbor(x, [h*ratio, w*ratio])

def upconv(x, num_layers, kernal, scale):
    upsample = upsample_nn(x, scale)
    conv = slim.conv2d(upsample, num_layers, [kernal, kernal], stride=1, activation_fn=tf.nn.elu)
    return conv

def disp_net_res50(tgt_image, is_training=True, reuse=False, get_feature=False):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net_res50') as sc:
        if reuse:
            sc.reuse_variables()
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.elu,
                            outputs_collections=end_points_collection):
            # Encoder
            conv1 = slim.conv2d(tgt_image, 64, [7, 7], stride=2)      # 1/2
            pool1 = slim.max_pool2d(conv1, [3, 3], padding='SAME')    # 1/4
            conv2 = resblock(pool1, 64, 3)                            # 1/8
            conv3 = resblock(conv2, 128, 4)                           # 1/16
            conv4 = resblock(conv3, 256, 6)                           # 1/32
            conv5 = resblock(conv4, 512, 3)                           # 1/64

            # Decoder
            upconv6 = upconv(conv5, 512, 3, 2)                        # 1/32
            #upconv6 = slim.conv2d_transpose(conv5, 512, [3, 3], stride=2)
            upconv6 = resize_like(upconv6, conv4)
            concat6 = tf.concat([upconv6, conv4], 3)
            iconv6  = slim.conv2d(concat6, 512, [3, 3], stride=1)

            upconv5 = upconv(iconv6, 256, 3, 2)                       # 1/16
            #upconv5 = slim.conv2d_transpose(iconv6, 256, [3, 3], stride=2)
            upconv5 = resize_like(upconv5, conv3)
            concat5 = tf.concat([upconv5, conv3], 3)
            iconv5  = slim.conv2d(concat5, 256, [3, 3], stride=1)

            upconv4 = upconv(iconv5, 128, 3, 2)                       # 1/8
            #upconv4 = slim.conv2d_transpose(iconv5, 128, [3, 3], stride=2)
            upconv4 = resize_like(upconv4, conv2)
            concat4 = tf.concat([upconv4, conv2], 3)
            iconv4  = slim.conv2d(concat4, 128, [3, 3], stride=1)
            disp4   = DISP_SCALING * slim.conv2d(iconv4, 1, [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upconv3  = upconv(iconv4, 64, 3, 2)                       # 1/4
            #upconv3  = slim.conv2d_transpose(iconv4, 64, [3, 3], stride=2)
            upconv3  = resize_like(upconv3, pool1)
            disp4_up = resize_like(disp4_up, pool1)
            concat3  = tf.concat([upconv3, disp4_up, pool1], 3)
            iconv3   = slim.conv2d(concat3, 64, [3, 3], stride=1)
            disp3    = DISP_SCALING * slim.conv2d(iconv3, 1, [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upconv2  = upconv(iconv3, 32, 3, 2)                       # 1/2
            #upconv2  = slim.conv2d_transpose(iconv3, 32, [3, 3], stride=2)
            upconv2  = resize_like(upconv2, conv1)
            disp3_up = resize_like(disp3_up, conv1)
            concat2  = tf.concat([upconv2, disp3_up, conv1], 3)
            iconv2   = slim.conv2d(concat2, 32, [3, 3], stride=1)
            disp2    = DISP_SCALING * slim.conv2d(iconv2, 1, [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upconv1 = upconv(iconv2, 16, 3, 2)
            #upconv1 = slim.conv2d_transpose(iconv2, 16, [3, 3], stride=2)
            upconv1 = resize_like(upconv1, disp2_up)
            concat1 = tf.concat([upconv1, disp2_up], 3)
            iconv1  = slim.conv2d(concat1, 16, [3, 3], stride=1)
            disp1   = DISP_SCALING * slim.conv2d(iconv1, 1, [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP

            end_points = utils.convert_collection_to_dict(end_points_collection)

            if not get_feature:
                return [disp1, disp2, disp3, disp4], end_points
            else:
                return [disp1, disp2, disp3, disp4], conv5, end_points


