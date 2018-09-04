from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .data_loader import DataLoader
from .nets import *
from .utils import *
from .flowlib import flow_to_image

class BaseLearner(object):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        raise NotImplementedError

    def collect_summaries(self):
        raise NotImplementedError

    def train(self, opt):
        raise NotImplementedError

    # Credit: https://github.com/mrharicot/monodepth/blob/master/average_gradients.py
    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                if g is not None:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
            if grads != []:
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
        return average_grads

    def get_dp_flow(self, opt, s, src_pixel_coords):
        x_base = tf.range(int(opt.img_width/(2**s)))
        y_base = tf.range(int(opt.img_height/(2**s)))
        x_base = tf.stack([x_base]*int(opt.img_height/(2**s)), axis=0)
        y_base = tf.transpose(tf.stack([y_base]*int(opt.img_width/(2**s)), axis=0))

        dp_flow_x = src_pixel_coords[:, :, :, 0] - tf.cast(x_base, tf.float32)
        dp_flow_y = src_pixel_coords[:, :, :, 1] - tf.cast(y_base, tf.float32)
        dp_flow = tf.stack([dp_flow_x, dp_flow_y], axis=-1)
        return dp_flow

    def get_in_range_mask(self, opt, s, flow):
        # 1 if the displacement is within the image
        x_min = 0.0
        x_max = int(opt.img_width/(2**s))-1
        y_min = 0.0
        y_max = int(opt.img_height/(2**s))-1

        x_base = tf.range(int(opt.img_width/(2**s)))
        y_base = tf.range(int(opt.img_height/(2**s)))
        x_base = tf.stack([x_base]*int(opt.img_height/(2**s)), axis=0)
        y_base = tf.transpose(tf.stack([y_base]*int(opt.img_width/(2**s)), axis=0))

        pos_x = flow[:,:,:,0]+tf.cast(x_base, tf.float32)
        pos_y = flow[:,:,:,1]+tf.cast(y_base, tf.float32)
        inside_x = tf.logical_and(pos_x <= tf.cast(x_max, tf.float32), pos_x >= x_min)
        inside_y = tf.logical_and(pos_y <= tf.cast(y_max, tf.float32), pos_y >= y_min)
        inside = tf.expand_dims(tf.logical_and(inside_x, inside_y), axis=-1)
        return tf.stop_gradient(tf.cast(inside, tf.float32))

    def get_fb_mask(self, flow, warped_flow, alpha1=0.01, alpha2=0.5):
        temp1 = tf.reduce_sum(tf.square(flow+warped_flow), axis=3, keep_dims=True)
        temp2 = tf.reduce_sum(tf.square(flow), axis=3, keep_dims=True)+tf.reduce_sum(tf.square(warped_flow), axis=3, keep_dims=True)
        occ_mask = tf.greater(temp1, alpha1*temp2+alpha2)
        return tf.stop_gradient(tf.cast(occ_mask, tf.float32))

    # Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
    def ternary_loss(self, im1, im2_warped, valid_mask, max_distance=1):
        patch_size = 2*max_distance+1
        with tf.variable_scope('ternary_loss'):
            def _ternary_transform(image):
                intensities = tf.image.rgb_to_grayscale(image) * 255
                out_channels = patch_size * patch_size
                w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
                weights =  tf.constant(w, dtype=tf.float32)
                patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding='SAME')

                transf = patches - intensities
                transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
                return transf_norm

            def _hamming_distance(t1, t2):
                dist = tf.square(t1 - t2)
                dist_norm = dist / (0.1 + dist)
                dist_sum = tf.reduce_sum(dist_norm, 3, keep_dims=True)
                return dist_sum

        t1 = _ternary_transform(im1)
        t2 = _ternary_transform(im2_warped)
        dist = _hamming_distance(t1, t2)

        transform_mask = self.create_mask(valid_mask, [[max_distance, max_distance], [max_distance, max_distance]])
        return self.charbonnier_loss(dist, valid_mask * transform_mask), dist

    def charbonnier_loss(self, x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
        with tf.variable_scope('charbonnier_loss'):
            batch, height, width, channels = tf.unstack(tf.shape(x))
            normalization = tf.cast(batch * height * width * channels, tf.float32)

            error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

            if mask is not None:
                error = tf.multiply(mask, error)
            if truncate is not None:
                error = tf.minimum(error, truncate)

            return tf.reduce_sum(error) / normalization

    def create_mask(self, tensor, paddings):
        with tf.variable_scope('create_mask'):
            shape = tf.shape(tensor)
            inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
            inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
            inner = tf.ones([inner_width, inner_height])

            mask2d = tf.pad(inner, paddings)
            mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
            mask4d = tf.expand_dims(mask3d, 3)
            return tf.stop_gradient(mask4d)

    # Credit: https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def compute_edge_aware_smooth_loss(self, pred_disp, img):
        """
        Edge-aware L1-norm on first-order gradient
        """
        def gradient(pred):
            D_dx = -pred[:, :, 1:, :] + pred[:, :, :-1, :]
            D_dy = -pred[:, 1:, :, :] + pred[:, :-1, :, :]
            return D_dx, D_dy
        img_dx, img_dy = gradient(img)
        disp_dx, disp_dy = gradient(pred_disp)

        weight_x = tf.exp(-tf.reduce_mean(tf.abs(img_dx), 3, keep_dims=True))
        weight_y = tf.exp(-tf.reduce_mean(tf.abs(img_dy), 3, keep_dims=True))

        loss = tf.reduce_mean(weight_x*tf.abs(disp_dx)) + tf.reduce_mean(weight_y*tf.abs(disp_dy))
        return loss

    def compute_smooth_loss(self, pred_disp):
        """
        L1-norm on second-order gradient
        """
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))

    def flow_to_image_tf(self, flow):
        im_stack = []
        for i in range(self.opt.batch_size//self.opt.num_gpus):
            temp = tf.py_func(flow_to_image, [flow[i,:,:,:]], tf.uint8)
            im_stack.append(temp)
        return tf.stack(im_stack, axis=0)

    # Credit: https://github.com/yzcjtr/GeoNet/blob/master/geonet_model.py
    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keep_dims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net_res50(
                input_mc, is_training=False)
            pred_depth = [1./disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints

    # Forward-backward
    def build_pose_fb_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
            self.img_height, self.img_width * self.seq_length, 3], 
            name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _ = pose_net_fb(
                tgt_image, src_image_stack, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses[:, :, :6]    # Only the first half is used

    def preprocess_image(self, image, is_dp=True):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if is_dp:
            return image * 2. -1.
        else:
            mean = [104.920005, 110.1753, 114.785955]
            out = []
            for i in range(0, int(image.shape[-1]), 3):
                r = image[:,:,:,i] - mean[0]/255.
                g = image[:,:,:,i+1] - mean[1]/255.
                b = image[:,:,:,i+2] - mean[2]/255.
                out += [r, g, b]
            return tf.stack(out, axis=-1)

    def minus_imagenet_rgb(self, image):
        mean = [122.7717, 115.9465, 102.9801]
        image = tf.cast(image, tf.float32)
        out = []
        for i in range(0, int(image.shape[-1]), 3):
            r = image[:,:,:,i] - mean[0]
            g = image[:,:,:,i+1] - mean[1]
            b = image[:,:,:,i+2] - mean[2]
            out += [r, g, b]
        return tf.stack(out, axis=-1)

    def deprocess_image(self, image, is_dp=True):
        # Assuming input image is float32
        if is_dp:
            image = (image + 1.)/2.
        else:
            mean = [104.920005, 110.1753, 114.785955]
            r = image[:,:,:,0] + mean[0]/255.
            g = image[:,:,:,1] + mean[1]/255.
            b = image[:,:,:,2] + mean[2]/255.
            image = tf.stack([r, g, b], axis=-1)
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self, 
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_fb_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

if __name__ == '__main__':
    model = BaseLearner()
