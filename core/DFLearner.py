from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .BaseLearner import BaseLearner
from .data_loader import DataLoader
from .nets import *
from .utils import *
from .flowlib import flow_to_image
from .UnFlow import flownet

FLOW_SCALE = 5.0
EPS = 1e-3

class DFLearner(BaseLearner):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        opt = self.opt
        with tf.device('/cpu:0'):
            loader = DataLoader(opt.dataset_dir,
                                opt.batch_size,
                                opt.img_height,
                                opt.img_width,
                                opt.num_source,
                                opt.num_scales)
            with tf.name_scope("data_loading"):
                tgt_image, src_image_stack, intrinsics, tgt_image_aug, src_image_stack_aug = loader.load_train_batch()
                # Feed photometric augmented image into depth network, [-1, 1]
                tgt_image_dp = self.preprocess_image(tgt_image_aug)
                src_image_stack_dp = self.preprocess_image(src_image_stack_aug)
                # Feed photometric augmented image into flow network, minus channel mean
                tgt_image_flow = self.preprocess_image(tgt_image_aug, is_dp=False)
                src_image_stack_flow = self.preprocess_image(src_image_stack_aug, is_dp=False)               
                # Normal image is used to compute loss, [0, 1]
                tgt_image = tf.image.convert_image_dtype(tgt_image, dtype=tf.float32)
                src_image_stack = tf.image.convert_image_dtype(src_image_stack, dtype=tf.float32)

                # Downsample images for depth network, and images for loss computation
                tf_size = tf.constant([int(opt.img_height/2), int(opt.img_width/2)], dtype=tf.int32)
                tgt_image_dp = tf.image.resize_area(tgt_image_dp, size=tf_size)
                src_image_stack_dp = tf.image.resize_area(src_image_stack_dp, size=tf_size)
                tgt_image = tf.image.resize_area(tgt_image, size=tf_size)
                src_image_stack = tf.image.resize_area(src_image_stack, size=tf_size)
                up = intrinsics[:,:,:2,:]/2.
                bottom = intrinsics[:,:,2:3,:]
                intrinsics = tf.concat([up, bottom], axis=2)
                # split for each gpu
                tgt_image_dp_splits = tf.split(tgt_image_dp, opt.num_gpus, 0)
                src_image_stack_dp_splits = tf.split(src_image_stack_dp, opt.num_gpus, 0)
                tgt_image_flow_splits = tf.split(tgt_image_flow, opt.num_gpus, 0)
                src_image_stack_flow_splits = tf.split(src_image_stack_flow, opt.num_gpus, 0)
                tgt_image_splits = tf.split(tgt_image, opt.num_gpus, 0)
                src_image_stack_splits = tf.split(src_image_stack, opt.num_gpus, 0)
                intrinsics_splits = tf.split(intrinsics, opt.num_gpus, 0)

            with tf.name_scope("train_op"):
                ## Learning rate decay
                boundaries = list(range(20000, opt.max_steps-1, 100000))
                values = [opt.learning_rate/(2**s) for s in range(len(boundaries)+1)]
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                lr = tf.train.piecewise_constant(self.global_step, boundaries, values)
                optim = tf.train.AdamOptimizer(lr, opt.beta1)

            tower_grads = []
            tower_dp_pixel_losses = []
            tower_dp_smooth_losses = []
            tower_depth_consistency_losses = []
            tower_flow_pixel_losses = []
            tower_flow_smooth_losses = []
            tower_flow_consistency_losses = []
            tower_cross_consistency_losses = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(opt.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        losses, grads = self.single_tower_operation(optim, tgt_image_dp_splits[i], src_image_stack_dp_splits[i], tgt_image_flow_splits[i], src_image_stack_flow_splits[i], tgt_image_splits[i], src_image_stack_splits[i], intrinsics_splits[i], model_idx=i)
                        dp_pixel_loss, dp_smooth_loss, depth_consistency_loss, flow_pixel_loss, flow_smooth_loss, flow_consistency_loss, cross_consistency_loss = losses
                        tower_dp_pixel_losses.append(dp_pixel_loss)
                        tower_dp_smooth_losses.append(dp_smooth_loss)
                        tower_depth_consistency_losses.append(depth_consistency_loss)
                        tower_flow_pixel_losses.append(flow_pixel_loss)
                        tower_flow_smooth_losses.append(flow_smooth_loss)
                        tower_flow_consistency_losses.append(flow_consistency_loss)
                        tower_cross_consistency_losses.append(cross_consistency_loss)
                        tower_grads.append(grads)

            grads = self.average_gradients(tower_grads)
            # Gradient clip
            grads = [(tf.clip_by_norm(grad, 10.), var) for grad, var in grads]
            self.train_op = optim.apply_gradients(grads, global_step=self.global_step)

            dp_pixel_loss = tf.reduce_mean(tower_dp_pixel_losses)
            dp_smooth_loss = tf.reduce_mean(tower_dp_smooth_losses)
            depth_consistency_loss = tf.reduce_mean(tower_depth_consistency_losses)
            flow_pixel_loss = tf.reduce_mean(tower_flow_pixel_losses)
            flow_smooth_loss = tf.reduce_mean(tower_flow_smooth_losses)
            flow_consistency_loss = tf.reduce_mean(tower_flow_consistency_losses)
            cross_consistency_loss = tf.reduce_mean(tower_cross_consistency_losses)
            total_loss = dp_pixel_loss+dp_smooth_loss+depth_consistency_loss+flow_pixel_loss+flow_smooth_loss+flow_consistency_loss+cross_consistency_loss
            # Collect tensors that are useful later (e.g. tf summary)
            self.steps_per_epoch = loader.steps_per_epoch
            self.total_loss = total_loss
            self.dp_pixel_loss = dp_pixel_loss
            self.dp_smooth_loss = dp_smooth_loss
            self.depth_consistency_loss = depth_consistency_loss
            self.flow_pixel_loss = flow_pixel_loss
            self.flow_smooth_loss = flow_smooth_loss
            self.flow_consistency_loss = flow_consistency_loss
            self.cross_consistency_loss = cross_consistency_loss

    def single_tower_operation(self, optim, tgt_image_dp, src_image_stack_dp, tgt_image_flow, src_image_stack_flow, tgt_image, src_image_stack, intrinsics, model_idx=0):
        opt = self.opt
        if model_idx > 0:
            reuse_variables = True
        else:
            reuse_variables = False

        with tf.name_scope("depth_prediction"):
            tgt_pred_disp, _ = disp_net_res50(tgt_image_dp, is_training=True, reuse=reuse_variables)
            if opt.scale_normalize:
                tgt_pred_disp = [self.spatial_normalize(disp) for disp in tgt_pred_disp]
            tgt_pred_depth = [1./d for d in tgt_pred_disp]
            src_pred_disps = []
            src_pred_depths = []
            for i in range(opt.num_source):
                temp_disp, _ = disp_net_res50(src_image_stack_dp[:,:,:,3*i:3*(i+1)], is_training=True, reuse=True)
                if opt.scale_normalize:
                    temp_disp = [self.spatial_normalize(disp) for disp in temp_disp]
                src_pred_disps.append(temp_disp)
                src_pred_depths.append([1./d for d in temp_disp])

        with tf.name_scope("pose_prediction"):
            pred_poses, _ = pose_net_fb(tgt_image_dp, src_image_stack_dp, is_training=True, reuse=reuse_variables)
            if opt.fix_pose:
                pred_poses = tf.stop_gradient(pred_poses)

        with tf.name_scope("flow_prediction"):
            pred_flows_tgt2src = []
            pred_flows_src2tgt = []
            for i in range(opt.num_source):
                if i > 0:
                    reuse_variables = True
                tgt2src, src2tgt = flownet(tgt_image_flow, src_image_stack_flow[:,:,:,3*i:3*(i+1)], flownet_spec='C', backward_flow=True, reuse=reuse_variables)
                # Only take the last stack of flownet (i.e. 2nd 'S' in 'CSS')
                pred_flows_tgt2src.append(tgt2src[-1])
                pred_flows_src2tgt.append(src2tgt[-1])

        # For census transform
        max_dists = [3, 2, 2, 1, 1]
        layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
        with tf.name_scope("compute_loss"):
            dp_pixel_loss = 0
            dp_smooth_loss = 0
            depth_consistency_loss = 0
            flow_pixel_loss = 0
            flow_smooth_loss = 0
            flow_consistency_loss = 0
            cross_consistency_loss = 0
            if model_idx == 0:
                tgt_image_all = []
                src_image_stack_all = []
                # For dp
                proj_image_src2tgt_stack_all = []
                proj_depth_src2tgt_stack_all = []
                proj_error_tgt_stack_all = []
                proj_image_tgt2src_stack_all = []
                proj_depth_tgt2src_stack_all = []
                proj_error_src_stack_all = []
                dp_inrange_mask_tgt_stack_all = []
                dp_inrange_mask_src_stack_all = []
                dp_occ_mask_tgt_stack_all = []
                dp_occ_mask_src_stack_all = []
                # For flow
                flow_image_src2tgt_stack_all = []
                flow_image_tgt2src_stack_all = []
                flow_src2tgt_stack_all = []
                flow_tgt2src_stack_all = []
                flow_error_tgt_stack_all = []
                flow_error_src_stack_all = []
                flow_inrange_mask_tgt_stack_all = []
                flow_inrange_mask_src_stack_all = []
                flow_occ_mask_tgt_stack_all = []
                flow_occ_mask_src_stack_all = []
            for s in range(opt.num_scales):
                # Scale the source and target images for computing loss at the 
                # according scale.
                curr_tgt_image = tf.image.resize_area(tgt_image, 
                    [int(opt.img_height/(2*2**s)), int(opt.img_width/(2*2**s))])                
                curr_src_image_stack = tf.image.resize_area(src_image_stack, 
                    [int(opt.img_height/(2*2**s)), int(opt.img_width/(2*2**s))])

                if opt.smooth_weight > 0 and s < 4:
                    # Edge-aware first-order
                    dp_smooth_loss += opt.smooth_weight*self.compute_edge_aware_smooth_loss(tgt_pred_disp[s], curr_tgt_image)
                    for i in range(opt.num_source):
                        dp_smooth_loss += opt.smooth_weight*self.compute_edge_aware_smooth_loss(src_pred_disps[i][s], curr_src_image_stack[:,:,:,3*i:3*(i+1)])

                for i in range(opt.num_source):
                    ### For DP (Only 4 scales)
                    if s < 4:
                        # Pose: tgt->src, warp src->tgt
                        curr_proj_image_src2tgt, coords_tgt2src = projective_inverse_warp(
                            curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                            tf.squeeze(tgt_pred_depth[s], axis=3), 
                            pred_poses[:,i,:6], 
                            intrinsics[:,s,:,:])
                        curr_proj_depth_src2tgt = bilinear_sampler(src_pred_depths[i][s], coords_tgt2src)

                        # Pose: src->tgt, warp tgt->src
                        curr_proj_image_tgt2src, coords_src2tgt = projective_inverse_warp(
                            curr_tgt_image,
                            tf.squeeze(src_pred_depths[i][s], axis=3),
                            pred_poses[:,i,6:],
                            intrinsics[:,s,:,:])
                        curr_proj_depth_tgt2src = bilinear_sampler(tgt_pred_depth[s], coords_src2tgt)

                        # Occlusion mask (and in-range mask)
                        curr_dp_flow_tgt2src = self.get_dp_flow(opt, s+1, coords_tgt2src)
                        curr_dp_flow_src2tgt = self.get_dp_flow(opt, s+1, coords_src2tgt)

                        curr_inrange_mask_tgt = self.get_in_range_mask(opt, s+1, curr_dp_flow_tgt2src)
                        curr_inrange_mask_src = self.get_in_range_mask(opt, s+1, curr_dp_flow_src2tgt)

                        # Forward-backward occlusion check, cannot apply on pre-training
                        curr_dp_flow_tgt2src_by_src2tgt = bilinear_sampler(curr_dp_flow_src2tgt, coords_tgt2src)
                        curr_dp_flow_src2tgt_by_tgt2src = bilinear_sampler(curr_dp_flow_tgt2src, coords_src2tgt)
                        curr_occ_mask_tgt = self.get_fb_mask(curr_dp_flow_tgt2src, curr_dp_flow_tgt2src_by_src2tgt)
                        curr_occ_mask_src = self.get_fb_mask(curr_dp_flow_src2tgt, curr_dp_flow_src2tgt_by_tgt2src)
                        curr_valid_mask_tgt = curr_inrange_mask_tgt*(1-curr_occ_mask_tgt)
                        curr_valid_mask_src = curr_inrange_mask_src*(1-curr_occ_mask_src)

                        # Pixel loss
                        curr_proj_error_tgt = tf.abs(curr_proj_image_src2tgt-curr_tgt_image)
                        curr_proj_error_src = tf.abs(curr_proj_image_tgt2src-curr_src_image_stack[:,:,:,3*i:3*(i+1)])
                        curr_pixel_loss, dp_proj_error_tgt = self.ternary_loss(curr_tgt_image, curr_proj_image_src2tgt, curr_valid_mask_tgt, max_dists[s+1])
                        dp_pixel_loss += curr_pixel_loss*layer_weights[s]
                        curr_pixel_loss, dp_proj_error_src = self.ternary_loss(curr_src_image_stack[:,:,:,3*i:3*(i+1)], curr_proj_image_tgt2src, curr_valid_mask_src, max_dists[s+1])
                        dp_pixel_loss += curr_pixel_loss*layer_weights[s]

                        # Forward-backward depth consistency loss
                        depth_error_tgt = tf.abs(curr_proj_depth_src2tgt-tgt_pred_depth[s])
                        depth_consistency_loss += opt.depth_consistency*tf.reduce_sum(depth_error_tgt*curr_valid_mask_tgt)/(tf.reduce_sum(curr_valid_mask_tgt)+EPS)
                        depth_error_src = tf.abs(curr_proj_depth_tgt2src-src_pred_depths[i][s])
                        depth_consistency_loss += opt.depth_consistency*tf.reduce_sum(depth_error_src*curr_valid_mask_src)/(tf.reduce_sum(curr_valid_mask_src)+EPS)

                    ### For flow
                    # 2x upsample flow, to match size with depth
                    curr_flow_tgt2src = resize_like(pred_flows_tgt2src[i][s], curr_tgt_image, type='bilinear')*FLOW_SCALE*2.0
                    curr_flow_src2tgt = resize_like(pred_flows_src2tgt[i][s], curr_tgt_image, type='bilinear')*FLOW_SCALE*2.0

                    if opt.flow_smooth_weight > 0:
                        # 2nd order
                        flow_smooth_loss += layer_weights[s]*opt.flow_smooth_weight*self.compute_smooth_loss(curr_flow_tgt2src)
                        flow_smooth_loss += layer_weights[s]*opt.flow_smooth_weight*self.compute_smooth_loss(curr_flow_src2tgt)

                    flow_proj_image_src2tgt = flow_inverse_warp(curr_src_image_stack[:,:,:,3*i:3*(i+1)], curr_flow_tgt2src)
                    flow_proj_image_tgt2src = flow_inverse_warp(curr_tgt_image, curr_flow_src2tgt)

                    # Occlusion
                    curr_flow_tgt2src_by_src2tgt = flow_inverse_warp(curr_flow_src2tgt, curr_flow_tgt2src)
                    curr_flow_src2tgt_by_tgt2src = flow_inverse_warp(curr_flow_tgt2src, curr_flow_src2tgt)

                    curr_flow_inrange_mask_tgt = self.get_in_range_mask(opt, s+1, curr_flow_tgt2src)
                    curr_flow_inrange_mask_src = self.get_in_range_mask(opt, s+1, curr_flow_src2tgt)

                    # Forward-backward occlusion check, cannot apply on pre-training
                    curr_flow_occ_mask_tgt = self.get_fb_mask(curr_flow_tgt2src, curr_flow_tgt2src_by_src2tgt)
                    curr_flow_occ_mask_src = self.get_fb_mask(curr_flow_src2tgt, curr_flow_src2tgt_by_tgt2src)
                    curr_flow_valid_mask_tgt = curr_flow_inrange_mask_tgt*(1-curr_flow_occ_mask_tgt)
                    curr_flow_valid_mask_src = curr_flow_inrange_mask_src*(1-curr_flow_occ_mask_src)

                    # census transform
                    flow_proj_error_tgt = tf.abs(flow_proj_image_src2tgt-curr_tgt_image)
                    flow_proj_error_src = tf.abs(flow_proj_image_tgt2src-curr_src_image_stack[:,:,:,3*i:3*(i+1)])
                    curr_pixel_loss, flow_proj_error_tgt = self.ternary_loss(curr_tgt_image, flow_proj_image_src2tgt, curr_flow_valid_mask_tgt, max_dists[s])
                    flow_pixel_loss += layer_weights[s]*curr_pixel_loss
                    curr_pixel_loss, flow_proj_error_src = self.ternary_loss(curr_src_image_stack[:,:,:,3*i:3*(i+1)], flow_proj_image_tgt2src, curr_flow_valid_mask_src, max_dists[s])
                    flow_pixel_loss += layer_weights[s]*curr_pixel_loss

                    ### (Occlusion-aware) Flow consistency loss
                    # curr_flow12 and curr_flow12by21 have opposite directions!!!
                    flow_error_tgt2src = tf.abs(curr_flow_tgt2src+curr_flow_tgt2src_by_src2tgt)
                    flow_consistency_loss += layer_weights[s]*opt.flow_consistency*tf.reduce_sum(flow_error_tgt2src*curr_flow_valid_mask_tgt)/(tf.reduce_sum(curr_flow_valid_mask_tgt)+EPS)
                    flow_error_src2tgt = tf.abs(curr_flow_src2tgt + curr_flow_src2tgt_by_tgt2src)
                    flow_consistency_loss += layer_weights[s]*opt.flow_consistency*tf.reduce_sum(flow_error_src2tgt*curr_flow_valid_mask_src)/(tf.reduce_sum(curr_flow_valid_mask_src)+EPS)

                    ### Cross
                    if s < 4:
                        # Create valid mask from both branches
                        cross_valid_mask_tgt = curr_valid_mask_tgt*curr_flow_valid_mask_tgt
                        cross_valid_mask_src = curr_valid_mask_src*curr_flow_valid_mask_src

                        cross_error_tgt = tf.abs(curr_dp_flow_tgt2src-curr_flow_tgt2src)
                        cross_error_src = tf.abs(curr_dp_flow_src2tgt-curr_flow_src2tgt)
                        cross_consistency_loss += layer_weights[s]*opt.cross_consistency*tf.reduce_sum(cross_error_tgt*cross_valid_mask_tgt)/(tf.reduce_sum(cross_valid_mask_tgt)+EPS)
                        cross_consistency_loss += layer_weights[s]*opt.cross_consistency*tf.reduce_sum(cross_error_src*cross_valid_mask_src)/(tf.reduce_sum(cross_valid_mask_src)+EPS)

                    # Prepare images for tensorboard summaries
                    if model_idx == 0:
                        if i == 0:
                            proj_image_src2tgt_stack = curr_proj_image_src2tgt
                            proj_depth_src2tgt_stack = curr_proj_depth_src2tgt
                            proj_error_tgt_stack = curr_proj_error_tgt
                            proj_image_tgt2src_stack = curr_proj_image_tgt2src
                            proj_depth_tgt2src_stack = curr_proj_depth_tgt2src
                            proj_error_src_stack = curr_proj_error_src
                            dp_inrange_mask_tgt_stack = curr_inrange_mask_tgt
                            dp_inrange_mask_src_stack = curr_inrange_mask_src
                            dp_occ_mask_tgt_stack = curr_occ_mask_tgt
                            dp_occ_mask_src_stack = curr_occ_mask_src
                            # For flow
                            flow_image_src2tgt_stack = flow_proj_image_src2tgt
                            flow_image_tgt2src_stack = flow_proj_image_tgt2src
                            flow_src2tgt_stack = curr_flow_src2tgt
                            flow_tgt2src_stack = curr_flow_tgt2src
                            flow_error_tgt_stack = flow_proj_error_tgt
                            flow_error_src_stack = flow_proj_error_src
                            flow_inrange_mask_tgt_stack = curr_flow_inrange_mask_tgt
                            flow_inrange_mask_src_stack = curr_flow_inrange_mask_src
                            flow_occ_mask_tgt_stack = curr_flow_occ_mask_tgt
                            flow_occ_mask_src_stack = curr_flow_occ_mask_src
                        else:
                            proj_image_src2tgt_stack = tf.concat([proj_image_src2tgt_stack, curr_proj_image_src2tgt], axis=3)
                            proj_depth_src2tgt_stack = tf.concat([proj_depth_src2tgt_stack, curr_proj_depth_src2tgt], axis=3)
                            proj_error_tgt_stack = tf.concat([proj_error_tgt_stack, curr_proj_error_tgt], axis=3)
                            proj_image_tgt2src_stack = tf.concat([proj_image_tgt2src_stack, curr_proj_image_tgt2src], axis=3)
                            proj_depth_tgt2src_stack = tf.concat([proj_depth_tgt2src_stack, curr_proj_depth_tgt2src], axis=3)
                            proj_error_src_stack = tf.concat([proj_error_src_stack, curr_proj_error_src], axis=3)
                            dp_inrange_mask_tgt_stack = tf.concat([dp_inrange_mask_tgt_stack, curr_inrange_mask_tgt], axis=3)
                            dp_inrange_mask_src_stack = tf.concat([dp_inrange_mask_src_stack, curr_inrange_mask_src], axis=3)
                            dp_occ_mask_tgt_stack = tf.concat([dp_occ_mask_tgt_stack, curr_occ_mask_tgt], axis=3)
                            dp_occ_mask_src_stack = tf.concat([dp_occ_mask_src_stack, curr_occ_mask_src], axis=3)
                            # For flow
                            flow_image_src2tgt_stack = tf.concat([flow_image_src2tgt_stack, flow_proj_image_src2tgt], axis=3)
                            flow_image_tgt2src_stack = tf.concat([flow_image_tgt2src_stack, flow_proj_image_tgt2src], axis=3)
                            flow_src2tgt_stack = tf.concat([flow_src2tgt_stack, curr_flow_src2tgt], axis=3)
                            flow_tgt2src_stack = tf.concat([flow_tgt2src_stack, curr_flow_tgt2src], axis=3)
                            flow_error_tgt_stack = tf.concat([flow_error_tgt_stack, flow_proj_error_tgt], axis=3)
                            flow_error_src_stack = tf.concat([flow_error_src_stack, flow_proj_error_src], axis=3)
                            flow_inrange_mask_tgt_stack = tf.concat([flow_inrange_mask_tgt_stack, curr_flow_inrange_mask_tgt], axis=3)
                            flow_inrange_mask_src_stack = tf.concat([flow_inrange_mask_src_stack, curr_flow_inrange_mask_src], axis=3)
                            flow_occ_mask_tgt_stack = tf.concat([flow_occ_mask_tgt_stack, curr_flow_occ_mask_tgt], axis=3)
                            flow_occ_mask_src_stack = tf.concat([flow_occ_mask_src_stack, curr_flow_occ_mask_src], axis=3)
                if model_idx == 0:
                    tgt_image_all.append(curr_tgt_image)
                    src_image_stack_all.append(curr_src_image_stack)
                    proj_image_src2tgt_stack_all.append(proj_image_src2tgt_stack)
                    proj_depth_src2tgt_stack_all.append(proj_depth_src2tgt_stack)
                    proj_error_tgt_stack_all.append(proj_error_tgt_stack)
                    proj_image_tgt2src_stack_all.append(proj_image_tgt2src_stack)
                    proj_depth_tgt2src_stack_all.append(proj_depth_tgt2src_stack)
                    proj_error_src_stack_all.append(proj_error_src_stack)
                    dp_inrange_mask_tgt_stack_all.append(dp_inrange_mask_tgt_stack)
                    dp_inrange_mask_src_stack_all.append(dp_inrange_mask_src_stack)
                    dp_occ_mask_tgt_stack_all.append(dp_occ_mask_tgt_stack)
                    dp_occ_mask_src_stack_all.append(dp_occ_mask_src_stack)
                    # For flow
                    flow_image_src2tgt_stack_all.append(flow_image_src2tgt_stack)
                    flow_image_tgt2src_stack_all.append(flow_image_tgt2src_stack)
                    flow_src2tgt_stack_all.append(flow_src2tgt_stack)
                    flow_tgt2src_stack_all.append(flow_tgt2src_stack)
                    flow_error_tgt_stack_all.append(flow_error_tgt_stack)
                    flow_error_src_stack_all.append(flow_error_src_stack)
                    flow_inrange_mask_tgt_stack_all.append(flow_inrange_mask_tgt_stack)
                    flow_inrange_mask_src_stack_all.append(flow_inrange_mask_src_stack)
                    flow_occ_mask_tgt_stack_all.append(flow_occ_mask_tgt_stack)
                    flow_occ_mask_src_stack_all.append(flow_occ_mask_src_stack)
            total_loss = dp_pixel_loss+dp_smooth_loss+depth_consistency_loss+flow_pixel_loss+flow_smooth_loss+flow_consistency_loss+cross_consistency_loss

        if model_idx == 0:
            # Collect tensors that are useful later (e.g. tf summary)
            self.tgt_pred_depth = tgt_pred_depth
            self.src_pred_depths = src_pred_depths
            self.pred_poses = pred_poses
            self.tgt_image_all = tgt_image_all
            self.src_image_stack_all = src_image_stack_all
            self.proj_image_src2tgt_stack_all = proj_image_src2tgt_stack_all
            self.proj_depth_src2tgt_stack_all = proj_depth_src2tgt_stack_all
            self.proj_error_tgt_stack_all = proj_error_tgt_stack_all
            self.proj_image_tgt2src_stack_all = proj_image_tgt2src_stack_all
            self.proj_depth_tgt2src_stack_all = proj_depth_tgt2src_stack_all
            self.proj_error_src_stack_all = proj_error_src_stack_all
            self.dp_inrange_mask_tgt_stack_all = dp_inrange_mask_tgt_stack_all
            self.dp_inrange_mask_src_stack_all = dp_inrange_mask_src_stack_all
            self.dp_occ_mask_tgt_stack_all = dp_occ_mask_tgt_stack_all
            self.dp_occ_mask_src_stack_all = dp_occ_mask_src_stack_all
            # For flow
            self.flow_image_src2tgt_stack_all = flow_image_src2tgt_stack_all
            self.flow_image_tgt2src_stack_all = flow_image_tgt2src_stack_all
            self.flow_src2tgt_stack_all = flow_src2tgt_stack_all
            self.flow_tgt2src_stack_all = flow_tgt2src_stack_all
            self.flow_error_tgt_stack_all = flow_error_tgt_stack_all
            self.flow_error_src_stack_all = flow_error_src_stack_all
            self.flow_inrange_mask_tgt_stack_all = flow_inrange_mask_tgt_stack_all
            self.flow_inrange_mask_src_stack_all = flow_inrange_mask_src_stack_all
            self.flow_occ_mask_tgt_stack_all = flow_occ_mask_tgt_stack_all
            self.flow_occ_mask_src_stack_all = flow_occ_mask_src_stack_all

        grads = optim.compute_gradients(total_loss)
        return [dp_pixel_loss, dp_smooth_loss, depth_consistency_loss, flow_pixel_loss, flow_smooth_loss, flow_consistency_loss, cross_consistency_loss], grads

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("dp_pixel_loss", self.dp_pixel_loss)
        tf.summary.scalar("dp_smooth_loss", self.dp_smooth_loss)
        tf.summary.scalar("depth_consistency_loss", self.depth_consistency_loss)
        tf.summary.scalar("flow_pixel_loss", self.flow_pixel_loss)
        tf.summary.scalar("flow_smooth_loss", self.flow_smooth_loss)
        tf.summary.scalar("flow_consistency_loss", self.flow_consistency_loss)
        tf.summary.scalar("cross_consistency_loss", self.cross_consistency_loss)
        for s in range(opt.num_scales):
            if s < 4:
                tf.summary.histogram("scale%d_tgt_depth" % s, self.tgt_pred_depth[s])
                tf.summary.image('scale%d_tgt_disparity_image' % s, 1./self.tgt_pred_depth[s])
            # For [0, 1]
            tf.summary.image('scale%d_target_image' % s, \
                             tf.image.convert_image_dtype(self.tgt_image_all[s], dtype=tf.uint8))

            for i in range(opt.num_source):
                # For [0, 1]
                tf.summary.image('scale%d_source_image_%d' % (s, i),
                    tf.image.convert_image_dtype(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3], dtype=tf.uint8))
                if s < 4:
                    tf.summary.image('scale%d_projected_image_src2tgt_%d' % (s, i),
                        tf.image.convert_image_dtype(self.proj_image_src2tgt_stack_all[s][:, :, :, i*3:(i+1)*3], dtype=tf.uint8))
                    tf.summary.image('scale%d_projected_image_tgt2src_%d' % (s, i),
                        tf.image.convert_image_dtype(self.proj_image_tgt2src_stack_all[s][:, :, :, i*3:(i+1)*3], dtype=tf.uint8))
                    tf.summary.image('scale%d_proj_error_tgt_%d' % (s, i),
                        self.deprocess_image(tf.clip_by_value(self.proj_error_tgt_stack_all[s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)))
                    tf.summary.image('scale%d_proj_error_src_%d' % (s, i),
                        self.deprocess_image(tf.clip_by_value(self.proj_error_src_stack_all[s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)))
                    # Depth
                    #tf.summary.image('scale%d_proj_depth_src2tgt_%d' % (s, i), 1./self.proj_depth_src2tgt_stack_all[s][:,:,:,i:i+1])
                    #tf.summary.image('scale%d_proj_depth_tgt2src_%d' % (s, i), 1./self.proj_depth_tgt2src_stack_all[s][:,:,:,i:i+1])
                    #tf.summary.image('scale%d_dp_inrange_mask_tgt_%d' % (s, i), self.dp_inrange_mask_tgt_stack_all[s][:,:,:,i:i+1])
                    #tf.summary.image('scale%d_dp_inrange_mask_src_%d' % (s, i), self.dp_inrange_mask_src_stack_all[s][:,:,:,i:i+1])
                    #tf.summary.image('scale%d_dp_occ_mask_tgt_%d' % (s, i), self.dp_occ_mask_tgt_stack_all[s][:,:,:,i:i+1])
                    #tf.summary.image('scale%d_dp_occ_mask_src_%d' % (s, i), self.dp_occ_mask_src_stack_all[s][:,:,:,i:i+1])
                # For flow
                tf.summary.image('scale%d_flow_image_src2tgt_%d' % (s, i), 
                    tf.image.convert_image_dtype(self.flow_image_src2tgt_stack_all[s][:, :, :, i*3:(i+1)*3], dtype=tf.uint8))
                tf.summary.image('scale%d_flow_image_tgt2src_%d' % (s, i),
                    tf.image.convert_image_dtype(self.flow_image_tgt2src_stack_all[s][:, :, :, i*3:(i+1)*3], dtype=tf.uint8))
                tf.summary.image('scale%d_flow_src2tgt_%d' % (s, i),
                    self.flow_to_image_tf(self.flow_src2tgt_stack_all[s][:,:,:,i*2:(i+1)*2]))
                tf.summary.image('scale%d_flow_tgt2src_%d' % (s, i),
                    self.flow_to_image_tf(self.flow_tgt2src_stack_all[s][:,:,:,i*2:(i+1)*2]))
                #tf.summary.image('scale%d_flow_inrange_mask_tgt_%d' % (s, i), self.flow_inrange_mask_tgt_stack_all[s][:,:,:,i:i+1])
                #tf.summary.image('scale%d_flow_inrange_mask_src_%d' % (s, i), self.flow_inrange_mask_src_stack_all[s][:,:,:,i:i+1])
                #tf.summary.image('scale%d_flow_occ_mask_tgt_%d' % (s, i), self.flow_occ_mask_tgt_stack_all[s][:,:,:,i:i+1])
                #tf.summary.image('scale%d_flow_occ_mask_src_%d' % (s, i), self.flow_occ_mask_src_stack_all[s][:,:,:,i:i+1])


    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 5. (5 for FlowNet-C, 4 for SfMLearner)
        opt.num_scales = 5
        self.opt = opt

        with tf.device('/cpu:0'):
            self.build_train_graph()
            self.collect_summaries()
            with tf.name_scope("parameter_count"):
                parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                                for v in tf.trainable_variables()])
            self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                        [self.global_step],
                                         max_to_keep=20)
            saver_flow = tf.train.Saver([var for var in tf.all_variables() if 'flownet' in var.name and 'Adam' not in var.name and 'Momentum' not in var.name])
            if opt.ckpt_pose is None:
                # Use same ckpt for both depth and pose
                saver_dp   = tf.train.Saver([var for var in tf.trainable_variables() if 'depth_net' in var.name or 'pose_net' in var.name])
            else:
                # Use diffrent ckpts for depth and pose
                saver_dp   = tf.train.Saver([var for var in tf.trainable_variables() if 'depth_net' in var.name])
                saver_pose = tf.train.Saver([var for var in tf.trainable_variables() if 'pose_net' in var.name])

            sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                     save_summaries_secs=0, 
                                     saver=None)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with sv.managed_session(config=config) as sess:
                print('Trainable variables: ')
                for var in tf.trainable_variables():
                    print(var.name)
                print("parameter_count =", sess.run(parameter_count))
                if opt.continue_train:
                    ckpt_flow = opt.ckpt_flow
                    ckpt_dp = opt.ckpt_dp
                    ckpt_pose = opt.ckpt_pose
                    # Load weights from different checkpoints
                    print("Resume flow_net from previous checkpoint: %s" % ckpt_flow)
                    saver_flow.restore(sess, ckpt_flow)
                    if ckpt_pose is None:
                        # Use same ckpt for both depth and pose
                        print("Resume depth_net and pose_net from previous checkpoint: %s" % ckpt_dp)
                        saver_dp.restore(sess, ckpt_dp)
                    else:
                        # Use diffrent ckpts for depth and pose
                        print("Resume depth_net from previous checkpoint: %s" % ckpt_dp)
                        saver_dp.restore(sess, ckpt_dp)
                        print("Resume pose_net from previous checkpoint: %s" % ckpt_pose)
                        saver_pose.restore(sess, ckpt_pose)

                start_time = time.time()
                for step in range(1, opt.max_steps):
                    fetches = {
                        "train": self.train_op,
                        "global_step": self.global_step
                    }

                    if step % opt.summary_freq == 0:
                        fetches["loss"] = self.total_loss
                        fetches["summary"] = sv.summary_op

                    results = sess.run(fetches)
                    gs = results["global_step"]

                    
                    if step % opt.summary_freq == 0:
                        sv.summary_writer.add_summary(results['summary'], gs)
                        train_epoch = math.ceil(gs / self.steps_per_epoch)
                        train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                        print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                                % (train_epoch, train_step, self.steps_per_epoch, \
                                    (time.time() - start_time)/opt.summary_freq, 
                                    results['loss']))
                        start_time = time.time()
                    
                    if step % opt.save_latest_freq == 0:
                        self.save(sess, opt.checkpoint_dir, gs)
                    if step % self.steps_per_epoch == 0 or step == opt.max_steps-1:
                        self.save(sess, opt.checkpoint_dir, gs)

