from __future__ import division
import cv2
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
import png
import scipy

from core import flow_to_image
from core import flownet

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 384, "Image height")
flags.DEFINE_integer("img_width", 1280, "Image width")
flags.DEFINE_string("dataset_dir", './dataset/KITTI/flow2012/training/', "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", 'pretrained/unflowc_pre', "checkpoint file")
FLAGS = flags.FLAGS

FLOW_SCALE = 5.0

# kitti 2012 has 194 training pairs, 195 test pairs
if 'train' in FLAGS.dataset_dir:
    NUM = 194
elif 'test' in FLAGS.dataset_dir:
    NUM = 195

def get_flow(path):
    bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    invalid = bgr[:, :, 0] == 0
    out_flow = (bgr[:, :, 2:0:-1].astype('f4') - 2**15) / 64.
    out_flow[invalid] = 0
    return out_flow, bgr[:, :, 0]

def compute_flow_error(gt_flow, pred_flow, mask):
    H, W, _ = gt_flow.shape
    old_H, old_W, _ = pred_flow.shape
    # Reshape predicted flow to have same size as ground truth
    pred0 = cv2.resize(pred_flow[:,:,0], (W, H), interpolation=cv2.INTER_LINEAR) * (1.0*W/old_W)
    pred1 = cv2.resize(pred_flow[:,:,1], (W, H), interpolation=cv2.INTER_LINEAR) * (1.0*H/old_H)
    pred = np.stack((pred0, pred1), axis=-1) * FLOW_SCALE

    err = np.sqrt(np.sum(np.square(gt_flow - pred), axis=-1))
    err_valid = np.sum(err * mask) / np.sum(mask)
    return err_valid, pred

def write_flow_png(name, flow):
    H, W, _ = flow.shape
    out = np.ones((H, W, 3), dtype=np.uint64)
    out[:,:,1] = np.minimum(np.maximum(flow[:,:,1]*64.+2**15, 0), 2**16).astype(np.uint64)
    out[:,:,0] = np.minimum(np.maximum(flow[:,:,0]*64.+2**15, 0), 2**16).astype(np.uint64)
    with open(name, 'wb') as f:
        writer = png.Writer(width=W, height=H, bitdepth=16)
        im2list = out.reshape(-1, out.shape[1]*out.shape[2]).tolist()
        writer.write(f, im2list)


def pick_frame(path):
    new_files = []
    # flow2012 dataset only has 194 pairs
    for i in range(NUM):
        frame1 = os.path.join(path, 'colored_0', '{:06d}'.format(i) + '_10.png')
        frame2 = os.path.join(path, 'colored_0', '{:06d}'.format(i) + '_11.png')
        new_files.append([frame1, frame2])
    return new_files

def main(_):
    new_files = pick_frame(FLAGS.dataset_dir)
    basename = os.path.basename(FLAGS.ckpt_file)

    im1_pl = tf.placeholder(dtype=tf.float32, shape=(1, FLAGS.img_height, FLAGS.img_width, 3))
    im2_pl = tf.placeholder(dtype=tf.float32, shape=(1, FLAGS.img_height, FLAGS.img_width, 3))
    pred_flows = flownet(im1_pl, im2_pl, flownet_spec='C')

    saver = tf.train.Saver([var for var in tf.all_variables() if 'flow' in var.name]) 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    errs = np.zeros(NUM)

    if not FLAGS.output_dir is None and not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        # For val set
        for t in range(0, len(new_files)):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(new_files)))
            raw_im0 = pil.open(new_files[t][0])
            raw_im1 = pil.open(new_files[t][1])
            scaled_im0 = raw_im0.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
            scaled_im1 = raw_im1.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
            # Minus ImageNet channel mean
            channel_mean = np.array([104.920005, 110.1753, 114.785955])
            scaled_im0 = (np.expand_dims(np.array(scaled_im0), axis=0).astype(np.float32)-channel_mean)/255.
            scaled_im1 = (np.expand_dims(np.array(scaled_im1), axis=0).astype(np.float32)-channel_mean)/255.
            feed_dict = {im1_pl: scaled_im0, im2_pl: scaled_im1}
            pred_flows_val = sess.run(pred_flows, feed_dict=feed_dict)           
            pred_flow_val = pred_flows_val[-1][0]

            # Only for training set
            if 'train' in FLAGS.dataset_dir:
                # no occlusion
                #gt_flow, mask = get_flow(new_files[t][0].replace('colored_0', 'flow_noc'))
                # all
                gt_flow, mask = get_flow(new_files[t][0].replace('colored_0', 'flow_occ'))
                errs[t], scaled_pred = compute_flow_error(gt_flow, pred_flow_val[0,:,:,:], mask)

            # Save for eval
            if 'test' in FLAGS.dataset_dir:
                _, scaled_pred = compute_flow_error(np.array(raw_im0)[:,:,:2], pred_flow_val[0,:,:,:], np.array(raw_im0)[:,:,0])
                png_name = os.path.join(FLAGS.output_dir, new_files[t][0].split('/')[-1])
                write_flow_png(png_name, scaled_pred)

            # Save for visual colormap
            if not 'test' in FLAGS.dataset_dir and not FLAGS.output_dir is None:
                flow_im = flow_to_image(scaled_pred)
                png_name = os.path.join(FLAGS.output_dir, new_files[t][0].split('/')[-1]).replace('png', 'jpg')
                cv2.imwrite(png_name, flow_im[:,:,::-1])

        print('{:>10}'.format('(valid) endpoint error'))
        print('{:10.4f}'.format(errs.mean()))

if __name__ == '__main__':
    tf.app.run()
