from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from core import DFLearner

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 160, "Image height")
flags.DEFINE_integer("img_width", 576, "Image width")
flags.DEFINE_string("dataset_dir", './dataset/KITTI/raw/data/', "Dataset directory")
flags.DEFINE_string("split", 'val', 'val or test')
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.split == 'val':
        txt_file = 'data/kitti/val_files_eigen.txt'
    elif FLAGS.split == 'test':
        txt_file = 'data/kitti/test_files_eigen.txt'
    else:
        assert False

    with open(txt_file, 'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    basename = os.path.basename(FLAGS.ckpt_file)
    output_file = os.path.join(FLAGS.output_dir, basename)
    model = DFLearner()
    model.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        mode='depth')
    saver = tf.train.Saver([var for var in tf.model_variables()]) 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        pred_all = []
        for t in range(0, len(test_files), FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))
            inputs = np.zeros(
                (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3), 
                dtype=np.uint8)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                raw_im = pil.open(test_files[idx])
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)
            pred = model.inference(inputs, sess, mode='depth')
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b,:,:,0])
        np.save(output_file, pred_all)

if __name__ == '__main__':
    tf.app.run()
