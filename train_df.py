from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from core import DFLearner
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "./dataset/kitti_5frame_1152_320/", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./ckpt/dfnet", "Directory name to save the checkpoints")
flags.DEFINE_string("ckpt_flow", "pretrained/unflowc_pre", "checkpoint for Flow Net")
flags.DEFINE_string("ckpt_dp", "pretrained/cs_5frame_pre", "checkpoint for Depth Net and Pose Net")
flags.DEFINE_string("ckpt_pose", None, "checkpoint for Pose Net, if not shared with Depth Net")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 3.0, "Weight for smoothness")
flags.DEFINE_float("alpha_image_loss", 0.85, "Weight between SSIM and L1 in the image loss")
flags.DEFINE_float("depth_consistency", 0.2, "Weight for forward-backward depth consistency loss.")
flags.DEFINE_float("flow_smooth_weight", 3.0, "Weight for flow smoothness")
flags.DEFINE_float("flow_consistency", 0.2, "Weight for forward-backward flow consistency loss.")
flags.DEFINE_float("cross_consistency", 0.5, "Weight for cross-network consistency loss")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch, must divisible by number of GPUs!")
flags.DEFINE_integer("num_gpus", 4, "Number of GPUs for training, starting from 0.")
flags.DEFINE_integer("img_height", 320, "Image height")
flags.DEFINE_integer("img_width", 1152, "Image width")
flags.DEFINE_integer("seq_length", 5, "Sequence length for each example") # Fixed. Don't change
flags.DEFINE_integer("max_steps", 100000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", True, "Continue training from previous checkpoint")
flags.DEFINE_boolean("scale_normalize", False, "Scale normalization for disparity.") # Set to True will break the training.
flags.DEFINE_boolean("fix_pose", False, "Fix pose network")
FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        
    learner = DFLearner()
    learner.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
