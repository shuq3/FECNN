import os
from datetime import datetime
import numpy as np

from model import FECNN

import tensorflow as tf

# parames
flags = tf.app.flags
# Learning params
flags.DEFINE_integer("epoch", 300, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.01]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
# Network params
flags.DEFINE_integer("display_step", 60, "How often we want to write the tf.summary data to disk [60]")
flags.DEFINE_integer("source_classes_num", 10, "The number of camera sources [10]")
flags.DEFINE_integer("resample_classes_num", 10, "The number of resample categories [5]")
# Path for tf.summary.FileWriter and to store model checkpoints
flags.DEFINE_string("checkpoint_dir", "E:\\shuqian\\resample\\checkpoints", "Directory name to save the checkpoints")
flags.DEFINE_string("graph_dir", "E:\\shuqian\\resample\\graph\\train", "Directory name to save graphs")
flags.DEFINE_boolean("istrain",False , "True for training, False for testing [True]")
# flags.DEFINE_string("train_graph_dir", "E:\\shuqian\\resample\\graph\\train", "Directory name to save train graph")
# flags.DEFINE_string("test_graph_dir", "E:\\shuqian\\resample\\graph\\test", "Directory name to save test graph")
FLAGS = flags.FLAGS

def main(_):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
  run_config = tf.ConfigProto(gpu_options = gpu_options)
  run_config.gpu_options.allow_growth=True

  with tf.Session(config =run_config) as sess:
    fecnn = FECNN(
      sess,
      batch_size = FLAGS.batch_size,
      checkpoint_dir = FLAGS.checkpoint_dir,
      graph_dir = FLAGS.graph_dir,
      source_classes_num = FLAGS.source_classes_num,
      resample_classes_num = FLAGS.resample_classes_num,
      istrain = FLAGS.istrain)

  if FLAGS.istrain:
    fecnn.train(FLAGS)
  else:
    fecnn.test()

if __name__ == '__main__':
  tf.app.run()
