from __future__ import division
import os
import time
import math
from glob import glob
import numpy as np
import re
import tensorflow as tf

from ops import *
from utils import *

class FECNN(object):
  def __init__(self, sess, batch_size, checkpoint_dir, graph_dir,
               source_classes_num,resample_classes_num, istrain):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      checkpoint_dir: Directory name to save the checkpoints.
      graph_dir: Directory name to save the graph.
      source_classes_num: The number of camera sources
      resample_classes_num: The number of resample operations
    """
    self.sess = sess
    self.batch_size = batch_size
    self.checkpoint_dir = checkpoint_dir
    self.graph_dir = graph_dir
    self.s_classes_num = source_classes_num
    self.r_classes_num = resample_classes_num
    self.istrain = istrain

    # batch normalization : deals with poor initialization helps gradient flow
    self.f_bn0 = Batch_norm(name='f_bn0')
    self.f_bn1 = Batch_norm(name='f_bn1')
    self.f_bn2 = Batch_norm(name='f_bn2')
    self.f_bn3 = Batch_norm(name='f_bn3')

    self.data_generator = DataGenerator(self.istrain, self.batch_size, self.s_classes_num, self.r_classes_num)
    self.batches_num = np.floor(self.data_generator.data_size / self.batch_size).astype(np.int16)

    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, 128, 128, 3], name = 'images')
    self.s_labels = tf.placeholder(tf.float32, [self.batch_size, self.s_classes_num], name = 's_labels')
    self.r_labels = tf.placeholder(tf.float32, [self.batch_size, self.r_classes_num], name = 'r_labels')

    self.FE                     = self.feature_generator()
    self.s_logits, self.s_pred  = self.source_discriminator(self.FE)
    self.r_logits, self.r_pred  = self.resample_discriminator(self.FE)

    # set loss
    with tf.name_scope("cross_ent"):
      self.s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                    logits = self.s_logits, labels = self.s_labels))
      class_w = tf.range(0, self.r_classes_num, 1, tf.float32)
      class_w = tf.tile(class_w, [self.batch_size])
      class_w = tf.reshape(class_w, shape=[self.batch_size, self.r_classes_num])
      true_r = tf.cast(tf.reshape(tf.argmax(self.r_labels, 1), [self.batch_size,1]), tf.float32)
      r_w = tf.square(tf.subtract(class_w, true_r))
      f_w = tf.cast(tf.square(self.r_classes_num-1), tf.float32)-r_w
      dif = tf.square(self.r_pred)
      self.r_loss = tf.reduce_mean(tf.reduce_sum(tf.cast(dif*r_w, tf.float32), 1))
      self.f_loss = tf.reduce_mean(tf.reduce_sum(tf.cast(dif*f_w, tf.float32), 1))+self.s_loss

    self.s_pred_idx = tf.argmax(self.s_pred, 1)
    self.s_label_idx = tf.argmax(self.s_labels, 1)

    # loss sum
    self.f_loss_sum = scalar_summary('feature_cross_entropy', self.f_loss)
    self.s_loss_sum = scalar_summary('source_cross_entropy', self.s_loss)
    self.r_loss_sum = scalar_summary('resample_cross_entropy', self.r_loss)

    # Evaluation op: Accuracy of the model
    def get_acc (preds, labels):
      corr_pred = tf.equal(tf.argmax(preds, 1),tf.argmax(labels, 1))
      return tf.reduce_mean(tf.cast(corr_pred, tf.float32))

    self.s_acc = get_acc(self.s_pred, self.s_labels)
    self.r_acc = get_acc(self.r_pred, self.r_labels)

    # acc sum
    self.s_acc_sum = scalar_summary('source_accuracy', self.s_acc)
    self.r_acc_sum = scalar_summary('resample_accuracy', self.r_acc)

    t_vars = tf.trainable_variables()
    self.f_vars = [var for var in t_vars if 'f_' in var.name]
    self.s_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='s_discriminator')
    self.r_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='r_discriminator')

    self.saver = tf.train.Saver()

  def train(self, config):
    print(" [*] Training Mode")
    # set optimizer
    f_optim = tf.train.MomentumOptimizer(learning_rate = config.learning_rate, momentum=0.9)\
              .minimize(self.f_loss, var_list=self.f_vars)
    s_optim = tf.train.MomentumOptimizer(learning_rate = config.learning_rate, momentum=0.9)\
              .minimize(self.s_loss, var_list=self.s_vars+self.f_vars)
    r_optim = tf.train.MomentumOptimizer(learning_rate = config.learning_rate, momentum=0.9)\
              .minimize(self.r_loss, var_list=self.r_vars)

    # set summary
    self.merged_summary = merge_summary([self.f_loss_sum, self.s_loss_sum, \
                          self.s_acc_sum, self.r_loss_sum, self.r_acc_sum])
    self.writer = SummaryWriter(self.graph_dir, self.sess.graph)

    # init variables
    self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

    counter = 1
    start_time = time.time()
    could_load, ckcount = self.load(self.checkpoint_dir)
    if could_load:
      counter = ckcount
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in range(config.epoch):
      for idx in range(self.batches_num):
        # First: train extract_feature network and source_network
        images_, s_labels_, r_labels_ = self.sess.run([self.data_generator.images,
                                                       self.data_generator.source_labels,
                                                       self.data_generator.resample_labels])
        s_optim_, s_loss_, s_acc_ = self.sess.run([s_optim, self.s_loss, self.s_acc],
                                   feed_dict={self.images: images_,
                                              self.s_labels: s_labels_})

        # Second: train resample network
        r_optim_, r_loss_, r_acc_ = self.sess.run([r_optim, self.r_loss, self.r_acc],
                                    feed_dict={self.images: images_,
                                               self.r_labels: r_labels_})

        # Third: train extract_feature network
        f_optim_ = self.sess.run(f_optim,
                  feed_dict={self.images: images_,
                             self.s_labels: s_labels_,
                             self.r_labels: r_labels_})

        counter += 1
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, s_loss: %.8f, s_acc: %.8f,\
        r_loss: %.8f, r_acc: %.8f" % (epoch, config.epoch, idx, self.batches_num,
        time.time() - start_time, s_loss_, s_acc_, r_loss_, r_acc_))

        if np.mod(epoch, 20) == 1:
          config.learning_rate /= 2;
          summary_str = self.sess.run(self.merged_summary,
                        feed_dict={self.images: images_,
                                   self.s_labels: s_labels_,
                                   self.r_labels: r_labels_})
          self.writer.add_summary(summary_str, counter)

      # save checkpoints
      self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'FECNN.model'), global_step=epoch)
    self.writer.close()
    coord.request_stop()
    coord.join(threads)

  def test(self):
    print(" [*] Testing Mode")
    could_load = self.load(self.checkpoint_dir)[0]
    if could_load:
      print(" [*] Load SUCCESS")
    else:
      raise Exception("[!] Train a model first, then run test mode")

    # init variables
    self.sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

    start_time = time.time()
    for idx in range(self.batches_num):
      val_images_, val_s_labels_,  val_r_labels_ = self.sess.run([self.data_generator.images,
                                                             self.data_generator.source_labels,
                                                             self.data_generator.resample_labels])
      s_loss_, s_acc_, r_loss_, r_acc_ = self.sess.run([self.s_loss, self.s_acc, self.r_loss, self.r_acc],
                                      feed_dict = {self.images: val_images_,
                                                   self.s_labels: val_s_labels_,
                                                   self.r_labels: val_r_labels_})

      print("Epoch: [%4d/%4d] time: %4.4f, s_loss: %.8f, s_acc: %.8f,\
      r_loss: %.8f, r_acc: %.8f" % (idx, self.batches_num, \
      time.time() - start_time, s_loss_, s_acc_, r_loss_, r_acc_))
      s_pred_idx_, s_label_idx_ = self.sess.run([self.s_pred_idx, self.s_label_idx],
                                  feed_dict = {self.images: val_images_,
                                               self.s_labels: val_s_labels_,
                                               self.r_labels: val_r_labels_})
      fp1 = open("E:\\shuqian\\info1.txt", 'a')
      fp2 = open("E:\\shuqian\\info2.txt", 'a')
      fp1.writelines(str(s_label_idx_))
      fp2.writelines(str(s_pred_idx_))
    coord.request_stop()
    coord.join(threads)
    # get predication

  def feature_generator(self):
    with tf.variable_scope("f_generator") as scope:
      # inception 0
      convincep0_1 = conv(self.images, 1, 1, 32, 1, 1, name = 'f_incep0_1')
      convincep0_2 = conv(self.images, 3, 3, 32, 1, 1, name = 'f_incep0_2')
      convincep0_3 = conv(self.images, 5, 5, 32, 1, 1, name = 'f_incep0_3')
      convincep0_4_1 = max_pool(self.images, 2, 2, 1, 1, name = 'f_incep0_4_1')
      convincep0_4_2 = conv(convincep0_4_1, 1, 1, 32, 1, 1, name = 'f_incep0_4_2')
      inception0 = tf.concat([convincep0_1, convincep0_2, convincep0_3, convincep0_4_2, self.images], 3)

      # 0st Layer: Conv -> pooling
      conv0 = conv(inception0, 3, 3, 32, 1, 1, name = 'f_conv0')
      relu0 = relu(self.f_bn0(conv0, self.istrain), name = 'f_relu0')
      pool0 = max_pool(relu0, 2, 2, 2, 2, padding = 'VALID', name = 'f_pool0')

      convincep1_1 = conv(pool0, 1, 1, 32, 1, 1, name = 'f_incep1_1')
      convincep1_2 = conv(pool0, 3, 3, 32, 1, 1, name = 'f_incep1_2')
      convincep1_3 = conv(pool0, 5, 5, 32, 1, 1, name = 'f_incep1_3')
      convincep1_4_1 = max_pool(pool0, 2, 2, 1, 1, name = 'f_incep1_4_1')
      convincep1_4_2 = conv(convincep1_4_1, 1, 1, 32, 1, 1, name = 'f_incep1_4_2')
      inception1 = tf.concat([convincep1_1, convincep1_2, convincep1_3, convincep1_4_2, pool0], 3)

      # 1st Layer: Conv -> pooling
      conv1 = conv(inception1, 3, 3, 48, 1, 1, name = 'f_conv1')
      relu1 = relu(self.f_bn1(conv1, self.istrain), name = 'f_relu1')
      pool1 = max_pool(relu1, 2, 2, 2, 2, padding = 'VALID', name = 'f_pool1')

      # 2nd Layer: Conv -> pooling
      conv2 = conv(pool1, 3, 3, 96, 1, 1, name = 'f_conv2')
      relu2 = relu(self.f_bn2(conv2, self.istrain), name = 'f_relu2')
      pool2 = max_pool(relu2, 2, 2, 2, 2, padding = 'VALID', name = 'f_pool2')

      # 3rd Layer: Conv -> pooling
      conv3 = conv(pool2, 3, 3, 128, 1, 1, name = 'f_conv3')
      relu3 = relu(self.f_bn3(conv3, self.istrain), name = 'f_relu3')
      pool3 = max_pool(relu3, 2, 2, 2, 2, padding = 'VALID', name = 'f_pool3')

      # 4th Layer: Conv
      conv4 = conv(pool3, 3, 3, 128, 1, 1, padding = 'VALID', name = 'f_conv4')
      relu4 = relu(conv4, name = 'f_relu4')

      # 5th Layer: Flatten -> FC (ReLu)
      flattened = tf.reshape(conv4, [-1, 128*6*6], name = 'f_flattened')
      fc = fully_connected(flattened, 512, name='f_fc')
      features = relu(fc, name = 'f_relu5')
      return features

  # network for discriminating camera source
  def source_discriminator(self, features):
    with tf.variable_scope("s_discriminator") as scope:
      fc1 = fully_connected(features, 128, name='fc1')
      relu1 = relu(fc1, name = 'relu1')
      fc2 = fully_connected(relu1, self.s_classes_num, name='fc2')
      source_logits = relu(fc2, name = 'relu2')
      source_pred = tf.nn.softmax(source_logits, name='source_logits')
      return source_logits, source_pred

  # network for discriminating resample operation
  def resample_discriminator(self, features):
    with tf.variable_scope("r_discriminator") as scope:
      fc1 = fully_connected(features, 128, name='fc1')
      relu1 = relu(fc1, name = 'relu1')
      fc2 = fully_connected(relu1, self.r_classes_num, name='fc2')
      resample_logits = relu(fc2, name = 'relu2')
      resample_pred = tf.nn.softmax(resample_logits, name='resample_logits')
      return resample_logits, resample_pred

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
