import math
import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops

from utils import *

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

class Batch_norm(object):
  def __init__(self, name="batch_norm"):
    self.name = name

  def __call__(self, x, train=True):
    with tf.variable_scope(self.name) as scope:
      params_shape = [x.get_shape()[-1]]
      beta = tf.get_variable('beta', params_shape, tf.float32,
        initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable('gamma', params_shape, tf.float32,
        initializer=tf.constant_initializer(1.0, tf.float32))

      if train:
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
        moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32),
          trainable=False)
        moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32),
          trainable=False)

        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.9)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.9)
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
          y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
          y.set_shape(x.get_shape())
          return y

      else:
        mean = tf.get_variable('moving_mean', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32),
          trainable=False)
        variance = tf.get_variable('moving_variance', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32),
          trainable=False)
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y

# convolution operation
def conv(inputs, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
  with tf.variable_scope(name) as scope:
    kernel = tf.get_variable(dtype=tf.float32,name='weights',
                             shape = [filter_height, filter_width, inputs.get_shape()[-1], num_filters],
                             initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(dtype=tf.float32, name='biases',
                             shape=[num_filters], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride_y, stride_x, 1], padding=padding)
    conv = tf.nn.bias_add(conv, biases)
    return conv

# relu opretation
def relu(inputs, name):
  return tf.nn.relu(inputs, name=name)

# max pooling
def max_pool(inputs, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  with tf.variable_scope(name) as scope:
    return tf.nn.max_pool(inputs, ksize=[1, filter_height, filter_width, 1], name = name,
                          strides = [1, stride_y, stride_x, 1], padding = padding)

# fully connected layers
def fully_connected(inputs, num_out, name):
  with tf.variable_scope(name) as scope:
    weights = tf.get_variable(shape = [inputs.get_shape()[1], num_out], dtype=tf.float32,
              name='fc_weights', initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(shape=[num_out], dtype=tf.float32,
              name='fc_biases', initializer=tf.constant_initializer(0.0))
    return tf.nn.xw_plus_b(inputs, weights, biases, name=name)
