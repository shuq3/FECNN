# -*- coding: UTF-8 -*-
import tensorflow as tf

class FECNN(object):
    def __init__(self, images, source_classes_num, resample_classes_num, istrain):
        # """
        # Args:
        #   images: Input arguments
        #   source_classes_num: The number of classes used as source labels
        #   resample_classes_num: The number of classes used as resample labels
        # """

        self.images = images
        self.source_classes_num = source_classes_num
        self.resample_classes_num = resample_classes_num
        self.istrain = istrain
        self.extract_feature()
        self.source_discriminator()
        self.resample_discriminator()
        self.extra_train_ops = []

    # network for feature extraction
    def extract_feature(self):
        with tf.variable_scope("extract_feature") as scope:
            # inception 0
            convincep0_1 = conv(self.images, 1, 1, 16, 1, 1, name = 'convincep0_1')
            convincep0_2_1 = conv(self.images, 3, 3, 32, 1, 1, name = 'convincep0_2_1')
            convincep0_2_2 = conv(convincep0_2_1, 3, 3, 16, 1, 1, name = 'convincep0_2_2')
            convincep0_3_1 = conv(self.images, 5, 5, 32, 1, 1, name = 'convincep0_3_1')
            convincep0_3_2 = conv(convincep0_3_1, 5, 5, 16, 1, 1, name = 'convincep0_3_2')
            convincep0_4_1 = max_pool(self.images, 2, 2, 1, 1, name = 'convincep0_4_1')
            convincep0_4_2 = conv(convincep0_4_1, 1, 1, 16, 1, 1, name = 'convincep0_4_2')
            inception0 = tf.concat([convincep0_1, convincep0_2_2, convincep0_3_2, convincep0_4_2], 3)

            # inception 1
            # convincep1_1 = conv(inception0, 1, 1, 32, 1, 1, name = 'convincep1_1')
            # convincep1_2_1 = conv(inception0, 1, 1, 32, 1, 1, name = 'convincep1_2_1')
            # convincep1_2_2 = conv(convincep1_2_1, 3, 3, 32, 1, 1, name = 'convincep1_2_2')
            # convincep1_3_1 = conv(inception0, 1, 1, 32, 1, 1, name = 'convincep1_3_1')
            # convincep1_3_2 = conv(convincep1_3_1, 5, 5, 32, 1, 1, name = 'convincep1_3_2')
            # convincep1_4_1 = max_pool(inception0, 2, 2, 1, 1, name = 'convincep1_4_1')
            # convincep1_4_2 = conv(convincep1_4_1, 1, 1, 32, 1, 1, name = 'convincep1_4_2')
            # inception1 = tf.concat([convincep1_1, convincep1_2_2, convincep1_3_2, convincep1_4_2], 3)

            # 0st Layer: Conv -> pooling
            conv0 = conv(inception0, 3, 3, 32, 1, 1, name = 'conv0')
            batch_norm0 = self.batch_norm(conv0, name = 'batch_norm0')
            relu0 = relu(batch_norm0, name = 'relu0')
            pool0 = max_pool(relu0, 2, 2, 2, 2, padding = 'VALID', name = 'pool0')

            # 1st Layer: Conv -> pooling
            conv1 = conv(pool0, 3, 3, 48, 1, 1, name = 'conv1')
            batch_norm1 = self.batch_norm(conv1, name = 'batch_norm1')
            relu1 = relu(batch_norm1, name = 'relu1')
            pool1 = max_pool(relu1, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')

            # 2nd Layer: Conv -> pooling
            conv2 = conv(pool1, 3, 3, 96, 1, 1, name = 'conv2')
            batch_norm2 = self.batch_norm(conv2, name = 'batch_norm2')
            relu2 = relu(batch_norm2, name = 'relu2')
            pool2 = max_pool(relu2, 2, 2, 2, 2, padding = 'VALID', name = 'pool2')

            # 3rd Layer: Conv -> pooling
            conv3 = conv(pool2, 3, 3, 128, 1, 1, name = 'conv3')
            batch_norm3 = self.batch_norm(conv3, name = 'batch_norm3')
            relu3 = relu(batch_norm3, name = 'relu3')
            pool3 = max_pool(relu3, 2, 2, 2, 2, padding = 'VALID', name = 'pool3')

            # 4th Layer: Conv
            conv4 = conv(pool3, 3, 3, 128, 1, 1, padding = 'VALID', name = 'conv4')
            relu4 = relu(conv4, name = 'relu4')

            # 5th Layer: Flatten -> FC (ReLu)
            flattened = tf.reshape(relu4, [-1, 128*6*6], name = 'flattened')
            fc0 = fully_connected(flattened, 512, name='fc0')
            self.features = relu(fc0, name = 'relu5')

    # network for discriminating camera source
    def source_discriminator(self):
        with tf.variable_scope("camera_source_discriminator") as scope:
            fc1 = fully_connected(self.features, 128, name='fc1')
            relu1 = relu(fc1, name = 'relu1')
            fc2 = fully_connected(relu1, self.source_classes_num, name='fc2')
            self.source_logits = relu(fc2, name = 'relu2')
            self.source_pred = tf.nn.softmax(self.source_logits, name='source_logits')
    #
    # # network for discriminating resample operation
    def resample_discriminator(self):
        with tf.variable_scope("resample_operation_discriminator") as scope:
            fc1 = fully_connected(self.features, 128, name='fc1')
            relu1 = relu(fc1, name = 'relu1')
            fc2 = fully_connected(relu1, self.resample_classes_num, name='fc2')
            self.resample_logits = relu(fc2, name = 'relu2')
            self.resample_pred = tf.nn.softmax(self.resample_logits, name='resample_logits')

    # batch normalization.
    def batch_norm(self, x, name):
        with tf.variable_scope(name):
          params_shape = [x.get_shape()[-1]]
          beta = tf.get_variable('beta', params_shape, tf.float32,
              initializer=tf.constant_initializer(0.0, tf.float32))
          gamma = tf.get_variable('gamma', params_shape, tf.float32,
              initializer=tf.constant_initializer(1.0, tf.float32))

          if self.istrain == True:
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

            self.extra_train_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            self.extra_train_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))
          else:
            mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            # tf.summary.histogram(mean.op.name, mean)
            # tf.summary.histogram(variance.op.name, variance)
          # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
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
        weights = tf.get_variable(shape = [inputs.get_shape()[1], num_out], dtype=tf.float32, name='weights',
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(shape=[num_out], dtype=tf.float32,
                                 name='fc_biases', initializer=tf.constant_initializer(0.0))
        return tf.nn.xw_plus_b(inputs, weights, biases, name=name)
