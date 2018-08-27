# -*- coding: UTF-8 -*-
import tensorflow as tf

class FECNN(object):
    def __init__(self, images, source_classes_num, resample_classes_num):
        # """
        # Args:
        #   images: Input arguments
        #   source_classes_num: The number of classes used as source labels
        #   resample_classes_num: The number of classes used as resample labels
        # """

        self.images = images
        self.source_classes_num = source_classes_num
        self.resample_classes_num = resample_classes_num
        self.extract_feature()
        # self.source_discriminator()
        #self.resample_discriminator()

    # network for feature extraction
    def extract_feature(self):
        # with tf.variable_scope("extract_feature") as scope:
            # 1st Layer: Conv -> pooling
            conv1 = conv(self.images, 3, 3, 32, 1, 1, name = 'conv1')
            relu1 = relu(conv1, name = 'relu1')
            pool1 = max_pool(relu1, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')

            # 2nd Layer: Conv -> pooling
            conv2 = conv(pool1, 3, 3, 48, 1, 1, name = 'conv2')
            relu2 = relu(conv2, name = 'relu2')
            pool2 = max_pool(relu2, 2, 2, 2, 2, padding = 'VALID', name = 'pool2')

            # 3rd Layer: Conv -> pooling
            conv3 = conv(pool2, 3, 3, 64, 1, 1, name = 'conv3')
            relu3 = relu(conv3, name = 'relu3')
            pool3 = max_pool(relu3, 2, 2, 2, 2, padding = 'VALID', name = 'pool3')

            # 4th Layer: Conv
            conv4 = conv(pool3, 3, 3, 72, 1, 1, padding = 'VALID', name = 'conv4')
            relu4 = relu(conv4, name = 'relu4')

            # 5th Layer: Flatten -> FC (ReLu)
            flattened = tf.reshape(conv4, [-1, 72*6*6], name = 'flattened')
            fc = fully_connected(flattened, 512, name='fc')
            self.features = relu(fc, name = 'relu5')
            fc1 = fully_connected(self.features, 128, name='fc1')
            relu_1 = relu(fc1, name = 'relu1')
            fc2 = fully_connected(relu_1, self.resample_classes_num, name='fc2')
            self.resample_logits = relu(fc2, name = 'relu2')
            self.resample_pred = tf.nn.softmax(self.resample_logits, name='source_logits')

    # network for discriminating camera source
    # def source_discriminator(self):
    #     with tf.variable_scope("camera_source_discriminator") as scope:
    #         fc1 = fully_connected(self.features, 128, name='fc1')
    #         relu1 = relu(fc1, name = 'relu1')
    #         fc2 = fully_connected(relu1, self.source_classes_num, name='fc2')
    #         self.source_logits = relu(fc2, name = 'relu2')
    #         self.source_pred = tf.nn.softmax(self.source_logits, name='source_logits')
    #
    # # network for discriminating resample operation
    # def resample_discriminator(self):
    #     with tf.variable_scope("resample_operation_discriminator") as scope:
    #         fc1 = fully_connected(self.features, 128, name='fc1')
    #         relu1 = relu(fc1, name = 'relu1')
    #         fc2 = fully_connected(relu1, self.resample_classes_num, name='fc2')
    #         self.resample_logits = relu(fc2, name = 'relu2')
    #         self.resample_pred = tf.nn.softmax(self.resample_logits, name='resample_logits')

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
