# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

class FECNN(object):
    def __init__(self, images, source_classes_num, resample_classes_num, mode, source_labels, resample_labels):
        # """
        # Args:
        #   images: Input arguments
        #   source_classes_num: The number of classes used as source labels
        #   resample_classes_num: The number of classes used as resample labels
        #   mode: train or val, true for train, false for val
        # """

        self.images = images
        self.mode = mode
        self.source_classes_num = source_classes_num
        self.resample_classes_num = resample_classes_num
        self.source_labels = source_labels
        self.resample_labels = resample_labels
        self.train_ops = []

    # build a whole graph for the model.
    def build_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.extract_feature()
        self.source_discriminator()
        self.resample_discriminator()
        if self.mode == True:
            self.build_train_op()
        self.summaries = tf.summary.merge_all()

    # network for feature extraction
    def extract_feature(self):
        with tf.variable_scope("extract_feature") as scope:
            # 1st Layer: Conv -> pooling
            conv1 = conv(self.images, 3, 3, 32, 1, 1, name = 'conv1')
            relu1 = relu(conv1, name = 'relu1')
            pool1 = max_pool(relu1, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')

            # 2nd Layer: Conv -> pooling
            conv2 = conv(pool1, 3, 3, 48, 1, 1, name = 'conv2')
            relu2 = relu(conv2, name = 'relu2')
            pool2 = max_pool(relu2, 2, 2, 2, 2, padding = 'VALID', name = 'pool2')

            # 3rd Layer: Conv -> pooling
            conv3 = conv(pool2, 3, 3, 72, 1, 1, name = 'conv3')
            relu3 = relu(conv3, name = 'relu3')
            pool3 = max_pool(relu3, 2, 2, 2, 2, padding = 'VALID', name = 'pool3')

            # 4th Layer: Conv
            conv4 = conv(pool3, 5, 5, 128, 1, 1, padding = 'VALID', name = 'conv4')
            relu4 = relu(conv4, name = 'relu4')

            # 5th Layer: Flatten -> FC (ReLu)
            flattened = tf.reshape(relu4, [-1, 128*4*4], name = 'flattened')
            fc = fully_connected(flattened, 512, name='fc')
            self.features = relu(fc, name = 'relu5')

    # network for discriminating camera source
    def source_discriminator(self):
        with tf.variable_scope("camera_source_discriminator") as scope:
            fc1 = fully_connected(self.features, 128, name='fc1')
            relu1 = relu(fc1, name = 'relu1')
            fc2 = fully_connected(relu1, self.source_classes_num, name='fc2')
            self.source_logits = relu(fc2, name = 'relu2')
            self.source_pred = tf.nn.softmax(self.source_logits, name='source_logits')
            self.resample_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = resample_logits, labels = resample_labels))

        # source accurac
        with tf.name_scope("source_accuracy"):
          source_correct_pred = tf.equal(tf.argmax(source_pred, 1), tf.argmax(source_labels, 1))
          self.source_accuracy = tf.reduce_mean(tf.cast(source_correct_pred, tf.float32))

    # network for discriminating resample operation
    def resample_discriminator(self):
        with tf.variable_scope("resample_operation_discriminator") as scope:
            fc1 = fully_connected(self.features, 128, name='fc1')
            relu1 = relu(fc1, name = 'relu1')
            fc2 = fully_connected(relu1, self.resample_classes_num, name='fc2')
            self.resample_logits = relu(fc2, name = 'relu2')
            self.resample_pred = tf.nn.softmax(self.resample_logits, name='resample_logits')

        # resample_accuracy
        with tf.name_scope("resample_accuracy"):
          resample_correct_pred = tf.equal(tf.argmax(resample_pred, 1), tf.argmax(resample_labels, 1))
          slef.resample_accuracy = tf.reduce_mean(tf.cast(resample_correct_pred, tf.float32))

    # Build training specific ops for the graph
    def build_train_op(self):
        tf.summary.scalar('learning_rate', FLAGS.learning_rate)

        feature_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='extract_feature')
        source_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='camera_source_discriminator')
        resample_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resample_operation_discriminator')

        # loss function
        with tf.name_scope("cross_ent"):
            # non_resample_labels = tf.constant(0.2, shape = ([FLAGS.batch_size, FLAGS.resample_classes_num]))
            # feature_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = resample_logits, labels = non_resample_labels));
            source_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = source_logits, labels = source_labels))
            resample_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = resample_logits, labels = resample_labels))
            feature_loss =tf.log(tf.div(resample_loss, source_loss))

        # optimizer
        with tf.name_scope("train_feature"):
            # Get gradients of all trainable variables
            feature_gradients = tf.gradients(feature_loss, feature_var_list)
            feature_gradients = list(zip(feature_gradients, feature_var_list))
            # Create optimizer and apply gradient descent to the trainable variables
            self.feature_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(feature_loss, var_list=feature_var_list)

        with tf.name_scope("train_source"):
            # Get gradients of all trainable variables
            source_gradients = tf.gradients(source_loss, source_var_list+feature_var_list)
            source_gradients = list(zip(source_gradients, source_var_list+feature_var_list))
            # Create optimizer and apply gradient descent to the trainable variables
            self.source_optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(source_loss, var_list= source_var_list + feature_var_list)

        with tf.name_scope("train_resample"):
            # Get gradients of all trainable variables
            resample_gradients = tf.gradients(resample_loss, resample_var_list+feature_var_list)
            resample_gradients = list(zip(resample_gradients, resample_var_list+feature_var_list))
            # Create optimizer and apply gradient descent to the trainable variables
            self.resample_optimizer = tf.train.MomentumOptimizer(learning_rate = FLAGS.learning_rate, momentum = 0.9).minimize(resample_loss, var_list=resample_var_list + feature_var_list)

        slef.train_ops.append(self.source_optimizer)
        self.train_ops.append(self.resample_optimizer)
        self.train_ops.append(self.feature_optimizer)
        self.train_op = tf.group(*self.train_ops)

# convolution operation
def conv(inputs, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(dtype=tf.float32,name='weights',
                                 shape = [filter_height, filter_width, inputs.get_shape()[-1], num_filters],
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(dtype=tf.float32, name='biases',
                                 shape=[num_filters], initializer=tf.constant_initializer(0.1))
        conv_ = tf.nn.conv2d(inputs, kernel, strides=[1, stride_y, stride_x, 1], padding=padding)
        conv = tf.nn.bias_add(conv_, biases)
        return conv

# relu opretation
def relu(inputs, name):
    with tf.variable_scope(name) as scope:
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
                                 name='fc_biases', initializer=tf.constant_initializer(0))
        return tf.nn.xw_plus_b(inputs, weights, biases, name=name)

# batch_normalization
def batch_normalization(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
