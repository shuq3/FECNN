import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from model import FECNN
from dataGenerator import DataGenerator

flags = tf.app.flags
# Learning params
flags.DEFINE_integer("epoch", 200, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.01]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")

# Network params
flags.DEFINE_integer("display_step", 60, "How often we want to write the tf.summary data to disk [60]")
flags.DEFINE_integer("source_classes_num", 10, "The number of camera sources [10]")
flags.DEFINE_integer("resample_classes_num", 10, "The number of resample categories [5]")

# Path for tf.summary.FileWriter and to store model checkpoints
flags.DEFINE_string("checkpoint_dir", "E:\\shuqian\\resample\\checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("filewriter_path", "E:\\shuqian\\resample\\graph", "Directory name to save graphs [E:\\shuqian\\resample\\graph]")
flags.DEFINE_string("train_graph_dir", "E:\\shuqian\\resample\\graph\\train", "Directory name to save train graph [E:\\shuqian\\resample\\graph\\train]")
flags.DEFINE_string("test_graph_dir", "E:\\shuqian\\resample\\graph\\test", "Directory name to save test graph [E:\\shuqian\\resample\\graph\\test]")
FLAGS = flags.FLAGS

# TF placeholder for graph input and output
images = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, 3], name = 'images')
source_labels = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.resample_classes_num], name = 'source_labels')
resample_labels = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.resample_classes_num], name = 'resample_labels')
istrain = tf.placeholder(bool, name = 'istrain')

# set train operations
def main(_):
    # Initialize model
    model = FECNN(images, FLAGS.source_classes_num, FLAGS.resample_classes_num, istrain)

    # Link variable to model output
    source_logits = model.source_logits
    source_pred = model.source_pred
    resample_logits = model.resample_logits
    resample_pred = model.resample_pred

    # List of trainable variables of the layers we want to train
    feature_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='extract_feature')
    source_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='camera_source_discriminator')
    resample_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resample_operation_discriminator')

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        source_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = source_logits, labels = source_labels))
        class_weigh = tf.range(0, FLAGS.resample_classes_num, 1, tf.float32)
        class_weigh = tf.tile(class_weigh, [FLAGS.batch_size])
        class_weigh = tf.reshape(class_weigh, shape=[FLAGS.batch_size, FLAGS.resample_classes_num])
        true_label = tf.cast(tf.reshape(tf.argmax(resample_labels, 1), [FLAGS.batch_size,1]), tf.float32)
        final_weigh = tf.square(tf.subtract(class_weigh, true_label))
        pred_dif = tf.square(model.resample_pred)
        resample_loss = tf.reduce_mean(tf.reduce_sum(tf.cast(pred_dif*final_weigh, tf.float32), 1))
        feature_weigh = tf.cast(tf.square(FLAGS.resample_classes_num-1), tf.float32)
        final_weigh_feature = feature_weigh-final_weigh
        feature_loss = tf.reduce_mean(tf.reduce_sum(tf.cast(pred_dif*final_weigh_feature, tf.float32), 1))+source_loss

    # Train op_
    with tf.name_scope("train_feature"):
        # Get gradients of all trainable variables
        feature_gradients = tf.gradients(feature_loss, feature_var_list)
        feature_gradients = list(zip(feature_gradients, feature_var_list))
        # Create optimizer and apply gradient descent to the trainable variables
        feature_optimizer = tf.train.MomentumOptimizer(learning_rate = FLAGS.learning_rate, momentum=0.9).minimize(feature_loss, var_list=feature_var_list)

    with tf.name_scope("train_source"):
        # Get gradients of all trainable variables
        source_gradients = tf.gradients(source_loss, source_var_list+feature_var_list)
        source_gradients = list(zip(source_gradients, source_var_list+feature_var_list))
        # Create optimizer and apply gradient descent to the trainable variables
        source_optimizer = tf.train.MomentumOptimizer(learning_rate = FLAGS.learning_rate, momentum=0.9).minimize(source_loss, var_list= source_var_list + feature_var_list)

    with tf.name_scope("train_resample"):
        # Get gradients of all trainable variables
        resample_gradients = tf.gradients(resample_loss, resample_var_list)
        resample_gradients = list(zip(resample_gradients, resample_var_list))
        resample_optimizer = tf.train.MomentumOptimizer(learning_rate = FLAGS.learning_rate, momentum=0.9).minimize(resample_loss, var_list=resample_var_list)

    # Add gradients to summary
    for feature_gradient, feature_var in feature_gradients:
        tf.summary.histogram(feature_var.name + '/feature_gradient', feature_gradient)

    for source_gradient, source_var in source_gradients:
        tf.summary.histogram(source_var.name + '/source_gradient', source_gradient)

    for resample_gradient, resample_var in resample_gradients:
        tf.summary.histogram(resample_var.name + '/resample_gradient', resample_gradient)

    # Add the variables we train to the summary
    for var in feature_var_list:
        tf.summary.histogram(var.name, var)

    for var in source_var_list:
        tf.summary.histogram(var.name, var)

    for var in resample_var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('feature_cross_entropy', feature_loss)
    tf.summary.scalar('source_cross_entropy', source_loss)
    tf.summary.scalar('resample_cross_entropy', resample_loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("source_accuracy"):
      source_correct_pred = tf.equal(tf.argmax(source_pred, 1), tf.argmax(source_labels, 1))
      source_accuracy = tf.reduce_mean(tf.cast(source_correct_pred, tf.float32))

    with tf.name_scope("resample_accuracy"):
      resample_correct_pred = tf.equal(tf.argmax(resample_pred, 1), tf.argmax(resample_labels, 1))
      resample_accuracy = tf.reduce_mean(tf.cast(resample_correct_pred, tf.float32))

    with tf.name_scope("resample_test_accuracy"):
      resample_test_correct_pred_1 = tf.equal(tf.argmax(resample_pred, 1), tf.argmax(resample_labels, 1))
      resample_test_correct_pred_2 = tf.equal(tf.argmax(resample_pred, 1), tf.argmax(resample_labels, 1)+1)
      resample_test_accuracy_1 = tf.reduce_mean(tf.cast(resample_test_correct_pred_1, tf.float32))
      resample_test_accuracy_2 = tf.reduce_mean(tf.cast(resample_test_correct_pred_2, tf.float32))
      resample_test_accuracy = tf.add(resample_test_accuracy_1, resample_test_accuracy_2)

    # Add the accuracy to the summary
    tf.summary.scalar('source_accuracy', source_accuracy)
    tf.summary.scalar('resample_accuracy', resample_accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    train_writer = tf.summary.FileWriter(FLAGS.train_graph_dir)
    test_writer = tf.summary.FileWriter(FLAGS.test_graph_dir)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Initalize the data generator seperately for the training and validation set
    train_generator = DataGenerator('train', FLAGS.batch_size, FLAGS.source_classes_num, FLAGS.resample_classes_num)
    val_generator = DataGenerator('test', FLAGS.batch_size, FLAGS.source_classes_num, FLAGS.resample_classes_num)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(train_generator.data_size / FLAGS.batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(val_generator.data_size / FLAGS.batch_size).astype(np.int16)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)

    with tf.device('/gpu:0'):
        # Start Tensorflow session
        with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

            # Initialize all variables
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Add the model graph to TensorBoard
            for epoch in range(FLAGS.epoch):
                train_writer.add_graph(sess.graph)

                print("{} Start training...".format(datetime.now()))
                print("{} Open Tensorboard at --logdir {}".format(datetime.now(), FLAGS.filewriter_path))

                # Loop over number of epochs
                print("{} Epoch number: {}".format(datetime.now(), epoch+1))
                if epoch == 10:
                    FLAGS.learning_rate = FLAGS.learning_rate / 2
                if epoch % 50 == 0:
                    FLAGS.learning_rate = FLAGS.learning_rate / 5
                step = 1
                source_train_acc = 0.0
                resample_train_acc = 0.0
                train_count = 0
                while step < train_batches_per_epoch:
                    # First: train extract_feature network and source_network
                    images_, source_labels_, resample_labels_ = sess.run([train_generator.images,
                                                                         train_generator.source_labels, train_generator.resample_labels])
                    source_op, source_acc, _ = sess.run([source_optimizer, source_accuracy, model.extra_train_ops],
                                             feed_dict={istrain: True,
                                                        images: images_,
                                                        source_labels: source_labels_})

                    # Second: train resample network
                    resample_op, resample_acc = sess.run([resample_optimizer, resample_accuracy],
                                                feed_dict={istrain: True,
                                                           images: images_,
                                                           resample_labels: resample_labels_})

                    # Third: train extract_feature network
                    feature_op = sess.run(feature_optimizer,
                                feed_dict={istrain: True,
                                           images: images_,
                                           source_labels: source_labels_,
                                           resample_labels: resample_labels_})

                    resample_train_acc += resample_acc
                    source_train_acc += source_acc
                    train_count += 1
                    # Generate summary with the current batch of data and write to file
                    if step % FLAGS.display_step == 0:
                        s = sess.run(merged_summary, feed_dict = {istrain: True,
                                                                  images: images_,
                                                                  source_labels: source_labels_,
                                                                  resample_labels: resample_labels_})
                        train_writer.add_summary(s, epoch*train_batches_per_epoch + step)
                    step += 1

                source_train_acc /= train_count
                resample_train_acc /= train_count
                print("{} Train source Accuracy = {:.4f}".format(datetime.now(), source_train_acc))
                print("{} Train resample Accuracy = {:.4f}".format(datetime.now(), resample_train_acc))

                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                source_test_acc = 0.0
                resample_test_acc = 0.0
                test_count = 0
                step = 1
                while step < val_batches_per_epoch:
                    val_images_, val_source_labels_,  val_resample_labels_ = sess.run([val_generator.images,
                                                                                      val_generator.source_labels, val_generator.resample_labels])
                    source_acc, resample_acc = sess.run([source_accuracy, resample_test_accuracy],
                                                    feed_dict = {istrain: False,
                                                                images: val_images_,
                                                                source_labels: val_source_labels_,
                                                                resample_labels: val_resample_labels_})
                    source_test_acc += source_acc
                    resample_test_acc += resample_acc
                    test_count += 1
                    # Generate summary with the current batch of data and write to file
                    if step % 30 == 0:
                        s = sess.run(merged_summary,
                                     feed_dict = {istrain: False,
                                                  images: val_images_,
                                                  source_labels: val_source_labels_,
                                                  resample_labels: val_resample_labels_})
                        test_writer.add_summary(s, epoch*val_batches_per_epoch + step)
                    step += 1

                source_test_acc /= test_count
                resample_test_acc /= test_count

                print("{} Validation source accuracy = {:.4f}".format(datetime.now(), source_test_acc))
                print("{} Validation resample accuracy = {:.4f}".format(datetime.now(), resample_test_acc))

                print("{} Saving checkpoint of model...".format(datetime.now()))

                #save checkpoint of the model
                checkpoint_name = os.path.join(FLAGS.checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
            train_writer.close()
            test_writer.close()
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
     tf.app.run()
