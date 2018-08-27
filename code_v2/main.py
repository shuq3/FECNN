import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from model import FECNN
from dataGenerator import DataGenerator

flags = tf.app.flags
# Learning params
flags.DEFINE_integer("epoch", 1000, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.01]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")

# Network params
flags.DEFINE_integer("display_step", 60, "How often we want to write the tf.summary data to disk [60]")
flags.DEFINE_integer("source_classes_num", 10, "The number of camera sources [10]")
flags.DEFINE_integer("resample_classes_num", 5, "The number of resample categories [5]")
flags.DEFINE_integer("mode", True, "True for train, false for val")

# Path for tf.summary.FileWriter and to store model checkpoints
flags.DEFINE_string("checkpoint_dir", "E:\\shuqian\\resample\\checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("filewriter_path", "E:\\shuqian\\resample\\graph", "Directory name to save graphs [E:\\shuqian\\resample\\graph]")
flags.DEFINE_string("train_graph_dir", "E:\\shuqian\\resample\\graph\\train", "Directory name to save train graph [E:\\shuqian\\resample\\graph\\train]")
flags.DEFINE_string("test_graph_dir", "E:\\shuqian\\resample\\graph\\test", "Directory name to save test graph [E:\\shuqian\\resample\\graph\\test]")
FLAGS = flags.FLAGS

# TF placeholder for graph input and output
images = tf.placeholder(tf.float32, [FLAGS.batch_size, 64, 64, 3], name = 'images')
source_labels = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.source_classes_num], name = 'source_labels')
resample_labels = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.resample_classes_num], name = 'resample_labels')

# set train operations
def train():
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), FLAGS.filewriter_path))

    train_generator = DataGenerator('train', FLAGS.batch_size, FLAGS.source_classes_num, FLAGS.resample_classes_num)
    # Initialize model
    model = FECNN(train_generator.images, FLAGS.source_classes_num, FLAGS.resample_classes_num,
            FLAGS.mode, train_generator.source_labels, train_generator.resample_labels)

    # save precision
    summary_hook = tf.train.SummarySaverHook(
                   save_steps=FLAGS.display_step,
                   output_dir=FLAGS.train_graph_dir,
                   summary_op=tf.summary.merge(model.summaries))
    # print log info
    logging_hook = tf.train.LoggingTensorHook(
                   tensors={'step': model.global_step,
                         'source_loss': model.source_loss,
                         'source_accuracy': model.source_accuracy,
                         'resample_loss': model.resample_loss,
                         'resample_accuracy': model.resample_accuracy},
                   every_n_iter=FLAGS.display_step)

    # TODO model.train_op
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.checkpoint_dir,
        hooks=[logging_hook, _LearningRateSetterHook()],
        chief_only_hooks=[summary_hook],
        save_summaries_steps=0,
        config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8))) as mon_sess:
    while not mon_sess.should_stop():
        mon_sess.run(model.train_op)


def val():
    print("{} Start validation".format(datetime.now()))
    saver = tf.train.Saver()
    # summary file
    test_writer = tf.summary.FileWriter(FLAGS.test_graph_dir)
    test_generator = DataGenerator('test', FLAGS.batch_size, FLAGS.source_classes_num, FLAGS.resample_classes_num)
    test_batches_per_epoch = np.floor(val_generator.data_size / FLAGS.batch_size).astype(np.int16)

    # Initialize model
    model = FECNN(test_generator.images, FLAGS.source_classes_num, FLAGS.resample_classes_num,
            FLAGS.mode=False, test_generator.source_labels, test_generator.resample_labels)

    with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8))) as sess:
        # Initialize all variables
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 逐Batch执行测试
        source_test_acc = 0.0
        resample_test_acc = 0.0
        test_count = 0
        step = 1
        while step < test_batches_per_epoch:
            (source_accuracy, resample_accuracy, global_step) = sess.run([model.source_accuracy,
                                                model.resample_accuracy, model.global_step])
            source_test_acc += source_accuracy
            resample_test_acc += resample_accuracy
            test_count += 1
            test_writer.add_summary(s, global_step*val_batches_per_epoch + step)
        source_test_acc /= test_count
        resample_test_acc /= test_count

        print("{} Validation source accuracy = {:.4f}".format(datetime.now(), source_test_acc))
        print("{} Validation resample accuracy = {:.4f}".format(datetime.now(), resample_test_acc))

        print("{} Saving checkpoint of model...".format(datetime.now()))

        #save checkpoint of the model
        checkpoint_name = os.path.join(FLAGS.checkpoint_dir, 'model_epoch'+str(global_step+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
    test_writer.close()
    coord.request_stop()
    coord.join(threads)

def main(_):
    # start training or evaluating
    with tf.device('/gpu:0'):
        for epoch in range(FLAGS.epoch):
            if epoch < 40:
                self._learning_rate = 0.1
            elif epoch < 60:
                self._learning_rate = 0.01
            elif epoch < 80:
                self._learning_rate = 0.001
            else:
                self._learning_rate = 0.0001
            # Loop over number of epochs
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))
            train()
            evaluate()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
