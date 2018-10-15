"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import os
from PIL import Image
import tensorflow as tf

class DataGenerator:
  def __init__(self, mode, batch_size, source_classes_num, resample_classes_num):
    if mode:
        self.data_size = 24220
    else:
        self.data_size = 18232
    # self.write_to_tfrecord(mode)
    self.read_from_tfrecord(batch_size, source_classes_num, resample_classes_num, mode)

  def write_to_tfrecord(self, mode):
    with tf.name_scope("write_image_to_tfrecord"):
      # largest image number in a tfrecord file
      bestnum = 1000
      # index of image in current tfrecord file
      num = 0
      # index of tfrecord file
      recordfilenum = 0
      # tfrecords filename
      tf_filepath = 'E:\\shuqian\\resample\\tfrecord\\'
      if mode:
        ftrecordfilename = ('train_image.tfrecords_%.2d' % recordfilenum)
        file = open('E:\\shuqian\\resample\\dataset\\train_rand.txt', 'r')
      else:
        ftrecordfilename = ('test_image.tfrecords_%.2d' % recordfilenum)
        file = open('E:\\shuqian\\resample\\dataset\\test_rand.txt', 'r')
      writer= tf.python_io.TFRecordWriter(tf_filepath+ftrecordfilename)

      lines = file.readlines()
      # random.shuffle(lines)
      for line  in lines:
        num = num + 1
        img_path = line.split()[0] # image pathname

        # write to next tfrecord file
        if num > bestnum:
          num = 1
          recordfilenum = recordfilenum + 1
          # tfrecords filename
          if mode:
            ftrecordfilename = ('train_image.tfrecords_%.2d' % recordfilenum)
          else:
            ftrecordfilename = ('test_image.tfrecords_%.2d' % recordfilenum)
          writer= tf.python_io.TFRecordWriter(tf_filepath+ftrecordfilename)

          # load image
          img=Image.open(img_path)
          img_raw=img.tobytes()# transform images to binary form
          example = tf.train.Example(features=tf.train.Features(feature={
          # store images and labels
               'source_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line.split()[1])])),
               'resample_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line.split()[2])])),
               'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
          }))
          writer.write(example.SerializeToString())

      writer.close()
      self.data_size = recordfilenum*bestnum + num
      print (mode, self.data_size)

  def read_from_tfrecord(self, batch_size, source_classes_num, resample_classes_num, mode):
    with tf.name_scope("read_image_from_tfrecord"):
      if mode:
        files = tf.train.match_filenames_once('E:\\shuqian\\resample\\tfrecord\\train_image.tfrecords*')
      else:
        files = tf.train.match_filenames_once('E:\\shuqian\\resample\\tfrecord\\test_image.tfrecords*')
      # read into queue
      filename_queue = tf.train.string_input_producer(files)
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
      # get images and labels
      features = tf.parse_single_example(serialized_example,
                 features={'source_label': tf.FixedLenFeature([], tf.int64),
                           'resample_label': tf.FixedLenFeature([], tf.int64),
                           'img_raw' : tf.FixedLenFeature([], tf.string)})
      # tf.decode_raw can transform strings to array
      image = tf.decode_raw(features['img_raw'], tf.uint8)
      image = tf.reshape(image, [128,128,3])
      image = tf.cast(image, tf.float32)
      image = tf.image.per_image_standardization(image)
      source_label = tf.cast(features['source_label'], tf.int32)
      resample_label = tf.cast(features['resample_label'], tf.int32)
      if mode:
        example_queue = tf.FIFOQueue(
          # capacity of this queue
          capacity = 5 * batch_size,
          dtypes = [tf.float32, tf.int32, tf.int32],
          # image size and label size
          shapes = [[128, 128, 3], (), ()])
        # 读线程的数量
        num_threads = 4
      else:
        example_queue = tf.FIFOQueue(
          capacity = 3 * batch_size,
          dtypes = [tf.float32, tf.int32, tf.int32],
          shapes = [[128, 128, 3], (), ()])
        num_threads = 2
      # enqueue data
      example_enqueue_op = example_queue.enqueue([image, source_label, resample_label])
      # queue runner
      tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
               example_queue, [example_enqueue_op] * num_threads))

      # dequeue data
      images, source_labels, resample_labels = example_queue.dequeue_many(batch_size)
      # transform labels from [1, batch_size] to [class_num, batch_size]
      source_labels = tf.reshape(source_labels, [batch_size, 1])
      source_indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
      source_labels = tf.sparse_to_dense(tf.concat(values=[source_indices, source_labels], axis=1),
                      [batch_size, source_classes_num], 1.0, 0.0)
      resample_labels = tf.reshape(resample_labels, [batch_size, 1])
      resample_indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
      resample_labels = tf.sparse_to_dense(tf.concat(values=[resample_indices,
                          resample_labels], axis=1),[batch_size, resample_classes_num], 1.0, 0.0)

      # check data dim
      assert len(images.get_shape()) == 4
      assert images.get_shape()[0] == batch_size
      assert images.get_shape()[-1] == 3
      assert len(source_labels.get_shape()) == 2
      assert source_labels.get_shape()[0] == batch_size
      assert source_labels.get_shape()[1] == source_classes_num

      self.images = images
      self.source_labels = source_labels
      self.resample_labels = resample_labels
