# -*- coding: UTF-8 -*-
import os
from PIL import Image
import tensorflow as tf

class DataGenerator:
    def __init__(self, mode, batch_size, source_classes_num, resample_classes_num):
        #self.write_to_tfrecord(mode)
        if mode == 'train':
            self.data_size = 24220
        else:
            self.data_size = 18232
        self.read_from_tfrecord(batch_size, source_classes_num, resample_classes_num, mode)

    def write_to_tfrecord(self, mode):
      with tf.name_scope("write_image_to_tfrecord"):
        #存放图片个数
        bestnum = 1000
        #第几个图片
        num = 0
        #第几个TFRecord文件
        recordfilenum = 0
        #tfrecords格式文件名以及数据集的名字
        tf_filepath = 'E:\\shuqian\\resample\\tfrecord\\'
        if mode == 'train':
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
            img_path = line.split()[0] #每一个图片的地址

            # 写入下一个文件
            if num > bestnum:
                num = 1
                recordfilenum = recordfilenum + 1
                #tfrecords格式文件名
                if mode == 'train':
                    ftrecordfilename = ('train_image.tfrecords_%.2d' % recordfilenum)
                else:
                    ftrecordfilename = ('test_image.tfrecords_%.2d' % recordfilenum)
                writer= tf.python_io.TFRecordWriter(tf_filepath+ftrecordfilename)

            # 加载文件
            img=Image.open(img_path)
            img_raw=img.tobytes()#将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                #标注图片数据的source和resample类型
                'source_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line.split()[1])])),
                'resample_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line.split()[2])])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  #序列化为字符串

        writer.close()
        self.data_size = recordfilenum*bestnum + num
        print (mode, self.data_size)

    def read_from_tfrecord(self, batch_size, source_classes_num, resample_classes_num, mode):
      with tf.name_scope("read_image_from_tfrecord"):
        if mode == 'train':
            files = tf.train.match_filenames_once('E:\\shuqian\\resample\\tfrecord\\train_image.tfrecords*')
        else:
            files = tf.train.match_filenames_once('E:\\shuqian\\resample\\tfrecord\\test_image.tfrecords*')
        filename_queue = tf.train.string_input_producer(files) #读入流中
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                       features={
                                           'source_label': tf.FixedLenFeature([], tf.int64),
                                           'resample_label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
        #tf.decode_raw可以将字符串解析成图像对应的像素数组
        image = tf.decode_raw(features['img_raw'], tf.uint8)
        image = tf.reshape(image, [128,128,3])
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        source_label = tf.cast(features['source_label'], tf.int32)
        resample_label = tf.cast(features['resample_label'], tf.int32)
        if mode == 'train':
            example_queue = tf.FIFOQueue(
                # 队列容量
                capacity = 5 * batch_size,
                dtypes = [tf.float32, tf.int32, tf.int32],
                # 图片数据尺寸，标签尺寸
                shapes = [[128, 128, 3], (), ()])
            # 读线程的数量
            num_threads = 4
        else:
            example_queue = tf.FIFOQueue(
                capacity = 3 * batch_size,
                dtypes = [tf.float32, tf.int32, tf.int32],
                shapes = [[128, 128, 3], (), ()])
            # 读线程的数量
            num_threads = 2
        # 数据入队操作
        example_enqueue_op = example_queue.enqueue([image, source_label, resample_label])
        # 队列执行器
        tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
          example_queue, [example_enqueue_op] * num_threads))

        # 数据出队操作，从队列读取Batch数据
        images, source_labels, resample_labels = example_queue.dequeue_many(batch_size)
        # 将标签数据由稀疏格式转换成稠密格式
        # [ 2,       [[0,1,0,0,0]
        #   4,        [0,0,0,1,0]
        #   3,   -->  [0,0,1,0,0]
        #   5,        [0,0,0,0,1]
        #   1 ]       [1,0,0,0,0]]
        source_labels = tf.reshape(source_labels, [batch_size, 1])
        source_indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
        source_labels = tf.sparse_to_dense(tf.concat(values=[source_indices, source_labels], axis=1),
                      [batch_size, source_classes_num], 1.0, 0.0)
        resample_labels = tf.reshape(resample_labels, [batch_size, 1])
        resample_indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
        resample_labels = tf.sparse_to_dense(tf.concat(values=[resample_indices,
                          resample_labels], axis=1),[batch_size, resample_classes_num], 1.0, 0.0)

        #检测数据维度
        assert len(images.get_shape()) == 4
        assert images.get_shape()[0] == batch_size
        assert images.get_shape()[-1] == 3
        assert len(source_labels.get_shape()) == 2
        assert source_labels.get_shape()[0] == batch_size
        assert source_labels.get_shape()[1] == source_classes_num

        self.images = images
        self.source_labels = source_labels
        self.resample_labels = resample_labels
