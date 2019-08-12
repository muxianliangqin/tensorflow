import tensorflow as tf
import numpy as np
from cnn.origin.layer import Layer
from cnn.origin.block import Block
from cnn.origin.section import Section
from cnn.origin.net import Net
import cnn.utils.base_util as util
import time

DATA_TRAIN = './data/cifar-100-python/train'
DATA_TEST = './data/cifar-100-python/test'
BATCH_SIZE = 100
TRAIN_TOTAL_SIZE = 50000
TEST_TOTAL_SIZE = 10000
TRAIN_TIMES_FOR_EPOCH = int(TRAIN_TOTAL_SIZE / BATCH_SIZE)
TEST_TIMES_FOR_EPOCH = int(TEST_TOTAL_SIZE / BATCH_SIZE)
PRINT_EVERY_TIMES = 50
EPOCH = 50
CLASSES = 100
FILTERS_SIZE = 32
W_HEIGHT = 16*FILTERS_SIZE*4
CHANNELS_1 = 64
CHANNELS_2 = CHANNELS_1 * 2
CHANNELS_3 = CHANNELS_1 * 4
CHANNELS_4 = CHANNELS_1 * 8
CHANNELS_5 = CHANNELS_1 * 16
LEARNING_RATE = 0.001


class ResNetLayer(Layer):
    def layer(self):
        self.inputs = Layer.do_bn(self.inputs, name='bn')
        self.inputs = tf.nn.relu(self.inputs, name='relu')
        self.inputs = tf.nn.conv2d(self.inputs, self.filters, self.strides, self.padding, name='conv2d')
        return self
            

class ResNetBlock(Block):
    def block(self):
        shortcut = self.inputs
        in_channels = self.inputs.get_shape().as_list()[-1]
        strides = 1
        if in_channels < self.out_channels:
            strides = 2
            shortcut = Layer().set_scope('shortcut').config(shortcut, self.out_channels, strides_size=strides).exec()
        reduce_channels = self.out_channels // 4
        self.inputs = self.layer.set_scope('layer1').config(self.inputs, reduce_channels, strides_size=strides).exec()
        self.inputs = self.layer.set_scope('layer2')\
                                .config(self.inputs, reduce_channels, kernel_size=3).exec()
        self.inputs = self.layer.set_scope('layer3').config(self.inputs, self.out_channels).exec()
        self.inputs = tf.add(shortcut, self.inputs, name='add')
        return self


class ResNetSection(Section):
    def section(self):
        self.inputs = self.block.set_scope('block_0').config(self.inputs, self.out_channels).exec()
        for i in range(1, self.blocks_num):
            scope = 'block_{}'.format(i)
            self.inputs = self.block.set_scope(scope).config(self.inputs, self.out_channels).exec()
        return self


class ResNet50(Net):
    layers_num = 50
    blocks_num_every_section = [3, 4, 6, 3]
    start_channels = 64

    def set_layer_num(self, num):
        if num == 18:
            self.blocks_num_every_section = [2, 2, 2, 2]
        elif num == 34:
            self.blocks_num_every_section = [3, 4, 6, 3]
        elif num == 50:
            self.blocks_num_every_section = [3, 4, 6, 3]
        elif num == 101:
            self.blocks_num_every_section = [3, 4, 23, 3]
        elif num == 152:
            self.blocks_num_every_section = [3, 4, 36, 3]
        else:
            self.blocks_num_every_section = [3, 4, 6, 3]

    def set_start_channels(self, channels):
        self.start_channels = channels

    def before_train(self):
        with tf.variable_scope('before_train', reuse=tf.AUTO_REUSE):
            self.inputs = tf.image.convert_image_dtype(self.inputs, dtype=tf.float32, name='convert_image_dtype')
            self.inputs = tf.image.random_flip_up_down(self.inputs)
            self.inputs = tf.image.random_flip_left_right(self.inputs)
            return self

    def net(self):
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.inputs = Layer().set_scope('section_0').config(self.inputs, self.start_channels, kernel_size=3).exec()
            self.inputs = self.section.set_scope('section_1').config(self.inputs, self.start_channels * 2,
                                                                     self.blocks_num_every_section[0]).exec()
            self.inputs = self.section.set_scope('section_2').config(self.inputs, self.start_channels * 4,
                                                                     self.blocks_num_every_section[1]).exec()
            self.inputs = self.section.set_scope('section_3').config(self.inputs, self.start_channels * 8,
                                                                     self.blocks_num_every_section[2]).exec()
            self.inputs = self.section.set_scope('section_4').config(self.inputs, self.start_channels * 16,
                                                                     self.blocks_num_every_section[3]).exec()
            return self


def get_batch(data, idx):
    file_names = np.asarray(data[b'filenames'])
    fine_labels = np.asarray(data[b'fine_labels'])
    coarse_labels = np.asarray(data[b'coarse_labels'])
    image = np.asarray(data[b'data'])
    file_names_batch = file_names[idx]
    fine_labels_batch = fine_labels[idx]
    coarse_labels_batch = coarse_labels[idx]
    image_batch = image[idx].reshape([-1, 32, 32, 3])
    return file_names_batch, fine_labels_batch, coarse_labels_batch, image_batch


def train():
    t1 = time.time()
    tf.reset_default_graph()
    x = tf.placeholder(tf.uint8, shape=[BATCH_SIZE, 32, 32, 3], name='images')
    y = tf.placeholder(tf.int64, shape=[BATCH_SIZE], name='labels')
    net = ResNet50().build(ResNetLayer(), ResNetBlock(), ResNetSection())\
                    .set_scope('res_net_50').set_lr(LEARNING_RATE)\
                    .config(x, y, CLASSES)
    train_op = net.exec()
    accuracy = net.accuracy
    graph = tf.get_default_graph()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session(graph=graph) as sess:
        writer_train = tf.summary.FileWriter('./log/train', sess.graph)
        writer_test = tf.summary.FileWriter('./log/test')
        data_train = util.load_data(DATA_TRAIN)
        data_test = util.load_data(DATA_TEST)
        sess.run(tf.global_variables_initializer())
        idx_train = np.linspace(0, TRAIN_TOTAL_SIZE - 1, TRAIN_TOTAL_SIZE, dtype=np.int32)
        step = 0
        for i in range(EPOCH):
            np.random.shuffle(idx_train)
            for j in range(TRAIN_TIMES_FOR_EPOCH):
                idx_j = np.linspace(j * BATCH_SIZE, (j + 1) * BATCH_SIZE - 1, BATCH_SIZE,
                                    dtype=np.int32)
                idx_train_batch = idx_train[idx_j]
                _, labels_train, _, images_train = get_batch(data_train, idx_train_batch)
                feed_dict_train = {x: images_train, y: labels_train}
                summary_train, _ = sess.run([summaries, train_op], feed_dict=feed_dict_train)

                # test
                idx_test_batch = np.random.randint(0, TEST_TOTAL_SIZE, [BATCH_SIZE])
                _, labels_test, _, images_test = get_batch(data_test, idx_test_batch)
                feed_dict_test = {x: images_test, y: labels_test}
                summary_test, _ = sess.run([summaries, accuracy], feed_dict=feed_dict_test)
                step += 1
                if step % PRINT_EVERY_TIMES == 0:
                    writer_train.add_summary(summary_train, step)
                    writer_test.add_summary(summary_test, step)
            print('EPOCH:{}'.format(i+1))
            saver.save(sess, save_path='./model/cifar-resnet.ckpt', global_step=step)

    t2 = time.time()
    print('耗时：{}'.format(util.str_time(t2 - t1)))


def graph_test():
    tf.reset_default_graph()
    x = tf.placeholder(tf.uint8, shape=[BATCH_SIZE, 32, 32, 3], name='images')
    y = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='labels')
    train_op = ResNet50().build(ResNetLayer(), ResNetBlock(), ResNetSection())\
                         .set_scope('res_net_50').set_lr(LEARNING_RATE)\
                         .config(x, y, CLASSES).exec()
    tf.summary.FileWriter('./log', tf.get_default_graph())
    # [print(i) for i in tf.global_variables()]


train()
