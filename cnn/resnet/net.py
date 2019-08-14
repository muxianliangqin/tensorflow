import tensorflow as tf
import numpy as np
from cnn.origin.layer import Layer
from cnn.origin.block import Block
from cnn.origin.section import Section
from cnn.origin.net import Net
from cnn.origin.batch import Batch
import cnn.utils.base_util as util
import time

DATA_TRAIN = './data/cifar-100-python/train'
DATA_TEST = './data/cifar-100-python/test'
MODEL_SAVE_PATH = './model/cifar-resnet.ckpt'
LOG_TRAIN = './log/train'
LOG_TEST = './log/test'
BATCH_SIZE = 100
TRAIN_TOTAL_SIZE = 50000
TEST_TOTAL_SIZE = 10000
PRINT_EVERY_TIMES = 1
EPOCH = 50
CLASSES = 100
FILTERS_SIZE = 32
LEARNING_RATE = 0.001


class ResNetLayer(Layer):
    def layer(self):
        self.inputs = Layer.do_bn(self.inputs, name='bn')
        self.inputs = tf.nn.relu(self.inputs, name='relu')
        self.inputs = tf.nn.conv2d(self.inputs, self.filters, self.strides, self.padding, name='conv2d')
        return self
            

class ResNetBlock(Block):
    def compute_shortcut(self):
        in_channels = self.inputs.get_shape().as_list()[-1]
        if in_channels < self.out_channels:
            strides = 2
            self.shortcut = Layer().set_scope('shortcut')\
                                   .config(self.shortcut, self.out_channels, strides_size=strides).exec()
        return self

    def compute_residual(self):
        in_channels = self.inputs.get_shape().as_list()[-1]
        strides = 1
        if in_channels < self.out_channels:
            strides = 2
        reduce_channels = self.out_channels // 4
        layers_num = 50
        net = self.sup.sup
        if hasattr(net, 'layers_num'):
            layers_num = net.layers_num
        if layers_num in [18, 34]:
            self.inputs = self.layer.set_scope('layer1') \
                .config(self.inputs, self.out_channels, strides_size=strides, kernel_size=3).exec()
            self.inputs = self.layer.set_scope('layer2') \
                .config(self.inputs, self.out_channels, kernel_size=3).exec()
        else:
            self.inputs = self.layer.set_scope('layer1') \
                .config(self.inputs, reduce_channels, strides_size=strides).exec()
            self.inputs = self.layer.set_scope('layer2') \
                .config(self.inputs, reduce_channels, kernel_size=3).exec()
            self.inputs = self.layer.set_scope('layer3') \
                .config(self.inputs, self.out_channels).exec()
        return self


class ResNetSection(Section):
    def section(self):
        self.inputs = self.block.set_scope('block_0').config(self.inputs, self.out_channels).exec()
        for i in range(1, self.blocks_num):
            scope = 'block_{}'.format(i)
            self.inputs = self.block.set_scope(scope).config(self.inputs, self.out_channels).exec()
        return self


class ResNet(Net):
    layers_num = 50
    blocks_num_every_section = [3, 4, 6, 3]
    start_channels = 256

    def set_layer_num(self, num):
        self.layers_num = num
        if num == 18:
            self.blocks_num_every_section = [2, 2, 2, 2]
            self.start_channels = 64
        elif num == 34:
            self.blocks_num_every_section = [3, 4, 6, 3]
            self.start_channels = 64
        elif num == 50:
            self.blocks_num_every_section = [3, 4, 6, 3]
            self.start_channels = 256
        elif num == 101:
            self.blocks_num_every_section = [3, 4, 23, 3]
            self.start_channels = 256
        elif num == 152:
            self.blocks_num_every_section = [3, 4, 36, 3]
            self.start_channels = 256
        else:
            self.blocks_num_every_section = [3, 4, 6, 3]
            self.start_channels = 256
        return self

    def before_train(self):
        with tf.variable_scope('before_train', reuse=tf.AUTO_REUSE):
            self.inputs = tf.image.convert_image_dtype(self.inputs, dtype=tf.float32, name='convert_image_dtype')
            self.inputs = tf.image.random_flip_up_down(self.inputs)
            self.inputs = tf.image.random_flip_left_right(self.inputs)
            return self

    def net(self):
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.inputs = Layer().set_scope('section_0').config(self.inputs, self.start_channels, kernel_size=5).exec()
            self.inputs = self.section.set_scope('section_1').config(self.inputs, self.start_channels,
                                                                     self.blocks_num_every_section[0]).exec()
            self.inputs = self.section.set_scope('section_2').config(self.inputs, self.start_channels * 2,
                                                                     self.blocks_num_every_section[1]).exec()
            self.inputs = self.section.set_scope('section_3').config(self.inputs, self.start_channels * 4,
                                                                     self.blocks_num_every_section[2]).exec()
            self.inputs = self.section.set_scope('section_4').config(self.inputs, self.start_channels * 8,
                                                                     self.blocks_num_every_section[3]).exec()
            return self


class TestBatch(Batch):
    def batch_index(self):
        self.index_batch = np.random.randint(0, TEST_TOTAL_SIZE, [BATCH_SIZE])


def train():
    t1 = time.time()
    tf.reset_default_graph()
    x = tf.placeholder(tf.uint8, shape=[BATCH_SIZE, 32, 32, 3], name='images')
    y = tf.placeholder(tf.int64, shape=[BATCH_SIZE], name='labels')
    net = ResNet().build(ResNetLayer(), ResNetBlock(), ResNetSection())\
                  .set_scope('res_net_50').set_lr(LEARNING_RATE).set_layer_num(34)\
                  .config(x, y, CLASSES)
    train_op = net.exec()
    accuracy = net.accuracy
    graph = tf.get_default_graph()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session(graph=graph) as sess:
        data_train = util.load_data(DATA_TRAIN)
        data_test = util.load_data(DATA_TEST)
        batch_train = Batch().config(data_train, TRAIN_TOTAL_SIZE, BATCH_SIZE, EPOCH)
        batch_test = TestBatch().config(data_test, TEST_TOTAL_SIZE, BATCH_SIZE)
        writer_train = tf.summary.FileWriter(LOG_TRAIN, sess.graph)
        writer_test = tf.summary.FileWriter(LOG_TEST)
        sess.run(tf.global_variables_initializer())
        while True:
            try:
                labels_train, images_train = batch_train.exec()
                feed_dict_train = {x: images_train, y: labels_train}
                summary_train, _ = sess.run([summaries, train_op], feed_dict=feed_dict_train)
                # test
                labels_test, images_test = batch_test.exec()
                feed_dict_test = {x: images_test, y: labels_test}
                summary_test, _ = sess.run([summaries, accuracy], feed_dict=feed_dict_test)
                step = batch_train.step
                if step % PRINT_EVERY_TIMES == 0:
                    writer_train.add_summary(summary_train, step)
                    writer_test.add_summary(summary_test, step)
                if batch_train.cycle_one_epoch:
                    print('EPOCH:{},save model'.format(batch_test.epoch + 1))
                    saver.save(sess, save_path=MODEL_SAVE_PATH, global_step=step)
            except Exception as e:
                break

    t2 = time.time()
    print('耗时：{}'.format(util.str_time(t2 - t1)))


def graph_test():
    tf.reset_default_graph()
    x = tf.placeholder(tf.uint8, shape=[BATCH_SIZE, 32, 32, 3], name='images')
    y = tf.placeholder(tf.int64, shape=[BATCH_SIZE], name='labels')
    train_op = ResNet().build(ResNetLayer(), ResNetBlock(), ResNetSection())\
                       .set_scope('res_net_50').set_lr(LEARNING_RATE).set_layer_num(34)\
                       .config(x, y, CLASSES).exec()
    tf.summary.FileWriter('./log', tf.get_default_graph())
    # [print(i) for i in tf.global_variables()]


train()
