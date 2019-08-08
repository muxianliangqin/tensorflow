import tensorflow as tf
import numpy as np
from cnn.origin.layer import Layer
from cnn.origin.block import Block
from cnn.origin.section import Section
from cnn.origin.net import Net
from cnn.utils.tf_util import do_bn, get_w
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
EPOCH = 150
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
        self.inputs = do_bn(self.inputs, name='bn')
        self.inputs = tf.nn.relu(self.inputs, name='relu')
        return tf.nn.conv2d(self.inputs, self.filters, self.strides, self.padding, name='conv2d')
            

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
        return self.inputs


class ResNetSection(Section):
    def section(self):
        self.inputs = self.block.set_scope('block_0').config(self.inputs, self.out_channels).exec()
        for i in range(1, self.blocks_num):
            scope = 'block_{}'.format(i)
            self.inputs = self.block.set_scope(scope).config(self.inputs, self.out_channels).exec()
        return self.inputs


class ResNet50(Net):
    def before_train(self):
        with tf.variable_scope('before_train', reuse=tf.AUTO_REUSE):
            self.inputs = tf.image.convert_image_dtype(self.inputs, dtype=tf.float32, name='convert_image_dtype')
            self.inputs = tf.image.random_flip_up_down(self.inputs)
            self.inputs = tf.image.random_flip_left_right(self.inputs)
            return self.inputs

    def net(self):
        self.inputs = Layer().set_scope('section_0').config(self.inputs, CHANNELS_1, kernel_size=3).exec()
        self.inputs = self.section.set_scope('section_1').config(self.inputs, CHANNELS_2, 3).exec()
        self.inputs = self.section.set_scope('section_2').config(self.inputs, CHANNELS_3, 4).exec()
        self.inputs = self.section.set_scope('section_3').config(self.inputs, CHANNELS_4, 6).exec()
        self.inputs = self.section.set_scope('section_4').config(self.inputs, CHANNELS_5, 3).exec()


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
    y = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='labels')
    cls = ResNet50().build(ResNetLayer(), ResNetBlock(), ResNetSection()).set_scope('res_net_50').config(x).exec()
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=cls, name='loss')
    loss_mean = tf.reduce_mean(loss, name='loss_mean')
    global_step = tf.Variable(0, name='global_step')
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.96,
                                               staircase=True, name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
    train_op = optimizer.minimize(loss_mean, global_step=global_step, name='train_op')

    data_train = util.load_data(DATA_TRAIN)
    data_test = util.load_data(DATA_TEST)
    graph = tf.get_default_graph()
    #     var_list = [i for i in tf.global_variables() if i.name.split('/')[1] == 'result']
    #     saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
    #     [print(i) for i in tf.global_variables()]
    #     [print(i.name) for i in graph.get_operations()]
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        idx_train = np.linspace(0, TRAIN_TOTAL_SIZE - 1, TRAIN_TOTAL_SIZE, dtype=np.int32)
        step = 0
        accuracies_train, accuracies_test, losses_train, losses_test = [], [], [], []
        for i in range(EPOCH):
            np.random.shuffle(idx_train)
            for j in range(TRAIN_TIMES_FOR_EPOCH):
                idx_j = np.linspace(j * BATCH_SIZE, (j + 1) * BATCH_SIZE - 1, BATCH_SIZE,
                                    dtype=np.int32)
                idx_train_batch = idx_train[idx_j]
                _, labels_train, _, images_train = get_batch(data_train, idx_train_batch)
                feed_dict_train = {
                    x: images_train,
                    y: labels_train
                }
                cls_train, loss_train, _ = sess.run([cls, loss_mean, train_op], feed_dict=feed_dict_train)
                arg_idx_train = np.argmax(cls_train, axis=1)
                accuracy_train = sum(labels_train == arg_idx_train) / BATCH_SIZE
                # test 
                idx_test_batch = np.random.randint(0, TEST_TOTAL_SIZE, [BATCH_SIZE])
                _, labels_test, _, images_test = get_batch(data_test, idx_test_batch)
                feed_dict_test = {
                    x: images_test,
                    y: labels_test
                }
                cls_test, loss_test = sess.run([cls, loss_mean], feed_dict=feed_dict_test)
                arg_idx_test = np.argmax(cls_test, axis=1)
                accuracy_test = sum(labels_test == arg_idx_test) / BATCH_SIZE

                step += 1
                if step % PRINT_EVERY_TIMES == 0:
                    print('time:{},epoch:{},loss_train:{},loss_test:{},accuracy_train:{:.2%},accuracy_test:{:.2%}'
                          .format(util.cur_time(), step, loss_train, loss_test, accuracy_train, accuracy_test))
                    accuracies_train.append(accuracy_train)
                    accuracies_test.append(accuracy_test)
                    losses_train.append(loss_train)
                    losses_test.append(loss_test)
            saver.save(sess, save_path='./model/resnet/cifar-resnet.ckpt', global_step=step)
        accuracy_map = {
            'accuracies_train': accuracies_train,
            'accuracies_test': accuracies_test,
            'losses_train': losses_train,
            'losses_test': losses_test,
        }
        util.dump_data(accuracy_map, './accuracy_map.pkl')

    t2 = time.time()
    print('耗时：{}'.format(util.str_time(t2 - t1)))


def graph_test():
    tf.reset_default_graph()
    x = tf.placeholder(tf.uint8, shape=[BATCH_SIZE, 32, 32, 3], name='images')
    y = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='labels')
    cls = ResNet50().build(ResNetLayer(), ResNetBlock(), ResNetSection()).set_scope('res_net_50').config(x).exec()
    tf.summary.FileWriter('./log')
    [print(i) for i in tf.global_variables()]


graph_test()
