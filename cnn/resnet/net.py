import tensorflow as tf
import cnn.origin.config as config
from cnn.origin.layer import Layer
from cnn.origin.block import Block
from cnn.origin.section import Section
from cnn.origin.net import Net


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
            self.shortcut = Layer().set_scope('shortcut')\
                                   .config(self.shortcut, self.out_channels, strides_size=2).exec()
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
            self.inputs = self.layer.set_scope('layer0') \
                .config(self.inputs, self.out_channels, strides_size=strides, kernel_size=3).exec()
            self.inputs = self.layer.set_scope('layer1') \
                .config(self.inputs, self.out_channels, kernel_size=3).exec()
        else:
            self.inputs = self.layer.set_scope('layer0') \
                .config(self.inputs, reduce_channels, strides_size=strides).exec()
            self.inputs = self.layer.set_scope('layer1') \
                .config(self.inputs, reduce_channels, kernel_size=3).exec()
            self.inputs = self.layer.set_scope('layer2') \
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
    start_channels = 64

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
            self.inputs = Layer().set_scope('section_0')\
                                 .config(self.inputs, self.start_channels, kernel_size=7).exec()
            self.inputs = Layer().set_scope('section_0_1')\
                                 .config(self.inputs, self.start_channels, kernel_size=5).exec()
            self.inputs = self.section.set_scope('section_1')\
                                      .config(self.inputs, self.start_channels, self.blocks_num_every_section[0])\
                                      .exec()
            self.inputs = self.section.set_scope('section_2')\
                                      .config(self.inputs, self.start_channels * 2, self.blocks_num_every_section[1])\
                                      .exec()
            # self.inputs = self.section.set_scope('section_3')\
            #                           .config(self.inputs, self.start_channels * 4, self.blocks_num_every_section[2])\
            #                           .exec()
            # self.inputs = self.section.set_scope('section_4')\
            #                           .config(self.inputs, self.start_channels * 8, self.blocks_num_every_section[3])\
            #                           .exec()
            return self
