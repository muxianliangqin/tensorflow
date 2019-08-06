from cnn.layer.common import Layer
from cnn.layer.resnet import ResNetLayer
from cnn.block.common import Block
import tensorflow as tf


class ResNetBlock(Block):
    default_scope = 'res_net_block'
    layer = ResNetLayer()

    def exec(self):
        with tf.variable_scope(self.default_scope):
            shortcut = self.inputs
            if self.in_channels < self.out_channels:
                layer = Layer()
                layer.config(shortcut, [self.kernel_size, self.kernel_size, self.in_channels, self.out_channels])
                shortcut = layer.set_scope('shortcut').exec()
                strides = 2
            reduce_kernel_size = 1
            reduce_channels = self.out_channels // 4
            filters1 = [reduce_kernel_size, reduce_kernel_size, self.in_channels, reduce_channels]
            self.inputs = self.layer.set_scope('layer1').config(self.inputs, filters1, strides).exec()
            filters2 = [self.kernel_size, self.kernel_size, reduce_channels, reduce_channels]
            self.inputs = self.layer.set_scope('layer2').config(self.inputs, filters2).exec()
            filters3 = [reduce_kernel_size, reduce_kernel_size, reduce_channels, self.out_channels]
            self.inputs = self.layer.set_scope('layer3').config(self.inputs, filters3).exec()
            self.inputs = tf.add(shortcut, self.inputs)
            return self.inputs
