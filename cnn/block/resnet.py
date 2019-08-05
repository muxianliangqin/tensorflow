from cnn.layer.resnet import ResNetLayer
from cnn.block.common import Block
import tensorflow as tf


class ResNetBlock(Block):
    default_name = 'res_net_block'

    def exec(self):
        with tf.variable_scope(self.default_name):
            kernel_size = 1
            out_channels = self.out_channels // 4
            layer = ResNetLayer(self.inputs, [kernel_size, kernel_size, self.in_channels, out_channels])
            layer = layer.exec()
