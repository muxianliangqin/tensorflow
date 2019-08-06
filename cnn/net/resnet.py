import tensorflow as tf
from cnn.net.common import Net
from cnn.block.resnet import ResNetBlock
from cnn.layer.common import Layer


class ResNetNet(Net):
    default_scope = 'res_net'
    block = ResNetBlock()
    channels_1 = 64
    channels_2 = channels_1 * 2
    channels_3 = channels_1 * 4
    channels_4 = channels_1 * 8
    channels_5 = channels_1 * 16

    def exec(self):
        with tf.variable_scope(self.default_scope):
            layer = Layer()
            self.inputs = layer.set_scope('conv_1').config(self.inputs, [3, 3, 3, self.channels_1], 2, "SAME").exec()
            self.block.set_scope('convx_1').config(self.inputs, 16, 64)

