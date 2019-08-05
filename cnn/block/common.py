from cnn.layer.common import Layer
import tensorflow as tf


class Block:
    default_name = 'block'
    kernel_size = 3

    def __init__(self, inputs, in_channels, out_channels):
        self.inputs = inputs
        self.in_channels = in_channels
        self.out_channels = out_channels

    def set_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size
        return self

    def exec(self):
        with tf.name_scope(self.default_name):
            return Layer(self.inputs, [self.kernel_size, self.kernel_size, self.in_channels, self.out_channels])
