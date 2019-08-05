from cnn.layer.common import Layer
import tensorflow as tf


class Block:
    default_name = 'block'

    def __init__(self, inputs, in_channels, out_channels, kernel_size=3):
        self.inputs = inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        with tf.variable_scope(self.default_name):
            self.layer = Layer(self.inputs, [self.kernel_size, self.kernel_size, self.in_channels, self.out_channels])

    def set_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size
        return self

