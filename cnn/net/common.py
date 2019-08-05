import tensorflow as tf
from cnn.block.common import Block


class Net:
    default_name = 'net'
    in_channels = 3
    out_channels = 16

    def __init__(self, inputs):
        self.inputs = inputs

    def exec(self):
        with tf.variable_scope(self.default_name):
            block = Block(self.inputs, self.in_channels, self.out_channels)
            return block.exec()
