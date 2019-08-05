import tensorflow as tf


class Layer:
    default_name = 'layer'

    def __init__(self, inputs, filters, strides=1, padding='SAME'):
        self.inputs = inputs
        self.filter = filters
        self.strides = strides
        self.padding = padding

    def exec(self):
        with tf.variable_scope(self.default_name):
            return tf.nn.conv2d(self.inputs, self.inputs, self.filters, self.strides, self.padding, name='conv2d')
