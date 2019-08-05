import tensorflow as tf
from cnn.layer.common import Layer
from cnn.utils.tf_util import do_bn,get_w


class ResNetLayer(Layer):
    default_name = 'res_net_layer'

    def exec(self):
        with tf.variable_scope(self.default_name):
            self.inputs = do_bn(self.inputs, name='bn')
            self.inputs = tf.nn.relu(self.inputs, name='relu')
            return tf.nn.conv2d(self.inputs, self.inputs, self.filters, self.strides, self.padding, name='conv2d')
