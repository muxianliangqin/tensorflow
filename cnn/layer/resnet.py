import tensorflow as tf
from cnn.layer.common import Layer
from cnn.utils.tf_util import do_bn, get_w


class ResNetLayer(Layer):
    default_scope = 'res_net_layer'

    def exec(self):
        with tf.variable_scope(self.default_scope):
            self.inputs = do_bn(self.inputs, name='bn')
            self.inputs = tf.nn.relu(self.inputs, name='relu')
            filters = get_w(self.filters, 'weight')
            return tf.nn.conv2d(self.inputs, filters, self.strides, self.padding, name='conv2d')
