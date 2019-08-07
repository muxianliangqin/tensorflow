import tensorflow as tf
import numpy as np
from cnn.origin.layer import Layer
from cnn.origin.block import Block
from cnn.origin.section import Section


class Net:
    default_scope = 'net'
    section = Section()
    inputs = None

    def __init__(self, inputs):
        if not isinstance(inputs, np.ndarray):
            raise Exception('参数:inputs不是ndarray类型', inputs)
        self.inputs = inputs

    def set_scope(self, scope):
        if not isinstance(scope, str):
            raise Exception('参数:scope不是str类型', scope)
        self.default_scope = scope
        return self

    def config(self, layer, block, section):
        if not isinstance(layer, Layer):
            raise Exception('参数:layer不是Layer类型', Layer)
        if not isinstance(block, Block):
            raise Exception('参数:block不是Block类型', block)
        if not isinstance(section, Section):
            raise Exception('参数:section不是Section类型', section)
        self.section = section.set_block(block.set_layer(layer))
        return self

    def before_train(self):
        return self.inputs

    def net(self):
        return self.inputs

    def exec(self):
        with tf.variable_scope(self.default_scope):
            self.inputs = self.before_train()
            return self.net()
