import tensorflow as tf
import numpy as np
from cnn.block.common import Block


class Net:
    default_scope = 'net'
    block = Block()
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

    def config(self, block):
        if not isinstance(block, Block):
            raise Exception('参数:block不是Block类型', block)
        self.block = block

    def exec(self):
        with tf.variable_scope(self.default_scope):
            block = Block(self.inputs, 16, 16)
            return block.exec()
