import numpy as np
import tensorflow as tf
from cnn.origin.block import Block


class Section:
    default_scope = 'section'
    block = Block()
    inputs = None
    in_channels = None
    out_channels = None
    blocks_num = None

    def __init__(self, inputs):
        if not isinstance(inputs, np.ndarray):
            raise Exception('参数:inputs不是ndarray类型', inputs)
        self.inputs = inputs

    def set_scope(self, scope):
        if not isinstance(scope, str):
            raise Exception('参数:scope不是str类型', scope)
        self.default_scope = scope
        return self

    def set_block(self, block):
        if not isinstance(block, Block):
            raise Exception('参数:block不是Block类型', block)
        self.block = block
        return self

    def config(self, in_channels, out_channels, blocks_num):
        if not isinstance(in_channels, int):
            raise Exception('参数:in_channels不是int类型', in_channels)
        if not isinstance(out_channels, int):
            raise Exception('参数:out_channels不是int类型', out_channels)
        if not isinstance(blocks_num, int):
            raise Exception('参数:blocks_num不是int类型', blocks_num)
        if blocks_num <= 0:
            raise Exception('参数:blocks_num必须大于0', blocks_num)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks_num = blocks_num
        return self

    def section(self):
        self.inputs = self.block.set_scope('block_0').config(self.inputs, self.in_channels, self.out_channels).exec()
        for i in range(1, self.blocks_num):
            self.inputs = self.block.set_scope('block_{}'.format(i)) \
                .config(self.inputs, self.out_channels, self.out_channels) \
                .exec()
        return self.inputs

    def exec(self):
        with tf.variable_scope(self.default_scope, reuse=tf.AUTO_REUSE):
            return self.section()

