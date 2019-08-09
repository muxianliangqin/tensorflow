import tensorflow as tf
from cnn.origin.block import Block


class Section:
    default_scope = 'section'
    block = Block()
    sup = None
    inputs = None
    out_channels = None
    blocks_num = None

    def set_block(self, block):
        if not isinstance(block, Block):
            raise Exception('参数:block不是Layer类型', Block)
        self.block = block
        return self

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

    def set_sup(self, sup):
        self.sup = sup
        return self

    def config(self, inputs, out_channels, blocks_num):
        if not isinstance(out_channels, int):
            raise Exception('参数:out_channels不是int类型', out_channels)
        if not isinstance(blocks_num, int):
            raise Exception('参数:blocks_num不是int类型', blocks_num)
        if blocks_num <= 0:
            raise Exception('参数:blocks_num必须大于0', blocks_num)
        self.inputs = inputs
        self.out_channels = out_channels
        self.blocks_num = blocks_num
        return self

    def section(self):
        return self

    def exec(self):
        with tf.variable_scope(self.default_scope, reuse=tf.AUTO_REUSE):
            self.section()
            return self.inputs

