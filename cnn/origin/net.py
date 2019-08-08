import tensorflow as tf
from cnn.origin.layer import Layer
from cnn.origin.block import Block
from cnn.origin.section import Section


class Net:
    default_scope = 'net'
    # 记录完整的scope路径
    scopes = [default_scope]
    section = Section()
    block = Block()
    layer = Layer()
    inputs = None

    def set_scope(self, scope):
        if not isinstance(scope, str):
            raise Exception('参数:scope不是str类型', scope)
        self.default_scope = scope
        return self

    def config(self, inputs):
        self.inputs = inputs
        return self

    def build(self, layer, block, section):
        if not isinstance(layer, Layer):
            raise Exception('参数:layer不是Layer类型', Layer)
        if not isinstance(block, Block):
            raise Exception('参数:block不是Block类型', block)
        if not isinstance(section, Section):
            raise Exception('参数:section不是Section类型', section)
        self.layer = layer
        self.block = block.set_layer(self.layer)
        self.section = section.set_block(self.block)
        self.layer.set_sup(self.block)
        self.block.set_sup(self.section)
        self.section.set_sup(self)
        return self

    def before_train(self):
        return self.inputs

    def net(self):
        return self.inputs

    def exec(self):
        print('layer.type:{},block.type:{},section.type:{},net.type:{}'
              .format(type(self.layer), type(self.block), type(self.section), type(self)))
        with tf.variable_scope(self.default_scope):
            self.inputs = self.before_train()
            return self.net()
