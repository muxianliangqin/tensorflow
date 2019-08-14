from cnn.origin.layer import Layer
import tensorflow as tf


class Block:
    default_scope = 'block'
    layer = Layer()
    sup = None
    inputs = None
    shortcut = inputs
    out_channels = None
    kernel_size = 1

    def set_scope(self, scope):
        if not isinstance(scope, str):
            raise Exception('参数:scope不是str类型', scope)
        self.default_scope = scope
        return self

    def set_layer(self, layer):
        if not isinstance(layer, Layer):
            raise Exception('参数:layer不是Layer类型', layer)
        self.layer = layer
        return self

    def set_sup(self, sup):
        self.sup = sup
        return self

    def config(self, inputs, out_channels, kernel_size=1):
        if not isinstance(out_channels, int):
            raise Exception('参数:out_channels不是int类型', out_channels)
        self.inputs = inputs
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        return self

    def compute_residual(self):
        """
        差分模块计算
        :return:
        """
        return self

    def compute_shortcut(self):
        """
        快捷连接模块计算
        :return:
        """
        return self

    def integrate(self):
        """
        整合模块
        快捷连接模块和差分
        :return:
        """
        self.inputs = tf.add(self.shortcut, self.inputs, name='add')

    def block(self):
        """
        具体实现block
        :return:
        """
        self.compute_shortcut()
        self.compute_residual()
        self.integrate()
        return self

    def exec(self):
        with tf.variable_scope(self.default_scope, reuse=tf.AUTO_REUSE):
            self.block()
            return self.inputs
