from cnn.layer.common import Layer


class Block:
    default_scope = 'block'
    layer = Layer()
    inputs = None
    in_channels = None
    out_channels = None
    kernel_size = 3

    def __init__(self, layer):
        if not isinstance(layer, Layer):
            raise Exception('参数:layer不是Layer类型', layer)
        self.layer = layer

    def set_scope(self, scope):
        if not isinstance(scope, str):
            raise Exception('参数:scope不是str类型', scope)
        self.default_scope = scope
        return self

    def config(self, inputs, in_channels, out_channels, kernel_size):
        self.inputs = inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = kernel_size
        return self

    def exec(self):
        filters = [self.kernel_size, self.kernel_size, self.in_channels, self.out_channels]
        self.inputs = self.layer.config(self.inputs, filters).exec()
        return self.inputs
