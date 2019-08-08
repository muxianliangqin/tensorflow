import tensorflow as tf


class Layer:
    default_scope = 'layer'
    sup = None
    inputs = None
    out_channels = None
    filters = None
    strides = None
    filters_shape = None
    inputs_shape = None
    kernel_size = 1
    strides_size = 1
    padding = 'SAME'

    def set_scope(self, scope):
        if not isinstance(scope, str):
            raise Exception('参数:scope不是str类型', scope)
        self.default_scope = scope
        return self

    def set_sup(self, sup):
        self.sup = sup

    def config(self, inputs, out_channels, kernel_size=1, strides_size=1, padding='SAME'):
        """
        参数说明
        :param inputs: 输入
            tensor
            [batch, height, width, channels]
        :param kernel_size: 滤波器的卷积核尺寸
            int
        :param out_channels: 滤波器的输出通道数
            int
            default:
        :param strides_size: 步长
            int
        :param padding: 填充方式
            str
        :return:
        """
        if not isinstance(out_channels, int):
            raise Exception('参数:out_channels不是int类型', out_channels)
        self.inputs = inputs
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides_size = strides_size
        self.padding = padding
        return self

    def get_convolution_params(self):
        self.inputs_shape = self.inputs.get_shape().as_list()
        self.filters_shape = [self.kernel_size, self.kernel_size, self.inputs_shape[-1], self.out_channels]
        self.strides = [1, self.strides_size, self.strides_size, 1]
        self.filters = tf.get_variable(name='weight', shape=self.filters_shape,
                                       initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        return self

    def layer(self):
        self.inputs = tf.nn.conv2d(self.inputs, self.filters, self.strides, self.padding, name='conv2d')
        return self.inputs

    def exec(self):
        with tf.variable_scope(self.default_scope, reuse=tf.AUTO_REUSE):
            self.get_convolution_params()
            scopes = []
            obj = self
            while True:
                scopes.insert(0, obj.default_scope)
                if not hasattr(obj, 'sup') or obj.sup is None:
                    break
                obj = obj.sup
            print('{},inputs.shape:{},filter.shape:{},strides:{},padding:{}'
                  .format('/'.join(scopes), self.inputs.get_shape().as_list(),
                          self.filters_shape, self.strides, self.padding))
            return self.layer()
