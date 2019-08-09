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
    if_log = False

    @staticmethod
    def get_w(shape, name):
        return tf.get_variable(name=name, shape=shape,
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))

    @staticmethod
    def do_bn(inputs, name):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            init_shape = inputs.get_shape().as_list()[-1:]
            mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], name='moments')
            beta = tf.get_variable(name='beta', shape=init_shape, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(name='gamma', shape=init_shape, dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
            variance_epsilon = 1e-9
            return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, variance_epsilon,
                                             name='batch_normalization')

    def set_scope(self, scope):
        if not isinstance(scope, str):
            raise Exception('参数:scope不是str类型', scope)
        self.default_scope = scope
        return self

    def set_sup(self, sup):
        self.sup = sup
        return self

    def set_log(self, log):
        if not isinstance(log, bool):
            raise Exception('参数:log不是bool类型', log)
        self.if_log = log
        return self

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

    def get_filters(self):
        """
        计算卷积需要的滤波器参数:filters
        :return:
        """
        self.inputs_shape = self.inputs.get_shape().as_list()
        self.filters_shape = [self.kernel_size, self.kernel_size, self.inputs_shape[-1], self.out_channels]
        self.strides = [1, self.strides_size, self.strides_size, 1]
        self.filters = Layer.get_w(shape=self.filters_shape, name='weight')
        return self

    def layer(self):
        """
        具体实现layer
        :return:
        """
        self.inputs = tf.nn.conv2d(self.inputs, self.filters, self.strides, self.padding, name='conv2d')
        return self

    def log(self):
        """
        日志记录layer的一些具体配置
        :return:
        """
        if self.if_log:
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
        return self

    def exec(self):
        """
        当layer配置完成后，最后一步执行此方法，生成网络
        :return:
        """
        with tf.variable_scope(self.default_scope, reuse=tf.AUTO_REUSE):
            self.get_filters()
            self.log()
            self.layer()
            return self.inputs
