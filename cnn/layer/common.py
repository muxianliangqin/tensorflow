import tensorflow as tf
from cnn.utils.tf_util import get_w


class Layer:
    default_scope = 'layer'
    inputs = None
    filters = None
    strides = 1
    padding = 'SAME'

    def set_scope(self, scope):
        if not isinstance(scope, str):
            raise Exception('参数:scope不是str类型', scope)
        self.default_scope = scope
        return self

    def config(self, inputs, filters, strides, padding):
        '''
        参数说明
        :param inputs: 输入
            numpy.ndarray
            [batch, height, width, channels]
        :param filters: 滤波器的形状
            filter.shape->list
            [filter_height, filter_width, in_channels, output_channels]
        :param strides: 步长
            int
            default: 1
        :param padding: 填充方式
            str
            default: 'SAME'
        :return:
        '''

        self.inputs = inputs
        self.filters = filters
        if isinstance(strides, int):
            self.strides = strides
        if isinstance(padding, str):
            self.padding = padding
        return self

    def exec(self):
        with tf.variable_scope(self.default_scope):
            filters = get_w(self.filters, 'weight')
            self.inputs = tf.nn.conv2d(self.inputs, filters, self.strides, self.padding, name='conv2d')
            return self.inputs
