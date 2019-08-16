from cnn.resnet.net import ResNetBlock
import tensorflow as tf


class SEResNetBlock(ResNetBlock):
    reduce_rate = 16

    def compute_se(self):
        """
        Squeeze and Excitation
        se模块应在残差模块后执行
        :return:
        """
        re = tf.reduce_mean(self.inputs, axis=[1, 2], name='reduce_mean')
        se_shape = self.inputs.get_shape().as_list()
        batch_size, channels = se_shape[0], se_shape[-1]
        r = int(channels / 16)
        # 全连接
        dense = tf.layers.dense(re, r, activation=tf.nn.relu, name='dense_relu')
        dense = tf.layers.dense(dense, channels, activation=tf.nn.sigmoid, name='dense_sigmod')
        dense = tf.reshape(dense, [-1, 1, 1, channels])
        self.inputs = self.inputs * dense
        return self

    def block(self):
        self.compute_shortcut()
        self.compute_residual()
        self.compute_se()
        self.integrate()
        return self
