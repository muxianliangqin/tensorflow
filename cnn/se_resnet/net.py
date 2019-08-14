from cnn.resnet.net import ResNetBlock, train
import tensorflow as tf


class SEResNetBlock(ResNetBlock):
    reduce_rate = 16
    se = super().inputs

    def compute_se(self):
        """
        Squeeze and Excitation
        se模块应在差分模块后执行
        :return:
        """
        with tf.variable_scope('se', reuse=tf.AUTO_REUSE):
            re = tf.reduce_mean(self.se, axis=[1,2], name='reduce_mean')
            se_shape = self.se.get_shape().as_list()
            batch_size, channels = se_shape[0], se_shape[-1]
            r = int(channels / 16)
            # 全连接
            dense = tf.layers.dense(re, r, activation=tf.nn.relu, name='dense_relu')
            dense = tf.layers.dense(dense, channels, activation=tf.nn.sigmoid, name='dense_sigmod')
            self.se = self.se * dense
            return self

    def exec(self):
        self.compute_shortcut()
        self.compute_residual()
        self.compute_se()
        self.integrate()

train()
