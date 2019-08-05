import tensorflow as tf


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


def get_w(shape, name):
    w = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    return w
