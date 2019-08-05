import tensorflow as tf
import constant


def get_w(shape, name):
    w = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    return w


def do_bn(inputs, name):
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
        init_shape = inputs.get_shape().as_list()[-1:]
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], name='moments')
        beta = tf.get_variable(name='beta', shape=init_shape, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable(name='gamma', shape=init_shape, dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        variance_epsilon = 1e-9
        bn = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, variance_epsilon,
                                       name='batch_normalization')
        return bn


def resnet_block(x, kernel, in_channels, out_channels, scope_name, stride=1):
    '''
    x: [batch, height, width, channels]
    filters: [filter_height, filter_width, in_channels, out_channels]
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        strides = (1, 1)
        # shortcut 连接
        if out_channels > in_channels:
            strides = (stride, stride)
            shortcut = tf.layers.conv2d(inputs=x, filters=out_channels, kernel_size=[1, 1],
                                        strides=strides, activation=tf.nn.relu, padding='VALID', name='x_shortcut')
            shortcut = do_bn(shortcut, 'bn_shortcut')
        else:
            shortcut = x
        dim_reduce = int(out_channels / 4)
        # 卷积层 1
        #         print('{}：x shape:{}'.format(scope_name,x.get_shape().as_list()))
        # channels数增加时，[1,stride,stride,1],一般stride=2，图片尺寸 = /2

        # [1,1,in,out//4]
        conv1 = tf.layers.conv2d(inputs=x, filters=dim_reduce, kernel_size=[1, 1],
                                 strides=strides, activation=tf.nn.relu, padding='VALID', name='conv1')
        conv1 = do_bn(conv1, 'bn_conv1')
        # 卷积层 2
        strides = (1, 1)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=dim_reduce, kernel_size=[kernel, kernel],
                                 strides=strides, activation=tf.nn.relu, padding='SAME', name='conv2')
        conv2 = do_bn(conv2, 'bn_conv2')
        # 卷积层 3
        conv3 = tf.layers.conv2d(inputs=conv2, filters=out_channels, kernel_size=[1, 1],
                                 strides=strides, activation=tf.nn.relu, padding='VALID', name='conv3')
        conv3 = do_bn(conv3, 'bn_conv3')
        add4 = tf.add(conv3, shortcut)
        #         print('{}：relu4 shape:{}'.format(scope_name,relu4.get_shape().as_list()))
        return add4


def conv_1(inputs):
    with tf.variable_scope(name_or_scope='conv_1', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(inputs, constant.CHANNELS_1, [3, 3], padding='SAME', activation=tf.nn.relu, name='conv1')
        return conv1


def conv_x(inputs, kernel, in_channels, out_channels, stride, block, size):
    with tf.variable_scope(name_or_scope='conv_x', reuse=tf.AUTO_REUSE):
        inputs = resnet_block(inputs, kernel, in_channels, out_channels, 'conv_{}_0'.format(block), stride)
        in_channels = out_channels
        for i in range(1, size):
            inputs = resnet_block(inputs, kernel, in_channels, out_channels, 'conv_{}_{}'.format(block, (i)), stride)
        return inputs


def weighted_mean(inputs, weights):
    '''
    inputs: 一个由多个维度相同的张量组成的元组
    此方法将多个所得结果按权重求平均
    weights: 权重
    '''
    num = len(weights)
    cls = inputs[0] * weights[0]
    for n in range(1, num):
        cls = tf.add(cls, inputs[n] * weights[n], name='add_cls_{}'.format(n))
    return cls


def gap(inputs, classes, name):
    # global average pooling
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
        shape = inputs.get_shape().as_list()
        filters = [1, 1, shape[-1], classes]
        w = get_w(filters, 'w')
        c = tf.nn.conv2d(input=inputs, filter=w, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')
        g = tf.reduce_mean(c, axis=[1, 2], name='reduce_mean')
        return tf.squeeze(g, name='squeeze')


def deal(inputs, weights):
    conv1 = conv_1(inputs)
    cls1 = gap(conv1, constant.CLASSES, 'cls1')
    conv2 = conv_x(conv1, 3, constant.CHANNELS_1, constant.CHANNELS_2, 2, block=2, size=3)
    cls2 = gap(conv2, constant.CLASSES, 'cls2')
    conv3 = conv_x(conv2, 3, constant.CHANNELS_2, constant.CHANNELS_3, 2, block=3, size=4)
    cls3 = gap(conv3, constant.CLASSES, 'cls3')
    conv4 = conv_x(conv3, 3, constant.CHANNELS_3, constant.CHANNELS_4, 2, block=4, size=6)
    cls4 = gap(conv4, constant.CLASSES, 'cls4')
    conv5 = conv_x(conv4, 3, constant.CHANNELS_4, constant.CHANNELS_5, 1, block=5, size=3)
    cls5 = gap(conv5, constant.CLASSES, 'cls5')
    classify = (cls1, cls2, cls3, cls4, cls5)
    cls = weighted_mean(classify, weights)
    return cls


def before_train(inputs):
    with tf.variable_scope('before_train', reuse=tf.AUTO_REUSE):
        inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32, name='convert_image_dtype')
        inputs = tf.image.random_flip_up_down(inputs)
        inputs = tf.image.random_flip_left_right(inputs)
        return inputs


def result():
    with tf.variable_scope(name_or_scope='result', reuse=tf.AUTO_REUSE):
        x = tf.placeholder(tf.uint8, shape=[constant.BATCH_SIZE, 32, 32, 3], name='images')
        y = tf.placeholder(tf.int32, shape=[constant.BATCH_SIZE], name='labels')
        x2 = before_train(x)
        w0 = tf.get_variable(name='cls_weight_0', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1))
        w1 = tf.get_variable(name='cls_weight_1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1))
        w2 = tf.get_variable(name='cls_weight_2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1))
        w3 = tf.get_variable(name='cls_weight_3', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1))
        w4 = tf.get_variable(name='cls_weight_4', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1))
        weights = (w0, w1, w2, w3, w4)
        cls = deal(x2, weights)
        return cls, (x, y), (w0, w1, w2, w3, w4)


def assign_weight(wa, ws):
    assigns = []
    with tf.variable_scope(name_or_scope='assign', reuse=tf.AUTO_REUSE):
        for i, w in enumerate(ws):
            assigns.append(tf.assign(w, wa.weights[i], name='assign_w_{}'.format(i)))
        return assigns

