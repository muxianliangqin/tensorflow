import tensorflow as tf
from cnn.origin.layer import Layer
from cnn.origin.block import Block
from cnn.origin.section import Section


class Net:
    """
    本类主要用于创建类resnet类型的cnn网络，网络从小到大为Layer->Block->Section->Net
    Layer：基础层，重写方法layer()
    Block：Layer组成的块，重写方法block()
    Section：Block组成的域，重写方法section()
    Net：整个网络，重写方法net()
    创建新的类型的网络时，一般重写相应同名的方法即可
    """
    default_scope = 'net'
    section = Section()
    block = Block()
    layer = Layer()
    learn_rate = 0.001
    inputs = None
    labels = None
    classes = None

    def set_scope(self, scope):
        if not isinstance(scope, str):
            raise Exception('参数:scope不是str类型', scope)
        self.default_scope = scope
        return self

    def config(self, inputs, labels, classes):
        self.inputs = inputs
        self.labels = labels
        self.classes = classes
        return self

    def set_lr(self, learn_rate):
        if not isinstance(learn_rate, float):
            raise Exception('参数:learn_rate不是float类型', learn_rate)
        self.learn_rate = learn_rate
        return self

    def build(self, layer, block, section):
        """
        网络创建前设置Layer，Block，Section
        :param layer:
        :param block:
        :param section:
        :return:
        """
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
        """
        训练前对图像进行一些处理
        :return:
        """
        return self

    def net(self):
        """
        具体实现net
        :return:
        """
        return self

    def validation(self):
        """
        计算模型的精确度
        这里使用全局平均池化替换全连接，global average pooling
        :return:
        """
        with tf.variable_scope(name_or_scope='validation', reuse=tf.AUTO_REUSE):
            self.inputs = Layer().set_scope('global_average_pooling').config(self.inputs, self.classes).exec()
            gap = tf.reduce_mean(self.inputs, axis=[1, 2], name='reduce_mean')
            self.inputs = tf.squeeze(gap, name='squeeze')
            result = tf.equal(tf.argmax(self.inputs, axis=-1, name='compute_result'), self.labels, name='equal')
            accuracy = tf.reduce_mean(tf.cast(result, dtype=tf.float16, name='cast'), name='accuracy')
            tf.summary.scalar('accuracy', accuracy)
            return self

    def optimizer(self):
        """
        损失函数及优化函数
        :return:
        """
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.inputs, name='loss')
            global_step = tf.Variable(0, name='global_step')
            learning_rate = tf.train.exponential_decay(self.learn_rate, global_step, 1000, 0.96,
                                                       staircase=True, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')
            loss_mean = tf.reduce_mean(loss, name='loss_mean')
            tf.summary.scalar('loss', loss_mean)
            return train_op

    def exec(self):
        print('layer.type:{},block.type:{},section.type:{},net.type:{}'
              .format(type(self.layer), type(self.block), type(self.section), type(self)))
        with tf.variable_scope(self.default_scope, reuse=tf.AUTO_REUSE):
            self.before_train()
            self.net()
            self.validation()
            train_op = self.optimizer()
            return train_op
