import tensorflow as tf
from cnn.origin.layer import Layer
from cnn.origin.block import Block
from cnn.origin.section import Section


class Net:
    default_scope = 'net'
    # 记录完整的scope路径
    scopes = [default_scope]
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
        return self

    def net(self):
        return self

    def classify(self):
        # global average pooling
        with tf.variable_scope(name_or_scope='classify', reuse=tf.AUTO_REUSE):
            self.inputs = Layer().set_scope('global_average_pooling').config(self.inputs, self.classes).exec()
            gap = tf.reduce_mean(self.inputs, axis=[1, 2], name='reduce_mean')
            self.inputs = tf.squeeze(gap, name='squeeze')
            return self

    def optimizer(self):
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.inputs, name='loss')
            loss_mean = tf.reduce_mean(loss, name='loss_mean')
            result = tf.equal(tf.argmax(self.inputs, axis=-1), tf.argmax(self.labels, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(result, dtype=tf.float16, name='cast'), name='accuracy')
            global_step = tf.Variable(0, name='global_step')
            learning_rate = tf.train.exponential_decay(self.learn_rate, global_step, 1000, 0.96,
                                                       staircase=True, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')
            tf.summary.scalar('loss', loss_mean)
            tf.summary.scalar('accuracy', accuracy)
            return train_op

    def exec(self):
        print('layer.type:{},block.type:{},section.type:{},net.type:{}'
              .format(type(self.layer), type(self.block), type(self.section), type(self)))
        with tf.variable_scope(self.default_scope, reuse=tf.AUTO_REUSE):
            self.before_train()
            self.net()
            self.classify()
            train_op = self.optimizer()
            return train_op
