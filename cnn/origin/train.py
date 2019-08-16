import time
import tensorflow as tf
import cnn.utils.base_util as util
import cnn.origin.config as config


class Train:
    net = None
    batch_train = None
    batch_test = None

    def build(self, net, batch_train, batch_test):
        self.net = net
        self.batch_train = batch_train
        self.batch_test = batch_test
        return self

    def train(self):
        t1 = time.time()
        tf.reset_default_graph()
        x = tf.placeholder(tf.uint8, shape=[config.BATCH_SIZE, 32, 32, 3], name='images')
        y = tf.placeholder(tf.int64, shape=[config.BATCH_SIZE], name='labels')
        self.net.config(x, y, config.CLASSES)
        train_op = self.net.exec()
        accuracy = self.net.accuracy
        self.batch_train.config(config.DATA_TRAIN, config.TRAIN_TOTAL_SIZE, config.BATCH_SIZE, config.EPOCH)
        self.batch_test.config(config.DATA_TEST, config.TEST_TOTAL_SIZE, config.BATCH_SIZE)
        graph = tf.get_default_graph()
        summaries = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=5)
        with tf.Session(graph=graph) as sess:
            writer_train = tf.summary.FileWriter(config.LOG_TRAIN, sess.graph)
            writer_test = tf.summary.FileWriter(config.LOG_TEST)
            sess.run(tf.global_variables_initializer())
            while True:
                try:
                    labels_train, images_train = self.batch_train.exec()
                    feed_dict_train = {x: images_train, y: labels_train}
                    summary_train, _ = sess.run([summaries, train_op], feed_dict=feed_dict_train)
                    # test
                    labels_test, images_test = self.batch_test.exec()
                    feed_dict_test = {x: images_test, y: labels_test}
                    summary_test, _ = sess.run([summaries, accuracy], feed_dict=feed_dict_test)
                    step = self.batch_train.step
                    if step % config.PRINT_EVERY_TIMES == 0:
                        writer_train.add_summary(summary_train, step)
                        writer_test.add_summary(summary_test, step)
                    if self.batch_train.cycle_one_epoch:
                        print('EPOCH:{},save model'.format(self.batch_train.epoch + 1))
                        saver.save(sess, save_path=config.MODEL_SAVE_PATH, global_step=step)
                except StopIteration as e:
                    print(e.args)
                    break
        t2 = time.time()
        print('耗时：{}'.format(util.str_time(t2 - t1)))

    def exec(self):
        self.train()
