import tensorflow as tf
import time
import numpy as np

import graph_meta as gm
import util
import constant
from weight_adjust import WeightAdjust


def train():
    t1 = time.time()
    tf.reset_default_graph()
    with tf.variable_scope(name_or_scope='train', reuse=tf.AUTO_REUSE):
        cls, (x, y), (w0, w1, w2, w3, w4) = gm.result()
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=cls, name='loss')
        loss_mean = tf.reduce_mean(loss, name='loss_mean')
        global_step = tf.Variable(0, name='global_step')
        learning_rate = tf.train.exponential_decay(constant.LEARNING_RATE, global_step, 1000, 0.96,
                                                   staircase=True, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
        train_op = optimizer.minimize(loss_mean, global_step=global_step, name='train_op')
    data_train = util.load_data(constant.DATA_TRAIN)
    data_test = util.load_data(constant.DATA_TEST)
    graph = tf.get_default_graph()
    #     var_list = [i for i in tf.global_variables() if i.name.split('/')[1] == 'result']
    #     saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
    #     [print(i) for i in tf.global_variables()]
    #     [print(i.name) for i in graph.get_operations()]
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        idx_train = np.linspace(0, constant.TRAIN_TOTAL_SIZE - 1, constant.TRAIN_TOTAL_SIZE, dtype=np.int32)
        step = 0
        accuracies_train = []
        accuracies_test = []
        losses = []
        ws = (w0, w1, w2, w3, w4)
        wa = WeightAdjust()
        wa.init(len(ws))
        for i in range(constant.EPOCH):
            np.random.shuffle(idx_train)
            for j in range(constant.TRAIN_TIMES_FOR_EPOCH):
                idx_j = np.linspace(j * constant.BATCH_SIZE, (j + 1) * constant.BATCH_SIZE - 1, constant.BATCH_SIZE, dtype=np.int32)
                idx_train_batch = idx_train[idx_j]
                _, labels_train, _, images_train = util.get_batch(data_train, idx_train_batch)
                feed_dict_train = {
                    x: images_train,
                    y: labels_train
                }
                cls_train, _loss, _ = sess.run([cls, loss_mean, train_op], feed_dict=feed_dict_train)
                arg_idx_train = np.argmax(cls_train, axis=1)
                accuracy_train = sum(labels_train == arg_idx_train) / constant.BATCH_SIZE
                # test 
                idx_test_batch = np.random.randint(0, constant.TEST_TOTAL_SIZE, [constant.BATCH_SIZE])
                _, labels_test, _, images_test = util.get_batch(data_test, idx_test_batch)
                feed_dict_test = {
                    x: images_test,
                    y: labels_test
                }
                cls_test = sess.run(cls, feed_dict=feed_dict_test)
                arg_idx_test = np.argmax(cls_test, axis=1)
                accuracy_test = sum(labels_test == arg_idx_test) / constant.BATCH_SIZE

                step += 1
                if step % constant.PRINT_EVERY_TIMES == 0:
                    print('time:{},epoch:{},loss:{},accuracy_train:{:.2%},accuracy_test:{:.2%}'
                          .format(util.cur_time(), step, _loss, accuracy_train, accuracy_test))
                    accuracies_train.append(accuracy_train)
                    accuracies_test.append(accuracy_test)
                    losses.append(_loss)

            times = int(constant.TRAIN_TIMES_FOR_EPOCH / constant.PRINT_EVERY_TIMES)
            train_mean = util.mean(accuracies_train[-times:])
            test_mean = util.mean(accuracies_test[-times:])
            print('save model,step: {},train_mean:{},test_mean:{}'.format(step, train_mean, test_mean))
            saver.save(sess, save_path='./model/resnet/cifar-resnet.ckpt', global_step=step)
            wa.adjust(train_mean, test_mean, step)
            print(wa.action)
            if wa.action == 'adjust':
                print('本次迭代权重经过调整：{}'.format(wa.weights))
                assigns = gm.assign_weight(wa, ws)
                sess.run(assigns)
            elif wa.action == 'stop':
                break
            else:
                pass
        accuracy_map = {'accuracies_train': accuracies_train,
                        'accuracies_test': accuracies_test,
                        'losses': losses,
                        'weights': wa}
        util.dump_data(accuracy_map, './accuracy_map.pkl')

    t2 = time.time()
    print('耗时：{}'.format(util.str_time(t2 - t1)))


train()
