import numpy as np
import tensorflow as tf


class Batch:
    """
    训练时获取批次数据
    次类适合通过随机索引的方式获取训练批次
    """
    epoch = 0
    step = 0
    data = None
    total_size = None
    index_total = None
    batch_size = None
    index_batch = None
    data_batch = None

    def config(self, data, total, batch, epoch=1):
        self.data = data
        self.total_size = total
        self.batch_size = batch
        self.index_total = np.linspace(0, total - 1, total, dtype=np.int32)
        np.random.shuffle(self.index_total)
        self.epoch = epoch
        self.step = 0
        return self

    def batch_index(self):
        batch_num_of_epoch = self.total_size // self.batch_size
        if self.step != 0 and self.step % batch_num_of_epoch == 0:
            self.epoch += 1
            np.random.shuffle(self.index_total)
        step_in_epoch = self.step % batch_num_of_epoch
        self.index_batch = np.linspace(step_in_epoch, ((step_in_epoch + 1) * self.batch_size) - 1,
                                       self.batch_size, dtype=np.int32)
        if self.step >= self.epoch * batch_num_of_epoch:
            raise tf.errors.OutOfRangeError('批次结束')
        return self

    def batch_data(self):
        file_names = np.asarray(self.data[b'filenames'])
        fine_labels = np.asarray(self.data[b'fine_labels'])
        coarse_labels = np.asarray(self.data[b'coarse_labels'])
        image = np.asarray(self.data[b'data'])
        file_names_batch = file_names[self.index_batch]
        fine_labels_batch = fine_labels[self.index_batch]
        coarse_labels_batch = coarse_labels[self.index_batch]
        image_batch = image[self.index_batch].reshape([-1, 32, 32, 3])
        self.data_batch = (fine_labels_batch, image_batch)
        return self

    def exec(self):
        self.batch_index()
        self.batch_data()
        return self.data_batch
