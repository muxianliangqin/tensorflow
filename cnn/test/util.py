import pickle
import time
import numpy as np


def load_data(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f,encoding='bytes')
        return data


def dump_data(data,filename):
    with open(filename,'wb') as f:
        pickle.dump(data,f)


def get_batch(data,idx):
    filenames = np.asarray(data[b'filenames'])
    fine_labels = np.asarray(data[b'fine_labels'])
    coarse_labels = np.asarray(data[b'coarse_labels'])
    image = np.asarray(data[b'data'])
    filenames_batch = filenames[idx]
    fine_labels_batch = fine_labels[idx]
    coarse_labels_batch = coarse_labels[idx]
    image_batch = image[idx].reshape([-1,32,32,3])
    return filenames_batch, fine_labels_batch, coarse_labels_batch, image_batch


def cur_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def str_time(ti):
    hour = int(ti) // (60 * 60)
    ti -= hour * (60 * 60)
    minute = int(ti) // 60
    ti -= minute * 60
    second = int(ti)
    ti -= second
    millisecond = int(ti * 1000)
    return '{}小时 {}分钟 {}秒 {}毫秒'.format(hour,minute,second,millisecond)


def mean(lists):
    return sum(lists) / len(lists)
