import cv2
import numpy as np
import os
import tensorflow as tf
from config import config


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def get_image(img):
    black = np.reshape(img, [32, 32, 3],  order='F')
    np.reshape()
    #black = np.zeros((32, 32, 3), np.uint8)
    '''
    for channel in range(3):
        for i in range(32):
            for j in range(32):
                black[i, j, 2 - channel] = img[1024 * channel + i * 32 + j]
    '''
    return black


# print(len(image1))

def label2onehot(label):
    r = np.zeros(config.classes)
    r[label] = 1
    return r


def process_file(filename):
    data = unpickle(os.path.join(config.data_dir, filename))
    x = data[b'data']
    labels = data[b'labels']
    y = []
    for i in labels:
        y.append(label2onehot(i))

    y = np.array(y)
    return x, y


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == '__main__':
    x, y = process_file('data_batch_1')

    with tf.Session() as sess:
        cs = [0] * 10
        for i in range(30):
            img = x[i]
            a = tf.constant(img)
            b = tf.reshape(a, [3, 32, 32])
            c = tf.transpose(b, [1, 2, 0])
            img = sess.run(c)
            print(sess.run(tf.shape(c)))
            #img = get_image(img)

            # savename = int(str(labels[i], encoding = "utf8"))
            # cv2.imwrite('cifar1/%d_%d.bmp' % (label,cs[label]), img) # 保存图片
            cv2.imshow('image', img)
            cv2.waitKey(0)