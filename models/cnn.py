from config import config
from models.net import Network
import tensorflow as tf


class CNN(Network):
    def __init__(self, fc=256):
        Network.__init__(self)
        self.fc = fc

    def output(self, input, w_alpha=0.01, b_alpha=0.1):


        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, config.image_channel, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)

        # 全连接层
        # w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
        w_d = tf.Variable(w_alpha * tf.random_normal([config.image_height * config.image_width * 2, self.fc]))

        b_d = tf.Variable(b_alpha * tf.random_normal([self.fc]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        #dense = tf.layers.batch_normalization(dense, training=self.is_training)
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([self.fc, config.classes]))
        b_out = tf.Variable(b_alpha * tf.random_normal([config.classes]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        #out = tf.layers.batch_normalization(out, training=self.is_training)

        # out = tf.nn.softmax(out)
        return out
