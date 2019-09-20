import tensorflow as tf

class Network(object):
    def __init__(self):
        self.is_training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)