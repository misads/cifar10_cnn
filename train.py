# encoding=utf-8
import tensorflow as tf
import argparse
import numpy as np
import time

from config import config
from data_loader import process_file, batch_iter
from models.cnn import CNN
from models.vgg import VGG
from utils import print_num_of_total_parameters


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

    parser.add_argument('--epoch', '-e', dest='epoch',
                        help='number of epochs to train',
                        default=25, type=int)

    parser.add_argument('--net', '-n', dest='net',
                        help='vgg, res50, res101, res152, mobile, cnn',
                        choices=['vgg', 'res50', 'res101', 'res152', 'mobile', 'cnn'],
                        default='cnn', type=str)

    args = parser.parse_args()

    return args


def train(net='cnn', epoch=20):
    # 读取训练集
    x_train, y_train = None, None

    for i in range(1, 6):
        x, y = process_file('data_batch_%d' % i)
        if x_train is None:
            x_train = x
            y_train = y
        else:
            x_train = np.append(x_train, x, axis=0)
            y_train = np.append(y_train, y, axis=0)
        del x, y

    x_dev, y_dev = process_file('test_batch')

    x = tf.placeholder(tf.float32, [None, config.image_width * config.image_height * config.image_channel])
    x_reshape = tf.reshape(x, [-1, config.image_channel, config.image_height, config.image_width])
    # [batch, depth, height, width] => [batch, height, width, depth]
    x_reshape = tf.transpose(x_reshape, [0, 2, 3, 1])

    if net == 'cnn':
        model = CNN()
    elif net == 'vgg':
        model = VGG()
    else:
        pass

    out = model.output(input=x_reshape)
    y_ = tf.placeholder(tf.float32, [None, config.classes])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))
    tf.summary.scalar('loss', loss)

    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs/", sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        '''
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)

        print_num_of_total_parameters()
        '''
        step = 1

        best_acc = 0.
        start_time = time.time()
        for e in range(1, epoch + 1):
            for x_batch, y_batch in batch_iter(x=x_train, y=y_train, batch_size=config.batch_size):
                step = step + 1
                _, trainloss, train_acc = sess.run([optimizer, loss, accuracy],
                                                   feed_dict={x: x_batch, y_: y_batch, model.keep_prob: 0.5,
                                                              model.is_training: True})

                if step % 20 == 0:
                    pass
                    #  print('Iterator:%d loss:%f train acc:%f' % (step, trainloss, train_acc))

                if step % 781 == 0:
                    train_acc, summary = sess.run([accuracy, merged],
                                                  feed_dict={x: x_train[:10000], y_: y_train[:10000],
                                                             model.keep_prob: 1.,
                                                             model.is_training: False})
                    writer.add_summary(summary, e)

                    acc = sess.run(accuracy,
                                   feed_dict={x: x_dev, y_: y_dev, model.keep_prob: 1., model.is_training: False})
                    print('Iterator:%d loss:%f train acc:%f' % (step, trainloss, train_acc))
                    elapsed_time = time.time() - start_time
                    print('\033[1;32mepoch:%d/%d' % (e, config.epoch))
                    print('\033[1;32mvalidation accuracy:%f\033[0m' % acc, end='')
                    if acc > best_acc:
                        best_acc = acc
                        print('\033[1;35m(new best acc!)\033[0m')
                    else:
                        print('')


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    train(net=args.net, epoch=config.epoch)
