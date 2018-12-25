import tensorflow as tf
import numpy as np
import datetime

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

with tf.device('/gpu:0'):
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y = tf.placeholder(tf.int64, shape=[None])

    with tf.variable_scope("conv1"):
        w1 = tf.get_variable('w', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b', [32], initializer=tf.constant_initializer(0.01))
        a1 = tf.nn.relu(tf.nn.conv2d(X, filter=w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
        p1 = tf.nn.avg_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("conv2"):
        w2 = tf.get_variable('w', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b', [64], initializer=tf.constant_initializer(0.01))
        a2 = tf.nn.relu(tf.nn.conv2d(p1, filter=w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
        p2 = tf.nn.avg_pool(a2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        p2 = tf.reshape(p2, [-1, 7 * 7 * 64])

    with tf.variable_scope("fc1"):
        w3 = tf.get_variable('w', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('b', [1024], initializer=tf.constant_initializer(0.01))
        a3 = tf.nn.relu(tf.matmul(p2, w3) + b3)

    with tf.variable_scope("softmax"):
        w4 = tf.get_variable('w', [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('b', [10], initializer=tf.constant_initializer(0.01))
        logits = tf.matmul(a3, w4) + b4

    y_hat = tf.argmax(tf.nn.softmax(logits), 1)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_hat, y), tf.float32))
    trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    epoch = 1000
    batch_size = 200
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(epoch):
            batch = mnist.train.next_batch(batch_size)
            _, current_loss = sess.run([trainer, loss],
                                       feed_dict={X: batch[0].reshape(batch_size, 28, 28, 1), y: batch[1]})
            if i % 100 == 0:
                batch = mnist.test.next_batch(batch_size)
                accuracy = sess.run(acc, feed_dict={X: batch[0].reshape(batch_size, 28, 28, 1), y: batch[1]})
                print('done epoch ', i, 'accu: ', accuracy)