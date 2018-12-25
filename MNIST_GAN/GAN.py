import tensorflow as tf
import numpy as np
import datetime
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


def discriminator(X, name="d", reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope("conv1"):
            w1 = tf.get_variable('w', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('b', [32], initializer=tf.constant_initializer(0))
            a1 = tf.nn.relu(tf.nn.conv2d(X, filter=w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
            p1 = tf.nn.avg_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("conv2"):
            w2 = tf.get_variable('w', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable('b', [64], initializer=tf.constant_initializer(0))
            a2 = tf.nn.relu(tf.nn.conv2d(p1, filter=w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
            p2 = tf.nn.avg_pool(a2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            p2 = tf.reshape(p2, [-1, 7 * 7 * 64])

        with tf.variable_scope("fc1"):
            w3 = tf.get_variable('w', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b3 = tf.get_variable('b', [1024], initializer=tf.constant_initializer(0))
            a3 = tf.nn.relu(tf.matmul(p2, w3) + b3)

        with tf.variable_scope("fc2"):
            w4 = tf.get_variable('w', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b4 = tf.get_variable('b', [1], initializer=tf.constant_initializer(0))
            a4 = tf.nn.relu(tf.matmul(a3, w4) + b4)

    return a4


def generator(X, name='g'):
    dim = int(X.shape[-1])

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fc1"):
            w1 = tf.get_variable('w', [dim, 3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('b', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z1 = tf.matmul(X, w1) + b1
            z1 = tf.reshape(z1, [-1, 56, 56, 1])
            z1 = tf.contrib.layers.batch_norm(z1, epsilon=1e-5, scope='bn')
            a1 = tf.nn.relu(z1)

        # Generate 50 features
        with tf.variable_scope("conv1"):
            w2 = tf.get_variable('w', [3, 3, 1, dim / 2], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable('b', [dim / 2], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z2 = tf.nn.conv2d(a1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2
            z2 = tf.contrib.layers.batch_norm(z2, epsilon=1e-5, scope='bn')
            a2 = tf.nn.relu(z2)
            a2 = tf.image.resize_images(a2, [56, 56])

        with tf.variable_scope("conv2"):
            w3 = tf.get_variable('w', [3, 3, dim / 2, dim / 4],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b3 = tf.get_variable('b', [dim / 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z3 = tf.nn.conv2d(a2, w3, strides=[1, 2, 2, 1], padding='SAME') + b3
            z3 = tf.contrib.layers.batch_norm(z3, epsilon=1e-5, scope='bn')
            a3 = tf.nn.relu(z3)
            a3 = tf.image.resize_images(a3, [56, 56])

        with tf.variable_scope("conv3"):
            w4 = tf.get_variable('w', [1, 1, dim / 4, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b4 = tf.get_variable('b', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z4 = tf.nn.conv2d(a3, w4, strides=[1, 2, 2, 1], padding='SAME') + b4
            a4 = tf.sigmoid(z4)

    return a4


tf.reset_default_graph()

seed_dim = 100
seed_placeholder = tf.placeholder(tf.float32, [None, seed_dim], name='seed_placeholder')
real_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='real_placeholder')
G = generator(seed_placeholder)
Dr = discriminator(real_placeholder)
Df = discriminator(G, reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dr, labels = tf.ones_like(Dr)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Df, labels = tf.zeros_like(Df)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Df, labels = tf.ones_like(Df)))

params = tf.trainable_variables()
d_params = [p for p in params if 'd/' in p.name]
g_params = [p for p in params if 'g/' in p.name]

d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_params)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_params)
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_params)

sess = tf.Session()
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(seed_placeholder)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

batch_size = 200
sess.run(tf.global_variables_initializer())

# Pre-train discriminator
for i in range(120):
    print(i)
    seed_batch = np.random.normal(0, 1, size=[batch_size, seed_dim])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, _, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {real_placeholder: real_image_batch, seed_placeholder: seed_batch})

    if(i % 100 == 0):
        print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

# Train generator and discriminator together
for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    seed_batch = np.random.normal(0, 1, size=[batch_size, seed_dim])

    # Train discriminator on both real and fake images
    _, _, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {real_placeholder: real_image_batch, seed_placeholder: seed_batch})

    # Train generator
    seed_batch = np.random.normal(0, 1, size=[batch_size, seed_dim])
    _ = sess.run(g_trainer, feed_dict={seed_placeholder: seed_batch})

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        seed_batch = np.random.normal(0, 1, size=[batch_size, seed_dim])
        summary = sess.run(merged, {seed_placeholder: seed_batch, real_placeholder: real_image_batch})
        writer.add_summary(summary, i)

    if i % 100 == 0:
        # Every 100 iterations, show a generated image
        print("Iteration:", i, "at", datetime.datetime.now())
        seed_batch = np.random.normal(0, 1, size=[1, seed_dim])
        generated_images = generator(seed_placeholder)
        images = sess.run(generated_images, {seed_placeholder: seed_batch})

        # Show discriminator's estimate
        im = images[0].reshape([1, 28, 28, 1])
        result = discriminator(real_placeholder)
        estimate = sess.run(result, {real_placeholder: im})
        print("Estimate:", estimate)