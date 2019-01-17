import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 

import sys
import argparse
from vis import *

# import mnist-- as always
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyperparameters
disc_lr = 0.01
gen_lr = 0.01
batch_size = 128
num_epochs = 10000

# build GAN
x = tf.placeholder(tf.float32, [None, 784], name="inputs_x")
z = tf.placeholder(tf.float32, [None, 100], name="inputs_z")

def generator(z):
    with tf.variable_scope("generator"):
        gen_fc1 = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        gen_output = tf.layers.dense(gen_fc1, units=784, activation=tf.nn.sigmoid)
    return gen_output

def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        disc_fc1 = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        disc_logit = tf.layers.dense(disc_fc1, units=1)
        disc_prob = tf.nn.sigmoid(disc_logit)
    return disc_logit, disc_prob

# the magical loss function
noise_samples = generator(z)

"""
Our GAN discriminator will see a batch of real samples and fake (noise) samples and pump out the probabilities
of them being real. We will then feed these probabilities into the loss function, which will be backpropagated
through both generator/discriminator networks to train them.
"""
disc_real_logits, disc_real_prob = discriminator(x)
disc_fake_logits, disc_fake_prob = discriminator(noise_samples)

eps = 1e-6
disc_loss = -tf.reduce_mean(tf.log(eps+disc_real_prob) + tf.log(eps+1-disc_fake_prob))
gen_loss = tf.reduce_mean(tf.log(eps+1-disc_fake_prob))

disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

# gradient train
disc_optimizer = tf.train.AdamOptimizer(disc_lr).minimize(disc_loss, var_list=disc_vars)
gen_optimizer = tf.train.AdamOptimizer(gen_lr).minimize(gen_loss, var_list=gen_vars)

# sampling helper function
def sample_noise(num_samples):
    return np.random.uniform(-1.0, 1.0, [num_samples, 100])

# training loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())
samples = sess.run(noise_samples, feed_dict={z: sample_noise(25)})
marker = 0

for epoch in range(num_epochs):
    input_batch, _ = mnist.train.next_batch(batch_size)
    z_noise = sample_noise(batch_size)

    d_loss, _ = sess.run([disc_loss, disc_optimizer], 
        feed_dict={x:input_batch, z:z_noise})
    g_loss, _ = sess.run([gen_loss, gen_optimizer],
        feed_dict={z:z_noise})

    if epoch % 1000 == 0:
        print("epoch %d, disc_loss = %.08f / gen_loss = %.08f" % (epoch, d_loss, g_loss))
        take_snapshot(samples, marker)
        marker += 1

# build gif
build_gif("trained")


