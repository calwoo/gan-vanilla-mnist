import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# import mnist-- as always
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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

disc_loss = -tf.reduce_mean(tf.log(disc_real_prob) + tf.log(1-disc_fake_prob))
gen_loss = tf.reduce_mean(tf.log(1-disc_fake_prob))

disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

# gradient train
disc_optimizer = tf.train.AdamOptimizer().minimize(disc_loss, var_list=disc_vars)

for var in disc_vars:
    print(var)

