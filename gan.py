import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# import mnist-- as always
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# build GAN
x = tf.placeholder(tf.float32, [None, 784], name="inputs_x")
z = tf.placeholder(tf.float32, [None, 100], name="inputs_z")

def generator(z):
    gen_fc1 = tf.layers.Dense(z, 128, tf.nn.relu)
    gen_output = tf.layers.Dense(gen_fc1, 784, tf.nn.sigmoid)
    return gen_output

def discriminator(x):
    disc_fc1 = tf.layers.Dense(x, 128, tf.nn.relu)
    disc_logit = tf.layers.Dense(disc_fc1, 1)
    disc_prob = tf.nn.sigmoid(disc_logit)
    return disc_logit, disc_prob

# the magical loss function


