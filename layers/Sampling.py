from keras import layers
from keras import random
from keras import saving
import tensorflow as tf


@saving.register_keras_serializable()
class Sampling(layers.Layer):
    """a layer that maps the distribution on the latent space to a point in the latent space"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]  # how many inputs are we processing
        dim = tf.shape(z_mean)[1]  # how many dimensions is the latent space

        eps = random.normal(shape=(batch, dim))

        return z_mean + tf.math.exp(0.5 * z_log_var) * eps  # using the parameterisation trick
