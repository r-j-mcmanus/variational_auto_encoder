from keras import Model
from keras import metrics
from keras import losses
from keras import saving

import tensorflow as tf

@saving.register_keras_serializable()
class VAE(Model):
    """A model for a variational auto encoder"""

    def __init__(self, encoder: Model, decoder: Model, beta=500, **kwargs):
        # pass args for model to Model
        super(VAE, self).__init__(**kwargs)

        self.encoder: Model = encoder
        self.decoder: Model = decoder

        self.total_loss_tracker = metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = metrics.Mean(name='kl_loss')

        self.beta = beta

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(self, inputs):
        """The variational auto encoder takes an image array as an input and returns the encoded -> decoded image
        array"""

        # get the latent space params from the encoder
        z_mean, z_log_var, z = self.encoder(inputs)

        # decode back to the image
        reconstruction = self.decoder(z)

        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # run the model against the data
            z_mean, z_log_var, reconstruction = self(data)

            # for the reconstruction we use binary cross entropy as ...
            # we use the weight beta for this loss as ...
            reconstruction_loss = tf.reduce_mean(
                self.beta * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3))
            )

            # for the distribution we use kl_loss, which is a measure of difference in distributions, below is the
            # closed form for two normal distributions
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.math.exp(z_log_var))
                )
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        """This is needed as encoder and decorder are passed to the init, so we need to save their values"""
        config = super(VAE, self).get_config()
        config.update({
            'encoder': saving.serialize_keras_object(self.encoder),
            'decoder': saving.serialize_keras_object(self.decoder)
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        encoder = saving.deserialize_keras_object(config.pop('encoder'))
        decoder = saving.deserialize_keras_object(config.pop('decoder'))
        return cls(encoder=encoder, decoder=decoder, **config)