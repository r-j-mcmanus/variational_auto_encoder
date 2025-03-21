import logging

import tensorflow as tf
import numpy as np
from keras import datasets, Model
from keras import layers, models, optimizers
import os

import graphs
from layers.Sampling import Sampling
from layers.LeakyConv2D import LeakyConv2D
from layers.LeakyConv2DTranspose import LeakyConv2DTranspose
from layers.LeakyDense import LeakyDense
from models.VAE import VAE
from load_kagle_celeba_dataset import get_datasets


def main():
    """
    Here we make a variational Autoencoder.

    The difference between this and the autoencoder is that we map images to a point in the autoencoder, but here we
    map images to a random variable, expressed as value drawn from a normal distribution. To model this, rather than
    output being a point in the latent space, we map to a distribution defined on the latent space, with the final
    layer drawing from that distribution.

    The benefit of this is that the model can learn not to associate an image with a point, but with nearby neighboring
    points through the distribution. In so doing the latent space becomes continuous with respect to the decoded images
    in the sense that latent vectors corresponding to fictional images near training vectors will apear to look like
    those training vectors.

    An auto encoder consists of both a model that is responsible for encoding, and a model that is responsible for decoding.

    The encoder maps the input to a chosen feature space.

    The feature space is called `the latent embedding space`, this is a space with dimensions specified in the arguments.

    The decoder maps the feature space back to the input space.

    The use of these models is the decoder, where we can co from the embedding space to the input space.

    """
    vae_container = VAEContainer()


class VAEContainer:
    def __init__(self,
                 latent_dimensions: int = 2,
                 image_size: int = 64,
                 batch_size: int = 128,
                 channels: int = 3,
                 epochs: int = 5):
        self.latent_dimensions = latent_dimensions
        self.image_size = image_size
        self.batch_size = batch_size
        self.channels = channels
        self.epochs = epochs

        self.train_data_set: tf.data.Dataset = None
        self._shape = (0, 0, 0)

        self.encoder: Model = None
        self.decoder: Model = None
        self.autoencoder: Model = None

        self.encoder_path = "saved_models/var_encoder_larger.keras"
        self.decoder_path = "saved_models/var_decoder_larger.keras"
        self.autoencoder_path = "saved_models/var_autoencoder_larger.keras"

        self._init_make_model()

    def _init_make_model(self):
        self.train_data_set = get_datasets(self.image_size, self.batch_size)
        self._get_models()

    def _get_models(self):
        if os.path.exists(self.encoder_path) and os.path.exists(self.decoder_path) and os.path.exists(
                self.autoencoder_path):
            logging.debug("Loading models")
            self.encoder = models.load_model(self.encoder_path)
            self.decoder = models.load_model(self.decoder_path)
            self.autoencoder = models.load_model(self.autoencoder_path)
        else:
            logging.debug("Making models")
            self._make_model()

    def _make_model(self):
        self._make_encoder()
        self.encoder.summary()

        self._make_decoder()
        self.decoder.summary()

        # the autoencoder stacks the two models
        vae = VAE(self.encoder, self.decoder)
        vae.compile()
        vae.fit(self.train_data_set, epochs=self.epochs, batch_size=self.batch_size)
        vae.summary()
        self.autoencoder = vae

        self.encoder.save(self.encoder_path)
        self.decoder.save(self.decoder_path)
        self.autoencoder.save(self.autoencoder_path)

    def _make_encoder(self):
        """Our encoder uses cov layers as the input is an image. We output to """
        encoder_input = layers.Input(
            shape=(self.image_size, self.image_size, self.channels),  # shape of the images we are using
            name="encoder_input"
        )
        # find geometric features
        x = LeakyConv2D(128, (3, 3), strides=2, activation='relu', padding="same")(encoder_input)
        x = LeakyConv2D(128, (3, 3), strides=2, activation='relu', padding="same")(x)
        x = LeakyConv2D(128, (3, 3), strides=2, activation='relu', padding="same")(x)
        x = LeakyConv2D(128, (3, 3), strides=2, activation='relu', padding="same")(x)

        # we need this for the decoder to know how many nodes we go back to from the latent space
        self._shape = x.shape[1:]

        x = layers.Flatten()(x)

        # to the distribution
        z_mean = layers.Dense(self.latent_dimensions, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dimensions, name='z_log_var')(x)

        z = Sampling()([z_mean, z_log_var])

        # the model will output the mean, log_var and the drawn point
        self.encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

    def _make_decoder(self):
        decoder_input = layers.Input(
            shape=(self.latent_dimensions,),
            name="decoder_input"
        )

        # allow for a layer to interact directly with the input to make enough data to make an image
        x = LeakyDense(np.prod(self._shape))(decoder_input)
        x = layers.Reshape(self._shape)(x)

        # invert layers from encoder
        x = LeakyConv2DTranspose(128, (3, 3), strides=2, activation='relu', padding="same")(x)
        x = LeakyConv2DTranspose(128, (3, 3), strides=2, activation='relu', padding="same")(x)
        x = LeakyConv2DTranspose(128, (3, 3), strides=2, activation='relu', padding="same")(x)
        x = LeakyConv2DTranspose(128, (3, 3), strides=2, activation='relu', padding="same")(x)
        # at this point is a 32 x 32 array

        # remove artifacts, Conv2DTranspose layers are known for producing artifacts like checkerboard patterns, especially when upsampling.
        decoder_output = layers.Conv2D(self.channels, (3, 3), strides=1,
                                       activation='sigmoid',  # normalises the output values to [0,1]
                                       padding="same", name="decorator_output")(x)

        self.decoder = Model(decoder_input, decoder_output)


if __name__ == "__main__":
    main()
