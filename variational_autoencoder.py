import logging

import numpy as np
from keras import datasets, Model
from keras import layers, models, optimizers
import os

import graphs
from layers.Sampling import Sampling
from models.VAE import VAE


def main(latent_dimensions: int = 2):
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
    encoder, decoder, autoencoder = get_models(latent_dimensions)

    x_train, y_train, x_test, y_test = load_data()

    graphs.autoencoded_images(autoencoder, x_test)
    # this is very blury and clearly mixes images. This is becasue we are moving to distributions not points, and
    # forcing the means to be near the origin. As such the latent feature space isnt large enogh for independent
    # features ot carve out their own space.
    graphs.latent_space(encoder, x_test, y_test)
    graphs.generated_image(decoder, np.array([1.0, 0.0]))
    graphs.generated_image(decoder, np.array([0, -8]))


def get_models(latent_dimensions) -> tuple[Model, Model, Model]:
    encoder_path = "saved_models/var_encoder.keras"
    decoder_path = "saved_models/var_decoder.keras"
    autoencoder_path = "saved_models/var_autoencoder.keras"

    if os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(autoencoder_path):
        logging.debug("Loading models")
        encoder: Model = models.load_model(encoder_path)
        decoder: Model = models.load_model(decoder_path)
        autoencoder: Model = models.load_model(autoencoder_path)
    else:
        logging.debug("Making models")
        encoder, decoder, autoencoder = make_model(latent_dimensions)
        encoder.save(encoder_path)
        decoder.save(decoder_path)
        autoencoder.save(autoencoder_path)

    return encoder, decoder, autoencoder


def make_model(latent_dimensions) -> tuple[Model, Model, Model]:
    x_train, y_train, x_test, y_test = load_data()

    encoder, shape = make_encoder(latent_dimensions)
    encoder.summary()

    decoder = make_decoder(latent_dimensions, shape)
    decoder.summary()

    # the autoencoder stacks the two models
    vae = VAE(encoder, decoder)
    vae.compile()
    vae.fit(x_train, epochs=5, batch_size=100)
    vae.summary()
    return encoder, decoder, vae


def preprocess(images: np.array) -> np.array:
    images = images.astype("float32") / 255.0  # 0-1 range
    images = np.pad(images, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)  # add 2 to the left, right, above and bellow
    images = np.expand_dims(images, -1)
    return images


def load_data() -> tuple[np.array, np.array, np.array, np.array]:
    # imports 28x28 grey pixel images
    ((x_train, y_train), (x_test, y_test)) = datasets.fashion_mnist.load_data()
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    return x_train, y_train, x_test, y_test


def make_encoder(latent_dimensions: int) -> tuple[Model, np.array]:
    """Our encoder uses cov layers as the input is an image. We output to """
    encoder_input = layers.Input(
        shape=(32, 32, 1),  # shape of the images we are using
        name="encoder_input"
    )
    # find geometric features
    x = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding="same")(encoder_input)
    x = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same")(x)

    # we need this for the decoder to know how many nodes we go back to from the latent space
    shape = x.shape[1:]

    x = layers.Flatten()(x)

    # to the distribution
    z_mean = layers.Dense(latent_dimensions, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dimensions, name='z_log_var')(x)

    z = Sampling()([z_mean, z_log_var])

    # the model will output the mean, log_var and the drawn point
    model = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

    return model, shape


def make_decoder(latent_dimensions: int, shape: tuple[int, int, int]) -> Model:
    decoder_input = layers.Input(
        shape=(latent_dimensions,),
        name="decoder_input"
    )

    # allow for a layer to interact directly with the input to make enough data to make an image
    x = layers.Dense(np.prod(shape))(decoder_input)
    x = layers.Reshape(shape)(x)

    # invert layers from encoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding="same")(x)
    # at this point is a 32 x 32 array

    # remove artifacts, Conv2DTranspose layers are known for producing artifacts like checkerboard patterns, especially when upsampling.

    decoder_output = layers.Conv2D(1, (3, 3), strides=1,
                                   activation='sigmoid',  # normalises the output values to [0,1]
                                   padding="same", name="decorator_output")(x)

    return Model(decoder_input, decoder_output)


def train_autoencoder(autoencoder: Model, x_train: np.array, x_test: np.array) -> Model:
    # binary_crossentropy leads to asymmetric penalisation, which will push the pixel value towards 0.5.
    # this leads to blurry images but smoother edges
    optimizer = optimizers.Adam(learning_rate=0.0005)
    autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy")

    autoencoder.fit(
        x_train,
        x_train,  # as the output aims to be the same as the input!
        epochs=5,
        batch_size=100,
        shuffle=True,
        validation_data=(x_test, x_test)
    )

    return autoencoder


if __name__ == "__main__":
    main()
