import numpy as np
from keras import datasets, Model
from keras import layers, models
import os


def main(latent_dimensions: int = 2):
    """
    Here we make a basic Autoencoder. 

    An auto encoder consists of both a model that is responsible for encoding, and a model that is responsible for decoding.

    The encoder maps the input to a chosen feature space. 

    The feature space is called `the latent embedding space`, this is a space with dimensions specified in the arguments.

    The decoder maps the feature space back to the input space.

    The use of these models is the decoder, where we can co from the embedding space to the input space. 
    
    """
    encoder, decoder, autoencoder = get_models(latent_dimensions)




def get_models(latent_dimensions) -> tuple[Model, Model, Model]:
    encoder_path = "saved_models/encoder.keras"
    decoder_path = "saved_models/decoder.keras"
    autoencoder_path = "saved_models/autoencoder.keras"

    if os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(autoencoder_path):
        encoder = models.load_model(encoder_path)
        decoder = models.load_model(decoder_path)
        autoencoder = models.load_model(autoencoder)
    else:
        encoder, decoder, autoencoder = make_model(latent_dimensions)
        encoder.save(encoder_path)
        decoder.save(decoder_path)
        autoencoder.save(autoencoder_path)

    return encoder, decoder, autoencoder

def make_model(latent_dimensions) -> tuple[Model, Model, Model]:
    x_train, y_train, x_test, y_test = load_data()

    encoder, encoder_input, encoder_output, shape = make_encoder(latent_dimensions)
    encoder.summary()

    decoder = make_decoder(latent_dimensions, shape)
    decoder.summary()

    # the autoencoder stacks the two models
    autoencoder = models.Model(encoder_input, decoder(encoder_output))
    autoencoder = train_autoencoder(autoencoder, x_train, x_test)
    autoencoder.summary()
    return encoder, decoder, autoencoder


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


def make_encoder(latent_dimensions: int) -> tuple[Model, layers.Layer, layers.Layer, tuple[int, int, int]]:
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

    encoder_output = layers.Dense(latent_dimensions, name="encoder_output")(x)

    return Model(encoder_input, encoder_output), encoder_input, encoder_output, shape


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
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

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
