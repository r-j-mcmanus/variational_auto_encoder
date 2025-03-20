from matplotlib import pyplot as plt
import numpy as np
from keras import Model


def autoencoded_images(autoencoder: Model, x_test: np.array):
    n_images = 5

    images = x_test[:n_images]
    x_mean, x_log_var, reconstructed = autoencoder.predict(images)

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 2))  # Create n subplots in a row
    for i in range(n_images):
        axes[0][i].imshow(images[i], cmap='gray')
        axes[1][i].imshow(reconstructed[i], cmap='gray')
    for ax in axes[0]:
        ax.axis('off')  # Turn off axes for better visualization
    for ax in axes[1]:
        ax.axis('off')  # Turn off axes for better visualization

    plt.tight_layout()
    plt.show()


def latent_space(encoder: Model, x_test: np.array, labels: np.array):
    """Note the asymetric distribution, large empty spaces, but also clustering of points"""
    n = 5000

    images = x_test[:n]
    x_mean, x_log_var, x_drawn = encoder.predict(images)

    plt.scatter(x_mean[:, 0], x_mean[:, 1], s=20, c=labels[:n], alpha=0.7)
    plt.show()


def generated_image(decoder: Model, latent_vector: np.array):
    """Note the asymetric distribution, large empty spaces, but also clustering of points"""
    if latent_vector.shape == (2,):
        latent_vector = latent_vector.reshape(1, -1)
    image = decoder.predict(latent_vector)

    plt.imshow(image[0], cmap='gray')
    plt.show()


