import kagglehub
import tensorflow as tf
from keras import utils


def _get_data_path() -> str:
    path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print("Path to dataset files:", path)
    return path


def _preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img


def get_datasets(image_size: int, batch_size: int):
    path = _get_data_path()
    path = path + '\\img_align_celeba\\img_align_celeba'

    train_data = utils.image_dataset_from_directory(
        path,
        labels=None,
        color_mode="rgb",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        interpolation="bilinear",
    )

    train = train_data.map(lambda x: _preprocess(x))
    return train
