from keras import saving
from tensorflow.keras.layers import Layer, Conv2DTranspose, BatchNormalization, LeakyReLU


@saving.register_keras_serializable()
class LeakyConv2DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), activation='relu',
                 padding='same', use_bias=True, negative_slope=0.2, **kwargs):
        super(LeakyConv2DTranspose, self).__init__(**kwargs)
        self._conv_t = Conv2DTranspose(filters, kernel_size, strides=strides, activation='relu',
                                       padding=padding, use_bias=use_bias)
        self._bn = BatchNormalization()
        self._leaky_relu = LeakyReLU(negative_slope=negative_slope)

    def call(self, inputs, training=False):
        x = self._conv_t(inputs)
        x = self._bn(x, training=training)
        return self._leaky_relu(x)

    def get_config(self):
        config = super(LeakyConv2DTranspose, self).get_config()
        config.update({
            'conv_t': self._conv_t.get_config(),
            'bn': self._bn.get_config(),
            'leaky_relu': self._leaky_relu.get_config()
        })
        return config
