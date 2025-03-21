from keras import saving
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU


@saving.register_keras_serializable()
class LeakyDense(Layer):
    def __init__(self, units, alpha=0.2, **kwargs):
        super(LeakyDense, self).__init__(**kwargs)
        self._dense = Dense(units=units)
        self._bn = BatchNormalization()
        self._leaky_relu = LeakyReLU(alpha=alpha)

    def call(self, inputs, training=False):
        x = self._dense(inputs)
        x = self._bn(x, training=training)
        return self._leaky_relu(x)

    def get_config(self):
        config = super(LeakyDense, self).get_config()
        config.update({
            'dense': self._dense.get_config(),
            'bn': self._bn.get_config(),
            'leaky_relu': self._leaky_relu.get_config()
        })
        return config
