import tensorflow as tf
from tensorflow import keras

from nobuco.commons import ChannelOrderingStrategy
from nobuco.layers.channel_order import ChangeOrderingLayer


class WeightLayer(keras.layers.Layer):
    def __init__(self, weight_shape, weight_dtype, trainable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_shape = weight_shape
        self.weight_dtype = weight_dtype
        self.weight = self.add_weight('weight', shape=weight_shape, dtype=weight_dtype, trainable=trainable, initializer=keras.initializers.Zeros()
)

    def get_config(self):
        config = super().get_config()
        config.update({
            'weight_shape': self.weight_shape,
            'weight_dtype': self.weight_dtype,
        })
        return config

    @classmethod
    def create(cls, weight, trainable):
        const_layer = WeightLayer(weight.shape, weight.dtype, trainable=trainable)
        const_layer.set_weights([weight])
        const_layer = ChangeOrderingLayer(const_layer, channel_ordering_strategy=ChannelOrderingStrategy.OUTPUT_FORCE_PYTORCH_ORDER, autocast=False)
        return const_layer

    @tf.function
    def call(self, *args, **kwargs):
        return self.weight
