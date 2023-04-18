import tensorflow as tf

from nobuco.commons import ChannelOrderingStrategy
from nobuco.layers.channel_order import ChangeOrderingLayer


class WeightLayer(tf.keras.layers.Layer):
    def __init__(self, weight_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_shape = weight_shape
        self.weight = self.add_weight('weight', shape=weight_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'weight_shape': self.weight_shape})
        return config

    @classmethod
    def create(cls, weight):
        const_layer = WeightLayer(weight.shape)
        const_layer.set_weights([weight])
        const_layer = ChangeOrderingLayer(const_layer, channel_ordering_strategy=ChannelOrderingStrategy.OUTPUT_FORCE_PYTORCH_ORDER, autocast=False)
        return const_layer

    def call(self, *args, **kwargs):
        return self.weight
