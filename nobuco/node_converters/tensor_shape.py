import torch
import tensorflow as tf

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(torch._shape_as_tensor, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_shape_as_tensor(self):
    def func(self):
        return tf.shape(self)
    return func
