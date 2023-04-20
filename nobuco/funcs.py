import torch
from torch import Tensor

import tensorflow as tf
from nobuco.converters.tensor import permute_pytorch2keras, permute_keras2pytorch

from nobuco.converters.channel_ordering import get_channel_order

from nobuco.commons import ChannelOrderingStrategy, ChannelOrder
from nobuco.converters.node_converter import converter
from nobuco.trace.trace import traceable


@traceable
def force_tensorflow_order(inputs):
    return inputs


@converter(force_tensorflow_order, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_force_tensorflow_order(inputs):
    return lambda inputs: inputs


@traceable
def force_pytorch_order(inputs):
    return inputs


@converter(force_pytorch_order, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_force_pytorch_order(inputs):
    return lambda inputs: inputs


@traceable
def shape(x: Tensor):
    return tuple(torch.tensor(d, dtype=torch.int32) for d in x.shape)


@converter(shape, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_shape(x: Tensor):
    def func(x):
        shape = tf.unstack(tf.shape(x))
        if get_channel_order(x) == ChannelOrder.TENSORFLOW:
            shape = permute_keras2pytorch(shape)
        return tuple(shape)
    return func
