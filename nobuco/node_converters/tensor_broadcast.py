import tensorflow as tf
import torch
from nobuco.converters.tensor import permute_pytorch2keras
from torch import Tensor

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter


def broadcast(tensors):
    shape = tf.shape(tensors[0])
    for tens in tensors[1:]:
        shape = tf.broadcast_dynamic_shape(shape, tf.shape(tens))
    tensors = [tf.broadcast_to(t, shape) for t in tensors]
    return tensors


@converter(torch.broadcast_tensors, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH)
def converter_broadcast_tensors(*tensors):
    def func(*tensors):
        return broadcast(tensors)
    return func


@converter(torch.broadcast_to, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_broadcast_to(input: Tensor, size):
    def func(input: Tensor, size):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            size = permute_pytorch2keras(size)
        return tf.broadcast_to(input, size)
    return func
