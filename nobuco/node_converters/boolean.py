from torch import Tensor

import tensorflow as tf
import torch

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(torch.Tensor.__and__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_logical_and(input: Tensor, other):
    def func(input, other):
        return tf.logical_and(input, other)
    return func


@converter(torch.Tensor.__or__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_logical_or(input: Tensor, other):
    def func(input, other):
        return tf.logical_or(input, other)
    return func


@converter(torch.Tensor.__invert__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_invert(input: Tensor):
    def func(input):
        return tf.math.logical_not(input)
    return func


@converter(torch.Tensor.__ne__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_ne(self, *args, **kwargs):
    def func(self, *args, **kwargs):
        val = args[0]
        return tf.math.not_equal(self, val)
    return func
