from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor

import tensorflow as tf
import torch
from torch import nn

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(torch.fill, torch.Tensor.fill_, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_fill(input, value):
    def func(input, value):
        result = tf.fill(tf.shape(input), value)
        result = tf.cast(result, dtype=input.dtype)
        return result
    return func


@converter(torch.meshgrid, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_meshgrid(*tensors, indexing: Optional[str] = None):
    def func(*tensors, indexing=None):
        return tf.meshgrid(*tensors, indexing=indexing)
    return func


@converter(torch.isnan, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_isnan(input: Tensor):
    def func(input):
        return tf.math.is_nan(input)
    return func


@converter(nn.Identity, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_Identity(self, input: Tensor):
    def func(input):
        return input
    return func


@converter(torch.Tensor.copy_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH)
def converter_copy_(self, src, non_blocking=False):
    def func(self, src, non_blocking=False):
        return self * 0 + src
    return func


@converter(torch.clone, torch.Tensor.clone, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_clone(input: Tensor, *, memory_format=None):
    def func(input: Tensor, *, memory_format=None):
        return tf.identity(input)
    return func
