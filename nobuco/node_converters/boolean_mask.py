from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
import torch

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(torch.masked_select, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_masked_select(input: Tensor, mask: Tensor, *, out: Optional[Tensor]=None):
    def func(input, mask, *, out=None):
        return input[mask]
    return func


@converter(torch.masked_fill, torch.Tensor.masked_fill, torch.Tensor.masked_fill_, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_masked_fill(input: Tensor, mask: Tensor, value: Number):
    def func(input, mask, value):
        value = tf.convert_to_tensor(value, dtype=input.dtype)
        return tf.where(mask, value, input)
    return func


@converter(torch.where, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_where(condition: Tensor, input=None, other=None):
    def func(condition, input=None, other=None):
        if input is not None and other is not None:
            return tf.where(condition, x=input, y=other)
        else:
            return tf.where(condition)[..., 0]
    return func
