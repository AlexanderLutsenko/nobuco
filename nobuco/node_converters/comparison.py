from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
import torch

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy, TF_TENSOR_CLASSES
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import dim_pytorch2keras


@converter(torch.greater, torch.Tensor.__gt__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_gt(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.greater(input, other)
    return func


@converter(torch.greater_equal, torch.Tensor.__ge__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_ge(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.greater_equal(input, other)
    return func


@converter(torch.less, torch.Tensor.__lt__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_lt(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.less(input, other)
    return func


@converter(torch.less_equal, torch.Tensor.__le__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_le(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.less_equal(input, other)
    return func


@converter(torch.eq, torch.equal, torch.Tensor.__eq__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_eq(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.equal(input, other)
    return func


@converter(torch.topk, torch.Tensor.topk, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_topk(input, k, dim=-1, largest=True, sorted=True, *, out=None):
    def func(input, k, dim=-1, largest=True, sorted=True, *, out=None):
        if isinstance(k, TF_TENSOR_CLASSES) and k.dtype != tf.int32:
            # TopK in TFLite can only work with int32
            k = tf.cast(k, tf.int32)
        result = tf.math.top_k(input, k=k, sorted=sorted)
        indices = tf.cast(result.indices, tf.int64)
        return result.values, indices
    return func


@converter(torch.Tensor.sort, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_sort(self, dim=-1, descending=False):
    n_dims = self.dim()

    def func(self, dim=-1, descending=False):
        if get_channel_order(self) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        if descending:
            direction = 'DESCENDING'
        else:
            direction = 'ASCENDING'
        sorted = tf.sort(self, axis=dim, direction=direction)
        argsorted = tf.argsort(self, axis=dim, direction=direction)
        argsorted = tf.cast(argsorted, tf.int64)
        return sorted, argsorted
    return func


@converter(torch.unique, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    assert input.dim() == 1
    assert return_inverse is False

    def func(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
        x, _ = tf.unique(input)
        if sorted:
            x = tf.sort(x)
        return x
    return func
