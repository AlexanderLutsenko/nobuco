import numbers
from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
import torch

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import _dim_make_positive, dim_pytorch2keras


@converter(torch.sum, torch.Tensor.sum, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_sum(input: Tensor, dim: Sequence, keepdim: _bool=False, *, dtype: Optional[_dtype]=None, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim, keepdim=False, *, dtype=None, out=None):
        dim = _dim_make_positive(dim, n_dims)
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.reduce_sum(input, axis=dim, keepdims=keepdim)
    return func


@converter(torch.mean, torch.Tensor.mean, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_mean(input: Tensor, dim=None, keepdim: _bool = False, *, dtype: Optional[_dtype] = None, out: Optional[Tensor] = None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, dtype=None, out=None):
        if dim is not None:
            dim = _dim_make_positive(dim, n_dims)
            if get_channel_order(input) == ChannelOrder.TENSORFLOW:
                dim = dim_pytorch2keras(dim, n_dims)
        return tf.reduce_mean(input, axis=dim, keepdims=keepdim)
    return func


@converter(torch.Tensor.add, torch.Tensor.__add__, torch.Tensor.__iadd__, torch.Tensor.__radd__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_add(input, other, *args, **kwargs):
    def func(input, other, *args, **kwargs):
        return input + other
    return func


@converter(torch.sub, torch.subtract, torch.Tensor.sub, torch.Tensor.__sub__, torch.Tensor.__isub__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_sub(input: Union[Tensor, Number], other: Union[Tensor, Number], *, alpha: Optional[Number]=1, out: Optional[Tensor]=None):
    def func(input, other, *, alpha=1, out=None):
        return input - other
    return func


@converter(torch.Tensor.__rsub__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_rsub(self, other):
    def func(self, other):
        return other - self
    return func


@converter(torch.mul, torch.Tensor.__mul__, torch.Tensor.__imul__, torch.Tensor.__rmul__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_mul(input: Union[Tensor, Number], other: Union[Tensor, Number], *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return input * other
    return func


@converter(torch.Tensor.mul, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_mul(self, value):
    def func(self, value):
        return self * value
    return func


@converter(torch.Tensor.__truediv__, torch.Tensor.__idiv__, torch.div, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_div(input: Union[Tensor, Number], other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None, out: Optional[Tensor]=None):
    def func(input, other, *, rounding_mode=None, out=None):
        return input / other
    return func


@converter(torch.Tensor.div, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_div(self, value, *args, **kwargs):
    def func(self, value, *args, **kwargs):
        return self / value
    return func


@converter(torch.Tensor.__rdiv__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_rdiv(self, other):
    def func(self, other):
        return other / self
    return func


@converter(torch.floor_divide, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_floor_divide(input: Union[Tensor, Number], other: Union[Tensor, Number], *, out: Optional[Tensor]=None):
    def func(input: Union[Tensor, Number], other: Union[Tensor, Number], *, out: Optional[Tensor]=None):
        return input // other
    return func


@converter(torch.sqrt, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_sqrt(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.sqrt(input)
    return func


@converter(torch.Tensor.rsqrt, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_rsqrt(self):
    def func(self):
        # return tf.math.rsqrt(self)
        return 1 / tf.math.sqrt(self)
    return func


@converter(torch.Tensor.__pow__, torch.pow, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_pow(input: Tensor, exponent: Number, *, out: Optional[Tensor]=None):
    def func(input, exponent, *, out=None):
        return input ** exponent
    return func


@converter(torch.Tensor.__rpow__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_rpow(self, other):
    def func(self, other):
        return other ** self
    return func


@converter(torch.exp, torch.Tensor.exp, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_exp(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return tf.exp(input)
    return func


@converter(torch.log, torch.Tensor.log, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_log(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return tf.experimental.numpy.log(input)
    return func


@converter(torch.log2, torch.Tensor.log2, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_log2(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return tf.experimental.numpy.log2(input)
    return func


@converter(torch.abs, torch.Tensor.abs, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_abs(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.abs(input)
    return func


@converter(torch.ceil, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_ceil(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.math.ceil(input)
    return func


@converter(torch.floor, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_floor(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.floor(input)
    return func


@converter(torch.round, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_round(input: Tensor, *, decimals: _int, out=None):
    assert decimals == 0

    def func(input: Tensor, *, decimals: _int, out=None):
        return tf.round(input)
    return func


@converter(torch.Tensor.round, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_t_round(self, decimals=0):
    assert decimals == 0

    def func(self, decimals=0):
        return tf.round(self)
    return func


@converter(torch.clamp, torch.Tensor.clamp, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_clamp(input: Tensor, min: Optional[Number]=None, max: Optional[Number]=None, *, out: Optional[Tensor]=None):
    def func(input, min=None, max=None, *, out=None):
        return tf.keras.backend.clip(input, min_value=min, max_value=max)
    return func


@converter(torch.min, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_min(input: Tensor, dim=None, keepdim: _bool=False, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor]]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        if dim is not None and get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.keras.backend.min(input, axis=dim, keepdims=keepdim)
    return func


@converter(torch.max, torch.Tensor.max, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_max(input: Tensor, dim=None, keepdim: _bool=False, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor]]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        if dim is not None and get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.keras.backend.max(input, axis=dim, keepdims=keepdim)
    return func


@converter(torch.argmax, torch.Tensor.argmax, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_argmax(input: Tensor, dim: Optional[_int]=None, keepdim: _bool=False, *, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        if dim is not None and get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.keras.backend.argmax(input, axis=dim)
    return func
