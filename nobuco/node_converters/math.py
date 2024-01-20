import math
from typing import Optional, Union, List, Tuple, Sequence, Any

import keras.layers
from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size, _layout, _device

import tensorflow as tf
import torch

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import _dim_make_positive, dim_pytorch2keras, perm_keras2pytorch, _permute, \
    perm_pytorch2keras, _ensure_iterable, _dims_make_positive, dims_pytorch2keras
from nobuco.converters.type_cast import dtype_pytorch2keras


@converter(torch.sin, torch.Tensor.sin, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_sin(input, *args, **kwargs):
    def func(input, *args, **kwargs):
        return tf.math.sin(input)
    return func


@converter(torch.cos, torch.Tensor.cos, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_cos(input, *args, **kwargs):
    def func(input, *args, **kwargs):
        return tf.math.cos(input)
    return func


@converter(torch.add, torch.Tensor.add, torch.Tensor.add_, torch.Tensor.__add__, torch.Tensor.__iadd__, torch.Tensor.__radd__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_add(input, other, *args, **kwargs):
    def func(input, other, *args, **kwargs):
        return input + other
    return func


@converter(torch.sub, torch.subtract, torch.Tensor.sub, torch.Tensor.sub_, torch.Tensor.__sub__, torch.Tensor.__isub__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_sub(input: Union[Tensor, Number], other: Union[Tensor, Number], *, alpha: Optional[Number]=1, out: Optional[Tensor]=None):
    def func(input, other, *, alpha=1, out=None):
        return input - other
    return func


@converter(torch.Tensor.__rsub__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_rsub(self, other):
    def func(self, other):
        return other - self
    return func


@converter(torch.Tensor.neg, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_neg(self):
    def func(self):
        return -self
    return func


@converter(torch.mul, torch.Tensor.__mul__, torch.Tensor.__imul__, torch.Tensor.__rmul__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_mul(input: Union[Tensor, Number], other: Union[Tensor, Number], *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return input * other
    return func


@converter(torch.Tensor.mul, torch.Tensor.mul_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_mul(self, value):
    def func(self, value):
        return self * value
    return func


def divide(x, y, rounding_mode):
    if rounding_mode is None:
        return x / y
    elif rounding_mode == 'trunc':
        return tf.truncatediv(x, y)
    elif rounding_mode == 'floor':
        return x // y


@converter(torch.Tensor.__truediv__, torch.Tensor.__idiv__, torch.div, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_div(input: Union[Tensor, Number], other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None, out: Optional[Tensor]=None):
    def func(input, other, *, rounding_mode=None, out=None):
        return divide(input, other, rounding_mode)
    return func


@converter(torch.Tensor.div, torch.Tensor.div_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_div(self, value, rounding_mode=None, *args, **kwargs):
    def func(self, value, rounding_mode=None, *args, **kwargs):
        return divide(self, value, rounding_mode)
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


@converter(torch.Tensor.__mod__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
def converter_mod(self, value, *args, **kwargs):
    def func(self, value, *args, **kwargs):
        return self % value
    return func


@converter(torch.sqrt, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_sqrt(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.sqrt(input)
    return func


@converter(torch.rsqrt, torch.Tensor.rsqrt, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_rsqrt(input: Tensor, *, out: Optional[Tensor] = None):
    def func(input: Tensor, *, out: Optional[Tensor] = None):
        # return tf.math.rsqrt(input)
        return 1 / tf.math.sqrt(input)
    return func


@converter(torch.pow, torch.Tensor.pow, torch.Tensor.__pow__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS, autocast=True)
def converter_pow(input: Tensor, exponent: Number, *, out: Optional[Tensor]=None):
    def func(input, exponent, *, out=None):
        return input ** exponent
    return func


@converter(torch.Tensor.__rpow__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS, autocast=True)
def converter_rpow(self, other):
    def func(self, other):
        return other ** self
    return func


@converter(torch.exp, torch.Tensor.exp, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_exp(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return tf.exp(input)
    return func


@converter(torch.log, torch.Tensor.log, torch.log_, torch.Tensor.log_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_log(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return tf.math.log(input)
    return func


@converter(torch.log2, torch.Tensor.log2, torch.log2_, torch.Tensor.log2_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_log2(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return keras.layers.Lambda(lambda x: tf.experimental.numpy.log2(x))(input)
    return func


@converter(torch.log10, torch.Tensor.log10, torch.log10_, torch.Tensor.log10_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_log10(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return keras.layers.Lambda(lambda x: tf.experimental.numpy.log10(x))(input)
    return func


@converter(torch.log1p, torch.Tensor.log1p, torch.log1p_, torch.Tensor.log1p_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_log1p(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return keras.layers.Lambda(lambda x: tf.experimental.numpy.log1p(x))(input)
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


@converter(torch.clamp, torch.Tensor.clamp, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS, autocast=True)
def converter_clamp(input: Tensor, min: Optional[Number]=None, max: Optional[Number]=None, *, out: Optional[Tensor]=None):
    def func(input, min=None, max=None, *, out=None):
        # return tf.keras.backend.clip(input, min_value=min, max_value=max)
        if min is not None:
            input = tf.maximum(input, min)
        if max is not None:
            input = tf.minimum(input, max)
        return input
    return func


@converter(torch.clamp_min, torch.Tensor.clamp_min, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS, autocast=True)
def converter_clamp_min(input: Tensor, min, *, out = None):
    def func(input, min, *, out = None):
        # return tf.keras.backend.clip(self, min_value=min, max_value=None)
        return tf.maximum(input, min)
    return func


@converter(torch.clamp_max, torch.Tensor.clamp_max, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS, autocast=True)
def converter_clamp_max(input: Tensor, max, *, out = None):
    def func(input, max, *, out = None):
        # return tf.keras.backend.clip(self, min_value=None, max_value=max)
        return tf.minimum(input, max)
    return func


@converter(torch.minimum, torch.Tensor.minimum, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS, autocast=True)
def converter_minimum(input: Tensor, other: Tensor, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.minimum(input, other)
    return func


@converter(torch.maximum, torch.Tensor.maximum, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS, autocast=True)
def converter_maximum(input: Tensor, other: Tensor, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.maximum(input, other)
    return func


@converter(torch.cumsum, torch.Tensor.cumsum, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_cumsum(input: Tensor, dim, *, dtype: Optional[_dtype] = None, out: Optional[Tensor] = None):
    num_dims = input.dim()

    def func(input, dim, *, dtype = None, out = None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        if dtype is not None:
            tf_type = dtype_pytorch2keras(dtype)
            input =  tf.cast(input, tf_type)
        return tf.cumsum(input, axis=dim)
    return func


@converter(torch.arange, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_arange(start: Number, end: Number=None, step: Number=1, *, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False):
    def func(start, end=None, step=1, *, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        return tf.range(start, limit=end, delta=step)
    return func
