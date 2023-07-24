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


@converter(torch.sum, torch.Tensor.sum, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_sum(input: Tensor, dim=None, keepdim: _bool=False, *, dtype: Optional[_dtype]=None, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, dtype=None, out=None):
        if dim is not None:
            dim = _ensure_iterable(dim)

        order = get_channel_order(input)

        if keepdim:
            dim = _dims_make_positive(dim, n_dims)
            if order == ChannelOrder.TENSORFLOW:
                dim = dims_pytorch2keras(dim, n_dims)
            out = tf.reduce_sum(input, axis=dim, keepdims=keepdim)
            out = set_channel_order(out, order)
            return out
        else:
            if order == ChannelOrder.TENSORFLOW:
                perm = perm_keras2pytorch(n_dims)
                input = _permute(perm)(input)
            out = tf.reduce_sum(input, axis=dim, keepdims=keepdim)
            out = set_channel_order(out, ChannelOrder.PYTORCH)
            return out

    return func


@converter(torch.mean, torch.Tensor.mean, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_mean(input: Tensor, dim=None, keepdim: _bool=False, *, dtype: Optional[_dtype]=None, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, dtype=None, out=None):
        if dim is not None:
            dim = _ensure_iterable(dim)

        order = get_channel_order(input)

        if keepdim:
            dim = _dims_make_positive(dim, n_dims)
            if order == ChannelOrder.TENSORFLOW:
                dim = dims_pytorch2keras(dim, n_dims)
            out = tf.reduce_mean(input, axis=dim, keepdims=keepdim)
            out = set_channel_order(out, order)
            return out
        else:
            if order == ChannelOrder.TENSORFLOW:
                perm = perm_keras2pytorch(n_dims)
                input = _permute(perm)(input)
            out = tf.reduce_mean(input, axis=dim, keepdims=keepdim)
            out = set_channel_order(out, ChannelOrder.PYTORCH)
            return out
    return func


@converter(torch.std, torch.Tensor.std, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_std(input: Tensor, dim, unbiased: _bool=True, keepdim: _bool=False, *, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def var_unbiased(input, axis, keepdims=False):
        input_means = tf.reduce_mean(input, axis=axis, keepdims=True)
        squared_deviations = tf.square(input - input_means)

        if input.dtype in (tf.complex64, tf.complex128):
            squared_deviations = tf.abs(squared_deviations)

        sum = tf.reduce_sum(squared_deviations, axis=axis, keepdims=keepdims)
        shape = tf.shape(input)
        n = math.prod(shape[i] for i in axis)
        # [sic] This implementation follows Pytorch behaviour, i.e. makes NaNs when n==1
        return sum / tf.cast(n - 1, dtype=sum.dtype)

    def std(input, axis, unbiased=True, keepdims=False):
        if unbiased:
            var = var_unbiased(input, axis=axis, keepdims=keepdims)
            return tf.sqrt(var)
        else:
            return tf.math.reduce_std(input, axis=axis, keepdims=keepdims)

    def func(input, dim, unbiased=True, keepdim=False, *, out=None):
        if dim is not None:
            dim = _ensure_iterable(dim)

        order = get_channel_order(input)

        if keepdim:
            dim = _dims_make_positive(dim, n_dims)
            if order == ChannelOrder.TENSORFLOW:
                dim = dims_pytorch2keras(dim, n_dims)
            out = std(input, axis=dim, unbiased=unbiased, keepdims=keepdim)
            out = set_channel_order(out, order)
            return out
        else:
            if order == ChannelOrder.TENSORFLOW:
                perm = perm_keras2pytorch(n_dims)
                input = _permute(perm)(input)
            out = std(input, axis=dim, unbiased=unbiased, keepdims=keepdim)
            out = set_channel_order(out, ChannelOrder.PYTORCH)
            return out
    return func


@converter(torch.sin, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_sin(input, *args, **kwargs):
    def func(input, *args, **kwargs):
        return tf.math.sin(input)
    return func


@converter(torch.cos, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_cos(input, *args, **kwargs):
    def func(input, *args, **kwargs):
        return tf.math.cos(input)
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


@converter(torch.Tensor.clamp_min, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_clamp(self, min):
    def func(self, min):
        return tf.keras.backend.clip(self, min_value=min, max_value=None)
    return func


@converter(torch.Tensor.clamp_max, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_clamp(self, max):
    def func(self, max):
        return tf.keras.backend.clip(self, min_value=None, max_value=max)
    return func


@converter(torch.minimum, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_minimum(input: Tensor, other: Tensor, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.minimum(input, other)
    return func


@converter(torch.maximum, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_maximum(input: Tensor, other: Tensor, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.maximum(input, other)
    return func


@converter(torch.min, torch.Tensor.min, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_min(input: Tensor, dim=None, keepdim: _bool=False, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor]]=None):
    n_dims = input.dim()

    if isinstance(dim, Tensor):
        return converter_minimum.convert(input, dim, out=out)

    def func(input, dim=None, keepdim=False, *, out=None):
        if dim is None:
            x = tf.keras.backend.min(input, axis=dim, keepdims=keepdim)
            return set_channel_order(x, get_channel_order(input))
        else:
            if get_channel_order(input) == ChannelOrder.TENSORFLOW:
                input = _permute(perm_keras2pytorch(n_dims))(input)
                input = set_channel_order(input, ChannelOrder.PYTORCH)
            x = tf.keras.backend.min(input, axis=dim, keepdims=keepdim)
            a = tf.keras.backend.argmin(input, axis=dim)
            order = get_channel_order(input)
            return set_channel_order(x, order), set_channel_order(a, order)
    return func


@converter(torch.max, torch.Tensor.max, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_max(input: Tensor, dim=None, keepdim: _bool=False, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor]]=None):
    n_dims = input.dim()

    if isinstance(dim, Tensor):
        return converter_maximum.convert(input, dim, out=out)

    def func(input, dim=None, keepdim=False, *, out=None):
        if dim is None:
            x = tf.keras.backend.max(input, axis=dim, keepdims=keepdim)
            return set_channel_order(x, get_channel_order(input))
        else:
            if get_channel_order(input) == ChannelOrder.TENSORFLOW:
                input = _permute(perm_keras2pytorch(n_dims))(input)
                input = set_channel_order(input, ChannelOrder.PYTORCH)
            x = tf.keras.backend.max(input, axis=dim, keepdims=keepdim)
            a = tf.keras.backend.argmax(input, axis=dim)
            order = get_channel_order(input)
            return set_channel_order(x, order), set_channel_order(a, order)
    return func


@converter(torch.argmin, torch.Tensor.argmin, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_argmin(input: Tensor, dim: Optional[_int]=None, keepdim: _bool=False, *, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.keras.backend.argmin(input, axis=dim)
    return func


@converter(torch.argmax, torch.Tensor.argmax, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_argmax(input: Tensor, dim: Optional[_int]=None, keepdim: _bool=False, *, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.keras.backend.argmax(input, axis=dim)
    return func


@converter(torch.arange, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_arange(start: Number, end: Number=None, step: Number=1, *, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False):
    def func(start, end=None, step=1, *, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        return tf.range(start, limit=end, delta=step)
    return func
