import math
from typing import Optional, Union, List, Tuple, Sequence, Any, Callable

from nobuco.node_converters.math import converter_maximum, converter_minimum

from nobuco.layers.channel_order import tf_annotate_recursively
from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size, _layout, _device

import tensorflow as tf
from tensorflow import keras
import torch

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import perm_keras2pytorch, _permute, _ensure_iterable, _dims_make_positive, dims_pytorch2keras


# Converters for reduce functions are made so complex to minimize the number of transpositions.
# Hopefully, the end user will never need to see this.
def reduce_func(inner_func: Callable, n_dims: int):
    def func(input, dim, keepdim, **kwargs):
        if dim is not None:
            dim = _ensure_iterable(dim)

        order = get_channel_order(input)

        if keepdim:
            if dim is not None:
                dim = _dims_make_positive(dim, n_dims)
                if order == ChannelOrder.TENSORFLOW:
                    dim = dims_pytorch2keras(dim, n_dims)
            outputs = inner_func(input, dim, keepdim, **kwargs)
            outputs = tf_annotate_recursively(outputs, order)
            return outputs
        else:
            if dim is not None and order == ChannelOrder.TENSORFLOW:
                perm = perm_keras2pytorch(n_dims)
                input = _permute(perm)(input)
            outputs = inner_func(input, dim, keepdim, **kwargs)
            outputs = tf_annotate_recursively(outputs, ChannelOrder.PYTORCH)
            return outputs
    return func


@converter(torch.sum, torch.Tensor.sum, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_sum(input: Tensor, dim=None, keepdim: _bool=False, *, dtype: Optional[_dtype]=None, out: Optional[Tensor]=None):
    def inner_func(input, dim, keepdim):
        return tf.reduce_sum(input, axis=dim, keepdims=keepdim)

    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, dtype=None, out=None):
        return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func


@converter(torch.mean, torch.Tensor.mean, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_mean(input: Tensor, dim=None, keepdim: _bool=False, *, dtype: Optional[_dtype]=None, out: Optional[Tensor]=None):
    def inner_func(input, dim, keepdim):
        return tf.reduce_mean(input, axis=dim, keepdims=keepdim)

    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, dtype=None, out=None):
        return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func


@converter(torch.std, torch.Tensor.std, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_std(input: Tensor, dim, unbiased: _bool=True, keepdim: _bool=False, *, out: Optional[Tensor]=None):
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

    def inner_func(input, dim, keepdim, unbiased):
        return std(input, axis=dim, unbiased=unbiased, keepdims=keepdim)

    n_dims = input.dim()

    def func(input, dim, unbiased=True, keepdim=False, *, out=None):
        return reduce_func(inner_func, n_dims)(input, dim, keepdim, unbiased=unbiased)
    return func


@converter(torch.any, torch.Tensor.any, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_any(input: Tensor, dim=None, keepdim: _bool=False, *, out: Optional[Tensor]=None):
    def inner_func(input, dim, keepdim):
        return tf.reduce_any(input, axis=dim, keepdims=keepdim)

    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func


@converter(torch.all, torch.Tensor.all, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_all(input: Tensor, dim=None, keepdim: _bool=False, *, out: Optional[Tensor]=None):
    def inner_func(input, dim, keepdim):
        return tf.reduce_all(input, axis=dim, keepdims=keepdim)

    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func


@converter(torch.min, torch.Tensor.min, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_min(input: Tensor, dim=None, keepdim: _bool=False, *, out=None):
    def inner_func(input, dim, keepdim):
        x = tf.keras.backend.min(input, axis=dim, keepdims=keepdim)
        if dim is None:
            return x
        else:
            dim = dim[0]
            a = tf.keras.backend.argmin(input, axis=dim)
            if keepdim:
                a = tf.expand_dims(a, dim)
            return x, a

    n_dims = input.dim()

    if isinstance(dim, torch.Tensor):
        def func(input, dim=None, keepdim=False, *, out=None):
            return converter_minimum.convert(input, other=dim)(input, dim)
    else:
        def func(input, dim=None, keepdim=False, *, out=None):
            return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func


@converter(torch.max, torch.Tensor.max, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_max(input: Tensor, dim=None, keepdim: _bool=False, *, out=None):
    def inner_func(input, dim, keepdim):
        x = tf.keras.backend.max(input, axis=dim, keepdims=keepdim)
        if dim is None:
            return x
        else:
            dim = dim[0]
            a = tf.keras.backend.argmax(input, axis=dim)
            if keepdim:
                a = tf.expand_dims(a, dim)
            return x, a

    n_dims = input.dim()

    if isinstance(dim, torch.Tensor):
        def func(input, dim=None, keepdim=False, *, out=None):
            return converter_maximum.convert(input, other=dim)(input, dim)
    else:
        def func(input, dim=None, keepdim=False, *, out=None):
            return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func


@converter(torch.argmin, torch.Tensor.argmin, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_argmin(input: Tensor, dim=None, keepdim: _bool=False, *, out=None):
    def inner_func(input, dim, keepdim):
        if dim is None:
            if get_channel_order(input) == ChannelOrder.TENSORFLOW:
                perm = perm_keras2pytorch(n_dims)
                input = _permute(perm)(input)
            return tf.math.argmin(tf.reshape(input, shape=(-1,)))
        else:
            dim = dim[0]
            a = tf.keras.backend.argmin(input, axis=dim)
            if keepdim:
                a = tf.expand_dims(a, dim)
            return a

    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func


@converter(torch.argmax, torch.Tensor.argmax, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_argmax(input: Tensor, dim=None, keepdim: _bool=False, *, out=None):
    def inner_func(input, dim, keepdim):
        if dim is None:
            if get_channel_order(input) == ChannelOrder.TENSORFLOW:
                perm = perm_keras2pytorch(n_dims)
                input = _permute(perm)(input)
            return tf.math.argmax(tf.reshape(input, shape=(-1,)))
        else:
            dim = dim[0]
            a = tf.keras.backend.argmax(input, axis=dim)
            if keepdim:
                a = tf.expand_dims(a, dim)
            return a

    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func


@converter(torch.amin, torch.Tensor.amin, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_amin(input: Tensor, dim: Union[_int, _size]=(), keepdim: _bool=False, *, out: Optional[Tensor]=None):
    def inner_func(input, dim, keepdim):
        return keras.layers.Lambda(lambda x: tf.experimental.numpy.amin(x, axis=dim, keepdims=keepdim))(input)

    n_dims = input.dim()

    def func(input, dim=(), keepdim=False, *, out=None):
        return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func


@converter(torch.amax, torch.Tensor.amax, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_amax(input: Tensor, dim: Union[_int, _size]=(), keepdim: _bool=False, *, out: Optional[Tensor]=None):
    def inner_func(input, dim, keepdim):
        return keras.layers.Lambda(lambda x: tf.experimental.numpy.amax(x, axis=dim, keepdims=keepdim))(input)

    n_dims = input.dim()

    def func(input, dim=(), keepdim=False, *, out=None):
        return reduce_func(inner_func, n_dims)(input, dim, keepdim)
    return func
