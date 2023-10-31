from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
import torch
from torch import nn

import numpy as np

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import dims_pytorch2keras, perm_keras2pytorch, \
    _dim_make_positive, dim_pytorch2keras, _permute, _flatten, perm_pytorch2keras, perm_compose, \
    is_identity_perm, permute_pytorch2keras, perm_identity, _ensure_iterable


def _permute_inner(perm_original, allow_lazy=True):
    def func(x):
        input_channel_order = get_channel_order(x)

        if allow_lazy:
            if input_channel_order == ChannelOrder.TENSORFLOW and list(perm_original) == perm_pytorch2keras(len(perm_original)):
                x = tf.identity(x)
                x = set_channel_order(x, ChannelOrder.PYTORCH)
                return x
            elif input_channel_order == ChannelOrder.PYTORCH and list(perm_original) == perm_keras2pytorch(len(perm_original)):
                x = tf.identity(x)
                x = set_channel_order(x, ChannelOrder.TENSORFLOW)
                return x

        if input_channel_order == ChannelOrder.TENSORFLOW:
            perm = perm_compose(perm_original, perm_keras2pytorch(len(perm_original)))
            perm = perm_compose(perm_pytorch2keras(len(perm)), perm)
        else:
            perm = perm_original

        if is_identity_perm(perm):
            return x
        else:
            x = tf.transpose(x, perm)
            x = set_channel_order(x, input_channel_order)
            return x
    return func


@converter(torch.Tensor.permute, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_t_permute(self, *dims):
    def func(self, *dims):
        return _permute_inner(dims)(self)
    return func


@converter(torch.permute, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_permute(input: Tensor, dims: _size):
    def func(input, dims):
        return _permute_inner(dims)(input)
    return func


@converter(torch.Tensor.transpose, torch.transpose, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_transpose(input: Tensor, dim0, dim1):
    a = np.zeros(shape=perm_identity(input.dim()))
    perm = np.swapaxes(a, dim0, dim1).shape

    def func(input, dim0, dim1):
        return _permute_inner(perm)(input)
    return func


@converter(torch.Tensor.t, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_t(self: Tensor):
    assert self.dim() == 2
    def func(self):
        return tf.transpose(self, perm=(1, 0))
    return func


# tensor.T
@converter(torch.Tensor.__getattribute__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_getattribute(self, attribute):
    if attribute == 'T':
        def func(self, attribute):
            return _permute_inner([1, 0])(self)
    elif attribute == 'data':
        def func(self, attribute):
            return self
    else:
        raise Exception(f'Unsupported attribute: {attribute}')
    return func


@converter(torch.moveaxis, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_moveaxis(input: Tensor, source: _size, destination: _size):
    a = np.zeros(shape=perm_identity(input.dim()))
    perm = np.moveaxis(a, source, destination).shape

    def func(input, source, destination):
        return _permute_inner(perm)(input)
    return func


@converter(torch.Tensor.view, torch.Tensor.reshape, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_reshape(self, *shape):
    def func(self, *shape):
        shape = _flatten(shape)
        return tf.reshape(self, tuple(shape))
    return func


@converter(torch.reshape, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_reshape(input, *shape):
    def func(input, *shape):
        shape = _flatten(shape)
        return tf.reshape(input, shape)
    return func


@converter(torch.cat, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim=0, *, out: Optional[Tensor]=None):
    num_dims = tensors[0].dim()
    dim_keras = dim_pytorch2keras(dim, num_dims)

    def func(tensors, dim=0, *, out=None):
        if get_channel_order(tensors[0]) == ChannelOrder.TENSORFLOW:
            return tf.concat(tensors, axis=dim_keras)
        else:
            return tf.concat(tensors, axis=dim)
    return func


@converter(torch.stack, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_stack(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: _int=0, *, out: Optional[Tensor]=None):
    def func(tensors, dim=0, *, out=None):
        return tf.stack(tensors, axis=dim)
    return func


@converter(torch.Tensor.split, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_split(self, split_size, dim=0):
    num_dims = self.dim()

    def func(self, split_size, dim=0):
        if get_channel_order(self) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.split(self, num_or_size_splits=split_size, axis=dim)
    return func


@converter(torch.chunk, torch.Tensor.chunk, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_chunk(input, chunks, dim=0):
    num_dims = input.dim()

    def func(input, chunks, dim=0):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.split(input, num_or_size_splits=chunks, axis=dim)
    return func


@converter(torch.Tensor.repeat, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_repeat(self, *sizes):
    def func(self, *sizes):
        if get_channel_order(self) == ChannelOrder.TENSORFLOW:
            sizes = permute_pytorch2keras(sizes)
        return tf.tile(self, sizes)
    return func


@converter(torch.Tensor.expand, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_expand(self, *sizes):
    def get_broadcast_shape(sizes, tensor_shape):
        return [ts if s == -1 else s for s, ts in zip(sizes, tensor_shape)]

    def func(self, *sizes):
        sizes = _flatten(sizes)
        broadcast_shape = get_broadcast_shape(sizes, self.shape)
        return tf.broadcast_to(self, broadcast_shape)
    return func


@converter(torch.Tensor.expand_as, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_expand_as(self, other):
    def func(self, other):
        return tf.broadcast_to(self, other.shape)
    return func


@converter(torch.roll, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_roll(input: Tensor, shifts: Union[_int, _size], dims: Union[_int, _size]=()):
    assert dims != (), 'Assertion error'
    n_dims = input.dim()

    def func(input, shifts, dims=()):
        shifts = _ensure_iterable(shifts)
        dims = _ensure_iterable(dims)

        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dims = dims_pytorch2keras(dims, n_dims)

        x = input
        for shift, dim in zip(shifts, dims):
            x = tf.roll(x, shift=shift, axis=dim)
        return x
    return func


@converter(torch.Tensor.unbind, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_unbind(self, dim=0):
    def func(self, dim=0):
        return tf.unstack(self, axis=dim)
    return func


@converter(torch.Tensor.flatten, torch.flatten, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_t_flatten(self, start_dim=0, end_dim=-1):
    def func(self, start_dim=0, end_dim=-1):
        start_shape = self.shape[:start_dim]

        n_dims = len(self.shape)
        end_dim = _dim_make_positive(end_dim, n_dims)
        if end_dim < n_dims-1:
            end_shape = self.shape[end_dim+1:]
        else:
            end_shape = []

        return tf.reshape(self, (*start_shape, -1, *end_shape))
    return func


@converter(torch.Tensor.narrow, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_narrow(self, dimension, start, length):
    n_dims = self.dim()

    def func(self, dimension, start, length):
        x = self
        dimension = _dim_make_positive(dimension, n_dims)

        if get_channel_order(x) == ChannelOrder.TENSORFLOW:
            perm = perm_keras2pytorch(n_dims)
            x = _permute(perm)(x)
        slices = (*[slice(None)]*dimension, slice(start, start+length))
        x = x.__getitem__(slices)
        x = set_channel_order(x, ChannelOrder.PYTORCH)
        return x
    return func


@converter(torch.squeeze, torch.Tensor.squeeze, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_squeeze(input: Tensor, dim):
    n_dims = input.dim()
    def func(input, dim):
        x = input
        if get_channel_order(x) == ChannelOrder.TENSORFLOW:
            perm = perm_keras2pytorch(n_dims)
            x = _permute(perm)(x)
        x = tf.squeeze(x, axis=dim)
        x = set_channel_order(x, ChannelOrder.PYTORCH)
        return x
    return func


@converter(torch.unsqueeze, torch.Tensor.unsqueeze, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_unsqueeze(input, dim):
    n_dims = input.dim()
    def func(input, dim):
        x = input
        if get_channel_order(x) == ChannelOrder.TENSORFLOW:
            perm = perm_keras2pytorch(n_dims)
            x = _permute(perm)(x)
        x = tf.expand_dims(x, axis=dim)
        x = set_channel_order(x, ChannelOrder.PYTORCH)
        return x
    return func


def torch_gather(x, indices, gather_axis):
    """
    See: https://stackoverflow.com/a/71037111/4850610
    """

    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(tf.cast(gather_locations, dtype=tf.int64))
        else:
            gather_indices.append(tf.cast(all_indices[:, axis], dtype=tf.int64))

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped


@converter(torch.gather, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_gather(input: Tensor, dim, index: Tensor, *, sparse_grad: _bool=False, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim, index, *, sparse_grad=False, out=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return torch_gather(input, index, dim)
    return func
