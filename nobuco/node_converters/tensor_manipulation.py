from typing import Optional, Union, List, Tuple, Sequence, Any

import keras.layers
import torch
from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf

import numpy as np

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import dims_pytorch2keras, perm_keras2pytorch, \
    _dim_make_positive, dim_pytorch2keras, _permute, _flatten, perm_pytorch2keras, perm_compose, \
    is_identity_perm, permute_pytorch2keras, perm_identity, _ensure_iterable


def _permute_inner(perm_original, allow_lazy=True):
    perm_original = _ensure_iterable(perm_original)

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


@converter(torch.Tensor.split, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_split(self, split_size, dim=0):
    num_dims = self.dim()

    def func(self, split_size, dim=0):
        if get_channel_order(self) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.split(self, num_or_size_splits=split_size, axis=dim)
    return func


@converter(torch.chunk, torch.Tensor.chunk, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
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


@converter(torch.repeat_interleave, torch.Tensor.repeat_interleave, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_repeat_interleave(input: Tensor, repeats, dim: Optional[_int] = None, *, output_size: Optional[_int] = None):
    n_dims = input.dim()

    def func(input, repeats, dim = None, *, output_size = None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.repeat(input, repeats, axis=dim)
    return func


@converter(torch.Tensor.expand, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_expand(self, *sizes):
    n_dims = self.dim()

    def get_broadcast_shape(sizes, tensor_shape):
        sizes_hd, sizes_tl = sizes[:-n_dims], sizes[-n_dims:]
        sizes_tl = tf.where(sizes_tl == -1, tensor_shape, sizes_tl)
        return tf.concat([sizes_hd, sizes_tl], axis=0)

    def func(self, *sizes):
        sizes = _flatten(sizes)
        sizes = tf.convert_to_tensor(sizes)
        broadcast_shape = get_broadcast_shape(sizes, tf.shape(self))
        return tf.broadcast_to(self, broadcast_shape)

    return func


@converter(torch.Tensor.expand_as, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_expand_as(self, other):
    def func(self, other):
        return tf.broadcast_to(self, tf.shape(other))
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


@converter(torch.unbind, torch.Tensor.unbind, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_unbind(input, dim=0):
    num_partitions = input.shape[dim]

    def func(input, dim=0):
        outputs = tf.split(input, num_partitions, axis=dim)
        outputs = [tf.squeeze(out, dim) for out in outputs]
        return outputs
    return func


@converter(torch.Tensor.flatten, torch.flatten, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_t_flatten(input, start_dim=0, end_dim=-1):
    n_dims = input.dim()
    start_dim_pos = _dim_make_positive(start_dim, n_dims)
    end_dim_pos = _dim_make_positive(end_dim, n_dims)

    def func(input, start_dim=0, end_dim=-1):
        shape = tf.shape(input)
        start_shape = shape[:start_dim_pos]
        end_shape = shape[end_dim_pos+1:]
        return tf.reshape(input, (*start_shape, -1, *end_shape))
    return func


@converter(torch.Tensor.unflatten, torch.unflatten, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_unflatten(input: Tensor, dim: _int, sizes):
    n_dims = input.dim()
    dim_pos = _dim_make_positive(dim, n_dims)

    def func(input: Tensor, dim, sizes):
        shape = tf.shape(input)
        start_shape = shape[:dim_pos]
        end_shape = shape[dim_pos+1:]
        return tf.reshape(input, (*start_shape, *sizes, *end_shape))
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


@converter(torch.gather, torch.Tensor.gather, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_gather(input: Tensor, dim, index: Tensor, *, sparse_grad: _bool=False, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim, index, *, sparse_grad=False, out=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return torch_gather(input, index, dim)
    return func
