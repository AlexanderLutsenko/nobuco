import tensorflow as tf
import torch

import numpy as np
from nobuco.layers.channel_order import ChangeOrderingLayer

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy, TF_TENSOR_CLASSES
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import perm_keras2pytorch, _permute, _flatten, permute_pytorch2keras, _ensure_iterable, _ensure_tuple
from nobuco.layers.weight import WeightLayer
from nobuco.node_converters.boolean_mask import converter_masked_select


def slices_make_full(slices, n_dims):
    slices = _ensure_tuple(slices)
    n_notnone = len([slc for slc in slices if slc is not None])
    n_pads = n_dims - n_notnone
    slices_full = slices + (slice(None),) * n_pads
    return slices_full


def to_shape_and_dtype(assigned_tensor, shape, dtype):
    if assigned_tensor.dtype != dtype:
        assigned_tensor = tf.cast(assigned_tensor, dtype)
    assigned_tensor = tf.broadcast_to(assigned_tensor, shape)
    return assigned_tensor


def broadcast(tensors):
    shape = tensors[0].shape
    for tens in tensors:
        shape = tf.broadcast_dynamic_shape(shape, tens.shape)
    tensors = [tf.broadcast_to(t, shape) for t in tensors]
    return tensors


def slice_assign(sliced_tensor, slice_args, assigned_tensor, is_scatter=False):
    slice_args = _ensure_iterable(slice_args)
    """Assign a tensor to the slice of another tensor.
    No broadcast is performed.
    Args:
        - sliced_tensor (tf.Tensor): the tensor whose slice you want changed.
        - assigned_tensor (tf.Tensor): the tensor which you want assigned.
        - *slice_args (str or slice): the slices arguments. Can be ':', '...'
        or slice.
    Returns:
        - tf.Tensor: the original tensor with the slice correctly assigned.
    """

    shape = sliced_tensor.shape
    n_dims = len(shape)
    n_indexed_dims = 0
    # parsing the slice specifications
    n_slices = len(slice_args)
    dims_to_index = []
    corresponding_ranges = []
    ellipsis = False
    for i_dim, slice_spec in enumerate(slice_args):
        if slice_spec is Ellipsis:
            ellipsis = True
        else:
            if ellipsis:
                real_index = i_dim + (n_dims - n_slices)
            else:
                real_index = i_dim

            if isinstance(slice_spec, slice):
                start, stop, step = slice_spec.start, slice_spec.stop, slice_spec.step

                if start is None and stop is None and step is None:
                    continue

                if start is None:
                    start = 0
                if stop is None:
                    stop = shape[real_index]
                if step is None:
                    step = 1

                if start < 0:
                    start = shape[real_index] + start
                if stop < 0:
                    stop = shape[real_index] + stop

                corresponding_range = tf.cast(tf.range(start, stop, step), dtype=tf.int32)
            else:
                slice_spec = tf.convert_to_tensor(slice_spec)
                corresponding_range = tf.cast(slice_spec, dtype=tf.int32)
                n_indexed_dims += 1

            corresponding_ranges.append(corresponding_range)
            dims_to_index.append(real_index)

    if not isinstance(assigned_tensor, TF_TENSOR_CLASSES):
        assigned_tensor = tf.convert_to_tensor(assigned_tensor)
        assigned_tensor = WeightLayer.create(assigned_tensor)(sliced_tensor)

    if not dims_to_index:
        assigned_tensor = to_shape_and_dtype(assigned_tensor, sliced_tensor.shape, sliced_tensor.dtype)
        return assigned_tensor

    dims_left_out = [i_dim for i_dim in range(n_dims) if i_dim not in dims_to_index]
    scatted_nd_perm = dims_to_index + dims_left_out
    inverse_scatter_nd_perm = list(np.argsort(scatted_nd_perm))

    left_out_shape = [shape[i_dim] for i_dim in dims_left_out]

    if n_indexed_dims < 2:
        mesh_ranges = tf.meshgrid(*corresponding_ranges, indexing='ij')
        sliced_shape = [tf.size(r) for r in corresponding_ranges] + left_out_shape
    elif n_indexed_dims == len(corresponding_ranges):
        mesh_ranges = broadcast(corresponding_ranges)
        sliced_shape = [tf.size(mesh_ranges[0])] + left_out_shape
    else:
        raise Exception('This slice configuration is currently not supported')

    update_indices = tf.stack([
        tf.reshape(slicing_range, (-1,))
        for slicing_range in mesh_ranges
    ], axis=-1)

    if isinstance(assigned_tensor, TF_TENSOR_CLASSES) and len(assigned_tensor.shape) == len(scatted_nd_perm):
        assigned_tensor = tf.transpose(assigned_tensor, scatted_nd_perm)

    if is_scatter:
        assigned_tensor_reshaped = assigned_tensor
    else:
        assigned_tensor_reshaped = to_shape_and_dtype(assigned_tensor, sliced_shape, sliced_tensor.dtype)
        assigned_tensor_reshaped = tf.reshape(assigned_tensor_reshaped, [-1] + left_out_shape)

    # NOTE: the tensors are reshaped to allow for easier indexing with
    sliced_tensor_reshaped = tf.transpose(sliced_tensor, perm=scatted_nd_perm)

    # finalisation
    sliced_tensor_reshaped = tf.tensor_scatter_nd_update(
        tensor=sliced_tensor_reshaped,
        indices=update_indices,
        updates=assigned_tensor_reshaped,
    )
    sliced_tensor_updated = tf.transpose(sliced_tensor_reshaped, perm=inverse_scatter_nd_perm)
    return sliced_tensor_updated


def slice_assign_boolean_mask_scatter(sliced_tensor, slice_args, assigned_tensor):
    update_indices = tf.where(slice_args)
    sliced_tensor = tf.tensor_scatter_nd_update(
        tensor=sliced_tensor,
        indices=update_indices,
        updates=assigned_tensor,
    )
    return sliced_tensor


def slice_assign_boolean_mask_select(sliced_tensor, slice_args, assigned_tensor):
    if isinstance(assigned_tensor, TF_TENSOR_CLASSES):
        assigned_tensor = tf.cast(assigned_tensor, dtype=sliced_tensor.dtype)
    else:
        assigned_tensor = tf.convert_to_tensor(assigned_tensor, dtype=sliced_tensor.dtype)
    return tf.where(slice_args, assigned_tensor, sliced_tensor)


@converter(torch.Tensor.__setitem__, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_setitem(sliced_tensor, slice_args, assigned_tensor):
    if isinstance(slice_args, torch.Tensor) and slice_args.dtype == torch.bool:
        if isinstance(assigned_tensor, torch.Tensor) and assigned_tensor.numel() != 1:
            func = slice_assign_boolean_mask_scatter
            func = ChangeOrderingLayer(func, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
        else:
            func = slice_assign_boolean_mask_select
            func = ChangeOrderingLayer(func, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
    else:
        n_dims = sliced_tensor.dim()

        def func(sliced_tensor, slice_args, assigned_tensor):
            if get_channel_order(sliced_tensor) == ChannelOrder.TENSORFLOW:
                slice_args = slices_make_full(slice_args, n_dims)
                slice_args = permute_pytorch2keras(slice_args)
            return slice_assign(sliced_tensor, slice_args, assigned_tensor)
        func = ChangeOrderingLayer(func, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
    return func


@converter(channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def getitem_indexed(self, *slices):
    def func(self, *slices):
        slices = _ensure_iterable(slices)
        slices = broadcast(slices)
        slices = tf.stack(slices, axis=-1)
        return tf.gather_nd(self, slices)
    return func


@converter(torch.Tensor.__getitem__, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_getitem(self, *slices):
    n_dims = self.dim()

    slices = _flatten(slices)

    if isinstance(slices[0], torch.Tensor):
        if slices[0].dtype == torch.bool:
            return converter_masked_select.convert(self, slices[0])
        elif slices[0].dtype == torch.int64:
            return getitem_indexed.convert(self, slices)

    def is_light(slices):
        return all(isinstance(slc, slice) for slc in slices)

    if is_light(slices):
        def func(self, *slices):
            x = self
            slices = _ensure_iterable(slices)

            if get_channel_order(x) == ChannelOrder.TENSORFLOW:
                s = slices_make_full(slices, n_dims)
                s = permute_pytorch2keras(s)
                x = x.__getitem__(s)
                x = set_channel_order(x, ChannelOrder.TENSORFLOW)
            else:
                x = x.__getitem__(slices)
                x = set_channel_order(x, ChannelOrder.PYTORCH)
            return x
    else:
        def func(self, *slices):
            x = self
            slices = _ensure_iterable(slices)

            if get_channel_order(x) == ChannelOrder.TENSORFLOW:
                perm = perm_keras2pytorch(n_dims)
                x = _permute(perm)(x)
            x = x.__getitem__(slices)
            x = set_channel_order(x, ChannelOrder.PYTORCH)
            return x
    return func


# @converter(torch.Tensor.scatter, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
# def converter_scatter(self, dim, index, src):
#     assert dim == 0
#     def func(self, dim, index, src):
#         slice_args = (tf.reshape(index, -1),)
#         return slice_assign(self, src, slice_args, is_scatter=True)
#     return func
