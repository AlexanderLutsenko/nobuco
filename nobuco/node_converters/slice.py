import tensorflow as tf
import torch

import numpy as np

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import perm_keras2pytorch, _permute, _flatten, permute_pytorch2keras, _ensure_iterable
from nobuco.node_converters.boolean_mask import converter_masked_select


# def broadcast_to_dim(tensor, target_n_dims):
#     shape = tensor.shape()
#     n_dims = len(tensor.shape)
#     if get_channel_order(tensor) == ChannelOrder.TENSORFLOW:
#         shape = dims_keras2pytorch(shape, n_dims)
#         target_shape = shape_make_full(shape, target_n_dims)
#         target_shape = dims_pytorch2keras(target_shape, target_n_dims)
#     else:
#         target_shape = shape_make_full(shape, target_n_dims)
#
#     tensor = tf.reshape(tensor, target_shape)
#     return tensor


def slices_make_full(slices, n_dims):
    n_notnone = len([slc for slc in slices if slc is not None])
    n_pads = n_dims - n_notnone
    slices_full = slices + (slice(None),) * n_pads
    return slices_full


def slice_assign(sliced_tensor, assigned_tensor, *slice_args, verbose=0):
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
    # parsing the slice specifications
    n_slices = len(slice_args)
    dims_to_index = []
    corresponding_ranges = []
    ellipsis = False
    for i_dim, slice_spec in enumerate(slice_args):
        if slice_spec is Ellipsis:
            ellipsis = True
        else:
            if isinstance(slice_spec, int):
                start, stop, step = slice_spec, slice_spec + 1, None
            elif isinstance(slice_spec, slice):
                start, stop, step = slice_spec.start, slice_spec.stop, slice_spec.step
            else:
                raise Exception(f'Unrecognized slice spec: {slice_spec}')

            no_start = start is None or start == 0
            no_stop = stop is None or stop == -1
            no_step = step is None or step == 1
            if no_start and no_stop and no_step:
                continue
            if ellipsis:
                real_index = i_dim + (n_dims - n_slices)
            else:
                real_index = i_dim
            dims_to_index.append(real_index)
            if no_step:
                step = 1
            if no_stop:
                stop = shape[real_index]
            if no_start:
                start = 0
            corresponding_range = tf.range(start, stop, step)
            corresponding_ranges.append(corresponding_range)
    if not dims_to_index:
        if verbose > 0:
            print('Warning: no slicing performed')
        return assigned_tensor
    dims_left_out = [
        i_dim for i_dim in range(n_dims) if i_dim not in dims_to_index
    ]
    scatted_nd_perm = dims_to_index + dims_left_out
    inverse_scatter_nd_perm = list(np.argsort(scatted_nd_perm))
    # reshaping the tensors
    # NOTE: the tensors are reshaped to allow for easier indexing with
    # tensor_scatter_nd_update
    sliced_tensor_reshaped = tf.transpose(sliced_tensor, perm=scatted_nd_perm)
    assigned_tensor_reshaped = tf.transpose(assigned_tensor, perm=scatted_nd_perm)
    left_out_shape = [shape[i_dim] for i_dim in dims_left_out]
    assigned_tensor_reshaped = tf.reshape(assigned_tensor_reshaped, [-1] + left_out_shape)
    # creating the indices
    mesh_ranges = tf.meshgrid(*corresponding_ranges, indexing='ij')
    update_indices = tf.stack([
        tf.reshape(slicing_range, (-1,))
        for slicing_range in mesh_ranges
    ], axis=-1)

    # finalisation
    sliced_tensor_reshaped = tf.tensor_scatter_nd_update(
        tensor=sliced_tensor_reshaped,
        indices=update_indices,
        updates=assigned_tensor_reshaped,
    )
    sliced_tensor_updated = tf.transpose(
        sliced_tensor_reshaped,
        perm=inverse_scatter_nd_perm,
    )
    return sliced_tensor_updated


@converter(channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def getitem_indexed(self, *slices):
    slices = _flatten(slices)
    slices = torch.broadcast_tensors(*slices)
    slices_combined = torch.stack(slices, dim=-1).numpy()

    def func(self, *slices):
        return tf.gather_nd(self, slices_combined)
    return func


@converter(torch.Tensor.__getitem__, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_getitem(self, *slices):
    n_dims = self.dim()

    slices = _flatten(slices)

    if isinstance(slices[0], torch.Tensor):
        if slices[0].dtype == torch.bool:
            return converter_masked_select(self, slices[0])
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


@converter(torch.Tensor.__setitem__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_setitem(sliced_tensor, assigned_tensor, *slice_args):
    n_dims = sliced_tensor.dim()

    def func(sliced_tensor, slice_args, assigned_tensor):
        if get_channel_order(sliced_tensor) == ChannelOrder.TENSORFLOW:
            slice_args = _flatten(slice_args)
            slice_args = slices_make_full(slice_args, n_dims)
            slice_args = permute_pytorch2keras(slice_args)
        return slice_assign(sliced_tensor, assigned_tensor, *slice_args)
    return func
