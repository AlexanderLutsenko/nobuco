import numbers

import tensorflow as tf
from tensorflow import keras
import torch

import numpy as np
from nobuco.layers.channel_order import ChangeOrderingLayer, tf_set_order_recursively

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy, TF_TENSOR_CLASSES
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import perm_keras2pytorch, _permute, _flatten, permute_pytorch2keras, _ensure_iterable, \
    _ensure_tuple, dim_pytorch2keras
from nobuco.layers.weight import WeightLayer
from nobuco.node_converters.tensor_broadcast import broadcast


def slices_make_full(slices, n_dims):
    slices = _ensure_tuple(slices)

    def count_nonnone(slices):
        return len([slc for slc in slices if slc is not None])

    def handle_ellipsis(slices, n_dims):
        res = []
        for s in slices:
            if s is Ellipsis:
                n_nonone = count_nonnone(slices)
                for i in range(n_dims - n_nonone + 1):
                    res.append(slice(None, None, None))
            else:
                res.append(s)
        return tuple(res)

    slices = handle_ellipsis(slices, n_dims)

    n_pads = n_dims - count_nonnone(slices)
    slices_full = slices + (slice(None),) * n_pads
    return slices_full


def to_shape_and_dtype(assigned_tensor, shape, dtype):
    if assigned_tensor.dtype != dtype:
        assigned_tensor = tf.cast(assigned_tensor, dtype)
    assigned_tensor = tf.broadcast_to(assigned_tensor, shape)
    return assigned_tensor


def slice_assign(sliced_tensor, slice_args, assigned_tensor):
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
                idx = tf.convert_to_tensor(slice_spec)
                idx = tf.where(idx < 0, idx + shape[real_index], idx)
                corresponding_range = tf.cast(idx, dtype=tf.int32)
                n_indexed_dims += 1

            corresponding_ranges.append(corresponding_range)
            dims_to_index.append(real_index)

    if not isinstance(assigned_tensor, TF_TENSOR_CLASSES):
        assigned_tensor = tf.convert_to_tensor(assigned_tensor)
        assigned_tensor = tf.cast(assigned_tensor, sliced_tensor.dtype)
        assigned_tensor = WeightLayer.create(assigned_tensor, trainable=False)(sliced_tensor)

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

    if isinstance(assigned_tensor, TF_TENSOR_CLASSES):
        if 1 < len(assigned_tensor.shape) < len(scatted_nd_perm):
            new_shape = []
            k = 0
            for i in range(n_dims):
                if i in dims_to_index:
                    d = tf.size(corresponding_ranges[k])
                    k += 1
                else:
                    d = assigned_tensor.shape[i]
                new_shape.append(d)
            assigned_tensor = tf.reshape(assigned_tensor, new_shape)

        if len(assigned_tensor.shape) == len(scatted_nd_perm):
            assigned_tensor = tf.transpose(assigned_tensor, scatted_nd_perm)

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


def getitem_sophisticated(self, n_dims, *slices):
    slices = _ensure_iterable(slices)
    res = self

    # Assign corresponding dims to slice specs taking into account None dimensions (i.e. unsqueeze ops) and ellipses (...)
    def enumerate_properly(slices, n_dims):
        res = []
        i = 0
        inverse = False
        for s in slices:
            if s is Ellipsis:
                inverse = True
            else:
                res.append((i, s, inverse))
                if s is not None:
                    i += 1

        max_idx = res[len(res) - 1][0]
        res = [(n_dims - 1 + max_idx - i, s) if inverse else (i, s) for (i, s, inverse) in res]
        return res

    def collect_take_indices_and_specs(slices_enumerated):
        dims = []
        slice_specs = []
        for i, s in slices_enumerated:
            if not isinstance(s, (slice, numbers.Number)) and s is not None:
                s = tf.convert_to_tensor(s)
                if s.dtype is not tf.bool:
                    dims.append(i)
                    slice_specs.append(s)
        return dims, slice_specs

    def indices_make_non_negative(indices, d):
        dim_size = tf.cast(tf.shape(res)[d], indices.dtype)
        return tf.where(indices < 0, dim_size + indices, indices)

    # ND variant of tf.experimental.numpy.take
    def take_nd(x, indices_list, start_dim):
        n_idx_dims = len(indices_list)
        shape = tf.cast(tf.shape(x), tf.int64)
        indices_list = [tf.cast(idxs, tf.int64) for idxs in indices_list]

        dim_mults = [shape[start_dim + i] for i in range(n_idx_dims)]
        dim_mults = tf.stack(dim_mults, axis=-1)
        dim_mults = tf.math.cumprod(dim_mults, reverse=True, exclusive=True)
        idxs = tf.stack(indices_list, axis=-1)
        linear_coordinates = tf.reduce_sum(dim_mults * idxs, axis=-1)

        x = tf.reshape(x, shape=(*shape[:start_dim], -1, *shape[start_dim + n_idx_dims:]))
        out = tf.gather(params=x, indices=linear_coordinates, axis=start_dim)
        return out

    def take_nd_disjoint(x, dims, indices_list, n_dims):
        start_dim = dims[0]
        perm_head = list(range(start_dim))
        perm_tail = [i for i in range(start_dim, n_dims) if i not in dims]
        perm = perm_head + dims + perm_tail
        x = tf.transpose(x, perm)
        indices_list = broadcast(indices_list)
        x = take_nd(x, indices_list, start_dim)
        return x

    slices_enumerated = enumerate_properly(slices, n_dims)

    def slices_adjust(slices_enumerated, start_i, delta):
        return [(i + delta, s) if i >= start_i else (i, s) for (i, s) in slices_enumerated]

    slices_adjusted = slices_enumerated[:]

    for i, slice_spec in reversed(list(slices_enumerated)):
        pads = [slice(None, None, None)] * i
        if isinstance(slice_spec, slice):
            if not (slice_spec.start is None and slice_spec.stop is None and slice_spec.step is None):
                res = res.__getitem__([*pads, slice_spec])
        elif isinstance(slice_spec, numbers.Number):
            res = res.__getitem__([*pads, slice_spec])
            # Remove one dimension
            slices_adjusted = slices_adjust(slices_adjusted, i, -1)
            n_dims -= 1
        elif slice_spec is None:
            res = res.__getitem__([*pads, slice_spec])
            # Add one dimension
            slices_adjusted = slices_adjust(slices_adjusted, i, +1)
            n_dims += 1
        else:
            slice_spec = tf.convert_to_tensor(slice_spec)
            if slice_spec.dtype is tf.bool:
                # Select by boolean mask
                res = tf.boolean_mask(res, slice_spec, axis=i)

    take_dims, take_slice_specs = collect_take_indices_and_specs(slices_adjusted)

    # Select by integer indices
    if len(take_dims) == 1:
        # Simple case
        d, indices = take_dims[0], take_slice_specs[0]
        indices = indices_make_non_negative(indices, d)
        res = keras.layers.Lambda(lambda x: tf.experimental.numpy.take(x[0], x[1], axis=d))([res, indices])
    elif len(take_dims) > 1:
        # Complicated case
        take_slice_specs = [indices_make_non_negative(s, d) for d, s in zip(take_dims, take_slice_specs)]
        res = take_nd_disjoint(res, take_dims, take_slice_specs, n_dims)

    return res


@converter(torch.Tensor.__getitem__, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def converter_getitem(self, *slices):
    n_dims = self.dim()
    slices = _flatten(slices)

    def is_light(slices):
        return all(isinstance(slc, slice) or slc is Ellipsis for slc in slices)

    if is_light(slices):
        def func(self, *slices):
            x = self
            slices = _ensure_iterable(slices)

            if get_channel_order(x) == ChannelOrder.TENSORFLOW:
                slices = slices_make_full(slices, n_dims)
                slices = permute_pytorch2keras(slices)
                x = x.__getitem__(slices)
                x = set_channel_order(x, ChannelOrder.TENSORFLOW)
            else:
                x = x.__getitem__(slices)
                x = set_channel_order(x, ChannelOrder.PYTORCH)
            return x
    else:
        def func(self, *slices):
            x = self
            slices = _ensure_iterable(slices)
            x, slices = tf_set_order_recursively((x, slices), channel_order=ChannelOrder.PYTORCH)

            x = getitem_sophisticated(x, n_dims, slices)
            x = set_channel_order(x, ChannelOrder.PYTORCH)
            return x
    return func


@converter(torch.scatter, torch.Tensor.scatter, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_scatter(input, dim, index, src):
    if input.dim() == 1:
        def func(input, dim, index, src):
            return tf.tensor_scatter_nd_update(input, index[..., None], src)
    else:
        def func(input, dim, index, src):
            if get_channel_order(input) == ChannelOrder.TENSORFLOW:
                dim = dim_pytorch2keras(dim, tf.rank(input))

            grids = tf.where(tf.ones_like(index))
            rank = tf.shape(grids)[-1:]
            grids = tf.reshape(grids, tf.concat([tf.shape(index), rank], axis=0))

            # Clean but not fully supported by TFLite
            # grids = tf.unstack(grids, axis=-1)
            # grids[dim] = index
            # multi_index = tf.stack(grids, axis=-1)

            # TFLite-compatible variant
            # - avoid operating tensors of dimensionality 5 and higher
            # - avoid tf.unstack
            grids = tf.reshape(grids, (-1, rank[0]))
            index = tf.reshape(index, (-1,))
            grids_before = grids[..., :dim]
            grids_after = grids[..., dim + 1:]
            multi_index = tf.concat([grids_before, index[..., None], grids_after], axis=-1)
            multi_index = tf.reshape(multi_index, tf.concat([tf.shape(src), rank], axis=0))

            return tf.tensor_scatter_nd_update(input, multi_index, src)
    return func
