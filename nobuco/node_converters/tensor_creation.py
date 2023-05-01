from typing import Optional

from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
import torch
import torch.nn.functional as F

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter
from nobuco.node_converters.tensor_cast import dtype_pytorch2keras


@converter(torch.zeros_like, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_zeros_like(input: Tensor, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
    def func(input: Tensor, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        tf_type = dtype_pytorch2keras(dtype)
        return tf.zeros_like(input, dtype=tf_type)
    return func


@converter(torch.Tensor.new_empty, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_new_empty(self, size, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
    def func(self, size, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
        if dtype is not None:
            dtype = dtype_pytorch2keras(dtype)
        else:
            dtype = self.dtype
        return tf.zeros(size, dtype=dtype)
    return func


@converter(torch.Tensor.new_full, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
    def func(self, size, fill_value, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
        if dtype is not None:
            dtype = dtype_pytorch2keras(dtype)
        else:
            dtype = self.dtype
        res = tf.fill(size, fill_value)
        res = tf.cast(res, dtype)
        return res
    return func


@converter(torch.full_like, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_full_like(input: Tensor, fill_value: Number, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
    def func(input: Tensor, fill_value: Number, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        if dtype is not None:
            dtype = dtype_pytorch2keras(dtype)
        else:
            dtype = input.dtype
        res = tf.fill(input.shape, fill_value)
        res = tf.cast(res, dtype)
        return res
    return func


@converter(F._canonical_mask, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_canonical_mask(mask: Optional[Tensor], mask_name: str, other_type: Optional, other_name: str, target_type, check_other: bool = True):
    if mask is not None:
        _mask_is_float = torch.is_floating_point(mask)
        if not _mask_is_float:
            raise Exception('Not supported yet')

    def func(mask, mask_name, other_type, other_name, target_type, check_other=True):
        return mask
    return func
