import numbers
from typing import Optional, Union, Any, Sequence

from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size, _device, _layout, Device

import tensorflow as tf
import torch
import torch.nn.functional as F

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy, TF_TENSOR_CLASSES
from nobuco.converters.node_converter import converter
from nobuco.node_converters.tensor_cast import dtype_pytorch2keras


@converter(torch.tensor, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_tensor(data: Any, dtype: Optional[_dtype]=None, device: Device=None, requires_grad: _bool=False):
    def func(data, dtype=None, device=None, requires_grad=False):
        dtype = dtype_pytorch2keras(dtype)
        if dtype is not None and isinstance(data, TF_TENSOR_CLASSES):
            return tf.cast(data, dtype=dtype)
        else:
            # Sic!
            if dtype is None:
                return tf.convert_to_tensor(data)
            else:
                return tf.convert_to_tensor(data, dtype=dtype)
    return func


@converter(torch.scalar_tensor, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_scalar_tensor(s: Number, *, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False):
    def func(s, *, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        dtype = dtype_pytorch2keras(dtype)
        if dtype is not None and isinstance(s, TF_TENSOR_CLASSES):
            return tf.cast(s, dtype=dtype)
        else:
            # Sic!
            if dtype is None:
                return tf.convert_to_tensor(s)
            else:
                return tf.convert_to_tensor(s, dtype=dtype)
    return func


@converter(torch.zeros, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_zeros(*size: _int, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False):
    def func(*size, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        tf_type = dtype_pytorch2keras(dtype)
        # Sic!
        if dtype is None:
            return tf.zeros(shape=size)
        else:
            return tf.zeros(shape=size, dtype=tf_type)
    return func


@converter(torch.Tensor.zero_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_zero_(self):
    def func(self):
        return tf.zeros_like(self)
    return func


@converter(torch.zeros_like, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_zeros_like(input: Tensor, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
    def func(input: Tensor, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        tf_type = dtype_pytorch2keras(dtype)
        return tf.zeros_like(input, dtype=tf_type)
    return func


@converter(torch.Tensor.new_zeros, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_new_zeros(self, size, *args, **kwargs):
    def func(self, size, *args, **kwargs):
        return tf.zeros(shape=size, dtype=self.dtype)
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


@converter(torch.full, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_full(size: _size, fill_value: Number, *, names=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False):
    def func(size, fill_value, *, names=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        dtype = dtype_pytorch2keras(dtype)
        res = tf.fill(size, fill_value)
        res = tf.cast(res, dtype)
        return res
    return func


@converter(torch.full_like, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
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


@converter(F._canonical_mask, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_canonical_mask(mask: Optional[Tensor], mask_name: str, other_type: Optional, other_name: str, target_type, check_other: bool = True):
    if mask is not None:
        _mask_is_float = torch.is_floating_point(mask)
        if not _mask_is_float:
            raise Exception('Not supported yet')

    def func(mask, mask_name, other_type, other_name, target_type, check_other=True):
        return mask
    return func


@converter(torch.complex, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_complex(real: Tensor, imag: Tensor, *, out: Optional[Tensor]=None):
    def func(real, imag, *, out=None):
        return tf.complex(real, imag)
    return func
