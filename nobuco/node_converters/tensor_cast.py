from torch import Tensor

import tensorflow as tf
import torch

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter
from nobuco.converters.type_cast import dtype_pytorch2keras


@converter(torch.Tensor.contiguous, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_contiguous(self, memory_format=None):
    def func(self, memory_format=None):
        return self
    return func


@converter(torch.Tensor.detach, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_detach(self):
    def func(self):
        return tf.stop_gradient(self)
    return func


@converter(torch.Tensor.cpu, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_cpu(self, memory_format=None):
    def func(self, memory_format=None):
        return self
    return func


def type_func(self, dtype=None, non_blocking=False, **kwargs):
    tf_type = dtype_pytorch2keras(dtype)
    return tf.cast(self, tf_type)


@converter(torch.Tensor.type, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_type(self, dtype=None, non_blocking=False, **kwargs):
    return type_func


@converter(torch.Tensor.bool, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_bool(self, memory_format=None):
    return lambda x: type_func(x, dtype=torch.bool)


@converter(torch.Tensor.int, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_int(self, memory_format=None):
    return lambda x: type_func(x, dtype=torch.int32)


@converter(torch.Tensor.long, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_long(self, memory_format=None):
    return lambda x: type_func(x, dtype=torch.int64)


@converter(torch.Tensor.half, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_half(self, memory_format=None):
    return lambda x: type_func(x, dtype=torch.float16)


@converter(torch.Tensor.float, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_float(self, memory_format=None):
    return lambda x: type_func(x, dtype=torch.float32)


@converter(torch.Tensor.double, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_double(self, memory_format=None):
    return lambda x: type_func(x, dtype=torch.float64)


@converter(torch.Tensor.to, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_to(self, device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format):
    def func(self, device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format):
        if dtype is not None:
            return type_func(self, dtype=dtype, non_blocking=non_blocking)
        elif isinstance(device, torch._C.dtype):
            return type_func(self, dtype=device, non_blocking=non_blocking)
        elif device is not None:
            return self
        else:
            raise Exception('Unsupported params')
    return func


@converter(torch.Tensor.type_as, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_type_as(self, tensor):
    def func(self, tensor):
        return tf.cast(self, tensor.dtype)
    return func


@converter(torch.view_as_complex, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_view_as_complex(input: Tensor):
    def func(input):
        return tf.complex(input[..., 0], input[..., 1])
    return func


@converter(torch.view_as_real, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_view_as_real(input: Tensor):
    def func(input):
        return tf.stack([tf.math.real(input), tf.math.imag(input)], axis=-1)
    return func
