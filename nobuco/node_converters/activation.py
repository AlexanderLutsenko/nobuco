from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F
from torch import nn

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import dim_pytorch2keras


def hard_sigmoid_pytorch_compatible(x):
  x = tf.clip_by_value(x/6 + 1/2, clip_value_min=0, clip_value_max=1)
  return x


def hard_swish_pytorch_compatible(x):
  x = x * hard_sigmoid_pytorch_compatible(x)
  return x


def hard_tanh_pytorch_compatible(x, min_val, max_val):
  x = tf.clip_by_value(x, clip_value_min=min_val, clip_value_max=max_val)
  return x


@converter(torch.sigmoid, torch.Tensor.sigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_sigmoid(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return keras.layers.Activation(keras.activations.sigmoid)(input)
    return func


@converter(F.logsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_logsigmoid(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.math.log_sigmoid(input)
    return func


@converter(torch.tanh, torch.Tensor.tanh, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_tanh(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input: Tensor, *, out=None):
        return keras.layers.Activation(keras.activations.tanh)(input)
    return func


@converter(nn.ReLU, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_ReLU(self, input: Tensor):
    return keras.layers.ReLU()


@converter(F.relu, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_relu(input: Tensor, inplace: bool = False):
    def func(input, inplace=False):
        return tf.nn.relu(input)
    return func


@converter(torch.relu_, torch.Tensor.relu_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_relu_(input: Tensor):
    def func(input):
        return tf.nn.relu(input)
    return func


@converter(nn.LeakyReLU, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_LeakyRelu(self, input: Tensor):
    return keras.layers.LeakyReLU(alpha=self.negative_slope)


@converter(F.leaky_relu, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False):
    def func(input, negative_slope=0.01, inplace=False):
        return keras.layers.LeakyReLU(alpha=negative_slope)(input)
    return func


def prelu_prepare_params(weight_np, input_dim, input_channels, input_channel_order):
    if weight_np.size == 1:
        w_shape = [1] * input_dim
        shared_axes = [i for i in range(1, input_dim)]
        weights = weight_np.reshape(w_shape)
    elif weight_np.size == input_channels:
        if input_channel_order == ChannelOrder.TENSORFLOW:
            channel_dim = dim_pytorch2keras(1, input_dim)
        else:
            channel_dim = 1
        w_shape = [1] * input_dim
        w_shape[channel_dim] = weight_np.size
        shared_axes = [i for i in range(1, input_dim) if i != channel_dim]
        weights = weight_np.reshape(w_shape)
    else:
        raise Exception('Unsupported weight shape')
    return shared_axes, weights


@converter(nn.PReLU, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_PReLU(self, input: Tensor):
    input_dim = input.dim()
    input_channels = input.shape[1] if input_dim > 1 else 1
    weight_np = self.weight.cpu().detach().numpy()

    def func(input):
        shared_axes, weights = prelu_prepare_params(weight_np, input_dim, input_channels, get_channel_order(input))
        return keras.layers.PReLU(shared_axes=shared_axes, weights=weights)(input)
    return func


@converter(F.prelu, torch.prelu, torch.Tensor.prelu, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_prelu(input: Tensor, weight: Tensor):
    input_dim = input.dim()
    input_channels = input.shape[1] if input_dim > 1 else 1
    weight_np = weight.cpu().detach().numpy()

    def func(input, weight):
        shared_axes, weights = prelu_prepare_params(weight_np, input_dim, input_channels, get_channel_order(input))
        return keras.layers.PReLU(shared_axes=shared_axes, weights=weights)(input)
    return func


@converter(F.hardsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_hardsigmoid(input: Tensor, inplace: bool = False):
    def func(input, inplace=False):
        return hard_sigmoid_pytorch_compatible(input)
    return func


@converter(F.hardtanh, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False):
    def func(input, min_val=-1.0, max_val=1.0, inplace=False):
        return hard_tanh_pytorch_compatible(input, min_val, max_val)
    return func


@converter(F.hardswish, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_hardswish(input: Tensor, inplace: bool = False):
    def func(input, inplace=False):
        return hard_swish_pytorch_compatible(input)
    return func


@converter(torch.softmax, torch.Tensor.softmax, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_softmax(input: Tensor, dim: Union[str, None], *, dtype: Optional[_dtype]=None):
    num_dims = input.dim()

    def func(input: Tensor, dim, *, dtype=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.nn.softmax(input, axis=dim)
    return func


@converter(torch.nn.functional.softmax, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype = None):
    num_dims = input.dim()

    def func(input, dim = None, _stacklevel = 3, dtype = None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.nn.softmax(input, axis=dim)
    return func


@converter(F.log_softmax, torch.log_softmax, torch.Tensor.log_softmax, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_log_softmax(input: Tensor, dim, *, dtype: Optional[_dtype]=None):
    num_dims = input.dim()

    def func(input, dim, *, dtype=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.nn.log_softmax(input, axis=dim)
    return func


@converter(torch.clip, torch.Tensor.clip, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_clip(input: Tensor, min: Optional[Tensor]=None, max: Optional[Tensor]=None, *, out: Optional[Tensor]=None):
    def func(input, min=None, max=None, *, out=None):
        if min is None:
            return tf.minimum(input, max)
        elif max is None:
            return tf.maximum(input, min)
        else:
            return tf.clip_by_value(input, min, max)
    return func


@converter(F.silu, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_silu(input: Tensor, inplace=False):
    def func(input: Tensor, inplace=False):
        return tf.nn.silu(input, beta=1.0)
    return func

      
@converter(F.gelu, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_gelu(input: Tensor, approximate='none'):
    def func(input: Tensor, approximate='none'):
        if approximate.lower() == 'none':
            # Gaussian Error Linear Units (GELUs)
            # https://arxiv.org/abs/1606.08415
            return input * 0.5 * (1 + tf.math.erf(input / tf.math.sqrt(2.)))
        else:
            return tf.nn.gelu(input, approximate=approximate)
    return func
