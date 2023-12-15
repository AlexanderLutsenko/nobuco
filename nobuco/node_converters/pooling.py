import numbers
from typing import Optional, Union, List, Tuple, Sequence, Any

from nobuco.commons import ChannelOrderingStrategy
from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn
import torch.nn.functional as F

from nobuco.converters.node_converter import converter


@converter(F.max_pool2d, torch.max_pool2d, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_max_pool_2d(input: Tensor, kernel_size: Union[_int, _size], stride: Union[_int, _size]=(), padding: Union[_int, _size]=0, dilation: Union[_int, _size]=1, ceil_mode: _bool=False,
                          *args, **kwargs):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if padding != (0, 0):
        paddings = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
        pad_func = lambda x: tf.pad(x, paddings, constant_values=float('-inf'), )
    else:
        pad_func = None

    if not stride:
        stride = None

    layer = keras.layers.MaxPool2D(pool_size=kernel_size, strides=stride)

    def func(input, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False, *args, **kwargs):
        if pad_func is not None:
            input = pad_func(input)
        return layer(input)
    return func


@converter(F.avg_pool2d, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if not stride:
        stride = None

    if padding != (0, 0):
        pad_layer = keras.layers.ZeroPadding2D(padding)
    else:
        pad_layer = None

    layer = keras.layers.AvgPool2D(pool_size=kernel_size, strides=stride)

    def func(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        if pad_layer is not None:
            input = pad_layer(input)
        return layer(input)
    return func


@converter(F.adaptive_avg_pool2d, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_adaptiveAvgPool2D(input: Tensor, output_size):
    if output_size == (1, 1) or output_size == 1:
        def func(input, output_size):
            return keras.layers.GlobalAvgPool2D(keepdims=True)(input)
    else:
        def func(input, output_size):
            import tensorflow_addons as tfa
            return tfa.layers.AdaptiveAveragePooling2D(output_size=output_size)(input)
    return func


def channel_interleave2d(x, block_size: int, reverse: bool):
    b, h, w, c = x.shape
    n_blocks = block_size ** 2

    if reverse:
        x = tf.reshape(x, (b, h, w, n_blocks, c // n_blocks))
    else:
        x = tf.reshape(x, (b, h, w, c // n_blocks, n_blocks))

    x = tf.transpose(x, (0, 1, 2, 4, 3))
    x = tf.reshape(x, (b, h, w, c))
    return x


@converter(F.pixel_shuffle, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_pixel_shuffle(input: Tensor, upscale_factor: _int):
    def func(input, upscale_factor):
        x = channel_interleave2d(input, upscale_factor, reverse=False)
        x = tf.nn.depth_to_space(x, upscale_factor)
        return x
    return func


@converter(F.pixel_unshuffle, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_pixel_unshuffle(input: Tensor, downscale_factor: _int):
    def func(input, downscale_factor):
        x = tf.nn.space_to_depth(input, downscale_factor)
        x = channel_interleave2d(x, downscale_factor, reverse=True)
        return x
    return func
