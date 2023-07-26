import numbers
from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F

from nobuco.converters.node_converter import converter


@converter(torch.max_pool2d)
def converter_max_pool_2d(input: Tensor, kernel_size: Union[_int, _size], stride: Union[_int, _size]=(), padding: Union[_int, _size]=0, dilation: Union[_int, _size]=1, ceil_mode: _bool=False):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if padding != (0, 0):
        pad_layer = keras.layers.ZeroPadding2D(padding)
    else:
        pad_layer = None

    def func(input, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
        if pad_layer is not None:
            input = pad_layer(input)
        return keras.layers.MaxPool2D(pool_size=kernel_size, strides=stride, padding='valid')(input)
    return func


@converter(F.avg_pool2d)
def converter_avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if padding != (0, 0):
        pad_layer = keras.layers.ZeroPadding2D(padding)
    else:
        pad_layer = None

    def func(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        if pad_layer is not None:
            input = pad_layer(input)
        return keras.layers.AvgPool2D(pool_size=kernel_size, strides=stride)(input)
    return func


@converter(F.adaptive_avg_pool2d)
def converter_adaptiveAvgPool2D(input: Tensor, output_size):
    if output_size == (1, 1) or output_size == 1:
        def func(input, output_size):
            return keras.layers.GlobalAvgPool2D(keepdims=True)(input)
    else:
        def func(input, output_size):
            import tensorflow_addons as tfa
            return tfa.layers.AdaptiveAveragePooling2D(output_size=output_size)(input)
    return func


@converter(F.pixel_shuffle)
def converter_pixel_shuffle(input: Tensor, upscale_factor: _int):
    def func(input, upscale_factor):
        x = input
        x = tf.concat([x[..., i::upscale_factor**2] for i in range(upscale_factor**2)], axis=-1)
        x = tf.nn.depth_to_space(x, upscale_factor)
        return x
    return func
