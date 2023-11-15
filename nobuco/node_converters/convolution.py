import numbers
from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
from nobuco.converters.node_converter import converter


@converter(nn.Conv1d)
def converter_Conv1d(self, input: Tensor):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation

    out_filters, in_filters, kw = weight.shape
    weights = weight.cpu().detach().numpy()
    weights = tf.transpose(weights, (2, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if isinstance(padding, numbers.Number):
        padding = (padding,)
    if padding != (0,) and padding != 'valid':
        pad_layer = keras.layers.ZeroPadding1D(padding[0])
    else:
        pad_layer = None

    conv = keras.layers.Conv1D(filters=out_filters,
                               kernel_size=kw,
                               strides=stride,
                               padding='valid',
                               dilation_rate=dilation,
                               groups=groups,
                               use_bias=use_bias,
                               weights=params
                               )

    def func(input):
        if pad_layer is not None:
            input = pad_layer(input)
        output = conv(input)
        return output
    return func


@converter(F.conv1d)
def converter_conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[_int, _size]=1, padding: str="valid", dilation: Union[_int, _size]=1, groups: _int=1):
    out_filters, in_filters, kw = weight.shape
    weights = weight.cpu().detach().numpy()
    weights = tf.transpose(weights, (2, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if isinstance(padding, numbers.Number):
        padding = (padding,)
    if padding != (0,) and padding != 'valid':
        pad_layer = keras.layers.ZeroPadding1D(padding[0])
    else:
        pad_layer = None

    conv = keras.layers.Conv1D(filters=out_filters,
                               kernel_size=kw,
                               strides=stride,
                               padding='valid',
                               dilation_rate=dilation,
                               groups=groups,
                               use_bias=use_bias,
                               weights=params
                               )

    def func(input, *args, **kwargs):
        if pad_layer is not None:
            input = pad_layer(input)
        output = conv(input)
        return output
    return func


@converter(nn.Conv2d)
def converter_Conv2d(self, input: Tensor):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation

    out_filters, in_filters, kh, kw = weight.shape

    weights = weight.cpu().detach().numpy()
    weights = tf.transpose(weights, (2, 3, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if padding != 0 and padding != (0, 0) and padding != 'valid':
        pad_layer = keras.layers.ZeroPadding2D(padding)
    else:
        pad_layer = None

    conv = keras.layers.Conv2D(filters=out_filters,
                               kernel_size=(kh, kw),
                               strides=stride,
                               padding='valid',
                               dilation_rate=dilation,
                               groups=groups,
                               use_bias=use_bias,
                               weights=params
                               )

    def func(input):
        if pad_layer is not None:
            input = pad_layer(input)
        output = conv(input)
        return output
    return func


@converter(F.conv2d)
def converter_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[_int, _size] = 1,
                    padding: str = "valid", dilation: Union[_int, _size] = 1, groups: _int = 1):

    out_filters, in_filters, kh, kw = weight.shape

    weights = weight.cpu().detach().numpy()
    weights = tf.transpose(weights, (2, 3, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if padding != 0 and padding != (0, 0) and padding != 'valid':
        pad_layer = keras.layers.ZeroPadding2D(padding)
    else:
        pad_layer = None

    conv = keras.layers.Conv2D(filters=out_filters,
                               kernel_size=(kh, kw),
                               strides=stride,
                               padding='valid',
                               dilation_rate=dilation,
                               groups=groups,
                               use_bias=use_bias,
                               weights=params
                               )

    def func(input, *args, **kwargs):
        if pad_layer is not None:
            input = pad_layer(input)
        output = conv(input)
        return output
    return func

# `groups` parameter in ConvTranspose2d is broken, see: https://github.com/tensorflow/tensorflow/issues/45216
@converter(nn.ConvTranspose2d)
def converter_ConvTranspose2d(self, input: Tensor, output_size: Optional[List[int]] = None):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation
    output_padding = self.output_padding

    in_filters, out_filters, kh, kw = weight.shape
    weights = weight.cpu().detach().numpy()

    if groups == 1:
        weights = weights.transpose((2, 3, 1, 0))
    else:
        weights = weights.transpose((2, 3, 0, 1))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if isinstance(output_padding, numbers.Number):
        output_padding = (output_padding, output_padding)

    assert output_padding == (0, 0), 'Output padding is not supported yet'

    if groups == 1:
        conv = keras.layers.Conv2DTranspose(out_filters,
                                            kernel_size=(kh, kw),
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )
    elif groups == in_filters and out_filters == 1:
        weights = params[0]

        weights_full = np.zeros(shape=(*weights.shape[:-1], groups))
        for i in range(groups):
            weights_full[..., i, i] = weights[..., i, 0]
        params[0] = weights_full

        conv = keras.layers.Conv2DTranspose(out_filters*groups,
                                            kernel_size=(kh, kw),
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )
    else:
        raise Exception('Unsupprorted # groups:', groups)

    def func(input: Tensor, output_size: Optional[List[int]] = None):
        assert output_size is None

        x = conv(input)

        if padding != (0, 0):
            x = x[:, padding[0]:-padding[0], padding[1]:-padding[1], :]

        return x
    return func


# `groups` parameter in ConvTranspose2d is broken, see: https://github.com/tensorflow/tensorflow/issues/45216
@converter(F.conv_transpose2d)
def converter_conv_transpose2d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None,
                     stride: Union[_int, _size]=1, padding: Union[_int, _size]=0, output_padding: Union[_int, _size]=0,
                     groups: _int=1, dilation: Union[_int, _size]=1):
    in_filters, out_filters, kh, kw = weight.shape
    weights = weight.cpu().detach().numpy()

    if groups == 1:
        weights = weights.transpose((2, 3, 1, 0))
    else:
        weights = weights.transpose((2, 3, 0, 1))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if isinstance(output_padding, numbers.Number):
        output_padding = (output_padding, output_padding)

    assert output_padding == (0, 0), 'Output padding is not supported yet'

    if groups == 1:
        conv = keras.layers.Conv2DTranspose(out_filters,
                                            kernel_size=(kh, kw),
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )
    elif groups == in_filters and out_filters == 1:
        weights = params[0]

        weights_full = np.zeros(shape=(*weights.shape[:-1], groups))
        for i in range(groups):
            weights_full[..., i, i] = weights[..., i, 0]
        params[0] = weights_full

        conv = keras.layers.Conv2DTranspose(out_filters * groups,
                                            kernel_size=(kh, kw),
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )
    else:
        raise Exception('Unsupprorted # groups:', groups)

    def func(input, *args, **kwargs):
        x = conv(input)

        if padding != (0, 0):
            x = x[:, padding[0]:-padding[0], padding[1]:-padding[1], :]

        return x

    return func
