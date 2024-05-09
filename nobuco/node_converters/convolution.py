import numbers
from typing import Optional, Union, List, Tuple, Sequence, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
import keras

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

    if isinstance(stride, numbers.Number):
        stride = (stride,)

    if isinstance(padding, numbers.Number):
        padding = (padding,)

    _, in_filters, _ = input.shape
    out_filters, _, kw = weight.shape

    use_depthwise = (groups == in_filters) and all([stride[0] == x for x in stride])

    weights = weight.cpu().detach().numpy()

    if use_depthwise:
        depth_multiplier = out_filters // groups
        weights = np.transpose(weights, (2, 0, 1))
        weights = np.reshape(weights, (kw, groups, depth_multiplier))
    else:
        weights = np.transpose(weights, (2, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    pad_str = 'valid'
    pad_layer = None

    if padding != 'valid':
        if padding == 'same':
            pad_str = 'same'
        elif padding != (0,):
            pad_layer = keras.layers.ZeroPadding1D(padding[0])

    if use_depthwise:
        conv = keras.layers.DepthwiseConv1D(
            kernel_size=kw,
            strides=stride,
            padding=pad_str,
            dilation_rate=dilation,
            groups=groups,
            depth_multiplier=out_filters // groups,
            use_bias=use_bias,
            weights=params
        )
    else:
        conv = keras.layers.Conv1D(
            filters=out_filters,
            kernel_size=kw,
            strides=stride,
            padding=pad_str,
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
def converter_conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[_int, _size] = 1,
                     padding: str = "valid", dilation: Union[_int, _size] = 1, groups: _int = 1):
    if isinstance(stride, numbers.Number):
        stride = (stride,)

    if isinstance(padding, numbers.Number):
        padding = (padding,)

    _, in_filters, _ = input.shape
    out_filters, _, kw = weight.shape

    use_depthwise = (groups == in_filters) and all([stride[0] == x for x in stride])

    weights = weight.cpu().detach().numpy()

    if use_depthwise:
        depth_multiplier = out_filters // groups
        weights = np.transpose(weights, (2, 0, 1))
        weights = np.reshape(weights, (kw, groups, depth_multiplier))
    else:
        weights = np.transpose(weights, (2, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    pad_str = 'valid'
    pad_layer = None

    if padding != 'valid':
        if padding == 'same':
            pad_str = 'same'
        elif padding != (0,):
            pad_layer = keras.layers.ZeroPadding1D(padding[0])

    if use_depthwise:
        conv = keras.layers.DepthwiseConv1D(
            kernel_size=kw,
            strides=stride,
            padding=pad_str,
            dilation_rate=dilation,
            groups=groups,
            depth_multiplier=out_filters // groups,
            use_bias=use_bias,
            weights=params
        )
    else:
        conv = keras.layers.Conv1D(
            filters=out_filters,
            kernel_size=kw,
            strides=stride,
            padding=pad_str,
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


@converter(nn.ConvTranspose1d)
def converter_ConvTranspose1d(self, input: Tensor, output_size: Optional[List[int]] = None):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation
    output_padding = self.output_padding

    if isinstance(padding, numbers.Number):
        padding = (padding,)

    assert output_padding == (0,), 'Output padding is not supported yet'

    in_filters, depth_multiplier, kw = weight.shape
    out_filters = groups * depth_multiplier

    weights = weight.cpu().detach().numpy()
    weights = weights.transpose((2, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if groups == 1:
        conv = keras.layers.Conv1DTranspose(out_filters,
                                            kernel_size=kw,
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )
    else:
        weights = params[0]

        weights_full = np.zeros(shape=(kw, out_filters, in_filters))
        for i in range(in_filters):
            chunk = i // (in_filters // groups)
            for d in range(depth_multiplier):
                weights_full[..., chunk*depth_multiplier + d, i] = weights[..., d, i]
        params[0] = weights_full

        conv = keras.layers.Conv1DTranspose(out_filters,
                                            kernel_size=kw,
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )

    def func(input, output_size: Optional[List[int]] = None):
        assert output_size is None

        x = conv(input)

        if padding[0] != 0:
            x = x[:, padding[0]:-padding[0], :]

        return x

    return func


@converter(F.conv_transpose1d)
def converter_conv_transpose1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1):
    if isinstance(padding, numbers.Number):
        padding = (padding,)

    assert output_padding == (0,), 'Output padding is not supported yet'

    in_filters, depth_multiplier, kw = weight.shape
    out_filters = groups * depth_multiplier

    weights = weight.cpu().detach().numpy()
    weights = weights.transpose((2, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if groups == 1:
        conv = keras.layers.Conv1DTranspose(out_filters,
                                            kernel_size=kw,
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )
    else:
        weights = params[0]

        weights_full = np.zeros(shape=(kw, out_filters, in_filters))
        for i in range(in_filters):
            chunk = i // (in_filters // groups)
            for d in range(depth_multiplier):
                weights_full[..., chunk * depth_multiplier + d, i] = weights[..., d, i]
        params[0] = weights_full

        conv = keras.layers.Conv1DTranspose(out_filters,
                                            kernel_size=kw,
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )

    def func(input, *args, **kwargs):
        x = conv(input)

        if padding[0] != 0:
            x = x[:, padding[0]:-padding[0], :]

        return x

    return func


@converter(nn.Conv2d)
def converter_Conv2d(self, input: Tensor):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation

    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)

    if isinstance(stride, numbers.Number):
        stride = (stride, stride)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    _, in_filters, _, _ = input.shape
    out_filters, _, kh, kw = weight.shape

    use_depthwise = (groups == in_filters) and all([stride[0] == x for x in stride])

    weights = weight.cpu().detach().numpy()

    if use_depthwise:
        depth_multiplier = out_filters // groups
        weights = np.transpose(weights, (2, 3, 0, 1))
        weights = np.reshape(weights, (kh, kw, groups, depth_multiplier))
    else:
        weights = np.transpose(weights, (2, 3, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    pad_str = 'valid'
    pad_layer = None

    if padding != 'valid':
        if padding == 'same':
            pad_str = 'same'
        elif padding != (0, 0):
            pad_layer = keras.layers.ZeroPadding2D(padding)

    if use_depthwise:
        conv = keras.layers.DepthwiseConv2D(
            kernel_size=(kh, kw),
            strides=stride,
            padding=pad_str,
            dilation_rate=dilation,
            groups=groups,
            depth_multiplier=out_filters // groups,
            use_bias=use_bias,
            weights=params
        )
    else:
        conv = keras.layers.Conv2D(
            filters=out_filters,
            kernel_size=(kh, kw),
            strides=stride,
            padding=pad_str,
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
    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)

    if isinstance(stride, numbers.Number):
        stride = (stride, stride)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    _, in_filters, _, _ = input.shape
    out_filters, _, kh, kw = weight.shape

    use_depthwise = (groups == in_filters) and all([stride[0] == x for x in stride])

    weights = weight.cpu().detach().numpy()

    if use_depthwise:
        depth_multiplier = out_filters // groups
        weights = np.transpose(weights, (2, 3, 0, 1))
        weights = np.reshape(weights, (kh, kw, groups, depth_multiplier))
    else:
        weights = np.transpose(weights, (2, 3, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    pad_str = 'valid'
    pad_layer = None

    if padding != 'valid':
        if padding == 'same':
            pad_str = 'same'
        elif padding != (0, 0):
            pad_layer = keras.layers.ZeroPadding2D(padding)

    if use_depthwise:
        conv = keras.layers.DepthwiseConv2D(
            kernel_size=(kh, kw),
            strides=stride,
            padding=pad_str,
            dilation_rate=dilation,
            groups=groups,
            depth_multiplier=out_filters // groups,
            use_bias=use_bias,
            weights=params
        )
    else:
        conv = keras.layers.Conv2D(
            filters=out_filters,
            kernel_size=(kh, kw),
            strides=stride,
            padding=pad_str,
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

    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if isinstance(output_padding, numbers.Number):
        output_padding = (output_padding, output_padding)

    assert output_padding == (0, 0), 'Output padding is not supported yet'

    in_filters, depth_multiplier, kh, kw = weight.shape
    out_filters = groups * depth_multiplier

    weights = weight.cpu().detach().numpy()
    weights = weights.transpose((2, 3, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

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
    else:
        weights = params[0]

        weights_full = np.zeros(shape=(kh, kw, out_filters, in_filters))
        for i in range(in_filters):
            chunk = i // (in_filters // groups)
            for d in range(depth_multiplier):
                weights_full[..., chunk*depth_multiplier + d, i] = weights[..., d, i]
        params[0] = weights_full

        conv = keras.layers.Conv2DTranspose(out_filters,
                                            kernel_size=(kh, kw),
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )

    def func(input: Tensor, output_size: Optional[List[int]] = None):
        assert output_size is None

        x = conv(input)

        if padding != (0, 0):
            if padding[0] == 0:
                x = x[:, :, padding[1]:-padding[1], :]
            elif padding[1] == 0:
                x = x[:, padding[0]:-padding[0], :]
            else:
                x = x[:, padding[0]:-padding[0], padding[1]:-padding[1], :]

        return x

    return func


# `groups` parameter in ConvTranspose2d is broken, see: https://github.com/tensorflow/tensorflow/issues/45216
@converter(F.conv_transpose2d)
def converter_conv_transpose2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                               stride: Union[_int, _size] = 1, padding: Union[_int, _size] = 0,
                               output_padding: Union[_int, _size] = 0,
                               groups: _int = 1, dilation: Union[_int, _size] = 1):
    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if isinstance(output_padding, numbers.Number):
        output_padding = (output_padding, output_padding)

    assert output_padding == (0, 0), 'Output padding is not supported yet'

    in_filters, depth_multiplier, kh, kw = weight.shape
    out_filters = groups * depth_multiplier

    weights = weight.cpu().detach().numpy()
    weights = weights.transpose((2, 3, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

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
    else:
        weights = params[0]

        weights_full = np.zeros(shape=(kh, kw, out_filters, in_filters))
        for i in range(in_filters):
            chunk = i // (in_filters // groups)
            for d in range(depth_multiplier):
                weights_full[..., chunk * depth_multiplier + d, i] = weights[..., d, i]
        params[0] = weights_full

        conv = keras.layers.Conv2DTranspose(out_filters,
                                            kernel_size=(kh, kw),
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            groups=1,
                                            use_bias=use_bias,
                                            weights=params
                                            )

    def func(input, *args, **kwargs):
        x = conv(input)

        if padding != (0, 0):
            if padding[0] == 0:
                x = x[:, :, padding[1]:-padding[1], :]
            elif padding[1] == 0:
                x = x[:, padding[0]:-padding[0], :]
            else:
                x = x[:, padding[0]:-padding[0], padding[1]:-padding[1], :]

        return x

    return func
