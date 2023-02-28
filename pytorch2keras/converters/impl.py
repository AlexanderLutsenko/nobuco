import numbers
from typing import Optional, Union, List, Tuple, Sequence, Any

from tensorflow.python.ops.image_ops_impl import ResizeMethod
from torch import Tensor
from torch.types import _int, _bool, Number, _dtype, _size

import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from pytorch2keras.commons import ChannelOrder, ChannelOrderingStrategy
from pytorch2keras.funcs import force_tensorflow_order, force_pytorch_order
from pytorch2keras.converters.channel_ordering import set_channel_order, get_channel_order
from pytorch2keras.converters.node_converter import converter
from pytorch2keras.converters.tensor import dims_pytorch2keras, perm_keras2pytorch, \
    _dim_make_positive, dim_pytorch2keras, _permute, _flatten, perm_pytorch2keras, perm_compose, \
    is_identity_perm, permute_pytorch2keras, _ensure_iterable, perm_identity
from pytorch2keras.converters.ops import hard_sigmoid_pytorch_compatible, hard_swish_pytorch_compatible, \
    hard_tanh_pytorch_compatible


# @converter(torch.zeros, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
# def zeros(*size: _int, names: Optional[Sequence[Union[str, None]]]=None, dtype: _dtype=None, layout: Optional[_layout]=strided, device: Union[_device, str, None]=None, pin_memory: _bool=False, requires_grad: _bool=False):
#     def func(*args, **kwargs):
#         return tf.zeros(shape=size)
#     return func


@converter(nn.GRU, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def gru(self: nn.GRU, input, hx=None):
    assert not self.bidirectional

    def swap_order(param):
        assert param.shape[-1] % 3 == 0
        p1, p2, p3 = np.split(param, 3, axis=-1)
        return np.concatenate([p2, p1, p3], axis=-1)

    grus = []
    for i in range(self.num_layers):
        weight_ih = self.__getattr__(f'weight_ih_l{i}').detach().numpy().transpose((1, 0))
        weight_hh = self.__getattr__(f'weight_hh_l{i}').detach().numpy().transpose((1, 0))
        bias_ih = self.__getattr__(f'bias_ih_l{i}').detach().numpy()
        bias_hh = self.__getattr__(f'bias_hh_l{i}').detach().numpy()

        weight_ih = swap_order(weight_ih)
        weight_hh = swap_order(weight_hh)
        bias_ih = swap_order(bias_ih)
        bias_hh = swap_order(bias_hh)

        gru = keras.layers.GRU(
            units=self.hidden_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=self.bias,
            dropout=self.dropout,
            return_sequences=True,
            return_state=True,
            time_major=not self.batch_first,
            reset_after=True,
            unroll=True,
            weights=[weight_ih, weight_hh, tf.stack([bias_ih, bias_hh], axis=0)],
        )
        grus.append(gru)

    def func(input, hx=None):
        x = input
        hxs = []
        for i in range(len(grus)):
            initial_state = hx[i] if hx is not None else None
            x, hxo = grus[i](x, initial_state=initial_state)
            hxs.append(hxo)
        hxs = tf.stack(hxs, axis=0)
        return x, hxs
    return func


@converter(nn.LSTM, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def lstm(self: nn.LSTM, input, hx=None):
    assert not self.bidirectional

    lstms = []
    for i in range(self.num_layers):
        weight_ih = self.__getattr__(f'weight_ih_l{i}').detach().numpy().transpose((1, 0))
        weight_hh = self.__getattr__(f'weight_hh_l{i}').detach().numpy().transpose((1, 0))
        bias_ih = self.__getattr__(f'bias_ih_l{i}').detach().numpy()
        bias_hh = self.__getattr__(f'bias_hh_l{i}').detach().numpy()

        lstm = keras.layers.LSTM(
            units=self.hidden_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=self.bias,
            dropout=self.dropout,
            return_sequences=True,
            return_state=True,
            time_major=not self.batch_first,
            unroll=True,
            weights=[weight_ih, weight_hh, bias_ih + bias_hh],
        )
        lstms.append(lstm)

    def func(input, hx=None):
        x = input
        hxs = []
        cxs = []
        for i in range(len(lstms)):
            initial_state = (hx[0][i], hx[1][i]) if hx is not None else None
            x, hxo, cxo = lstms[i](x, initial_state=initial_state)
            hxs.append(hxo)
            cxs.append(cxo)
        hxs = tf.stack(hxs, axis=0)
        cxs = tf.stack(cxs, axis=0)
        return x, (hxs, cxs)
    return func


@converter(torch.cat, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim, *, out: Optional[Tensor]=None):
    num_dims = tensors[0].dim()
    dim_keras = dim_pytorch2keras(dim, num_dims)

    def func(tensors, dim, *, out=None):
        if get_channel_order(tensors[0]) == ChannelOrder.TENSORFLOW:
            return tf.concat(tensors, axis=dim_keras)
        else:
            return tf.concat(tensors, axis=dim)
    return func


@converter(torch.stack, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def stack(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: _int=0, *, out: Optional[Tensor]=None):
    def func(tensors, dim=0, *, out=None):
        return tf.stack(tensors, axis=dim)
    return func


@converter(torch.Tensor.repeat, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def repeat(self, *sizes):
    def func(self, *sizes):
        if get_channel_order(self) == ChannelOrder.TENSORFLOW:
            sizes = permute_pytorch2keras(sizes)
        return tf.tile(self, sizes)
    return func


@converter(torch.roll, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def roll(input: Tensor, shifts: Union[_int, _size], dims: Union[_int, _size]=()):
    assert isinstance(shifts, _int) and isinstance(dims, _int)
    n_dims = input.dim()

    def func(input, shifts, dims=()):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dims = dim_pytorch2keras(dims, n_dims)
        return tf.roll(input, shift=shifts, axis=dims)
    return func


# TODO: add support for 'negative' paddings
@converter(F.pad, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def pad(input: Tensor, pad: List[int], mode: str = "constant", value: float = 0.0):
    n_dims = input.dim()
    pad_dims = len(pad) // 2
    assert len(pad) % 2 == 0
    assert pad_dims <= n_dims

    pad_full = []
    for i in range(pad_dims):
        pad_full.append(pad[2*i:2*i + 2])
    for i in range(n_dims - pad_dims):
        pad_full.append([0, 0])
    pad_full = list(reversed(pad_full))

    # pad_full_pos = [(max(s, 0), max(e, 0)) for s, e in pad_full]
    # pad_full_neg = [(max(-s, 0), max(-e, 0)) for s, e in pad_full]

    def func(input, pad, mode="constant", value=0.0):
        pad = pad_full
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            pad = permute_pytorch2keras(pad)
        x = tf.pad(input, pad)
        return x
    return func


@converter(nn.Flatten)
def flatten(self, input: Tensor):
    # FIXME start_dim, end_dim
    return keras.layers.Flatten()


@converter(torch.Tensor.flatten, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def flatten(self, start_dim=0, end_dim=-1):
    # FIXME start_dim, end_dim
    def func(self, start_dim=0, end_dim=-1):
        return tf.reshape(self, (*self.shape[:start_dim], -1))
    return func


@converter(torch.mean, torch.Tensor.mean, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def mean(input: Tensor, dim=None, keepdim: _bool = False, *, dtype: Optional[_dtype] = None, out: Optional[Tensor] = None):
    num_dims = len(input.shape)

    def func(input, dim=None, keepdim=False, *, dtype=None, out=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW and dim is not None:
            dim = dims_pytorch2keras(dim, num_dims=num_dims)
        return tf.reduce_mean(input, axis=dim, keepdims=keepdim)
    return func


@converter(torch.abs, torch.Tensor.abs, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def abs(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.abs(input)
    return func


# def conv2D(module: nn.Conv2d, input_tensors, args, kwargs):
#     out_filters, in_filters, kh, kw = module.weight.shape
#     weights = module.weight.detach().numpy()
#     weights = weights.transpose((2, 3, 1, 0))
#     biases = module.bias.detach().numpy()
#
#     dilations = module.dilation
#     groups = module.groups
#     kernel_shape = module.kernel_size
#     padding = module.padding
#     strides = module.stride
#
#     pad_layer = None
#     ph, pw = padding
#     if ph != 0 or pw != 0:
#         if module.padding_mode == 'zeros':
#             pad_layer = layers.ZeroPadding2D((ph, pw))
#         else:
#             raise Exception('Unsupported padding mode')
#
#     conv = layers.Conv2D(out_filters, kernel_shape,
#                          strides=strides, padding='valid', dilation_rate=dilations, groups=groups,
#                          weights=[weights, biases]
#                          )
#     if pad_layer is None:
#         return conv
#     else:
#         return lambda x: conv(pad_layer(x))
# rule_dict[nn.Conv2d] = Pytorch2KerasLambdaConverter(conv2D)


@converter(F.conv1d)
def conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[_int, _size]=1, padding: str="valid", dilation: Union[_int, _size]=1, groups: _int=1):
    out_filters, in_filters, kw = weight.shape
    weights = weight.detach().numpy()
    weights = weights.transpose((2, 1, 0))

    if bias is not None:
        biases = bias.detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if padding != 0:
        pad_layer = keras.layers.ZeroPadding1D(padding)
    else:
        pad_layer = None

    conv = keras.layers.Conv1D(out_filters, kernel_size=kw,
                         strides=stride, padding='valid',
                         # dilation_rate=dilations, groups=groups,
                         use_bias=use_bias,
                         weights=params
                         )

    def func(input, *args, **kwargs):
        if pad_layer is not None:
            input = pad_layer(input)
        return conv(input)
    return func


@converter(F.conv2d)
def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[_int, _size] = 1,
                    padding: str = "valid", dilation: Union[_int, _size] = 1, groups: _int = 1):

    out_filters, in_filters, kh, kw = weight.shape

    weights = weight.detach().numpy()
    if groups == out_filters and groups != 1:
        weights = tf.transpose(weights, (2, 3, 0, 1))
    elif groups == 1:
        weights = tf.transpose(weights, (2, 3, 1, 0))
    else:
        weights = tf.transpose(weights, (2, 3, 1, 0))

    if bias is not None:
        biases = bias.detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if padding != 0 and padding != (0, 0) and padding != 'valid':
        pad_layer = keras.layers.ZeroPadding2D(padding)
    else:
        pad_layer = None

    if groups == out_filters and groups != 1:
        conv = keras.layers.DepthwiseConv2D(kernel_size=(kh, kw),
                                      strides=stride,
                                      padding='valid',
                                      dilation_rate=dilation,
                                      groups=groups,
                                      use_bias=use_bias,
                                      weights=params
                                      )
    elif groups == 1:
        conv = keras.layers.Conv2D(filters=out_filters,
                             kernel_size=(kh, kw),
                             strides=stride,
                             padding='valid',
                             dilation_rate=dilation,
                             groups=groups,
                             use_bias=use_bias,
                             weights=params
                             )
    else:
        def split_params(params, groups, axis):
            params_split = [np.split(p, groups, axis=axis) for p in params]
            return list(zip(*params_split))

        params_split = split_params(params, groups, axis=-1)

        def grouped_conv2d(inputs, filters, kernel_size, strides, groups, dilation=dilation):
            splits = tf.split(inputs, groups, axis=-1)
            convolved_splits = [
                keras.layers.Conv2D(filters // groups,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding='valid',
                                    dilation_rate=dilation,
                                    use_bias=use_bias,
                                    weights=params_split[i]
                                    )(split)
                for i, split in enumerate(splits)
            ]
            return tf.concat(convolved_splits, -1)

        conv = lambda x: grouped_conv2d(x, out_filters, kernel_size=(kh, kw), strides=stride, groups=groups, dilation=dilation)

    def func(input, *args, **kwargs):
        if pad_layer is not None:
            input = pad_layer(input)
        output = conv(input)
        return output
    return func


# FIXME: not thoroughly tested
@converter(F.conv_transpose2d)
def conv_transpose2d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None,
                     stride: Union[_int, _size]=1, padding: Union[_int, _size]=0, output_padding: Union[_int, _size]=0,
                     groups: _int=1, dilation: Union[_int, _size]=1):

    in_filters, out_filters, kh, kw = weight.shape
    weights = weight.detach().numpy()

    if groups == 1:
        weights = weights.transpose((2, 3, 1, 0))
    # elif groups == out_filters:
    elif groups == in_filters:
        weights = weights.transpose((2, 3, 0, 1))
    else:
        weights = weights.transpose((2, 3, 1, 0))

    if bias is not None:
        biases = bias.detach().numpy()
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

    if padding != (0, 0):
        pad = (dilation[0] * (kh - 1) - padding[0], dilation[1] * (kw - 1) - padding[1])
        pad_layer = keras.layers.ZeroPadding2D(pad)
        # print('!!!', (kh, kw), pad, output_padding)
    else:
        pad_layer = None
    pad_layer = None

    # print('???', output_padding)

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

        # def split_params(params, groups, axis):
        #     params_split = [np.split(p, groups, axis=axis) for p in params]
        #     return list(zip(*params_split))
        #
        # params_split = split_params(params, groups, axis=-2)
        #
        # def grouped_conv2d_transpose(inputs, filters, kernel_size, strides, groups, dilation=dilation):
        #     splits = tf.split(inputs, groups, axis=-1)
        #     convolved_splits = [
        #         keras.layers.Conv2DTranspose(filters // groups,
        #                                      kernel_size=kernel_size,
        #                                      strides=strides,
        #                                      padding='same',
        #                                      dilation_rate=dilation,
        #                                      use_bias=use_bias,
        #                                      weights=params_split[i]
        #                                      )(split)
        #         for i, split in enumerate(splits)
        #     ]
        #     return tf.concat(convolved_splits, -1)
        #
        # conv = lambda x: grouped_conv2d_transpose(x, out_filters*groups, kernel_size=(kh, kw), strides=stride, groups=groups, dilation=dilation)

    # FIXME!!!
    def func(input, *args, **kwargs):
        if pad_layer is not None:
            input = pad_layer(input)

        x = conv(input)

        if output_padding != (0, 0):
            x = x[:, output_padding[0]:, output_padding[1]:, :]

        return x
    return func


@converter(nn.Linear, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def linear(self, input: Tensor):
    out_filters, in_filters = self.weight.shape
    weights = self.weight.detach().numpy()
    weights = weights.transpose(1, 0)
    biases = self.bias.detach().numpy()
    return keras.layers.Dense(out_filters, weights=[weights, biases])


@converter(torch.nn.functional.linear, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def linear(input, weight, bias, out=None):
    out_filters, in_filters = weight.shape
    weights = weight.detach().numpy()
    weights = weights.transpose(1, 0)
    biases = bias.detach().numpy()
    layer = keras.layers.Dense(out_filters, weights=[weights, biases])

    def func(input, weight, bias, out=None):
        return layer(input)
    return func


@converter(torch.matmul, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.linalg.matmul(input, other)
    return func


@converter(torch.Tensor.matmul, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def matmul(self, tensor2):
    def func(self, tensor2):
        return tf.linalg.matmul(self, tensor2)
    return func


@converter(torch.dot, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def dot(input: Tensor, tensor: Tensor, *, out: Optional[Tensor]=None):
    def func(input, tensor, *, out=None):
        return tf.linalg.tensordot(input, tensor, axes=1)
    return func


@converter(torch.mv, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def mv(input: Tensor, vec: Tensor, *, out: Optional[Tensor]=None):
    def func(input, vec, *, out=None):
        return tf.linalg.tensordot(input, vec, axes=1)
    return func


@converter(torch.einsum, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def einsum(*args: Any):
    def func(*args: Any):
        equation = args[0]
        operands = args[1:]
        return tf.einsum(equation, *operands)
    return func


# NB: tensorflow and pytorch implementations of batchnorm behave differently in train mode
@converter(nn.BatchNorm1d, nn.BatchNorm2d)
def batchNormalization(self, input: Tensor):
    momentum = self.momentum
    epsilon = self.eps
    weight = self.weight.detach().numpy()
    biase = self.bias.detach().numpy()
    running_mean = self.running_mean.detach().numpy()
    running_var = self.running_var.detach().numpy()

    layer = keras.layers.BatchNormalization(momentum=1 - momentum, epsilon=epsilon, weights=[weight, biase, running_mean, running_var])
    return layer


@converter(F.batch_norm)
def batch_norm(input: Tensor,
               running_mean: Optional[Tensor], running_var: Optional[Tensor],
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               training: bool = False, momentum: float = 0.1, eps: float = 1e-5):
    weight = weight.detach().numpy()
    bias = bias.detach().numpy()
    running_mean = running_mean.detach().numpy()
    running_var = running_var.detach().numpy()
    bn = keras.layers.BatchNormalization(momentum=1 - momentum, epsilon=eps, weights=[weight, bias, running_mean, running_var])

    def func(input, *args, **kwargs):
        return bn(input)
        # return (input - running_mean) / (tf.sqrt(running_var + eps)) * weight + bias
    return func


@converter(nn.Identity, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def identity(self, input: Tensor):
    def func(input):
        return input
    return func


@converter(torch.nn.modules.dropout.Dropout, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def dropout(self, input: Tensor):
    return keras.layers.Dropout(rate=self.p)


@converter(nn.MaxPool2d)
def maxPool2D(self, input: Tensor):
    kernel_size = self.kernel_size
    stride = self.stride
    return keras.layers.MaxPool2D(pool_size=kernel_size, strides=stride)


@converter(torch.max_pool2d)
def max_pool_2d(input: Tensor, kernel_size: Union[_int, _size], stride: Union[_int, _size]=(), padding: Union[_int, _size]=0, dilation: Union[_int, _size]=1, ceil_mode: _bool=False):
    def func(input, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
        return keras.layers.MaxPool2D(pool_size=kernel_size, strides=stride)(input)
    return func


@converter(F.avg_pool2d)
def max_pool_2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    def func(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        return keras.layers.AvgPool2D(pool_size=kernel_size, strides=stride)(input)
    return func


@converter(F.adaptive_avg_pool2d)
def adaptiveAvgPool2D(input: Tensor, output_size):
    if output_size == (1, 1) or output_size == 1:
        pool_size = input.shape[2:]
        def func(input, output_size):
            return keras.layers.AvgPool2D(pool_size=pool_size)(input)
        return func
    else:
        raise Exception('Unsupported parameters for adaptive_avg_pool2d')


@converter(F.pixel_shuffle)
def pixel_shuffle(input: Tensor, upscale_factor: _int):
    def func(input, upscale_factor):
        x = input
        x = tf.concat([x[..., i::upscale_factor**2] for i in range(upscale_factor**2)], axis=-1)
        x = tf.nn.depth_to_space(x, upscale_factor)
        return x
    return func


@converter(F.interpolate)
def interpolate(input: Tensor, size: Optional[int] = None, scale_factor: Optional[List[float]] = None, mode: str = 'nearest',
                align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None, antialias: bool = False):
    if mode == 'bilinear':
        method = ResizeMethod.BILINEAR
    elif mode == 'nearest':
        method = ResizeMethod.NEAREST_NEIGHBOR
    else:
        raise Exception('Unsupported mode: ', mode)

    def func(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
        if isinstance(scale_factor, numbers.Number):
            scale_factor = (scale_factor, scale_factor)

        if isinstance(size, numbers.Number):
            size = (size, size)

        if size is None:
            _, h, w, _ = input.shape
            size = (int(h * scale_factor[0]), int(w * scale_factor[1]))

        return tf.image.resize(input, size=size, method=method, antialias=antialias)
    return func


@converter(torch.sigmoid, torch.Tensor.sigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def sigmoid(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return keras.layers.Activation(keras.activations.sigmoid)(input)
    return func


@converter(torch.tanh, torch.Tensor.tanh, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def tanh(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input: Tensor, *, out=None):
        return keras.layers.Activation(keras.activations.tanh)(input)
    return func


@converter(nn.ReLU, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def relu(self, input: Tensor):
    return keras.layers.ReLU()


@converter(F.relu, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def relu(input: Tensor, inplace: bool = False):
    def func(input, inplace=False):
        return keras.layers.ReLU(input)
    return func


@converter(nn.LeakyReLU, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def leaky_relu(self, input: Tensor):
    return keras.layers.LeakyReLU(alpha=self.negative_slope)


@converter(F.leaky_relu, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False):
    def func(input, negative_slope=0.01, inplace=False):
        return keras.layers.LeakyReLU(alpha=negative_slope)(input)
    return func


@converter(F.hardsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardsigmoid(input: Tensor, inplace: bool = False):
    def func(input, inplace=False):
        return hard_sigmoid_pytorch_compatible(input)
    return func


@converter(F.hardtanh, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False):
    def func(input, min_val=-1.0, max_val=1.0, inplace=False):
        return hard_tanh_pytorch_compatible(input, min_val, max_val)
    return func


@converter(F.hardswish, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardswish(input: Tensor, inplace: bool = False):
    def func(input, inplace=False):
        return hard_swish_pytorch_compatible(input)
    return func


@converter(torch.clip, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def clip(input: Tensor, min: Optional[Tensor]=None, max: Optional[Tensor]=None, *, out: Optional[Tensor]=None):
    def func(input, min=None, max=None, *, out=None):
        return tf.clip_by_value(input, min, max)
    return func


@converter(torch.sum, torch.Tensor.sum, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def sum(input: Tensor, dim: Sequence, keepdim: _bool=False, *, dtype: Optional[_dtype]=None, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim, keepdim=False, *, dtype=None, out=None):
        dim = _dim_make_positive(dim, n_dims)
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.reduce_sum(input, axis=dim, keepdims=keepdim)
    return func


@converter(torch.Tensor.__add__, torch.Tensor.__iadd__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def add(input, other):
    def func(input, other):
        return input + other
    return func


@converter(torch.Tensor.__radd__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def add(self, other):
    def func(self, other):
        return self + other
    return func


@converter(torch.Tensor.add, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def add(self, other, *args, **kwargs):
    def func(self, other, *args, **kwargs):
        return self + other
    return func


@converter(torch.Tensor.__sub__, torch.Tensor.__isub__, torch.sub, torch.subtract, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def sub(input: Union[Tensor, Number], other: Union[Tensor, Number], *, alpha: Optional[Number]=1, out: Optional[Tensor]=None):
    def func(input, other, *, alpha=1, out=None):
        return input - other
    return func


@converter(torch.Tensor.sub, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def sub(self, other, *args, **kwargs):
    def func(self, other, *args, **kwargs):
        return self - other
    return func


@converter(torch.Tensor.__rsub__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def rsub(self, other):
    def func(self, other):
        return other - self
    return func


@converter(torch.Tensor.__mul__, torch.Tensor.__imul__, torch.mul, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def mul(input: Union[Tensor, Number], other: Union[Tensor, Number], *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return input * other
    return func


@converter(torch.Tensor.mul, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def mul(self, value):
    def func(self, value):
        return self * value
    return func


@converter(torch.Tensor.__truediv__, torch.Tensor.__idiv__, torch.div, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def div(input: Union[Tensor, Number], other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None, out: Optional[Tensor]=None):
    def func(input, other, *, rounding_mode=None, out=None):
        return input / other
    return func


@converter(torch.Tensor.div, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def div(self, value, *args, **kwargs):
    def func(self, value, *args, **kwargs):
        return self / value
    return func


@converter(torch.Tensor.__rdiv__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def rdiv(self, other):
    def func(self, other):
        return other / self
    return func


@converter(torch.sqrt, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def sqrt(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.sqrt(input)
    return func


@converter(torch.Tensor.__pow__, torch.pow, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def pow(input: Tensor, exponent: Number, *, out: Optional[Tensor]=None):
    def func(input, exponent, *, out=None):
        return input ** exponent
    return func


@converter(torch.Tensor.__rpow__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def rpow(self, other):
    def func(self, other):
        return other ** self
    return func


def _permute_inner(perm_original, allow_lazy=True):
    def func(x):
        input_channel_order = get_channel_order(x)

        if allow_lazy:
            if input_channel_order == ChannelOrder.TENSORFLOW and list(perm_original) == perm_pytorch2keras(len(perm_original)):
                x = tf.identity(x)
                x = set_channel_order(x, ChannelOrder.PYTORCH)
                return x
            elif input_channel_order == ChannelOrder.PYTORCH and list(perm_original) == perm_keras2pytorch(len(perm_original)):
                x = tf.identity(x)
                x = set_channel_order(x, ChannelOrder.TENSORFLOW)
                return x

        if input_channel_order == ChannelOrder.TENSORFLOW:
            perm = perm_compose(perm_original, perm_keras2pytorch(len(perm_original)))
            perm = perm_compose(perm_pytorch2keras(len(perm)), perm)
        else:
            perm = perm_original

        if is_identity_perm(perm):
            return x
        else:
            x = tf.transpose(x, perm)
            x = set_channel_order(x, input_channel_order)
            return x
    return func


@converter(torch.Tensor.permute, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def permute_method(self, *dims):
    def func(self, *dims):
        return _permute_inner(dims)(self)
    return func


@converter(torch.permute, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def permute(input: Tensor, dims: _size):
    def func(input, dims):
        return _permute_inner(dims)(input)
    return func


@converter(torch.Tensor.transpose, torch.transpose, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def transpose(input: Tensor, dim0, dim1):
    a = np.zeros(shape=perm_identity(input.dim()))
    perm = np.swapaxes(a, dim0, dim1).shape

    def func(input, dim0, dim1):
        return _permute_inner(perm)(input)
    return func


@converter(torch.moveaxis, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def moveaxis(input: Tensor, source: _size, destination: _size):
    a = np.zeros(shape=perm_identity(input.dim()))
    perm = np.moveaxis(a, source, destination).shape

    def func(input, source, destination):
        return _permute_inner(perm)(input)
    return func


@converter(torch.Tensor.view, torch.Tensor.reshape, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def reshape(self, *shape):
    def func(self, *shape):
        shape = _flatten(shape)
        return tf.reshape(self, shape)
    return func


@converter(torch.reshape, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def reshape(input, *shape):
    def func(input, *shape):
        shape = _flatten(shape)
        return tf.reshape(input, shape)
    return func


@converter(torch.masked_select, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def masked_select(input: Tensor, mask: Tensor, *, out: Optional[Tensor]=None):
    def func(input, mask, *, out=None):
        return tf.boolean_mask(input, mask)
    return func


@converter(torch.masked_fill, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def masked_fill(input: Tensor, mask: Tensor, value: Number):
    def func(input, mask, value):
        value = tf.convert_to_tensor(value, dtype=input.dtype)
        return tf.where(mask, value, input)
    return func


@converter(torch.Tensor.__getitem__, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def getitem(self, *slices):
    n_dims = self.dim()

    if isinstance(slices[0], torch.Tensor) and slices[0].dtype == torch.bool:
        mask = slices[0]
        return masked_select(self, mask)

    slices = _flatten(slices)

    def is_light(slices):
        return all(isinstance(slc, slice) for slc in slices)

    def slices_make_full(slices, n_dims):
        n_notnone = len([slc for slc in slices if slc is not None])
        n_pads = n_dims - n_notnone
        slices_full = slices + (slice(None),) * n_pads
        return slices_full

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


@converter(torch.Tensor.narrow, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def narrow(self, dimension, start, length):
    n_dims = self.dim()

    def func(self, dimension, start, length):
        x = self
        dimension = _dim_make_positive(dimension, n_dims)

        if get_channel_order(x) == ChannelOrder.TENSORFLOW:
            perm = perm_keras2pytorch(n_dims)
            x = _permute(perm)(x)
        slices = (*[slice(None)]*dimension, slice(start, start+length))
        x = x.__getitem__(slices)
        x = set_channel_order(x, ChannelOrder.PYTORCH)
        return x
    return func


@converter(torch.squeeze, torch.Tensor.squeeze, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def squeeze(input: Tensor, dim):
    n_dims = input.dim()
    def func(input, dim):
        x = input
        if get_channel_order(x) == ChannelOrder.TENSORFLOW:
            perm = perm_keras2pytorch(n_dims)
            x = _permute(perm)(x)
        x = tf.squeeze(x, axis=dim)
        x = set_channel_order(x, ChannelOrder.PYTORCH)
        return x
    return func


@converter(torch.unsqueeze, torch.Tensor.unsqueeze, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def unsqueeze(input, dim):
    n_dims = input.dim()
    def func(input, dim):
        x = input
        if get_channel_order(x) == ChannelOrder.TENSORFLOW:
            perm = perm_keras2pytorch(n_dims)
            x = _permute(perm)(x)
        x = tf.expand_dims(x, axis=dim)
        x = set_channel_order(x, ChannelOrder.PYTORCH)
        return x
    return func


# # FIXME
# @converter(torch.Tensor.unfold, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
# def unfold(self, dimension, size, step):
#     def func(self, dimension, size, step):
#         x = self
#         b, c, h, w, cplx = x.shape
#         x = tf.reshape(x, shape=(b, c, h, w * cplx))
#         n_dims = len(x.shape)
#         dimension = dimension + 1
#
#         sizes = [1]*n_dims
#         strides = [1]*n_dims
#         rates = [1]*n_dims
#
#         sizes[dimension] = size
#         strides[dimension] = step
#         x = tf.image.extract_patches(x, sizes=sizes, strides=strides, rates=rates, padding='VALID')
#
#         x = tf.reshape(x, shape=(b, c, -1, size, w, cplx))
#         x = tf.transpose(x, (0, 1, 2, 4, 5, 3))
#         return x
#     return func


# FIXME
@converter(torch.Tensor.unfold, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def unfold(self, dimension, size, step):
    n_dims = self.dim()

    def func(self, dimension, size, step):
        sizes = [1]*n_dims
        strides = [1]*n_dims
        rates = [1]*n_dims

        sizes[dimension] = size
        strides[dimension] = step
        x = self
        x = tf.image.extract_patches(x, sizes=sizes, strides=strides, rates=rates, padding='VALID')

        b, c, h, w = x.shape
        x = tf.reshape(x, shape=[b, c, h, size, -1])
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        return x
    return func


@converter(torch.Tensor.detach, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def detach(self):
    def func(self):
        return tf.stop_gradient(self)
    return func


@converter(torch.Tensor.detach, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def detach(self):
    def func(self):
        return tf.stop_gradient(self)
    return func


@converter(torch.Tensor.cpu, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def cpu(self, memory_format=None):
    def func(self, memory_format=None):
        return self
    return func


@converter(torch.Tensor.type, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def type(self, dtype=None, non_blocking=False, **kwargs):
    def func(self, dtype=None, non_blocking=False, **kwargs):
        if dtype == torch.float32:
            tf_type = tf.float32
        else:
            raise Exception('Unsupported dtype: ', dtype)
        return tf.cast(self, tf_type)
    return func


@converter(torch.greater, torch.Tensor.__gt__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def gt(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.greater(input, other)
    return func


@converter(torch.greater_equal, torch.Tensor.__ge__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def ge(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.greater_equal(input, other)
    return func


@converter(torch.less, torch.Tensor.__lt__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def lt(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.less(input, other)
    return func


@converter(torch.less_equal, torch.Tensor.__le__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def le(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.less_equal(input, other)
    return func


@converter(torch.view_as_complex, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def view_as_complex(input: Tensor):
    def func(input):
        return tf.complex(input[..., 0], input[..., 1])
    return func


@converter(torch.view_as_real, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def view_as_real(input: Tensor):
    def func(input):
        return tf.stack([tf.math.real(input), tf.math.imag(input)], axis=-1)
    return func


@converter(force_tensorflow_order, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def force_tensorflow_order(input):
    return lambda input: input


@converter(force_pytorch_order, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def force_pytorch_order(input):
    return lambda input: input
