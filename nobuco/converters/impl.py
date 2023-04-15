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

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.funcs import force_tensorflow_order, force_pytorch_order
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import dims_pytorch2keras, perm_keras2pytorch, \
    _dim_make_positive, dim_pytorch2keras, _permute, _flatten, perm_pytorch2keras, perm_compose, \
    is_identity_perm, permute_pytorch2keras, _ensure_iterable, perm_identity
from nobuco.converters.ops import hard_sigmoid_pytorch_compatible, hard_swish_pytorch_compatible, \
    hard_tanh_pytorch_compatible


@converter(nn.GRU, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def gru(self: nn.GRU, input, hx=None):
    assert not self.bidirectional

    def reorder(param):
        assert param.shape[-1] % 3 == 0
        p1, p2, p3 = np.split(param, 3, axis=-1)
        return np.concatenate([p2, p1, p3], axis=-1)

    grus = []
    for i in range(self.num_layers):
        weight_ih = self.__getattr__(f'weight_ih_l{i}').detach().numpy().transpose((1, 0))
        weight_hh = self.__getattr__(f'weight_hh_l{i}').detach().numpy().transpose((1, 0))
        bias_ih = self.__getattr__(f'bias_ih_l{i}').detach().numpy()
        bias_hh = self.__getattr__(f'bias_hh_l{i}').detach().numpy()

        weight_ih = reorder(weight_ih)
        weight_hh = reorder(weight_hh)
        bias_ih = reorder(bias_ih)
        bias_hh = reorder(bias_hh)

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
def cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim=0, *, out: Optional[Tensor]=None):
    num_dims = tensors[0].dim()
    dim_keras = dim_pytorch2keras(dim, num_dims)

    def func(tensors, dim=0, *, out=None):
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


@converter(torch.Tensor.split, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def split(self, split_size, dim=0):
    num_dims = self.dim()

    def func(self, split_size, dim=0):
        if get_channel_order(self) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.split(self, num_or_size_splits=split_size, axis=dim)
    return func


@converter(torch.Tensor.repeat, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def repeat(self, *sizes):
    def func(self, *sizes):
        if get_channel_order(self) == ChannelOrder.TENSORFLOW:
            sizes = permute_pytorch2keras(sizes)
        return tf.tile(self, sizes)
    return func


@converter(torch.Tensor.expand, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def expand(self, *sizes):
    def get_broadcast_shape(sizes, tensor_shape):
        tensor_shape = list(reversed(tensor_shape))
        res = []
        for i, s in enumerate(reversed(sizes)):
            if s == -1:
                s = tensor_shape[i]
            res.append(s)
        return list(reversed(res))

    def func(self, *sizes):
        sizes = _flatten(sizes)
        broadcast_shape = get_broadcast_shape(sizes, self.shape)
        return tf.broadcast_to(self, broadcast_shape)
    return func


@converter(torch.Tensor.expand_as, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def expand_as(self, other):
    def get_broadcast_shape(sizes, tensor_shape):
        tensor_shape = list(reversed(tensor_shape))
        res = []
        for i, s in enumerate(reversed(sizes)):
            if s == -1:
                s = tensor_shape[i]
            res.append(s)
        return list(reversed(res))

    def func(self, other):
        broadcast_shape = get_broadcast_shape(other.shape, self.shape)
        return tf.broadcast_to(self, broadcast_shape)
    return func


@converter(torch.zeros_like, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def zeros_like(input: Tensor, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
    def func(input: Tensor, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        tf_type = dtype_pytorch2keras(dtype)
        return tf.zeros_like(input, dtype=tf_type)
    return func


@converter(torch.Tensor.new_empty, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def new_empty(self, size, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
    def func(self, size, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
        if dtype is not None:
            dtype = dtype_pytorch2keras(dtype)
        else:
            dtype = self.dtype
        return tf.zeros(size, dtype=dtype)
    return func


@converter(torch.Tensor.new_full, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False):
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
def full_like(input: Tensor, fill_value: Number, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
    def func(input: Tensor, fill_value: Number, *, memory_format=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        if dtype is not None:
            dtype = dtype_pytorch2keras(dtype)
        else:
            dtype = input.dtype
        res = tf.fill(input.shape, fill_value)
        res = tf.cast(res, dtype)
        return res
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


@converter(torch.Tensor.unbind, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def unbind(self, dim=0):
    def func(self, dim=0):
        return tf.unstack(self, axis=dim)
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


@converter(torch.Tensor.flatten, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def flatten(self, start_dim=0, end_dim=-1):
    def func(self, start_dim=0, end_dim=-1):
        start_shape = self.shape[:start_dim]

        n_dims = len(self.shape)
        end_dim = _dim_make_positive(end_dim, n_dims)
        if end_dim < n_dims-1:
            end_shape = self.shape[end_dim+1:]
        else:
            end_shape = []

        return tf.reshape(self, (*start_shape, -1, *end_shape))
    return func


@converter(torch.mean, torch.Tensor.mean, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def mean(input: Tensor, dim=None, keepdim: _bool = False, *, dtype: Optional[_dtype] = None, out: Optional[Tensor] = None):
    def func(input, dim=None, keepdim=False, *, dtype=None, out=None):
        if isinstance(dim, numbers.Number):
            dim = [dim]
        return tf.reduce_mean(input, axis=dim, keepdims=keepdim)
    return func


@converter(torch.abs, torch.Tensor.abs, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def abs(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.abs(input)
    return func


@converter(nn.Conv1d)
def conv1d(self, input: Tensor):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation

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

    if isinstance(padding, numbers.Number):
        padding = (padding,)
    if padding != (0,) and padding != 'valid':
        pad_layer = keras.layers.ZeroPadding1D(padding[0])
    else:
        pad_layer = None

    if groups == out_filters and groups != 1:
        conv = keras.layers.DepthwiseConv1D(kernel_size=kw,
                                      strides=stride,
                                      padding='valid',
                                      dilation_rate=dilation,
                                      groups=groups,
                                      use_bias=use_bias,
                                      weights=params
                                      )
    elif groups == 1:
        conv = keras.layers.Conv1D(filters=out_filters,
                             kernel_size=kw,
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

        def grouped_conv1d(inputs, filters, kernel_size, strides, groups, dilation=dilation):
            splits = tf.split(inputs, groups, axis=-1)
            convolved_splits = [
                keras.layers.Conv1D(filters // groups,
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

        conv = lambda x: grouped_conv1d(x, out_filters, kernel_size=kw, strides=stride, groups=groups, dilation=dilation)

    def func(input):
        if pad_layer is not None:
            input = pad_layer(input)
        output = conv(input)
        return output
    return func


# @converter(F.conv1d)
# def conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[_int, _size]=1, padding: str="valid", dilation: Union[_int, _size]=1, groups: _int=1):
#     out_filters, in_filters, kw = weight.shape
#     weights = weight.detach().numpy()
#     weights = weights.transpose((2, 1, 0))
#
#     if bias is not None:
#         biases = bias.detach().numpy()
#         params = [weights, biases]
#         use_bias = True
#     else:
#         params = [weights]
#         use_bias = False
#
#     if padding != 0:
#         pad_layer = keras.layers.ZeroPadding1D(padding)
#     else:
#         pad_layer = None
#
#     conv = keras.layers.Conv1D(out_filters, kernel_size=kw,
#                          strides=stride, padding='valid',
#                          # dilation_rate=dilations, groups=groups,
#                          use_bias=use_bias,
#                          weights=params
#                          )
#
#     def func(input, *args, **kwargs):
#         if pad_layer is not None:
#             input = pad_layer(input)
#         return conv(input)
#     return func


@converter(nn.Conv2d)
def conv2d(self, input: Tensor):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation

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

    def func(input):
        if pad_layer is not None:
            input = pad_layer(input)
        output = conv(input)
        return output
    return func


# @converter(F.conv2d)
# def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[_int, _size] = 1,
#                     padding: str = "valid", dilation: Union[_int, _size] = 1, groups: _int = 1):
#
#     out_filters, in_filters, kh, kw = weight.shape
#
#     weights = weight.detach().numpy()
#     if groups == out_filters and groups != 1:
#         weights = tf.transpose(weights, (2, 3, 0, 1))
#     elif groups == 1:
#         weights = tf.transpose(weights, (2, 3, 1, 0))
#     else:
#         weights = tf.transpose(weights, (2, 3, 1, 0))
#
#     if bias is not None:
#         biases = bias.detach().numpy()
#         params = [weights, biases]
#         use_bias = True
#     else:
#         params = [weights]
#         use_bias = False
#
#     if padding != 0 and padding != (0, 0) and padding != 'valid':
#         pad_layer = keras.layers.ZeroPadding2D(padding)
#     else:
#         pad_layer = None
#
#     if groups == out_filters and groups != 1:
#         conv = keras.layers.DepthwiseConv2D(kernel_size=(kh, kw),
#                                       strides=stride,
#                                       padding='valid',
#                                       dilation_rate=dilation,
#                                       groups=groups,
#                                       use_bias=use_bias,
#                                       weights=params
#                                       )
#     elif groups == 1:
#         conv = keras.layers.Conv2D(filters=out_filters,
#                              kernel_size=(kh, kw),
#                              strides=stride,
#                              padding='valid',
#                              dilation_rate=dilation,
#                              groups=groups,
#                              use_bias=use_bias,
#                              weights=params
#                              )
#     else:
#         def split_params(params, groups, axis):
#             params_split = [np.split(p, groups, axis=axis) for p in params]
#             return list(zip(*params_split))
#
#         params_split = split_params(params, groups, axis=-1)
#
#         def grouped_conv2d(inputs, filters, kernel_size, strides, groups, dilation=dilation):
#             splits = tf.split(inputs, groups, axis=-1)
#             convolved_splits = [
#                 keras.layers.Conv2D(filters // groups,
#                                     kernel_size=kernel_size,
#                                     strides=strides,
#                                     padding='valid',
#                                     dilation_rate=dilation,
#                                     use_bias=use_bias,
#                                     weights=params_split[i]
#                                     )(split)
#                 for i, split in enumerate(splits)
#             ]
#             return tf.concat(convolved_splits, -1)
#
#         conv = lambda x: grouped_conv2d(x, out_filters, kernel_size=(kh, kw), strides=stride, groups=groups, dilation=dilation)
#
#     def func(input, *args, **kwargs):
#         if pad_layer is not None:
#             input = pad_layer(input)
#         output = conv(input)
#         return output
#     return func


@converter(nn.ConvTranspose2d)
def convTranspose2d(self, input: Tensor, output_size: Optional[List[int]] = None):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation
    output_padding = self.output_padding

    in_filters, out_filters, kh, kw = weight.shape
    weights = weight.detach().numpy()

    if groups == 1:
        weights = weights.transpose((2, 3, 1, 0))
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
    else:
        pad_layer = None
    pad_layer = None

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

        if pad_layer is not None:
            input = pad_layer(input)

        x = conv(input)

        if output_padding != (0, 0):
            x = x[:, output_padding[0]:, output_padding[1]:, :]
        return x
    return func


# # FIXME: not thoroughly tested
# @converter(F.conv_transpose2d)
# def conv_transpose2d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None,
#                      stride: Union[_int, _size]=1, padding: Union[_int, _size]=0, output_padding: Union[_int, _size]=0,
#                      groups: _int=1, dilation: Union[_int, _size]=1):
#
#     in_filters, out_filters, kh, kw = weight.shape
#     weights = weight.detach().numpy()
#
#     if groups == 1:
#         weights = weights.transpose((2, 3, 1, 0))
#     elif groups == in_filters:
#         weights = weights.transpose((2, 3, 0, 1))
#     else:
#         weights = weights.transpose((2, 3, 1, 0))
#
#     if bias is not None:
#         biases = bias.detach().numpy()
#         params = [weights, biases]
#         use_bias = True
#     else:
#         params = [weights]
#         use_bias = False
#
#     if isinstance(dilation, numbers.Number):
#         dilation = (dilation, dilation)
#
#     if isinstance(padding, numbers.Number):
#         padding = (padding, padding)
#
#     if isinstance(output_padding, numbers.Number):
#         output_padding = (output_padding, output_padding)
#
#     if padding != (0, 0):
#         pad = (dilation[0] * (kh - 1) - padding[0], dilation[1] * (kw - 1) - padding[1])
#         pad_layer = keras.layers.ZeroPadding2D(pad)
#     else:
#         pad_layer = None
#     pad_layer = None
#
#     if groups == 1:
#         conv = keras.layers.Conv2DTranspose(out_filters,
#                                             kernel_size=(kh, kw),
#                                             strides=stride,
#                                             padding='valid',
#                                             dilation_rate=dilation,
#                                             groups=1,
#                                             use_bias=use_bias,
#                                             weights=params
#                                             )
#     elif groups == in_filters and out_filters == 1:
#         weights = params[0]
#
#         weights_full = np.zeros(shape=(*weights.shape[:-1], groups))
#         for i in range(groups):
#             weights_full[..., i, i] = weights[..., i, 0]
#         params[0] = weights_full
#
#         conv = keras.layers.Conv2DTranspose(out_filters*groups,
#                                             kernel_size=(kh, kw),
#                                             strides=stride,
#                                             padding='valid',
#                                             dilation_rate=dilation,
#                                             groups=1,
#                                             use_bias=use_bias,
#                                             weights=params
#                                             )
#     else:
#         raise Exception('Unsupprorted # groups:', groups)
#
#         # def split_params(params, groups, axis):
#         #     params_split = [np.split(p, groups, axis=axis) for p in params]
#         #     return list(zip(*params_split))
#         #
#         # params_split = split_params(params, groups, axis=-2)
#         #
#         # def grouped_conv2d_transpose(inputs, filters, kernel_size, strides, groups, dilation=dilation):
#         #     splits = tf.split(inputs, groups, axis=-1)
#         #     convolved_splits = [
#         #         keras.layers.Conv2DTranspose(filters // groups,
#         #                                      kernel_size=kernel_size,
#         #                                      strides=strides,
#         #                                      padding='same',
#         #                                      dilation_rate=dilation,
#         #                                      use_bias=use_bias,
#         #                                      weights=params_split[i]
#         #                                      )(split)
#         #         for i, split in enumerate(splits)
#         #     ]
#         #     return tf.concat(convolved_splits, -1)
#         #
#         # conv = lambda x: grouped_conv2d_transpose(x, out_filters*groups, kernel_size=(kh, kw), strides=stride, groups=groups, dilation=dilation)
#
#     # FIXME!
#     def func(input, *args, **kwargs):
#         if pad_layer is not None:
#             input = pad_layer(input)
#
#         x = conv(input)
#
#         if output_padding != (0, 0):
#             x = x[:, output_padding[0]:, output_padding[1]:, :]
#         return x
#     return func


@converter(nn.Linear, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def linear(self, input: Tensor):
    out_filters, in_filters = self.weight.shape
    weights = self.weight.detach().numpy()
    weights = weights.transpose(1, 0)

    biases = self.bias
    if biases is not None:
        biases = self.bias.detach().numpy()
        params = [weights, biases]
    else:
        params = [weights]
    return keras.layers.Dense(out_filters, weights=params)


# @converter(torch.nn.functional.linear, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
# def linear(input, weight, bias, out=None):
#     out_filters, in_filters = weight.shape
#     weights = weight.detach().numpy()
#     weights = weights.transpose(1, 0)
#
#     if bias is not None:
#         biases = bias.detach().numpy()
#     else:
#         biases = np.zeros(shape=(out_filters,))
#
#     layer = keras.layers.Dense(out_filters, weights=[weights, biases])
#
#     def func(input, weight, bias, out=None):
#         return layer(input)
#     return func


@converter(torch.matmul, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.linalg.matmul(input, other)
    return func


@converter(torch.Tensor.matmul, torch.Tensor.__matmul__, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def matmul(self, tensor2):
    def func(self, tensor2):
        return tf.linalg.matmul(self, tensor2)
    return func


@converter(torch.dot, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def dot(input: Tensor, tensor: Tensor, *, out: Optional[Tensor]=None):
    def func(input, tensor, *, out=None):
        return tf.linalg.tensordot(input, tensor, axes=1)
    return func


@converter(torch.mv, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def mv(input: Tensor, vec: Tensor, *, out: Optional[Tensor]=None):
    def func(input, vec, *, out=None):
        return tf.linalg.tensordot(input, vec, axes=1)
    return func


@converter(torch.bmm, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def bmm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor]=None):
    def func(input, mat2, *, out=None):
        return tf.linalg.matmul(input, mat2)
    return func


@converter(torch.einsum, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def einsum(*args: Any):
    def func(*args: Any):
        equation = args[0]
        operands = args[1:]
        return keras.layers.Lambda(lambda operands: tf.einsum(equation, *operands))(operands)
    return func


@converter(torch.Tensor.triu, torch.Tensor.triu_, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def triu(self, diagonal=0):
    def func(self, diagonal=0):
        return keras.layers.Lambda(lambda x: tf.experimental.numpy.triu(x, k=diagonal))(self)
    return func


# NB: tensorflow and pytorch implementations of batchnorm behave differently in train mode
@converter(nn.BatchNorm1d, nn.BatchNorm2d)
def batchNorm1d(self, input: Tensor):
    momentum = self.momentum
    epsilon = self.eps
    weight = self.weight.detach().numpy()
    bias = self.bias.detach().numpy()
    running_mean = self.running_mean.detach().numpy()
    running_var = self.running_var.detach().numpy()

    layer = keras.layers.BatchNormalization(momentum=1 - momentum, epsilon=epsilon, weights=[weight, bias, running_mean, running_var])
    return layer

    # def func(input, *args, **kwargs):
    #     return (input - running_mean) / (tf.sqrt(running_var + epsilon)) * weight + bias
    # return func


# @converter(F.batch_norm)
# def batch_norm(input: Tensor,
#                running_mean: Optional[Tensor], running_var: Optional[Tensor],
#                weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
#                training: bool = False, momentum: float = 0.1, eps: float = 1e-5):
#     weight = weight.detach().numpy()
#     bias = bias.detach().numpy()
#     running_mean = running_mean.detach().numpy()
#     running_var = running_var.detach().numpy()
#     bn = keras.layers.BatchNormalization(momentum=1 - momentum, epsilon=eps, weights=[weight, bias, running_mean, running_var])
#
#     def func(input, *args, **kwargs):
#         return bn(input)
#         # return (input - running_mean) / (tf.sqrt(running_var + eps)) * weight + bias
#     return func


@converter(F.layer_norm, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def layer_norm(input: Tensor,
               normalized_shape: List[int],
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: float = 1e-5
               ):
    assert len(normalized_shape) == 1

    weight = weight.detach().numpy()
    bias = bias.detach().numpy()
    layer = keras.layers.LayerNormalization(axis=-1, epsilon=eps, weights=[weight, bias])

    def func(input, *args, **kwargs):
        return layer(input)
    return func


@converter(F.embedding, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None, max_norm: Optional[float] = None,
              norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False):
    input_dim, output_dim = weight.shape
    weight = weight.detach().numpy()

    layer = keras.layers.Embedding(input_dim, output_dim, weights=[weight])

    def func(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
        return layer(input)
    return func


@converter(nn.Identity, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def identity(self, input: Tensor):
    def func(input):
        return input
    return func


@converter(torch.nn.modules.dropout.Dropout, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def dropout(self, input: Tensor):
    return keras.layers.Dropout(rate=self.p)


@converter(F.dropout, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    def func(input, p=0.5, training=True, inplace=False):
        return keras.layers.Dropout(rate=p)(input)
    return func


# @converter(nn.MaxPool2d)
# def maxPool2D(self, input: Tensor):
#     kernel_size = self.kernel_size
#     stride = self.stride
#     return keras.layers.MaxPool2D(pool_size=kernel_size, strides=stride)


@converter(torch.max_pool2d)
def max_pool_2d(input: Tensor, kernel_size: Union[_int, _size], stride: Union[_int, _size]=(), padding: Union[_int, _size]=0, dilation: Union[_int, _size]=1, ceil_mode: _bool=False):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if padding != (0, 0):
        kh, kw = kernel_size
        pad = (dilation[0] * (kh - 1) - padding[0], dilation[1] * (kw - 1) - padding[1])
        pad_layer = keras.layers.ZeroPadding2D(pad)
    else:
        pad_layer = None

    def func(input, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
        if pad_layer is not None:
            input = pad_layer(input)
        return keras.layers.MaxPool2D(pool_size=kernel_size, strides=stride, padding='valid')(input)
    return func


@converter(F.avg_pool2d)
def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(padding, numbers.Number):
        padding = (padding, padding)

    if padding != (0, 0):
        kh, kw = kernel_size
        pad = ((kh - 1) - padding[0], (kw - 1) - padding[1])
        pad_layer = keras.layers.ZeroPadding2D(pad)
    else:
        pad_layer = None

    def func(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        if pad_layer is not None:
            input = pad_layer(input)
        return keras.layers.AvgPool2D(pool_size=kernel_size, strides=stride)(input)
    return func


# @converter(F.avg_pool2d)
# def max_pool_2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
#     def func(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
#         return keras.layers.AvgPool2D(pool_size=kernel_size, strides=stride)(input)
#     return func


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
        return tf.nn.relu(input)
    return func


@converter(torch.relu_, torch.Tensor.relu_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def relu_(input: Tensor):
    def func(input):
        return tf.nn.relu(input)
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


@converter(torch.softmax, torch.Tensor.softmax, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def softmax(input: Tensor, dim: Union[str, None], *, dtype: Optional[_dtype]=None):
    num_dims = input.dim()

    def func(input: Tensor, dim, *, dtype=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.nn.softmax(input, axis=dim)
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


@converter(torch.Tensor.add, torch.Tensor.__add__, torch.Tensor.__iadd__, torch.Tensor.__radd__,
           channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH,
           autocast=True)
def add(input, other, *args, **kwargs):
    def func(input, other, *args, **kwargs):
        return input + other
    return func


@converter(torch.sub, torch.subtract, torch.Tensor.sub, torch.Tensor.__sub__, torch.Tensor.__isub__,
           channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH,
           autocast=True)
def sub(input: Union[Tensor, Number], other: Union[Tensor, Number], *, alpha: Optional[Number]=1, out: Optional[Tensor]=None):
    def func(input, other, *, alpha=1, out=None):
        return input - other
    return func


@converter(torch.Tensor.__rsub__,
           channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH,
           autocast=True)
def rsub(self, other):
    def func(self, other):
        return other - self
    return func


@converter(torch.mul, torch.Tensor.__mul__, torch.Tensor.__imul__,
           torch.Tensor.__rmul__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH,
           autocast=True)
def mul(input: Union[Tensor, Number], other: Union[Tensor, Number], *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return input * other
    return func


@converter(torch.Tensor.mul,
           channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH,
           autocast=True)
def mul(self, value):
    def func(self, value):
        return self * value
    return func


@converter(torch.Tensor.__truediv__, torch.Tensor.__idiv__, torch.div,
           channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH,
           autocast=True)
def div(input: Union[Tensor, Number], other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None, out: Optional[Tensor]=None):
    def func(input, other, *, rounding_mode=None, out=None):
        return input / other
    return func


@converter(torch.Tensor.div,
           channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH,
           autocast=True)
def div(self, value, *args, **kwargs):
    def func(self, value, *args, **kwargs):
        return self / value
    return func


@converter(torch.Tensor.__rdiv__,
           channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH,
           autocast=True)
def rdiv(self, other):
    def func(self, other):
        return other / self
    return func


@converter(torch.Tensor.copy_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH)
def copy_(self, src, non_blocking=False):
    def func(self, src, non_blocking=False):
        return self * 0 + src
    return func


@converter(torch.sqrt, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def sqrt(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.sqrt(input)
    return func


@converter(torch.Tensor.rsqrt, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def rsqrt(self):
    def func(self):
        # return tf.math.rsqrt(self)
        return 1 / tf.math.sqrt(self)
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


@converter(torch.exp, torch.Tensor.exp, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def exp(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return tf.exp(input)
    return func


@converter(torch.log, torch.Tensor.log, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def log(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return tf.experimental.numpy.log(input)
    return func


@converter(torch.log2, torch.Tensor.log2, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def log2(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out: Optional[Tensor]=None):
        return tf.experimental.numpy.log2(input)
    return func


@converter(torch.floor, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def floor(input: Tensor, *, out: Optional[Tensor]=None):
    def func(input, *, out=None):
        return tf.floor(input)
    return func


@converter(torch.round, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def round(input: Tensor, *, decimals: _int, out=None):
    assert decimals == 0

    def func(input: Tensor, *, decimals: _int, out=None):
        return tf.round(input)
    return func


@converter(torch.norm, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
    num_dims = input.dim()

    def func(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.norm(input, ord=p, axis=dim, keepdims=keepdim)
    return func


@converter(torch.Tensor.round, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def round(self, decimals=0):
    assert decimals == 0

    def func(self, decimals=0):
        return tf.round(self)
    return func


@converter(torch.clamp, torch.Tensor.clamp, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def clamp(input: Tensor, min: Optional[Number]=None, max: Optional[Number]=None, *, out: Optional[Tensor]=None):
    def func(input, min=None, max=None, *, out=None):
        return tf.keras.backend.clip(input, min_value=min, max_value=max)
    return func


@converter(torch.min, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def min(input: Tensor, dim=None, keepdim: _bool=False, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor]]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        if dim is not None and get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.keras.backend.min(input, axis=dim, keepdims=keepdim)
    return func


@converter(torch.max, torch.Tensor.max, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def max(input: Tensor, dim=None, keepdim: _bool=False, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor]]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        if dim is not None and get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.keras.backend.max(input, axis=dim, keepdims=keepdim)
    return func


@converter(torch.argmax, torch.Tensor.argmax, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def argmax(input: Tensor, dim: Optional[_int]=None, keepdim: _bool=False, *, out: Optional[Tensor]=None):
    n_dims = input.dim()

    def func(input, dim=None, keepdim=False, *, out=None):
        if dim is not None and get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        return tf.keras.backend.argmax(input, axis=dim)
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
        return tf.reshape(self, tuple(shape))
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


@converter(torch.masked_fill, torch.Tensor.masked_fill, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def masked_fill(input: Tensor, mask: Tensor, value: Number):
    def func(input, mask, value):
        value = tf.convert_to_tensor(value, dtype=input.dtype)
        return tf.where(mask, value, input)
    return func


@converter(torch.where, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def where(condition: Tensor):
    def func(condition):
        return tf.where(condition)[..., 0]
    return func


@converter(torch.fill, torch.Tensor.fill_, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def fill(input, value):
    def func(input, value):
        result = tf.fill(input.shape, value)
        result = tf.cast(result, dtype=input.dtype)
        return result
    return func


@converter(torch.meshgrid, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def meshgrid(*tensors, indexing: Optional[str] = None):
    def func(*tensors, indexing=None):
        return tf.meshgrid(*tensors, indexing=indexing)
    return func


@converter(None, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def getitem_indexed(self, *slices):
    slices = _flatten(slices)
    slices = torch.broadcast_tensors(*slices)
    slices_combined = torch.stack(slices, dim=-1).numpy()

    def func(self, *slices):
        return tf.gather_nd(self, slices_combined)
    return func


def slices_make_full(slices, n_dims):
    n_notnone = len([slc for slc in slices if slc is not None])
    n_pads = n_dims - n_notnone
    slices_full = slices + (slice(None),) * n_pads
    return slices_full


@converter(torch.Tensor.__getitem__, channel_ordering_strategy=ChannelOrderingStrategy.MANUAL)
def getitem(self, *slices):
    n_dims = self.dim()

    slices = _flatten(slices)

    if isinstance(slices[0], torch.Tensor):
        if slices[0].dtype == torch.bool:
            return masked_select(self, slices[0])
        elif slices[0].dtype == torch.int64:
            return getitem_indexed(self, slices)

    def is_light(slices):
        return all(isinstance(slc, slice) for slc in slices)

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


def slice_assign(sliced_tensor, assigned_tensor, *slice_args, verbose=0):
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
    # parsing the slice specifications
    n_slices = len(slice_args)
    dims_to_index = []
    corresponding_ranges = []
    ellipsis = False
    for i_dim, slice_spec in enumerate(slice_args):
        if slice_spec is Ellipsis:
            ellipsis = True
        else:
            if isinstance(slice_spec, int):
                start, stop, step = slice_spec, slice_spec + 1, None
            elif isinstance(slice_spec, slice):
                start, stop, step = slice_spec.start, slice_spec.stop, slice_spec.step
            else:
                raise Exception(f'Unrecognized slice spec: {slice_spec}')

            no_start = start is None or start == 0
            no_stop = stop is None or stop == -1
            no_step = step is None or step == 1
            if no_start and no_stop and no_step:
                continue
            if ellipsis:
                real_index = i_dim + (n_dims - n_slices)
            else:
                real_index = i_dim
            dims_to_index.append(real_index)
            if no_step:
                step = 1
            if no_stop:
                stop = shape[real_index]
            if no_start:
                start = 0
            corresponding_range = tf.range(start, stop, step)
            corresponding_ranges.append(corresponding_range)
    if not dims_to_index:
        if verbose > 0:
            print('Warning: no slicing performed')
        return assigned_tensor
    dims_left_out = [
        i_dim for i_dim in range(n_dims) if i_dim not in dims_to_index
    ]
    scatted_nd_perm = dims_to_index + dims_left_out
    inverse_scatter_nd_perm = list(np.argsort(scatted_nd_perm))
    # reshaping the tensors
    # NOTE: the tensors are reshaped to allow for easier indexing with
    # tensor_scatter_nd_update
    sliced_tensor_reshaped = tf.transpose(sliced_tensor, perm=scatted_nd_perm)
    assigned_tensor_reshaped = tf.transpose(assigned_tensor, perm=scatted_nd_perm)
    left_out_shape = [shape[i_dim] for i_dim in dims_left_out]
    assigned_tensor_reshaped = tf.reshape(assigned_tensor_reshaped, [-1] + left_out_shape)
    # creating the indices
    mesh_ranges = tf.meshgrid(*corresponding_ranges, indexing='ij')
    update_indices = tf.stack([
        tf.reshape(slicing_range, (-1,))
        for slicing_range in mesh_ranges
    ], axis=-1)

    # finalisation
    sliced_tensor_reshaped = tf.tensor_scatter_nd_update(
        tensor=sliced_tensor_reshaped,
        indices=update_indices,
        updates=assigned_tensor_reshaped,
    )
    sliced_tensor_updated = tf.transpose(
        sliced_tensor_reshaped,
        perm=inverse_scatter_nd_perm,
    )
    return sliced_tensor_updated


def broadcast_to_dim(tensor, target_n_dims):
    shape = tensor.shape()
    n_dims = len(tensor.shape)
    if get_channel_order(tensor) == ChannelOrder.TENSORFLOW:
        shape = dims_keras2pytorch(shape, n_dims)
        target_shape = shape_make_full(shape, target_n_dims)
        target_shape = dims_pytorch2keras(target_shape, target_n_dims)
    else:
        target_shape = shape_make_full(shape, target_n_dims)

    tensor = tf.reshape(tensor, target_shape)
    return tensor


@converter(torch.Tensor.__setitem__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def setitem(sliced_tensor, assigned_tensor, *slice_args):
    n_dims = sliced_tensor.dim()

    def func(sliced_tensor, slice_args, assigned_tensor):
        if get_channel_order(sliced_tensor) == ChannelOrder.TENSORFLOW:
            slice_args = _flatten(slice_args)
            slice_args = slices_make_full(slice_args, n_dims)
            slice_args = permute_pytorch2keras(slice_args)
        return slice_assign(sliced_tensor, assigned_tensor, *slice_args)
    return func


@converter(torch.Tensor.__getattribute__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def getattribute(self, attribute):
    if attribute == 'T':
        def func(self, attribute):
            return _permute_inner([1, 0])(self)
    else:
        raise Exception(f'Unsupported attribute: {attribute}')
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


@converter(torch.Tensor.contiguous, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def contiguous(self, memory_format=None):
    def func(self, memory_format=None):
        return self
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


def dtype_pytorch2keras(dtype):
    if dtype == torch.float32:
        tf_type = tf.float32
    elif dtype == torch.float64:
        tf_type = tf.float64
    elif dtype == torch.int32:
        tf_type = tf.int32
    elif dtype == torch.int64:
        tf_type = tf.int64
    elif dtype == torch.bool:
        tf_type = tf.bool
    elif dtype is None:
        tf_type = None
    else:
        raise Exception('Unsupported dtype: ', dtype)
    return tf_type


def type_func(self, dtype=None, non_blocking=False, **kwargs):
    tf_type = dtype_pytorch2keras(dtype)
    return tf.cast(self, tf_type)


@converter(torch.Tensor.type, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def type_converter(self, dtype=None, non_blocking=False, **kwargs):
    return type_func


@converter(torch.Tensor.to, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def to(self, device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format):
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


@converter(torch.eq, torch.equal, torch.Tensor.__eq__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def eq(input: Tensor, other: Number, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.equal(input, other)
    return func


@converter(torch.Tensor.__and__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def logical_and(input: Tensor, other):
    def func(input, other):
        return tf.logical_and(input, other)
    return func


@converter(torch.Tensor.__or__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def logical_or(input: Tensor, other):
    def func(input, other):
        return tf.logical_or(input, other)
    return func


@converter(torch.Tensor.topk, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def topk(self, k, dim=None, largest=True, sorted=True):
    def func(self, k, dim=None, largest=True, sorted=True):
        result = tf.math.top_k(self, k=k, sorted=sorted)
        indices = tf.cast(result.indices, tf.int64)
        return result.values, indices
    return func


@converter(torch.Tensor.sort, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def sort(self, dim=-1, descending=False):
    n_dims = self.dim()

    def func(self, dim=-1, descending=False):
        if get_channel_order(self) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, n_dims)
        if descending:
            direction = 'DESCENDING'
        else:
            direction = 'ASCENDING'
        return tf.sort(self, axis=dim, direction=direction)
    return func


@converter(torch.unique, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    def func(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
        assert len(input.shape) == 1
        assert return_inverse is False

        x, _ = tf.unique(input)
        if sorted:
            x = tf.sort(x)
        return x
    return func


@converter(torch.clone, torch.Tensor.clone, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def clone(input: Tensor, *, memory_format=None):
    def func(input: Tensor, *, memory_format=None):
        return tf.identity(input)
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
def force_tensorflow_order(inputs):
    return lambda inputs: inputs


@converter(force_pytorch_order, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def force_pytorch_order(inputs):
    return lambda inputs: inputs
