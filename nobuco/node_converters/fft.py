from typing import Optional

import torch
from nobuco.converters.channel_ordering import set_channel_order

from nobuco.node_converters.padding import converter_pad
from torch import nn, Tensor
import torch.nn.functional as F

import tensorflow as tf
import keras

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(torch.fft.rfft, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_rfft(input, n=None, dim=-1, norm=None, *, out=None):
    assert dim == -1
    assert norm is None

    def func(input, n=None, dim=-1, norm=None, *, out=None):
        return keras.layers.Lambda(lambda input: tf.signal.rfft(input, fft_length=n))(input)
    return func


@converter(torch.fft.irfft, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_irfft(input, n=None, dim=-1, norm=None, *, out=None):
    assert dim == -1
    assert norm is None

    def func(input, n=None, dim=-1, norm=None, *, out=None):
        return keras.layers.Lambda(lambda input: tf.signal.irfft(input, fft_length=n))(input)
    return func


@converter(torch.fft.rfft2, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    assert dim == (-2, -1)
    assert norm is None

    def func(input, s=None, dim=(-2, -1), norm=None, *, out=None):
        return keras.layers.Lambda(lambda input: tf.signal.rfft2d(input, fft_length=s))(input)
    return func


@converter(torch.fft.irfft2, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    assert dim == (-2, -1)
    assert norm is None

    def func(input, s=None, dim=(-2, -1), norm=None, *, out=None):
        return keras.layers.Lambda(lambda input: tf.signal.irfft2d(input, fft_length=s))(input)
    return func


@converter(torch.stft, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_stft(input: Tensor, n_fft: int, hop_length: Optional[int] = None,
         win_length: Optional[int] = None, window: Optional[Tensor] = None,
         center: bool = True, pad_mode: str = 'reflect', normalized: bool = False,
         onesided: Optional[bool] = None,
         return_complex: Optional[bool] = None):

    assert return_complex

    if center:
        signal_dim = input.dim()
        extended_shape = [1] * (3 - signal_dim) + list(input.size())
        pad = int(n_fft // 2)
        input = input.view(extended_shape)
        pad_tf = converter_pad.convert(input, [pad, pad], pad_mode)

    def func(input, n_fft: int, hop_length = None,
         win_length = None, window = None,
         center = True, pad_mode = 'reflect', normalized = False,
         onesided = None,
         return_complex = None):

        if center:
            input = tf.reshape(input, extended_shape)
            set_channel_order(input, ChannelOrder.PYTORCH)
            input = pad_tf(input, [pad, pad], pad_mode)
            input = tf.reshape(input, tf.shape(input)[-signal_dim:])

        def stft_fun(args):
            input, window = args
            window_fn = None
            if window is not None:
                window_fn = lambda *args, **kwargs: window
            return tf.signal.stft(input, frame_length=win_length, frame_step=hop_length, fft_length=n_fft, window_fn=window_fn)

        x = keras.layers.Lambda(stft_fun)([input, window])

        if not onesided:
            x = x[..., 1:]
            x_rev = tf.reverse(x, axis=(-1,))
            x = tf.concat([x, x_rev], axis=-1)

        x = tf.transpose(x, (0, 2, 1))

        if normalized:
            x = x / win_length**0.5

        return x

    return func