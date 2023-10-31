import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F
from torch import nn

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


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
