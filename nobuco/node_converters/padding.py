from typing import Optional, Union, List, Tuple, Sequence, Any

import torch
from torch import Tensor
import torch.nn.functional as F

import tensorflow as tf

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import permute_pytorch2keras


def tf_pad_replicate(x, pad):
    for dim, (pl, pr) in enumerate(pad):
        if pl != 0 or pr != 0:
            slice_left = [slice(None)]*dim + [slice(None, 1)]
            pad_left = x.__getitem__(slice_left)
            pad_left = tf.repeat(pad_left, pl, axis=dim)

            slice_right = [slice(None)]*dim + [slice(-1, None)]
            pad_right = x.__getitem__(slice_right)
            pad_right = tf.repeat(pad_right, pr, axis=dim)

            x = tf.concat([pad_left, x, pad_right], axis=dim)
    return x


def tf_pad_circular(x, pad):
    for dim, (pl, pr) in enumerate(pad):
        if pl != 0 or pr != 0:
            slice_left = [slice(None)]*dim + [slice(-pl, None)]
            pad_left = x.__getitem__(slice_left)

            slice_right = [slice(None)]*dim + [slice(None, pr)]
            pad_right = x.__getitem__(slice_right)

            x = tf.concat([pad_left, x, pad_right], axis=dim)
    return x


# TODO: add support for 'negative' paddings
@converter(F.pad, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_pad(input: Tensor, pad: List[int], mode: str = "constant", value: float = 0.0):
    n_dims = input.dim()

    def func(input, pad, mode="constant", value=0.0):
        pad_dims = len(pad) // 2
        assert len(pad) % 2 == 0
        assert pad_dims <= n_dims

        pad_full = []
        for i in range(pad_dims):
            pad_full.append(pad[2 * i:2 * i + 2])
        for i in range(n_dims - pad_dims):
            pad_full.append([0, 0])
        pad_full = list(reversed(pad_full))

        pad = pad_full
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            pad = permute_pytorch2keras(pad)

        if mode == 'replicate':
            x = tf_pad_replicate(input, pad)
        elif mode == 'circular':
            x = tf_pad_circular(input, pad)
        else:
            x = tf.pad(input, pad, mode=mode.upper(), constant_values=value)
        return x
    return func
