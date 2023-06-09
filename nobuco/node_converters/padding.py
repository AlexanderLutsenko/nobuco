from typing import Optional, Union, List, Tuple, Sequence, Any
from torch import Tensor

import tensorflow as tf
import torch
import torch.nn.functional as F

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.node_converter import converter
from nobuco.converters.tensor import permute_pytorch2keras


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
        x = tf.pad(input, pad)
        return x
    return func
