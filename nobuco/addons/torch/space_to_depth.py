import torch
from torch import nn
import torch.nn.functional as F

import tensorflow as tf

import nobuco
from nobuco import ChannelOrderingStrategy
from nobuco.addons.torch.util import channel_interleave2d


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, input):
        x = F.pixel_unshuffle(input, self.block_size)
        x = channel_interleave2d(x, self.block_size, reverse=False)
        return x


@nobuco.converter(SpaceToDepth, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_space_to_depth(self, input: torch.Tensor):
    def func(input):
        return tf.nn.space_to_depth(input, self.block_size)
    return func
