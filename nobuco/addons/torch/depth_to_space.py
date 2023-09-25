import torch
from torch import nn
import torch.nn.functional as F

import tensorflow as tf

import nobuco
from nobuco import ChannelOrderingStrategy
from nobuco.addons.torch.util import channel_interleave2d


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, input):
        x = channel_interleave2d(input, self.block_size, reverse=True)
        x = F.pixel_shuffle(x, self.block_size)
        return x


@nobuco.converter(DepthToSpace, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_depth_to_space(self, input: torch.Tensor):
    def func(input):
        return tf.nn.depth_to_space(input, self.block_size)
    return func
