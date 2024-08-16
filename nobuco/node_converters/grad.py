import torch
from torch import Tensor

import tensorflow as tf

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(torch.Tensor.detach, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_detach(self):
    def func(self):
        return tf.stop_gradient(self)
    return func


@converter(torch.Tensor.requires_grad, torch.Tensor.requires_grad_, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_requires_grad(self, requires_grad):
    def func(self, requires_grad):
        if requires_grad:
            return self
        else:
            return tf.stop_gradient(self)
    return func
