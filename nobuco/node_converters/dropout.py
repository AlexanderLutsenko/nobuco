from torch import Tensor

from tensorflow import keras
import torch
import torch.nn.functional as F

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(torch.nn.modules.dropout.Dropout, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_Dropout(self, input: Tensor):
    return keras.layers.Dropout(rate=self.p)


@converter(F.dropout, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    def func(input, p=0.5, training=True, inplace=False):
        return keras.layers.Dropout(rate=p)(input)
    return func


@converter(torch.nn.modules.dropout.Dropout2d, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_Dropout(self, input: Tensor):
    return keras.layers.SpatialDropout2D(rate=self.p)


@converter(F.dropout2d, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    def func(input, p=0.5, training=True, inplace=False):
        return keras.layers.SpatialDropout2D(rate=p)(input)
    return func
