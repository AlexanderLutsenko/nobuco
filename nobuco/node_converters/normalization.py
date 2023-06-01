from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor

from tensorflow import keras
import torch
import torch.nn.functional as F
from torch import nn

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


# NB: tensorflow and pytorch implementations of batchnorm behave differently in train mode
@converter(nn.BatchNorm1d, nn.BatchNorm2d)
def converter_BatchNorm(self, input: Tensor):
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


@converter(F.layer_norm, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_layer_norm(input: Tensor,
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
