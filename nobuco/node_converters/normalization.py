from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor

import tensorflow as tf
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


@converter(nn.InstanceNorm1d, nn.InstanceNorm2d)
def converter_InstanceNorm(self, input: Tensor):
    assert not self.track_running_stats

    num_features = self.num_features
    epsilon = self.eps
    affine = self.affine

    if affine:
        weight = self.weight.detach().numpy()
        bias = self.bias.detach().numpy()
        params = [weight, bias]
    else:
        params = []

    layer = keras.layers.GroupNormalization(groups=num_features, axis=-1, epsilon=epsilon, scale=affine, center=affine, weights=params)
    return layer


@converter(nn.GroupNorm)
def converter_GroupNorm(self, input: Tensor):
    num_groups = self.num_groups
    epsilon = self.eps
    affine = self.affine

    if affine:
        weight = self.weight.detach().numpy()
        bias = self.bias.detach().numpy()
        params = [weight, bias]
    else:
        params = []

    layer = keras.layers.GroupNormalization(groups=num_groups, epsilon=epsilon, center=affine, scale=affine, weights=params)
    return layer


@converter(F.layer_norm, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_layer_norm(input: Tensor,
               normalized_shape: List[int],
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: float = 1e-5
               ):
    assert len(normalized_shape) == 1

    params = []
    if weight is not None:
        weight = weight.detach().numpy()
        params.append(weight)
    if bias is not None:
        bias = bias.detach().numpy()
        params.append(bias)

    layer = keras.layers.LayerNormalization(axis=-1, epsilon=eps, scale=weight is not None, center=bias is not None, weights=params)

    def func(input, *args, **kwargs):
        return layer(input)
    return func
