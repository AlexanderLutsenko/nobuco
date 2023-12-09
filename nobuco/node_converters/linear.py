from numbers import Number
from typing import Optional, Union, List, Tuple, Sequence, Any

from nobuco.converters.tensor import dim_pytorch2keras
from nobuco.converters.channel_ordering import get_channel_order
from torch import Tensor

import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(nn.Linear, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_Linear(self, input: Tensor):
    out_filters, in_filters = self.weight.shape
    weights = self.weight.cpu().detach().numpy()
    weights = weights.transpose(1, 0)

    use_bias = self.bias is not None
    if use_bias:
        biases = self.bias.cpu().detach().numpy()
        params = [weights, biases]
    else:
        params = [weights]
    return keras.layers.Dense(out_filters, use_bias=use_bias, weights=params)


@converter(torch.nn.functional.linear, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_linear(input, weight, bias=None, out=None):
    out_filters, in_filters = weight.shape
    weights = weight.cpu().detach().numpy()
    weights = weights.transpose(1, 0)

    use_bias = bias is not None
    if use_bias:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
    else:
        params = [weights]

    layer = keras.layers.Dense(out_filters, use_bias=use_bias, weights=params)

    def func(input, weight, bias=None, out=None):
        return layer(input)
    return func


@converter(torch.matmul, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor]=None):
    def func(input, other, *, out=None):
        return tf.linalg.matmul(input, other)
    return func


@converter(torch.Tensor.matmul, torch.Tensor.__matmul__, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_matmul(self, tensor2):
    def func(self, tensor2):
        return tf.linalg.matmul(self, tensor2)
    return func


@converter(torch.dot, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_dot(input: Tensor, tensor: Tensor, *, out: Optional[Tensor]=None):
    def func(input, tensor, *, out=None):
        return tf.linalg.tensordot(input, tensor, axes=1)
    return func


@converter(torch.mv, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_mv(input: Tensor, vec: Tensor, *, out: Optional[Tensor]=None):
    def func(input, vec, *, out=None):
        return tf.linalg.tensordot(input, vec, axes=1)
    return func


@converter(torch.bmm, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_bmm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor]=None):
    def func(input, mat2, *, out=None):
        return tf.linalg.matmul(input, mat2)
    return func


@converter(torch.baddbmm, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_baddbmm(input: Tensor, batch1: Tensor, batch2: Tensor, *, beta: Number=1, alpha: Number=1, out: Optional[Tensor]=None):
    def func(input: Tensor, batch1, batch2, *, beta=1, alpha=1, out=None):
        return beta*input + alpha*tf.linalg.matmul(batch1, batch2)
    return func


@converter(torch.einsum, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_einsum(*args: Any):
    def func(*args: Any):
        equation = args[0]
        operands = args[1:]
        return keras.layers.Lambda(lambda operands: tf.einsum(equation, *operands))(operands)
    return func


@converter(torch.Tensor.triu, torch.Tensor.triu_, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_triu(self, diagonal=0):
    def func(self, diagonal=0):
        return keras.layers.Lambda(lambda x: tf.experimental.numpy.triu(x, k=diagonal))(self)
    return func


@converter(F.normalize, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out: Optional[Tensor] = None):
    num_dims = input.dim()

    def func(input, p=2.0, dim=1, eps=1e-12, out=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        norm = tf.norm(input, ord=p, axis=dim, keepdims=True)
        norm = tf.maximum(norm, eps)
        return input / norm
    return func


@converter(torch.norm, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
    num_dims = input.dim()

    def func(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
        if get_channel_order(input) == ChannelOrder.TENSORFLOW:
            dim = dim_pytorch2keras(dim, num_dims)
        return tf.norm(input, ord=p, axis=dim, keepdims=keepdim)
    return func


@converter(F.embedding, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None, max_norm: Optional[float] = None,
              norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False):
    input_dim, output_dim = weight.shape
    weight = weight.cpu().detach().numpy()

    layer = keras.layers.Embedding(input_dim, output_dim, weights=[weight])

    def func(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
        return layer(input)
    return func
