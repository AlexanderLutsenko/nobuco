from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor
from torch import nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow import keras

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(nn.modules.activation.MultiheadAttention, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_MultiheadAttention(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False):

    assert self._qkv_same_embed_dim, 'Different embed dims are not supported yet'
    assert self.batch_first, 'batch_first=False is not supported yet'

    dropout = self.dropout

    num_heads = self.num_heads
    head_dim = self.head_dim
    embed_dim = self.embed_dim

    key_dim = self.kdim // num_heads
    value_dim = self.vdim // num_heads
    use_bias = self.in_proj_bias is not None

    in_proj_weight = self.in_proj_weight.cpu().detach().numpy()
    in_proj_bias = self.in_proj_bias.cpu().detach().numpy()

    w1, w2, w3 = in_proj_weight.reshape(3, num_heads, head_dim, embed_dim).transpose(0, 3, 1, 2)
    b1, b2, b3 = in_proj_bias.reshape(3, num_heads, head_dim)

    w4 = self.out_proj.weight.cpu().detach().numpy()
    w4 = w4.reshape(embed_dim, num_heads, head_dim).transpose(1, 2, 0)
    b4 = self.out_proj.bias.cpu().detach().numpy()

    params = [w1, b1, w2, b2, w3, b3, w4, b4]

    def func(
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False
    ):
        layer = keras.layers.MultiHeadAttention(num_heads, key_dim, value_dim=value_dim, use_bias=use_bias, dropout=dropout)
        layer(query, value, key=key, attention_mask=attn_mask, return_attention_scores=need_weights, use_causal_mask=is_causal)
        layer.set_weights(params)
        output = layer(query, value, key=key, attention_mask=attn_mask, return_attention_scores=need_weights, use_causal_mask=is_causal)
        return output
    return func


def tril(h, w):
    y = tf.range(0, h)[:, None]
    x = tf.range(0, w)[None, :]
    return y >= x


@converter(F.scaled_dot_product_attention, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    def func(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        L, D = tf.shape(query)[-2:]
        S = tf.shape(key)[-2]

        if scale is None:
            scale = tf.cast(D, query.dtype) ** -0.5

        # Corby's numerically more stable attention
        # See: https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118
        s_scale = tf.cast(tf.sqrt(scale), query.dtype)
        query = query * s_scale
        key = key * s_scale

        sim = query @ tf.experimental.numpy.swapaxes(key, -2, -1)

        if attn_mask is not None:
            sim = tf.where(attn_mask, sim, float("-inf"))
        elif is_causal:
            causal_mask = tril(L, S)
            sim = tf.where(causal_mask, sim, float("-inf"))

        attn = tf.nn.softmax(sim, axis=-1)
        attn = keras.layers.Dropout(dropout_p)(attn)
        return attn @ value
    return func
