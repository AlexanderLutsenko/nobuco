from typing import Optional, Union, List, Tuple, Sequence, Any

from torch import Tensor
from torch import nn

import tensorflow as tf

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(nn.modules.activation.MultiheadAttention, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_sum(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False):

    assert self._qkv_same_embed_dim, 'Different embed dims are not supported yet'

    dropout = self.dropout

    num_heads = self.num_heads
    head_dim = self.head_dim
    embed_dim = self.embed_dim

    key_dim = self.kdim // num_heads
    value_dim = self.vdim // num_heads
    use_bias = self.in_proj_bias is not None

    in_proj_weight = self.in_proj_weight.detach().numpy()
    in_proj_bias = self.in_proj_bias.detach().numpy()

    w1, w2, w3 = in_proj_weight.reshape(3, num_heads, head_dim, embed_dim).transpose(0, 3, 1, 2)
    b1, b2, b3 = in_proj_bias.reshape(3, num_heads, head_dim)

    w4 = self.out_proj.weight.detach().numpy()
    w4 = w4.reshape(embed_dim, num_heads, head_dim).transpose(1, 2, 0)
    b4 = self.out_proj.bias.detach().numpy()

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
        layer = tf.keras.layers.MultiHeadAttention(num_heads, key_dim, value_dim=value_dim, use_bias=use_bias, dropout=dropout)
        layer(query, value, key=key, attention_mask=attn_mask, return_attention_scores=need_weights, use_causal_mask=is_causal)
        layer.set_weights(params)
        output = layer(query, value, key=key, attention_mask=attn_mask, return_attention_scores=need_weights, use_causal_mask=is_causal)
        return output
    return func
