from typing import Optional

import torch
from torch import Tensor

import tensorflow as tf

from nobuco.commons import ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(torch.Tensor.__getattribute__, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_getattr(self: Tensor, name: str, *, out: Optional[Tensor] = None):
    def func(self, name: str, *, out: Optional[Tensor] = None):
        if name == "real":
            return tf.math.real(self)
        elif name == "imag":
            return tf.math.imag(self)
        else:
            raise AttributeError(f"'Tensor' object has no attribute '{name}'")
    return func
