import torch

import tensorflow as tf

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter
from nobuco.node_converters.tensor_manipulation import _permute_inner


@converter(torch.Tensor.__getattribute__, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_getattribute(self, attribute: str):
    try:
        if attribute == 'T':
            def func(self, attribute):
                return _permute_inner([1, 0])(self)
        elif attribute == 'data':
            def func(self, attribute):
                return self
        else:
            def func(self, attribute):
                return tf.math.__getattribute__(attribute)(self)
    except:
        raise Exception(f'Unsupported attribute: {attribute}')
    return func
