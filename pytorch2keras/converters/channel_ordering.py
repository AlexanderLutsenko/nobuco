from dataclasses import dataclass

import tensorflow as tf
import torch

from pytorch2keras.commons import ChannelOrder
from pytorch2keras.converters.tensor import _permute, perm_pytorch2keras, perm_keras2pytorch
from pytorch2keras.util import replace_recursively_func


def set_channel_order(tensor, channel_order: ChannelOrder):
    tensor.channel_order = channel_order
    return tensor


def get_channel_order(tensor) -> ChannelOrder:
    return tensor.channel_order


def t_keras2pytorch(tensor_tf, restore_channel_order=False):
    if restore_channel_order and tensor_tf.channel_order == ChannelOrder.TENSORFLOW:
        tensor_tf = _permute(perm_keras2pytorch(len(tensor_tf.shape)))(tensor_tf)
    tensor_pt = torch.as_tensor(tensor_tf.numpy())
    return tensor_pt


def t_pytorch2keras(tensor_pt, channel_order=ChannelOrder.PYTORCH):
    tensor_tf = tf.convert_to_tensor(tensor_pt.detach().numpy())
    if channel_order == ChannelOrder.TENSORFLOW:
        tensor_tf = _permute(perm_pytorch2keras(tensor_pt.dim()))(tensor_tf)
    tensor_tf = set_channel_order(tensor_tf, channel_order=channel_order)
    return tensor_tf


def pytorch2keras_recursively(obj, channel_order=ChannelOrder.PYTORCH):
    def replace_func(obj):
        if isinstance(obj, torch.Tensor):
            tensor = t_pytorch2keras(obj, channel_order=channel_order)
            return True, tensor
        else:
            return False, obj

    return replace_recursively_func(obj, replace_func)


@dataclass
class TensorPlaceholder:
    idx: int


def make_template_recursively(obj):
    i = 0

    def replace_func(obj):
        if isinstance(obj, torch.Tensor):
            nonlocal i
            placeholder = TensorPlaceholder(i)
            i += 1
            return True, placeholder
        else:
            return False, obj
    return replace_recursively_func(obj, replace_func)


def template_insert_recursively(obj, tensors):
    def replace_func(obj):
        if isinstance(obj, TensorPlaceholder):
            tensor = tensors[obj.idx]
            return True, tensor
        else:
            return False, obj
    return replace_recursively_func(obj, replace_func)
