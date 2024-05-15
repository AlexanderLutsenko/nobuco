from dataclasses import dataclass

import torch
import tensorflow as tf
from nobuco.converters.type_cast import dtype_pytorch2keras

from nobuco.commons import ChannelOrder, TF_TENSOR_CLASSES
from nobuco.converters.tensor import _permute, perm_pytorch2keras, perm_keras2pytorch
from nobuco.util import replace_recursively_func


def set_channel_order(tensor, channel_order: ChannelOrder):
    tensor.channel_order = channel_order
    return tensor


def get_channel_order(tensor) -> ChannelOrder:
    return tensor.channel_order


def t_keras2pytorch(tensor_tf, restore_channel_order=False):
    if restore_channel_order and get_channel_order(tensor_tf) == ChannelOrder.TENSORFLOW:
        tensor_tf = _permute(perm_keras2pytorch(len(tensor_tf.shape)))(tensor_tf)

    if hasattr(tensor_tf, 'output_min'):
        return t_keras2pytorch_quant(tensor_tf)
    else:
        return t_keras2pytorch_normal(tensor_tf)


def t_keras2pytorch_quant(tensor_tf):
    min_range = tensor_tf.output_min
    max_range = tensor_tf.output_max
    tensor_tf = tf.quantization.dequantize(tensor_tf, min_range, max_range)
    tensor_pt = torch.as_tensor(tensor_tf.numpy())
    return tensor_pt


def t_keras2pytorch_normal(tensor_tf):
    tensor_pt = torch.as_tensor(tensor_tf.numpy())
    return tensor_pt


def t_pytorch2keras(tensor_pt: torch.Tensor, channel_order=ChannelOrder.PYTORCH):
    if tensor_pt.is_quantized:
        tensor_tf = t_pytorch2keras_quant(tensor_pt, channel_order)
    else:
        tensor_tf = t_pytorch2keras_normal(tensor_pt, channel_order)
    set_channel_order(tensor_tf, channel_order=channel_order)
    return tensor_tf


def t_pytorch2keras_quant(tensor_pt, channel_order=ChannelOrder.PYTORCH):
    zero = tensor_pt.q_zero_point()
    scale = tensor_pt.q_scale()

    min_q = -zero * scale
    max_q = (255 - zero) * scale

    tensor_pt_dequant = torch.dequantize(tensor_pt)
    tensor_tf_dequant = t_pytorch2keras_normal(tensor_pt_dequant, channel_order)
    q = tf.quantization.quantize(tensor_tf_dequant, min_q, max_q, dtype_pytorch2keras(tensor_pt.dtype))
    tensor_tf = q.output
    tensor_tf.output_min = q.output_min
    tensor_tf.output_max = q.output_max
    return tensor_tf


def t_pytorch2keras_normal(tensor_pt, channel_order=ChannelOrder.PYTORCH):
    tensor_tf = tf.convert_to_tensor(tensor_pt.detach().cpu().numpy())
    if channel_order == ChannelOrder.TENSORFLOW:
        tensor_tf = _permute(perm_pytorch2keras(tensor_pt.dim()))(tensor_tf)
    return tensor_tf


def pytorch2keras_recursively(obj, channel_order=ChannelOrder.PYTORCH):

    def collect_func(obj):
        return isinstance(obj, torch.Tensor)

    def replace_func(obj):
        return t_pytorch2keras(obj, channel_order=channel_order)

    return replace_recursively_func(obj, collect_func, replace_func)


def keras2pytorch_recursively(obj, restore_channel_order=False):

    def collect_func(obj):
        return isinstance(obj, TF_TENSOR_CLASSES)

    def replace_func(obj):
        return t_keras2pytorch(obj, restore_channel_order=restore_channel_order)

    return replace_recursively_func(obj, collect_func, replace_func)


@dataclass
class TensorPlaceholder:
    idx: int


def make_template_recursively(obj):
    i = 0

    def collect_func(obj):
        return isinstance(obj, torch.Tensor)

    def replace_func(obj):
        nonlocal i
        placeholder = TensorPlaceholder(i)
        i += 1
        return placeholder

    return replace_recursively_func(obj, collect_func, replace_func)


def template_insert_recursively(obj, tensors):

    def collect_func(obj):
        return isinstance(obj, TensorPlaceholder)

    def replace_func(obj):
        return tensors[obj.idx]

    return replace_recursively_func(obj, collect_func, replace_func)
