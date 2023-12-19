import sys

import tensorflow as tf
import torch

from nobuco.commons import TF_TENSOR_CLASSES
from nobuco.util import collect_recursively, replace_recursively_func


TF_TYPE_PRIORITY_LIST = [
    tf.complex128,
    tf.complex64,

    tf.float64,
    tf.float32,
    tf.float16,

    tf.int64,
    tf.int32,
    tf.int16,
    tf.int8,
]


def dtype_pytorch2keras(dtype):
    if dtype == torch.float16:
        tf_type = tf.float16
    elif dtype == torch.float32:
        tf_type = tf.float32
    elif dtype == torch.float64:
        tf_type = tf.float64
    elif dtype == torch.int8:
        tf_type = tf.int8
    elif dtype == torch.int16:
        tf_type = tf.int16
    elif dtype == torch.int32:
        tf_type = tf.int32
    elif dtype == torch.int64:
        tf_type = tf.int64
    elif dtype == torch.uint8:
        tf_type = tf.uint8
    elif dtype == torch.bool:
        tf_type = tf.bool
    elif dtype == torch.complex64:
        tf_type = tf.complex64
    elif dtype == torch.complex128:
        tf_type = tf.complex128
    elif dtype is None:
        tf_type = None
    else:
        raise Exception('Unsupported dtype: ', dtype)
    return tf_type


def tf_autocast_recursively(inputs, type_priority_list=None):
    if type_priority_list is None:
        type_priority_list = TF_TYPE_PRIORITY_LIST

    tensors = collect_recursively(inputs, TF_TENSOR_CLASSES)
    min_priority = sys.maxsize
    for t in tensors:
        if t.dtype in type_priority_list:
            p = type_priority_list.index(t.dtype)
            min_priority = min(min_priority, p)

    if min_priority >= len(type_priority_list):
        return inputs

    target_dtype = type_priority_list[min_priority]

    def collect_func(obj):
        return isinstance(obj, TF_TENSOR_CLASSES)

    def replace_func(obj):
        if obj.dtype in type_priority_list and obj.dtype != target_dtype:
            obj_cast = tf.cast(obj, target_dtype)
            if hasattr(obj, 'channel_order'):
                obj_cast.channel_order = obj.channel_order
            return obj_cast
        else:
            return obj

    return replace_recursively_func(inputs, collect_func, replace_func)


def tf_cast_recursively(obj, types):
    assert len(collect_recursively(obj, TF_TENSOR_CLASSES)) == len(types)

    i = 0

    def collect_func(obj):
        return isinstance(obj, TF_TENSOR_CLASSES)

    def replace_func(tf_tensor):
        nonlocal i
        if tf_tensor.dtype != types[i]:
            tf_tensor_cast = tf.cast(tf_tensor, types[i])
            if hasattr(obj, 'channel_order'):
                tf_tensor_cast.channel_order = tf_tensor.channel_order
            tf_tensor = tf_tensor_cast
        i += 1
        return tf_tensor

    return replace_recursively_func(obj, collect_func, replace_func)
