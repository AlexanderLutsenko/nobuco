import sys

import tensorflow as tf

from nobuco.commons import TF_TENSOR_CLASSES
from nobuco.util import collect_recursively, replace_recursively_func

TF_TYPE_PRIORITY_LIST = [
    tf.complex128,
    tf.complex64,

    tf.float64,
    tf.float32,
    # tf.float16,

    tf.int64,
    tf.int32,
    tf.int16,
    tf.int8,
]


def tf_cast_recursively(inputs, type_priority_list=None):
    if type_priority_list is None:
        type_priority_list = TF_TYPE_PRIORITY_LIST

    tensors = collect_recursively(inputs, TF_TENSOR_CLASSES)
    min_priority = sys.maxsize
    for t in tensors:
        if t.dtype in type_priority_list:
            p = type_priority_list.index(t.dtype)
        else:
            raise Exception(f'Unsupported dtype: {t.dtype}')
        min_priority = min(min_priority, p)

    if min_priority >= len(type_priority_list):
        return inputs

    target_dtype = type_priority_list[min_priority]

    def collect_func(obj):
        return isinstance(obj, TF_TENSOR_CLASSES)

    def replace_func(obj):
        if obj.dtype != target_dtype:
            obj_cast = tf.cast(obj, target_dtype)
            if hasattr(obj, 'channel_order'):
                obj_cast.channel_order = obj.channel_order
            return obj_cast
        else:
            return obj

    return replace_recursively_func(inputs, collect_func, replace_func)
