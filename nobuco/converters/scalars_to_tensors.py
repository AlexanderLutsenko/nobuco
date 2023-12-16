import numbers

import tensorflow as tf

from nobuco.commons import ChannelOrder, TF_TENSOR_CLASSES
from nobuco.converters.channel_ordering import set_channel_order
from nobuco.util import collect_recursively, replace_recursively_func


def tf_scalars_to_tensors_recursively(inputs):
    def collect_func(obj):
        return isinstance(obj, TF_TENSOR_CLASSES) or (isinstance(obj, numbers.Number) and not isinstance(obj, bool))

    def replace_func(obj):
        if not isinstance(obj, TF_TENSOR_CLASSES):
            obj = tf.convert_to_tensor(obj)
            obj = set_channel_order(obj, ChannelOrder.PYTORCH)
        return obj

    return replace_recursively_func(inputs, collect_func, replace_func)
