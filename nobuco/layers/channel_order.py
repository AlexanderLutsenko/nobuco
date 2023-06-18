from nobuco.commons import ChannelOrderingStrategy, ChannelOrder, TF_TENSOR_CLASSES
from nobuco.converters.channel_ordering import set_channel_order, get_channel_order
from nobuco.converters.tensor import _permute, perm_keras2pytorch, perm_pytorch2keras
from nobuco.converters.type_cast import tf_cast_recursively
from nobuco.util import collect_recursively, replace_recursively_func


class ChangeOrderingLayer:
    def __init__(self, func, channel_ordering_strategy, autocast=False):
        self.func = func
        self.channel_ordering_strategy = channel_ordering_strategy
        self.autocast = autocast

    def __call__(self, *args, **kwargs):
        tf_assert_has_attr_recursively((args, kwargs), 'channel_order')

        strategy = self.channel_ordering_strategy

        if strategy == ChannelOrderingStrategy.MANUAL:
            outputs = self.func(*args, **kwargs)
        elif strategy == ChannelOrderingStrategy.OUTPUT_FORCE_PYTORCH_ORDER:
            outputs = self.func(*args, **kwargs)
            outputs = tf_annotate_recursively(outputs, channel_order=ChannelOrder.PYTORCH)
        else:
            if strategy == ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH:
                def is_single_el(shape):
                    return all(s == 1 for s in shape)

                input_tensors = collect_recursively((args, kwargs), TF_TENSOR_CLASSES)
                input_tensors_not_single = [t for t in input_tensors if is_single_el(t.shape) != 1]
                # All tensors are of same dimensionality except tensors of size 1
                all_same = len({len(t.shape) for t in input_tensors_not_single}) == 1
                if all_same:
                    strategy = ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS
                else:
                    strategy = ChannelOrderingStrategy.FORCE_PYTORCH_ORDER

            if strategy == ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER:
                channel_order = ChannelOrder.TENSORFLOW
            elif strategy == ChannelOrderingStrategy.FORCE_PYTORCH_ORDER:
                channel_order = ChannelOrder.PYTORCH
            elif strategy == ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS:
                input_tensors = collect_recursively((args, kwargs), TF_TENSOR_CLASSES)
                num_reordered = sum(t.channel_order == ChannelOrder.TENSORFLOW for t in input_tensors)
                num_unchanged = len(input_tensors) - num_reordered
                if num_reordered >= num_unchanged:
                    channel_order = ChannelOrder.TENSORFLOW
                else:
                    channel_order = ChannelOrder.PYTORCH

            args, kwargs = tf_set_order_recursively((args, kwargs), channel_order=channel_order)

            if self.autocast:
                args, kwargs = tf_cast_recursively((args, kwargs))

            outputs = self.func(*args, **kwargs)
            outputs = tf_annotate_recursively(outputs, channel_order=channel_order)
        tf_assert_has_attr_recursively(outputs, 'channel_order')
        return outputs

    def __str__(self):
        return f"{self.__class__.__name__}(func={self.func})"


def tf_set_order_recursively(obj, channel_order: ChannelOrder):

    def collect_func(obj):
        return isinstance(obj, TF_TENSOR_CLASSES)

    def replace_func(obj):
        if channel_order == ChannelOrder.TENSORFLOW and get_channel_order(obj) != ChannelOrder.TENSORFLOW:
            n_dims = len(obj.shape)
            obj = _permute(perm_pytorch2keras(n_dims))(obj)
        elif channel_order == ChannelOrder.PYTORCH and get_channel_order(obj) != ChannelOrder.PYTORCH:
            n_dims = len(obj.shape)
            obj = _permute(perm_keras2pytorch(n_dims))(obj)
        set_channel_order(obj, channel_order)
        return obj

    return replace_recursively_func(obj, collect_func, replace_func)


def tf_annotate_recursively(obj, channel_order):

    def collect_func(obj):
        return isinstance(obj, TF_TENSOR_CLASSES)

    def replace_func(obj):
        set_channel_order(obj, channel_order)
        return obj

    return replace_recursively_func(obj, collect_func, replace_func)


def tf_assert_has_attr_recursively(obj, attr):

    def collect_func(obj):
        return isinstance(obj, TF_TENSOR_CLASSES)

    def replace_func(obj):
        assert hasattr(obj, attr)
        return obj

    return replace_recursively_func(obj, collect_func, replace_func)
