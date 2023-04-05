from typing import Callable

from nobuco.convert.layers.channel_order import ChangeOrderingLayer
from nobuco.convert.validation import validate_diff_default
from nobuco.commons import ChannelOrderingStrategy
from nobuco.trace.trace import Tracer

CONVERTER_DICT = {}


class Pytorch2KerasNodeConverter:
    def __call__(self, *args, **kwargs):
        keras_op = self.convert(*args, **kwargs)
        return keras_op

    def convert(self, *args, **kwargs):
        raise NotImplementedError()

    def validate(self, keras_op, pytorch_op, input_tensors_pt, args_pt, kwargs_pt, is_training=False):
        raise NotImplementedError()


class Pytorch2KerasLambdaConverter(Pytorch2KerasNodeConverter):
    def __init__(self, convert_func, validate_func, reusable):
        self.convert_func = convert_func
        self.validate_func = validate_func
        self.reusable = reusable

    def convert(self, *args, **kwargs):
        return self.convert_func(*args, **kwargs)

    def validate(self, keras_op, pytorch_op, input_tensors_pt, args_pt, kwargs_pt, is_training=False):
        raise self.validate_func(keras_op, pytorch_op, input_tensors_pt, args_pt, kwargs_pt, is_training=False)


def converter(*ops,
              validate_func=validate_diff_default,
              channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER,
              autocast: bool = False,
              reusable = True,
              converter_dict=None,
              ):
    if converter_dict is None:
        converter_dict = CONVERTER_DICT

    def channel_ordering_decorator(converter_func, channel_ordering_strategy):
        def decorator(*args, **kwargs):
            converter_result_func = converter_func(*args, **kwargs)
            return ChangeOrderingLayer(converter_result_func, channel_ordering_strategy, autocast)
        return decorator

    def inner(convert_func: Callable) -> Callable:
        convert_func = channel_ordering_decorator(convert_func, channel_ordering_strategy)
        node_converter = Pytorch2KerasLambdaConverter(convert_func, validate_func, reusable)
        for op in ops:
            op = Tracer.op_unwrap(op)
            converter_dict[op] = node_converter
        return node_converter

    return inner


def converter_unregister(op, converter_dict=None):
    if converter_dict is None:
        converter_dict = CONVERTER_DICT

    op = Tracer.op_unwrap(op)
    if op in converter_dict:
        del converter_dict[op]