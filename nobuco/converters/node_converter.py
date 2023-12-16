from typing import Callable

from nobuco.converters.validation import validate_diff_default
from nobuco.commons import ChannelOrderingStrategy, CONVERTER_DICT
from nobuco.entity.pytorch import PytorchNode
from nobuco.layers.channel_order import ChangeOrderingLayer
from nobuco.trace.trace import Tracer


class Pytorch2KerasNodeConverter:
    def __init__(self, convert_func, validate_func, channel_ordering_strategy, autocast, reusable):
        self.convert_func = convert_func
        self.validate_func = validate_func
        self.channel_ordering_strategy = channel_ordering_strategy
        self.autocast = autocast
        self.reusable = reusable

    def convert(self, *args, _pytorch_node: PytorchNode = None, **kwargs):
        converter_result_func = self.convert_func(*args, **kwargs)
        if _pytorch_node is not None:
            output_types = _pytorch_node.output_types
        else:
            output_types = None
        return ChangeOrderingLayer(converter_result_func, self.channel_ordering_strategy, output_types=output_types, autocast=self.autocast)

    def validate(self, keras_op, pytorch_op, input_tensors_pt, args_pt, kwargs_pt, is_training=False):
        raise self.validate_func(keras_op, pytorch_op, input_tensors_pt, args_pt, kwargs_pt, is_training=False)


def converter(*ops,
              validate_func=validate_diff_default,
              channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER,
              autocast: bool = False,
              reusable: bool = True,
              ):
    def inner(convert_func: Callable) -> Pytorch2KerasNodeConverter:
        node_converter = Pytorch2KerasNodeConverter(convert_func, validate_func, channel_ordering_strategy, autocast=autocast, reusable=reusable)
        for op in ops:
            op = Tracer.op_undecorate(op)
            CONVERTER_DICT[op] = node_converter
        return node_converter
    return inner


def unregister_converter(op):
    op = Tracer.op_undecorate(op)
    if op in CONVERTER_DICT:
        del CONVERTER_DICT[op]
