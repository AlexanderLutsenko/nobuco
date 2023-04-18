from nobuco.commons import ChannelOrderingStrategy
from nobuco.converters.node_converter import converter
from nobuco.trace.trace import traceable


@traceable
def force_tensorflow_order(inputs):
    return inputs


@traceable
def force_pytorch_order(inputs):
    return inputs


@converter(force_tensorflow_order, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_force_tensorflow_order(inputs):
    return lambda inputs: inputs


@converter(force_pytorch_order, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_force_pytorch_order(inputs):
    return lambda inputs: inputs
