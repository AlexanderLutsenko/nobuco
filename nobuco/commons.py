from enum import Enum
import tensorflow as tf
from keras.src.engine.keras_tensor import KerasTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

CONVERTER_DICT = {}

CONVERTED_OP_DICT = {}


TF_TENSOR_CLASSES = (
    tf.Tensor,
    tf.Variable,
    ResourceVariable,
    KerasTensor,
)


class TraceLevel(Enum):
    NEVER = 0
    DEFAULT = 1
    ALWAYS = 2


class ChannelOrder(Enum):
    PYTORCH = 1
    CHANNEL_FIRST = 1

    TENSORFLOW = 2
    CHANNEL_LAST = 2


class ChannelOrderingStrategy(Enum):
    FORCE_TENSORFLOW_ORDER = 1
    FORCE_PYTORCH_ORDER = 2
    MINIMUM_TRANSPOSITIONS = 3
    MINIMUM_TRANSPOSITIONS_OR_PYTORCH = 4
    MANUAL = 5
    OUTPUT_FORCE_PYTORCH_ORDER = 6


class ConnectivityStatus:
    def __init__(self, unused_inputs, unreached_outputs, unused_nodes, unprovided_inputs):
        self.unused_inputs = unused_inputs
        self.unreached_outputs = unreached_outputs
        self.unused_nodes = unused_nodes
        self.unprovided_inputs = unprovided_inputs

    def is_connected(self):
        return len(self.unreached_outputs) == 0
