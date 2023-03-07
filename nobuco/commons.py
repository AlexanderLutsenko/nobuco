from enum import Enum
import tensorflow as tf
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable


CONVERTED_OP_DICT = {}


TF_TENSOR_CLASSES = (
    tf.Tensor,
    tf.Variable,
    ResourceVariable,
    KerasTensor,
)
try:
    from keras.engine.keras_tensor import KerasTensor as KerasTensor2
    TF_TENSOR_CLASSES = (*TF_TENSOR_CLASSES, KerasTensor2)
except Exception:
    pass


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
