import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from nobuco.convert.converter import pytorch_to_keras
from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter

from onnx_tf.backend_rep import TensorflowRep

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteKerasModelConverterV2
from tensorflow import keras

import torch
from torch import nn

import onnx
from onnx_tf.backend import prepare


class AddByMask(nn.Module):
    def forward(self, x, mask):
        x[mask] += 1
        return x


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        x[x > 0] += 1
        # mask = x > 0
        # AddByMask()(x, mask)
        return x


# class MyModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))
#
#     def forward(self, x):
#         x = self.conv(x)
#         x[x > 0] += 1
#         return x


@converter(AddByMask, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER, reusable=False)
def converterAddByMask(self, x, mask):
    model_path = 'add_by_mask'
    onnx_path = model_path + '.onnx'

    # NB: onnx.export in implemented via tracing i.e. it may modify the inputs!
    torch.onnx.export(self, (x, mask), onnx_path, opset_version=12, input_names=['input', 'mask'], dynamic_axes={'input': [0, 1, 2, 3]})

    onnx_model = onnx.load(onnx_path)
    tf_rep: TensorflowRep = prepare(onnx_model)
    tf_rep.export_graph(model_path)
    model = tf.keras.models.load_model(model_path)
    return keras.layers.Lambda(lambda x, mask: model(input=x, mask=mask))


args = [
    torch.normal(0, 1, size=(1, 3, 128, 128))
]
pytorch_module = MyModule().eval()

keras_model = pytorch_to_keras(
    pytorch_module,
    args=args,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
    save_trace_html=True
)

model_path = 'converter_inside_converter'
converter = TFLiteKerasModelConverterV2(keras_model)
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
