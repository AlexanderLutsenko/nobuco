import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteKerasModelConverterV2, TFLiteKerasModelConverter
import keras

import onnx
from onnx_tf.backend import prepare

import numpy as np


class SliceReLU(nn.Module):
    def forward(self, x):
        # Gives incorrect result after conversion
        torch.relu_(x[:, 1:2, 16:25, 8::2])
        # That's the recommended approach, but we're not going for it now
        # x[:, 1:2, 16:25, 8::2] = torch.relu_(x[:, 1:2, 16:25, 8::2])
        return x


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        SliceReLU()(x)
        return x


@nobuco.converter(SliceReLU, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER, reusable=False)
def converter_SliceReLU(self, x):
    model_path = 'slice_relu'
    onnx_path = model_path + '.onnx'

    # NB: onnx.export in implemented via tracing i.e. it may modify the inputs!
    torch.onnx.export(self, (x,), onnx_path, opset_version=12, input_names=['input'],
                      dynamic_axes={'input': [0, 1, 2, 3]}
                      )
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(model_path)
    model = tf.keras.models.load_model(model_path)
    return keras.layers.Lambda(lambda x: model(input=x))


x = torch.normal(0, 1, size=(1, 3, 128, 128))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'converter_inside_converter'
converter = TFLiteKerasModelConverterV2(keras_model)
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)


input_np = np.random.normal(size=(1, 128, 128, 3)).astype(np.float32)
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

n = 1000
start = time.time()
for _ in range(n):
    interpreter.set_tensor(input_details[0]['index'], input_np)
    interpreter.invoke()
print('TFLite elapsed: ', (time.time() - start) / n)
