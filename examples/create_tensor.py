import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=(1, 1))

    def forward(self, x):
        shape = x.shape
        y1 = torch.zeros(shape)
        y2 = torch.ones(shape)
        y3 = torch.empty(shape)
        y4 = torch.full(shape, fill_value=42)

        y5 = x.new_zeros(shape)
        y6 = x.new_ones(shape)
        y7 = x.new_empty(shape)
        y8 = x.new_full(shape, fill_value=42)

        y9 = torch.zeros_like(x)
        y10 = torch.ones_like(x)
        y11 = torch.empty_like(x)
        y12 = torch.full_like(x, fill_value=42)

        y13 = x.zero_()
        y14 = x.fill_(value=42)

        y15 = torch.tensor(x)
        y16 = torch.as_tensor(x)

        return y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16


input = torch.normal(0, 1, size=(1, 3, 128, 128))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[input],
    input_shapes={input: (None, 3, None, None)},  # Annotate dynamic axes with None
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
    trace_shape=True,
)

model_path = 'create_tensor'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')

converter = TFLiteConverter.from_keras_model_file(model_path + '.h5', custom_objects=custom_objects)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
