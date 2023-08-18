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


class DynamicShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)

        # Produces static shape
        # b, c, h, w = x.shape

        # Allows for dynamic shape
        b, c, h, w = nobuco.shape(x)

        x = x[:, :, h//3:, w//3:]
        return x


input = torch.normal(0, 1, size=(1, 3, 128, 128))
pytorch_module = DynamicShape().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[input],
    input_shapes={input: (None, 3, None, None)}, # Annotate dynamic axes with None
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
    trace_shape=False,
    save_trace_html=True,
)

model_path = 'dynamic_shape'
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
