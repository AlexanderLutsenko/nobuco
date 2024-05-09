import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
import keras


class MyModule(nn.Module):
    def forward(self, x, y, z):
        x1, y1, z1 = torch.broadcast_tensors(x, y, z)
        z2, x2 = torch.broadcast_tensors(z, x1)
        z3 = torch.broadcast_to(z, x1.shape)
        return x1, y1, z1, z2, z3


x = torch.normal(0, 1, size=(128,))
y = torch.normal(0, 1, size=(8, 3, 1, 128))
z = torch.normal(0, 1, size=(8, 3, 96, 1))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x, y, z],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
    input_shapes={x: (8, None, None, None), y: (8, None, None, None), z: (8, None, None, None)},
    trace_shape=True,
)

model_path = 'broadcast'
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
