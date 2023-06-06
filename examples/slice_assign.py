import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras

import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, indices, index_x, index_y):
        x[:] = 0
        x[:, :2] = 1
        x[:, :2, 3:11:2] = (x[:, :2, 3:11:2] + 2)
        x[:, :2, 3:11:2, 1] = torch.asarray(3)
        x[torch.asarray(2, dtype=torch.int32)] = 4
        x[:, 2, index_x, index_y] = 1

        d = x[indices]
        z = x[index_x, index_y]
        return y + x, d, z


x = torch.normal(0, 1, size=(8, 3, 128, 128))
y = torch.normal(0, 1, size=(8, 3, 128, 128))
index = torch.tensor([[[[0, 0, 1, 0], [0, 1, 2, 1]]]], dtype=torch.int64)
index_x = torch.tensor([2, 3, 4], dtype=torch.int64)
index_y = torch.tensor([0, 1, 2], dtype=torch.int64)
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x, y, index, index_x, index_y],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'slice_assign'
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
