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


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y1 = x.split(split_size=1, dim=1)
        y2 = x.split(split_size=(48, 80), dim=2)
        y3 = torch.split(x, split_size_or_sections=(100, 128, 28), dim=3)
        return y1, y2, y3


model = MyModel()
x = torch.randn(1, 3, 128, 256)

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[x],
)

model_path = 'split'
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
