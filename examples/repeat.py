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


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, repeats):
        x = torch.repeat_interleave(x, repeats=repeats, dim=1)
        x = x.repeat_interleave(repeats=repeats, dim=3)
        return x


model = DummyModel()
x = torch.randn(1, 3, 128, 128) + 10
# repeats = torch.zeros((1,), dtype=torch.int32) + 5
repeats = 5

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[x, repeats],
    # inputs_channel_order=ChannelOrder.PYTORCH,
)

model_path = 'repeat'
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
