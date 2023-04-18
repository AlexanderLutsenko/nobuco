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
        self.gru = nn.GRU(32, 128, num_layers=1, batch_first=True, bidirectional=False)
        self.conv1 = nn.Conv1d(12, 40, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(12, 60, kernel_size=1, padding=0)

    def forward(self, x):
        x, hx = self.gru(x)
        x = nobuco.force_tensorflow_order(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1, x2


pytorch_module = MyModule().eval()

inputs = [
    torch.normal(0, 1, size=(1, 12, 32)),
]

keras_model = nobuco.pytorch_to_keras(
    pytorch_module, inputs,
    inputs_channel_order=ChannelOrder.PYTORCH,
)


model_path = 'channel_ordering'
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
