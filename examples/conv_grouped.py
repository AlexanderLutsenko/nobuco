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
        self.conv1 = nn.Conv2d(16, 16, kernel_size=(3, 1), groups=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(1, 3), groups=4)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), groups=16)

        self.convt1 = nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(2, 2), groups=1)
        self.convt2 = nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), stride=(1, 1), groups=16)

        self.conv1d1 = nn.Conv1d(16, 32, kernel_size=3, padding=2, groups=1)
        self.conv1d2 = nn.Conv1d(32, 64, kernel_size=2, padding=1, groups=16)
        self.conv1d3 = nn.Conv1d(64, 128, kernel_size=5, padding=3, groups=64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.convt1(x)
        x = self.convt2(x)

        b, c, h, w = x.shape
        x = x.view(b, c, h*w)
        x = self.conv1d1(x)
        x = self.conv1d2(x)
        x = self.conv1d3(x)
        return x


# nobuco.unregister_converter(nn.Conv1d)
# nobuco.unregister_converter(nn.Conv2d)
# nobuco.unregister_converter(nn.ConvTranspose2d)


input = torch.normal(0, 1, size=(1, 16, 128, 128))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[input],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'conv_grouped'
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
