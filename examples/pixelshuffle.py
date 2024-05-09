import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
import keras


import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

from nobuco.addons.torch.depth_to_space import DepthToSpace
from nobuco.addons.torch.space_to_depth import SpaceToDepth


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = 4
        self.conv = nn.Conv2d(3*self.factor**2, 3*self.factor**2, kernel_size=1)

    def forward(self, x):
        x = nn.PixelUnshuffle(self.factor)(x)
        x = self.conv(x)
        x = nn.PixelShuffle(self.factor)(x)
        return x


class MyModuleTFOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = 4
        self.conv = nn.Conv2d(3*self.factor**2, 3*self.factor**2, kernel_size=1)

    def forward(self, x):
        x = SpaceToDepth(self.factor)(x)
        x = self.conv(x)
        x = DepthToSpace(self.factor)(x)
        return x


input = torch.normal(0, 1, size=(1, 3, 128, 128))
# pytorch_module = MyModule()
pytorch_module = MyModuleTFOptimized()

pytorch_module.eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[input],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    trace_shape=False,
)

model_path = 'pixelshuffle'
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
