import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pytorch2keras.converters.node_converter import converter_unregister, converter
from pytorch2keras.convert.converter import pytorch_to_keras
from pytorch2keras.commons import ChannelOrder, ChannelOrderingStrategy
from pytorch2keras.convert.layers.weight import WeightLayer

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1))
        self.act1 = nn.Hardsigmoid()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x


converter_unregister(F.hardsigmoid)


@converter(F.hardsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardsigmoid(input: Tensor, inplace: bool = False):
    def func(input, inplace=False):
        return tf.keras.activations.hard_sigmoid(input)
    return func


h = 256
w = 256
image = torch.rand(size=(1, 3, h, w))

inputs = [image]
pytorch_module = MyModule().eval()

keras_model = pytorch_to_keras(
    pytorch_module, inputs,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)


model_path = 'my_module'
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
