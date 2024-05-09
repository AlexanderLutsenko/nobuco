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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 3, kernel_size=1)
        self.conv2 = nn.Conv1d(6, 6, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)

        # # Reshape in Pytorch (channel-first) order
        # x = x.reshape(-1, 6, 2)

        # BCH -> BHC
        x = x.permute(0, 2, 1)
        # Reshape in Tensorflow (channel-last) order
        x = x.reshape(-1, 2, 6)
        # BHC -> BCH
        x = x.permute(0, 2, 1)

        x = self.conv2(x)
        return x


input = torch.normal(0, 1, size=(1, 3, 4))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[input],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'efficient_reshape'
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
