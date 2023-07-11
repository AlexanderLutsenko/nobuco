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
import torch.nn.functional as F


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        m1 = x.mean()
        m2 = x.mean((2, 3), keepdim=False)
        m3 = x.mean((2, 3), keepdim=True)
        m4 = x.mean(1, keepdim=False)
        m5 = x.mean(2, keepdim=True)

        s1 = torch.sum(x)
        s2 = torch.sum(x, dim=(1, 2, 3), keepdim=False)
        s3 = torch.sum(x, dim=(1, 2, 3), keepdim=True)
        s4 = torch.sum(x, dim=(3,), keepdim=False)
        s5 = torch.sum(x, dim=1, keepdim=True)
        return (m1, m2, m3, m4, m5), (s1, s2, s3, s4, s5)


model = DummyModel()
dummy_image = torch.randn(1, 3, 100, 100)

model(dummy_image)

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_image],
    # inputs_channel_order=ChannelOrder.PYTORCH,
)

model_path = 'sum_mean'
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
