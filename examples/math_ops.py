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
        log = torch.log_(x)
        log2 = x.log2_()
        log10 = torch.log10(x)
        log1p = torch.log1p(x)
        return log, log2, log10, log1p


model = DummyModel()
dummy_image = torch.randn(1, 3, 100, 100)

model(dummy_image)

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_image],
    # inputs_channel_order=ChannelOrder.PYTORCH,
)

model_path = 'math_ops'
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
