import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.natch_norm = nn.BatchNorm2d(32)
        self.instance_norm = nn.InstanceNorm2d(32, affine=True)
        self.group_norm = nn.GroupNorm(num_groups=4, num_channels=32, affine=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=256, elementwise_affine=True)

    def forward(self, x):
        x1 = self.natch_norm(x)
        x2 = self.instance_norm(x)
        x3 = self.group_norm(x)
        x4 = self.layer_norm(x)
        return x1, x2, x3, x4


model = DummyModel().eval()
x = torch.randn(4, 32, 256, 256)

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[x],
    # inputs_channel_order=ChannelOrder.PYTORCH,
)

model_path = 'norm'
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
