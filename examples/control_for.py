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


class ControlFor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv2d(3, 16, kernel_size=(1, 1))
        self.conv_loop = nn.Conv2d(16, 16, kernel_size=(1, 1))
        self.conv_shared = nn.Conv2d(16, 16, kernel_size=(1, 1))

    @nobuco.traceable
    def loop_body(self, x):
        x = self.conv_shared(x)
        x = torch.tanh(x)
        x = self.conv_loop(x)
        return x * 10

    @nobuco.traceable
    def for_loop(self, x, n):
        for i in range(n):
            x = self.loop_body(x)
        return x

    def forward(self, x, n):
        x = self.conv_pre(x)
        x = self.for_loop(x, n)
        x = self.conv_shared(x)
        return x


class ForLoop(tf.keras.layers.Layer):
    def __init__(self, body, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.body = body

    def get_config(self):
        config = super().get_config()
        config.update({
            "body": self.body,
        })
        return config

    @tf.function
    def call(self, x, n):
        for i in range(n):
            x = self.body(x)
        return x


@nobuco.converter(ControlFor.for_loop, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def loop(self, x, n):
    order = ChannelOrder.TENSORFLOW
    body = nobuco.pytorch_to_keras(self.loop_body, [x], inputs_channel_order=order, outputs_channel_order=order)
    for_loop = ForLoop(body)

    def func(self, x, n):
        return for_loop(x, n)
    return func


inputs = [
    torch.normal(0, 1, size=(1, 3, 128, 128)),
    torch.asarray(5),
]
pytorch_module = ControlFor().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module, inputs,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)


model_path = 'control_for'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer, 'ForLoop': ForLoop}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')

converter = TFLiteConverter.from_keras_model_file(model_path + '.h5', custom_objects=custom_objects)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
