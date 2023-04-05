import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from nobuco.convert.converter import pytorch_to_keras
from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter
from nobuco.convert.layers.weight import WeightLayer

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras

import torch
from torch import nn


class ControlIf(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv2d(300, 300, kernel_size=(1, 1))
        self.conv_true = nn.Conv2d(300, 300, kernel_size=(1, 1))
        self.conv_false = nn.Conv2d(300, 300, kernel_size=(1, 1))
        self.conv_shared = nn.Conv2d(300, 300, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv_pre(x)
        if x.mean() > 0:
            x = self.conv_true(x)
            x = torch.tanh(x)
            x = self.conv_shared(x)
            x = x + 1
        else:
            x = self.conv_false(x)
            x = torch.sigmoid(x)
            x = self.conv_shared(x)
            x = x - 1
        x = self.conv_shared(x)
        return x


class ControlIfKeras(tf.keras.layers.Layer):
    def __init__(self, conv_pre, conv_true, conv_false, conv_shared, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_pre = conv_pre
        self.conv_true = conv_true
        self.conv_false = conv_false
        self.conv_shared = conv_shared

    def get_config(self):
        config = super().get_config()
        config.update({
            "conv_pre": self.conv_pre,
            "conv_true": self.conv_true,
            "conv_false": self.conv_false,
            "conv_shared": self.conv_shared,
        })
        return config

    @tf.function
    def call(self, x):
        x = self.conv_pre(x)
        if tf.reduce_mean(x) > 0:
            x = self.conv_true(x)
            x = tf.tanh(x)
            x = self.conv_shared(x)
            x = x + 1
        else:
            x = self.conv_false(x)
            x = tf.sigmoid(x)
            x = self.conv_shared(x)
            x = x - 1
        x = self.conv_shared(x)
        return x


@converter(ControlIf, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converterControlIf(self, x):
    order = ChannelOrder.TENSORFLOW
    conv_pre = pytorch_to_keras(self.conv_pre, [x], inputs_channel_order=order, outputs_channel_order=order)
    conv_true = pytorch_to_keras(self.conv_true, [x], inputs_channel_order=order, outputs_channel_order=order)
    conv_false = pytorch_to_keras(self.conv_false, [x], inputs_channel_order=order, outputs_channel_order=order)
    conv_shared = pytorch_to_keras(self.conv_shared, [x], inputs_channel_order=order, outputs_channel_order=order)
    layer = ControlIfKeras(conv_pre, conv_true, conv_false, conv_shared)
    return layer


inputs = [
    torch.normal(0, 1, size=(1, 300, 32, 32)),
]
pytorch_module = ControlIf().eval()

keras_model = pytorch_to_keras(
    pytorch_module, inputs,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)


model_path = 'control_if'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer, 'ControlIfKeras': ControlIfKeras}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')

converter = TFLiteConverter.from_keras_model_file(model_path + '.h5', custom_objects=custom_objects)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
