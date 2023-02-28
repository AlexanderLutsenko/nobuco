import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pytorch2keras.convert.converter import pytorch_to_keras
from pytorch2keras.commons import ChannelOrder, ChannelOrderingStrategy
from pytorch2keras.converters.node_converter import converter
from pytorch2keras.convert.layers.weight import WeightLayer
from pytorch2keras.trace.trace import Tracer

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras

import torch
from torch import nn


class ControlWhile(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv2d(300, 300, kernel_size=(1, 1))
        self.conv_loop = nn.Conv2d(300, 300, kernel_size=(1, 1))
        self.conv_shared = nn.Conv2d(300, 300, kernel_size=(1, 1))

    @Tracer.traceable()
    def loop_cond(self, x):
        return x.abs().mean() < 3

    @Tracer.traceable()
    def loop_body(self, x):
        x = self.conv_shared(x)
        x = torch.tanh(x)
        x = self.conv_loop(x)
        return x * 10

    @Tracer.traceable()
    def loop(self, x):
        while self.loop_cond(x):
            x = self.loop_body(x)
        return x

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.loop(x)
        x = self.conv_shared(x)
        return x


class Loop(tf.keras.layers.Layer):
    def __init__(self, cond, body, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond = cond
        self.body = body

    def get_config(self):
        config = super().get_config()
        config.update({
            "cond": self.cond,
            "body": self.body,
        })
        return config

    @tf.function
    def call(self, x):
        return tf.while_loop(lambda x: self.cond(x), lambda x: self.body(x), [x])

        # while self.cond(x):
        #     x = self.body(x)
        # return x


@converter(ControlWhile.loop, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def loop(self, x):
    order = ChannelOrder.TENSORFLOW
    cond = pytorch_to_keras(self.loop_cond, [x],
                            inputs_channel_order=order,
                            outputs_channel_order=order
                            )
    body = pytorch_to_keras(self.loop_body, [x],
                            inputs_channel_order=order,
                            outputs_channel_order=order
                            )
    loop = Loop(cond, body)

    def func(self, x):
        return loop(x)
    return func


inputs = [
    torch.normal(0, 1, size=(1, 300, 32, 32)),
]
pytorch_module = ControlWhile().eval()

keras_model = pytorch_to_keras(
    pytorch_module, inputs,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)


model_path = 'control_while'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer, 'Loop': Loop}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')

converter = TFLiteConverter.from_keras_model_file(model_path + '.h5', custom_objects=custom_objects)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
