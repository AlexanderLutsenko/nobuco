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


class ControlIf(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv2d(300, 300, kernel_size=(1, 1))
        self.conv_true = nn.Conv2d(300, 300, kernel_size=(1, 1))
        self.conv_false = nn.Conv2d(300, 300, kernel_size=(1, 1))
        self.conv_shared = nn.Conv2d(300, 300, kernel_size=(1, 1))

    @Tracer.traceable()
    def branch_true(self, x):
        x = self.conv_true(x)
        x = torch.tanh(x)
        x = self.conv_shared(x)
        return x + 1

    @Tracer.traceable()
    def branch_false(self, x):
        x = self.conv_false(x)
        x = torch.sigmoid(x)
        x = self.conv_shared(x)
        return x - 1

    @Tracer.traceable()
    def cond(self, inputs):
        pred, x = inputs
        if pred:
            return self.branch_true(x)
        else:
            return self.branch_false(x)

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.cond([x.mean() > 0, x])
        x = self.conv_shared(x)
        return x


class Cond(tf.keras.layers.Layer):
    def __init__(self, branch_true, branch_false, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch_true = branch_true
        self.branch_false = branch_false

    def get_config(self):
        config = super().get_config()
        config.update({
            "branch_true": self.branch_true,
            "branch_false": self.branch_false,
        })
        return config

    @tf.function
    def call(self, inputs):
        pred, x = inputs
        return tf.cond(pred, true_fn=lambda: self.branch_true(x), false_fn=lambda: self.branch_false(x))

        # if pred:
        #     return self.branch_true(x)
        # else:
        #     return self.branch_false(x)


@converter(ControlIf.cond, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def cond(self, inputs):
    pred, x = inputs
    order = ChannelOrder.TENSORFLOW
    branch_true = pytorch_to_keras(self.branch_true, [x], inputs_channel_order=order, outputs_channel_order=order)
    branch_false = pytorch_to_keras(self.branch_false, [x], inputs_channel_order=order, outputs_channel_order=order)
    cond = Cond(branch_true, branch_false)

    def func(self, inputs):
        return cond(inputs)
    return func


inputs = [
    torch.normal(0, 1, size=(1, 300, 32, 32)),
]
pytorch_module = ControlIf().eval()

keras_model = pytorch_to_keras(
    pytorch_module, inputs,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)


model_path = 'control_if2'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer, 'Cond': Cond}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')

converter = TFLiteConverter.from_keras_model_file(model_path + '.h5', custom_objects=custom_objects)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
