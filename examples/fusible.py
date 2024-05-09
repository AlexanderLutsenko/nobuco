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
import numpy as np


# class FusibleModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bn = nn.BatchNorm2d(3)
#         self.conv = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(0, 0))
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         x = self.bn(x)
#         x = self.conv(x)
#         x = self.act(x)
#         return x


class FusibleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.conv = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(0, 0))
        self.act = nn.ReLU()

    @nobuco.traceable
    def bn_conv(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        x = self.bn_conv(x)
        x = self.act(x)
        return x


x = torch.normal(0, 1, size=(1, 3, 128, 128))
pytorch_module = FusibleModule()

# Train for a bit to get more realistic parameters of the BatchNorm layer
optimizer = torch.optim.Adam(pytorch_module.parameters(), lr=1e-3)
for _ in range(100):
    x = torch.normal(0, 1, size=(1, 3, 128, 128))
    loss = -pytorch_module(x).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

pytorch_module.eval()


@nobuco.converter(FusibleModule.bn_conv, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_bn_conv(self, x):
    order = ChannelOrder.TENSORFLOW
    bn, out_bn = nobuco.pytorch_to_keras(self.bn, [x], inputs_channel_order=order, outputs_channel_order=order, return_outputs_pt=True)
    conv = nobuco.pytorch_to_keras(self.conv, [out_bn], inputs_channel_order=order, outputs_channel_order=order)

    gamma, beta, moving_mean, moving_variance = bn.get_weights()
    kernel, bias = conv.get_weights()
    eps = self.bn.eps

    '''
    y = gamma * (x - moving_mean) / sqrt(moving_variance + eps) + beta
    z = kernel * y + bias
    =>
    z = kernel_fused * x + bias_fused WHERE
    kernel_fused = kernel * gamma / sqrt(moving_variance + eps)
    bias_fused = -kernel_fused * moving_mean + kernel * beta + bias
    '''
    kernel_fused = kernel * (gamma / np.sqrt(moving_variance + eps))[None, None, :, None]
    bias_fused = (-kernel_fused * moving_mean[None, None, :, None] + kernel * beta[None, None, :, None]).sum(axis=(0, 1, 2)).flatten() + bias
    conv.set_weights([kernel_fused, bias_fused])
    return lambda self, x: conv(x)


keras_model = nobuco.pytorch_to_keras(
    pytorch_module, [x],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    save_trace_html=True,
)


model_path = 'fusible'
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
