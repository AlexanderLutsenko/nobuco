import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import copy

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras

import numpy as np


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.inner1 = nn.Conv1d(128, 128, kernel_size=1)
        self.inner1 = nn.TransformerEncoderLayer(512, 4, dim_feedforward=256, batch_first=True)

        self.inner2 = copy.deepcopy(self.inner1)
        self.inner3 = copy.deepcopy(self.inner1)
        self.inner4 = copy.deepcopy(self.inner3)

    def forward(self, x):
        x = self.inner1(x)
        x = self.inner2(x)
        x = self.inner3(x)
        x = self.inner4(x)
        return x

    def forward_same(self, x):
        for i in range(4):
            x = self.inner1(x)
        return x


def initialize_weights(m):
    if isinstance(m, (nn.Linear, _ConvNd)):
        nn.init.normal_(m.bias, 0, 0.01)
        if m.bias.data is not None:
            nn.init.normal_(m.bias, 0, 0.01)


# pytorch_module = nn.TransformerEncoder(
#     nn.TransformerEncoderLayer(512, 4, dim_feedforward=256, batch_first=True).eval(),
#     3).eval()

pytorch_module = MyModule().eval()
pytorch_module.apply(initialize_weights)

input = torch.normal(0, 1, size=(1, 128, 512))

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[input],
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH,
)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


print_trainable_parameters(pytorch_module)
keras_model.summary()

model_path = 'module_deepcopy'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')


input_pt = torch.normal(0, 1, size=(1, 128, 512))
output_pt = pytorch_module(input_pt).detach().numpy()
output_pt_same = pytorch_module.forward_same(input_pt).detach().numpy()


input_tf = tf.convert_to_tensor(input_pt.numpy())
output_tf = keras_model_restored(input_tf).numpy()

print('PT:', np.mean(np.abs(output_pt)))
print('Diff pt-tf:', np.max(np.abs(output_pt - output_tf)))

diff_pt_pt = np.max(np.abs(output_pt - output_pt_same))
print('Diff pt-pt:', np.max(np.abs(output_pt - output_pt_same)))

assert diff_pt_pt > 0


converter = TFLiteConverter.from_keras_model_file(model_path + '.h5', custom_objects=custom_objects)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
