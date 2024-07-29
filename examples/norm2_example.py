import numpy as np

import nobuco
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverterV2
import keras


class MyModule(nn.Module):
    def forward(self, x):
        outputs = []
        for p in [None, 'fro', 0, 1, 2, np.inf, -np.inf]:
            for dim in [None, 0, 1, 2, [0, 1], [1, 2], [0, 2]]:
                for keepdim in [False, True]:
                    y = torch.norm(x, p=p, dim=dim, keepdim=keepdim)
                    outputs.append(y)
        return outputs


x = torch.normal(0, 1, size=(2, 3, 4, 5))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x],
    inputs_channel_order=nobuco.ChannelOrder.TENSORFLOW,
)

model_path = 'norm2_example'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')

converter = TFLiteConverterV2.from_keras_model(keras_model)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
