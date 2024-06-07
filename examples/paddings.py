import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
import keras


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1d, x2d, x3d):
        pads = [
            (2, 3),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6, 7),
        ]

        outs = []
        for x, pad in zip([x1d, x2d, x3d], pads):
            y_con = F.pad(x1, pad, mode='constant', value=42)
            y_ref = F.pad(x, pad, mode='reflect')
            y_rec = F.pad(x, pad, mode='replicate')
            y_cir = F.pad(x, pad, mode='circular')
            outs += [y_con, y_ref, y_rec, y_cir]

        return outs


x1 = torch.normal(0, 1, size=(1, 3, 128))
x2 = torch.normal(0, 1, size=(1, 3, 128, 128))
x3 = torch.normal(0, 1, size=(1, 3, 128, 128, 4))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x1, x2, x3],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'paddings'
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
