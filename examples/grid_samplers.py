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


class GridSamplers(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, flow):
        outputs = []
        for mode in ('bilinear', 'nearest'):
            for padding_mode in ('zeros', 'border', 'reflection'):
                for align_corners in (True, False):
                    out = F.grid_sample(image, flow, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
                    outputs.append(out)
        return outputs


h = 64
w = 127
b = 8

torch.manual_seed(42)
image = torch.rand(size=(b, 3, h, w))
flow = torch.normal(0, 1, size=(b, h, w, 2))

pytorch_module = GridSamplers().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module, [image, flow],
    inputs_channel_order={image: ChannelOrder.TENSORFLOW, flow: ChannelOrder.PYTORCH},
)

model_path = 'grid_samplers'
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
