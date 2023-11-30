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
from tensorflow import keras


class DenseImageWarp2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, flow):
        _, _, h, w = nobuco.shape(flow)

        lin_w = torch.linspace(start=0, end=(w - 1), steps=w)[None, :].repeat(h, 1)
        lin_h = torch.linspace(start=0, end=(h - 1), steps=h)[:, None].repeat(1, w)
        lin = torch.stack([lin_w, lin_h], dim=0)[None, ...]

        scale = torch.stack([w, h], dim=0)[None, :, None, None]
        scale = (scale - 1) / 2

        '''
        grid = lin - flow
        grid = grid / scale - 1
        '''
        grid = lin / scale - 1 - (flow / scale)
        grid = grid.permute(0, 2, 3, 1)
        return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)


h = 128
w = 128
b = 8
image = torch.rand(size=(b, 3, h, w))
flow = torch.normal(0, 1, size=(b, 2, h, w))

pytorch_module = DenseImageWarp2().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module, [image, flow],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)


model_path = 'dense_image_warp2'
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
