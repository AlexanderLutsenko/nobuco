import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import tensorflow as tf
import tensorflow_addons
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras


class DenseImageWarp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, flow):
        b, _, h, w = flow.shape

        lin_h = torch.linspace(start=0., end=(h - 1.), steps=h)[:, None].repeat(1, w)
        lin_w = torch.linspace(start=0., end=(w - 1.), steps=w)[None, :].repeat(h, 1)
        lin = torch.stack([lin_h, lin_w], dim=0)[None, ...]

        scale = torch.tensor([(h-1)/2, (w-1)/2])[None, None, None, :]

        grid = lin - flow
        grid = grid.permute(0, 2, 3, 1)
        grid = grid / scale - 1
        grid = torch.roll(grid, shifts=1, dims=-1)
        return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)


@nobuco.converter(DenseImageWarp, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def dense_image_warp(self, image: Tensor, flow: Tensor):
    def func(image, flow):
        return keras.layers.Lambda(lambda args: tensorflow_addons.image.dense_image_warp(args[0], args[1]))([image, flow])
    return func


h = 256
w = 256
image = torch.rand(size=(1, 3, h, w))
flow = torch.rand(size=(1, 2, h, w))

inputs = [image, flow]
pytorch_module = DenseImageWarp().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module, inputs,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)


model_path = 'dense_image_warp'
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
