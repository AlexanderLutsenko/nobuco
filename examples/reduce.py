import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras

import torch
from torch import nn
import torch.nn.functional as F


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        m1 = x.mean()
        m2 = x.mean((2, 3), keepdim=False)
        m3 = x.mean((2, 3), keepdim=True)
        m4 = x.mean(1, keepdim=False)
        m5 = x.mean(2, keepdim=True)

        s1 = torch.sum(x)
        s2 = torch.sum(x, dim=(1, 2, 3), keepdim=False)
        s3 = torch.sum(x, dim=(1, 2, 3), keepdim=True)
        s4 = torch.sum(x, dim=(3,), keepdim=False)
        s5 = torch.sum(x, dim=1, keepdim=True)

        std1 = torch.std(x, dim=(1, 2), unbiased=False, keepdim=False)
        std2 = x.std(dim=3, unbiased=True, keepdim=True)

        mask = x > 0

        any1 = torch.any(mask, dim=1, keepdim=False)
        any2 = mask.any(dim=3, keepdim=True)

        all1 = torch.all(mask, dim=0, keepdim=True)
        all2 = mask.all(dim=2, keepdim=False)

        min1 = torch.min(x, dim=1, keepdim=False)
        min2 = x.min(dim=3, keepdim=True)
        min3 = torch.min(x)
        min4 = torch.min(x, x)

        max1 = torch.max(x, dim=1, keepdim=False)
        max2 = x.max(dim=3, keepdim=True)
        max3 = torch.max(x)
        max4 = torch.max(x, x)

        argmin1 = torch.argmin(x, dim=1, keepdim=False)
        argmin2 = x.argmin(dim=3, keepdim=True)
        argmin3 = torch.argmin(x)

        argmax1 = torch.argmax(x, dim=1, keepdim=False)
        argmax2 = x.argmax(dim=3, keepdim=True)
        argmax3 = torch.argmax(x)

        amin1 = torch.amin(x, dim=None, keepdim=True)
        amin2 = torch.amin(x, dim=1, keepdim=True)
        amin3 = torch.amin(x, dim=(1, 2), keepdim=True)

        amax1 = torch.amax(x, dim=None, keepdim=False)
        amax2 = torch.amax(x, dim=0, keepdim=False)
        amax3 = torch.amax(x, dim=(0, 1, 2), keepdim=False)

        return \
            (m1, m2, m3, m4, m5), (s1, s2, s3, s4, s5), \
            (std1, std2), \
            (any1, any2), (all1, all2), \
            (min1, min2, min3, min4), (max1, max2, max3, max4), \
            (argmin1, argmin2, argmin3), (argmax1, argmax2, argmax3), \
            (amin1, amin2, amin3), (amax1, amax2, amax3)


model = DummyModel()
dummy_image = torch.randn(1, 3, 100, 100)

model(dummy_image)

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_image],
    # inputs_channel_order=ChannelOrder.PYTORCH,
)

model_path = 'reduce'
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
