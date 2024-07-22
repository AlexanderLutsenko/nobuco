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


@nobuco.traceable
def get_crop(x, crop_y, crop_x, h, w):
    return x[:, :, crop_y: crop_y + h, crop_x: crop_x + w]


@nobuco.converter(get_crop, channel_ordering_strategy=nobuco.ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_get_crop(x, crop_y, crop_x, h, w):
    def func(x, crop_y, crop_x, h, w):
        return tf.image.crop_to_bounding_box(x, crop_y, crop_x, h, w)  # Calls tf.slice under the hood
    return func


class CroppingModule(nn.Module):
    def __init__(self, crop_height, crop_width):
        super().__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width

    def forward(self, x, crop_x, crop_y):
        _, _, h, w = x.shape
        crop_x = (crop_x * (w - self.crop_width)).int()
        crop_y = (crop_y * (h - self.crop_height)).int()
        crop = x[:, :, crop_y: crop_y + self.crop_height, crop_x: crop_x + self.crop_width]  # Poor way to crop
        # crop = x.narrow(2, crop_y, self.crop_height).narrow(3, crop_x, self.crop_width)  # Better way to crop
        # crop = get_crop(x, crop_y, crop_x, self.crop_height, self.crop_width)  # Best way to crop
        return crop


x = torch.normal(0, 1, size=(1, 3, 128, 128))
crop_y = torch.rand(())
crop_x = torch.rand(())
pytorch_module = CroppingModule(64, 32).eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x, crop_x, crop_y],
    inputs_channel_order=ChannelOrder.TENSORFLOW
)

model_path = 'static_crop'
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
