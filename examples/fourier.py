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

    def forward(self, x1d, x2d):
        y1d = torch.fft.rfft(x1d)
        z1d = torch.fft.irfft(y1d)

        y2d = torch.fft.rfft2(x2d)
        z2d = torch.fft.irfft2(y2d)

        window = torch.ones((64,))*3
        ys1d = torch.stft(x1d, n_fft=64, hop_length=32, win_length=64, center=True, window=window,
                          return_complex=True, onesided=False, normalized=True)

        print('!!!', ys1d)
        # zs1d = torch.istft(ys1d, n_fft=64, hop_length=32, win_length=64, center=True, window=window,
        #                   return_complex=True, onesided=False, normalized=True)

        return y1d, z1d, y2d, z2d, ys1d, #zs1d


# x1d = torch.normal(0, 1, size=(1, 128))
# x2d = torch.normal(0, 1, size=(1, 128, 128))

x1d = torch.ones(size=(1, 128))
x2d = torch.ones(size=(1, 128, 128))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x1d, x2d],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'fourier'
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
