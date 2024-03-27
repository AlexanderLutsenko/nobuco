import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflowjs

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=(3, 1), groups=1)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=(1, 3), groups=4)
        self.conv2_3 = nn.Conv2d(64, 128, kernel_size=(3, 3), groups=64)

        self.conv2s1 = nn.Conv2d(16, 32, kernel_size=(3, 1), groups=1, dilation=2, padding='same')
        self.conv2s2 = nn.Conv2d(32, 64, kernel_size=(1, 3), groups=4, padding='same')
        self.conv2s3 = nn.Conv2d(64, 128, kernel_size=(3, 3), groups=64, padding='same')

        self.conv2t1 = nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), groups=1, padding=(3, 3))
        self.conv2t2 = nn.ConvTranspose2d(32, 64, kernel_size=(5, 5), stride=(1, 1), groups=32, padding=(2, 2))
        self.conv2t3 = nn.ConvTranspose2d(64, 128, kernel_size=(2, 2), stride=(1, 1), groups=1, padding=(1, 1))

        self.conv1d1 = nn.Conv1d(16, 64, kernel_size=3, padding=2, groups=1)
        self.conv1d2 = nn.Conv1d(64, 64, kernel_size=2, padding=1, groups=16)
        self.conv1d3 = nn.Conv1d(64, 128, kernel_size=5, padding=3, groups=64)

        self.conv1s1 = nn.Conv1d(16, 64, kernel_size=3, groups=1, padding='same')
        self.conv1s2 = nn.Conv1d(64, 64, kernel_size=2, groups=16, padding='same')
        self.conv1s3 = nn.Conv1d(64, 128, kernel_size=5, groups=64, padding='same')

    def forward(self, x):
        x1 = self.conv2_1(x)
        x1 = self.conv2_2(x1)
        x1 = self.conv2_3(x1)

        x2 = self.conv2s1(x)
        x2 = self.conv2s2(x2)
        x2 = self.conv2s3(x2)

        x3 = self.conv2t1(x)
        x3 = self.conv2t2(x3)
        x3 = self.conv2t3(x3)

        b, c, h, w = x.shape
        x = x.view(b, c, h*w)

        x4 = self.conv1d1(x)
        x4 = self.conv1d2(x4)
        x4 = self.conv1d3(x4)

        x5 = self.conv1s1(x)
        x5 = self.conv1s2(x5)
        x5 = self.conv1s3(x5)
        return x1, x2, x3, x4, x5


# nobuco.unregister_converter(nn.Conv1d)
# nobuco.unregister_converter(nn.Conv2d)
# nobuco.unregister_converter(nn.ConvTranspose2d)


input = torch.normal(0, 1, size=(1, 16, 128, 128))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[input],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'conv_grouped'
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

# keras_model.save(model_path)
# tensorflowjs.converters.convert_tf_saved_model(model_path, model_path + '.js', skip_op_check=False)
