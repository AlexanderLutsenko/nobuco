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


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs_2d = nn.ModuleList()
        for out_channels in [8, 16, 32]:
            for kernel_size in [(1, 1), (3, 1), (1, 3), (3, 3)]:
                for padding in [(0, 0), (1, 0), (0, 1), (1, 1), 'same']:
                    for groups in [1, 4, 8]:
                        for dilation in [1, 2]:
                            for stride in [(1, 1)]:
                                conv = nn.Conv2d(16, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, dilation=dilation, stride=stride)
                                self.convs_2d.append(conv)

        self.convs_transpose_2d = nn.ModuleList()
        for out_channels in [8, 16, 32]:
            for kernel_size in [(1, 1), (3, 1), (1, 3), (3, 3)]:
                for padding in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                    for groups in [1, 4, 8]:
                        for dilation in [1, 2]:
                            conv = nn.ConvTranspose2d(16, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, dilation=dilation)
                            self.convs_transpose_2d.append(conv)

        self.convs_2d_strided = nn.ModuleList()
        for out_channels in [8, 16, 32]:
            for kernel_size in [(3, 1), (1, 3)]:
                for padding in [(1, 0), (0, 1)]:
                    for groups in [1, 4, 8]:
                        for dilation in [1]:
                            for stride in [(1, 2), (2, 1)]:
                                conv = nn.Conv2d(16, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, dilation=dilation, stride=stride)
                                self.convs_2d.append(conv)

        self.convs_1d = nn.ModuleList()
        for out_channels in [8, 16, 32]:
            for kernel_size in [1, 3]:
                for padding in [0, 1, 'same']:
                    for groups in [1, 4, 8]:
                        for dilation in [1, 2]:
                            conv = nn.Conv1d(16, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, dilation=dilation)
                            self.convs_1d.append(conv)

        self.convs_transpose_1d = nn.ModuleList()
        for out_channels in [8, 16, 32]:
            for kernel_size in [1, 3]:
                for padding in [0, 1]:
                    for groups in [1, 4, 8]:
                        for dilation in [1]:
                            conv = nn.ConvTranspose1d(16, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, dilation=dilation)
                            self.convs_transpose_1d.append(conv)

    def forward(self, x):
        outputs = []

        for conv in self.convs_2d:
            out = conv(x)
            outputs.append(out)

        for conv in self.convs_2d_strided:
            out = conv(x)
            outputs.append(out)

        for conv in self.convs_transpose_2d:
            out = conv(x)
            outputs.append(out)

        b, c, h, w = x.shape
        x = x.view(b, c, h*w)
        x = nobuco.force_tensorflow_order(x)

        for conv in self.convs_1d:
            out = conv(x)
            outputs.append(out)

        for conv in self.convs_transpose_1d:
            out = conv(x)
            outputs.append(out)

        return outputs


# nobuco.unregister_converter(nn.Conv1d)
# nobuco.unregister_converter(nn.Conv2d)
# nobuco.unregister_converter(nn.ConvTranspose1d)
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

# import tensorflowjs
# keras_model.save(model_path)
# tensorflowjs.converters.convert_tf_saved_model(model_path, model_path + '.js', skip_op_check=False)
