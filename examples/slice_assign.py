import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

    def forward(self, x, y, indices, index_x, index_y, index_z):
        x[:] = 0
        x[:, :2] = 1
        x[:, :2, 3:11:2] = (x[:, :2, 3:11:2] + 2)
        x[:, :2, 3:11:2, 1] = torch.asarray(3)
        x[torch.asarray(2, dtype=torch.int32)] = 4
        x[:, 2, index_x, index_y] = 1

        d = x[:, indices]

        mask = torch.ones(x.shape[1]) > 0.5
        mask[2:] = False

        z3 = x[:, None, [2, 0, -1, 1], None]
        z4 = x[:, mask, ..., None, 0::2]

        z5 = x[0, None, ..., 1::2]
        z6 = x[..., 1::2]

        z7 = x[x > 0]

        c1 = x[index_x, :, index_y, 8]
        c2 = x[torch.stack([index_x, index_y], dim=1)]
        c3 = x[index_x][index_y]
        c4 = x[:, index_x, index_y, index_z]
        c5 = x[index_x[1][None, None, None], index_y[None, None, None], index_z[2][None, None, None], 1::2]

        scores = torch.zeros(size=(1, 78, 58))
        assign = torch.ones(size=(1, 77))
        scores[:, 1:, -1] = assign

        return x, y, d, z3, z4, z5, z6, z7, c1, c2, c3, c4, c5, scores


x = torch.normal(0, 1, size=(8, 5, 96, 128))
y = torch.normal(0, 1, size=(8, 5, 96, 128))
index = torch.tensor([[[[0, 0, 1, 0], [0, 1, 2, 1]]]], dtype=torch.int64)
index_x = torch.tensor([2, 3, 4], dtype=torch.int64)
index_y = torch.tensor([0, 1, 2], dtype=torch.int64)
index_z = torch.tensor([2, 1, 0], dtype=torch.int64)

pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x, y, index, index_x, index_y, index_z],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'slice_assign'
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
