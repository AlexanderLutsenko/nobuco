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


class DummyModel(nn.Module):
    def forward(self, input, dim, index, src):
        return torch.scatter(input, dim, index, src)


model = DummyModel().eval()

x_shape = (1000, 256, 7, 7)
index_shape = (4, 256, 7, 7)

# x_shape = (1000,)
# index_shape = (4,)

x = torch.randn(x_shape)
dim = 0
index = torch.randint(low=0, high=x_shape[0]-1, size=index_shape, dtype=torch.int64)
src = torch.rand(index_shape)


keras_model = nobuco.pytorch_to_keras(
    model,
    args=[x, dim, index, src],
    input_shapes={x: [None, None, None, None], index: [None, None, None, None], src: [None, None, None, None]}
)

model_path = 'scatter'
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
