import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
import torch.nn.functional as F
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras


class MyModule(nn.Module):
    def forward(self, x, y):
        return torch.cdist(x, y)


x = torch.normal(0, 1, size=(1, 3, 50, 512))
y = torch.normal(0, 1, size=(2, 1, 1024, 512))

pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x, y],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'distance'
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
