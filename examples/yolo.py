import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy, TraceLevel
from nobuco.layers.weight import WeightLayer

import torch

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow import keras


device = 'cpu'

x = torch.rand(size=(1, 3, 320, 320)).to(device)
pytorch_module = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).eval().to(device)  # or yolov5n - yolov5x6, custom

# Dry run to create model buffers which creation we don't want to trace
pytorch_module(x)


keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'yolo'

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
