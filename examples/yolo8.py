import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy, TraceLevel
from nobuco.layers.weight import WeightLayer

import torch

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter, TFLiteConverterV2
import keras

import ultralytics


device = 'cpu'

x = torch.rand(size=(1, 3, 320, 320)).to(device)
pytorch_module = ultralytics.YOLO("yolov8n.yaml").model.eval().to(device)

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

# Keras just cannot save/load some models correctly, so we choose SavedModel format and then load the model by Tensorflow
model_path = 'yolo8'

keras_model.save(model_path)
print('Model saved')

# Use TF API, not Keras
keras_model_restored = tf.saved_model.load(model_path)
print('Model loaded')

converter = TFLiteConverterV2.from_keras_model(keras_model)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
