import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy, TraceLevel

import torch
import torchvision

from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter, TFLiteConverterV2
from tensorflow import keras


torch.manual_seed(0)
device = 'cpu'

pytorch_module = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)\
    .eval().to(device)

x = torch.rand(size=(1, 3, 320, 320)).to(device)

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x],
    enable_torch_tracing=True,
)

x_tf = tf.random.uniform(shape=(1, 320, 320, 3))
outputs = keras_model(x_tf)
print(outputs)

model_path = 'ssdlite320'

keras_model.save(model_path)
print('Model saved')

keras_model_restored = tf.saved_model.load(model_path)
print('Model loaded')

converter = TFLiteConverterV2.from_keras_model(keras_model)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
