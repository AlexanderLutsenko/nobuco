import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn
import torchvision

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
import keras


class MyModule(nn.Module):
    def __init__(self, iou_threshold):
        super().__init__()
        self.iou_threshold = iou_threshold

    def forward(self, boxes, scores, idxs):
        out1 = torch.ops.torchvision.nms(boxes, scores, self.iou_threshold)
        out2 = torchvision.ops.nms(boxes, scores, self.iou_threshold)
        out3 = torch.ops.torchvision.nms(boxes, scores, self.iou_threshold)
        out4 = torchvision.ops.boxes.nms(boxes, scores, self.iou_threshold)
        out5 = torchvision.ops.boxes.batched_nms(boxes, scores, idxs, self.iou_threshold)
        return out1, out2, out3, out4, out5


boxes = torch.normal(0, 1, size=(128, 4))
boxes[:, 2:] += 10
scores = torch.normal(0, 1, size=(128,))
idxs = torch.randint(0, 127, size=(128,))
pytorch_module = MyModule(0.7).eval()


keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[boxes, scores, idxs],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
    trace_shape=True,
)

model_path = 'nms'
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
