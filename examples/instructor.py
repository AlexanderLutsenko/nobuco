import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverterV2
import keras

from InstructorEmbedding import INSTRUCTOR

import torch
from torch import nn


device = 'cpu'

instructor = INSTRUCTOR('hkunlp/instructor-large').eval().to(device)

input_ids = torch.zeros(size=(2, 16), dtype=torch.int64)
attention_mask = torch.ones(size=(2, 16), dtype=torch.int64)


keras_model = nobuco.pytorch_to_keras(
    instructor,
    args=[{'input_ids': input_ids, 'attention_mask': attention_mask}],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'instructor'

# keras_model.save(model_path + '.keras')
# print('Model saved')

# custom_objects = {'WeightLayer': WeightLayer}

# keras_model_restored = keras.models.load_model(model_path + '.keras', custom_objects=custom_objects)
# print('Model loaded')

converter = TFLiteConverterV2.from_keras_model(keras_model)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
