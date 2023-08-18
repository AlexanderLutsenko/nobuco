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
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=4,
            num_layers=6,
            batch_first=True,
            bidirectional=False,
            bias=False
        )
        self.lstm_bidirectional = nn.LSTM(
            input_size=4,
            hidden_size=4,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.gru = nn.GRU(
            input_size=4,
            hidden_size=4,
            num_layers=6,
            batch_first=True,
            bidirectional=False,
            bias=False,
        )
        self.gru_bidirectional = nn.GRU(
            input_size=4,
            hidden_size=4,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x, h0, c0):
        y1, _ = self.lstm(x, (h0, c0))
        y2, _ = self.lstm_bidirectional(x, (h0, c0))

        z1, _ = self.gru(x, h0)
        z2, _ = self.gru_bidirectional(x, h0)
        return y1, y2, z1, z2


x = torch.normal(0, 1, size=(8, 3, 4))
h0 = torch.rand(6, 8, 4)
c0 = torch.rand(6, 8, 4)

pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x, h0, c0],
    inputs_channel_order=ChannelOrder.PYTORCH,
)

model_path = 'rnn'
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
