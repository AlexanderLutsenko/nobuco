import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

import nobuco

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverterV2


class MyModel(nn.Module):
    def forward(self, x1, x2):
        y1 = torch.cat([x1, x2], dim=1)
        y2 = x1.mean()
        y3 = x2.max()
        return y1, y2, y3


model = MyModel()

x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 5, 32, 32)

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[x1, x2],
    input_names={x1: 'x1', x2: 'x2'},
    output_names={0: 'y1', 1: 'y2', 2: 'y3'},
)

model_path = 'signature'
keras_model.save(model_path + '.h5')
print('Model saved')

# Convert with V2
converter = TFLiteConverterV2.from_keras_model(keras_model)
tflite_model = converter.convert()

with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)

# Validate signatures
interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()
print('Signatures:', signatures)

x1_np = np.random.uniform(size=(1, 32, 32, 3)).astype(np.float32)
x2_np = np.random.uniform(size=(1, 32, 32, 5)).astype(np.float32)

# Run inference with signature_runner
my_signature = interpreter.get_signature_runner()
outputs = my_signature(x1=x1_np, x2=x2_np)
print('Outputs:', [(k, t.shape) for k, t in outputs.items()])
