import nobuco
from nobuco.layers.weight import WeightLayer

import torch
from tensorflow.lite.python.lite import TFLiteConverter
from torch import nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow import keras


class MyModule(nn.Module):
    def forward(self, q, k, v, attn_mask):
        x1 = F.scaled_dot_product_attention(q, k, v)
        x2 = F.scaled_dot_product_attention(q, k, v, scale=0.1)
        x3 = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x4 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return x1, x2, x3, x4


device = 'cuda'

l1 = 100
l2 = 170
d = 64
q = torch.normal(0, 1, size=(1, 4, l1, d)).to(device)
k = torch.normal(0, 1, size=(1, 4, l2, d)).to(device)
v = torch.normal(0, 1, size=(1, 4, l2, d)).to(device)
attn_mask = torch.normal(0, 1, size=(1, 4, l1, l2), dtype=torch.float16).to(device) > 0

pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[q, k, v, attn_mask],
    input_shapes={q: (1, 4, None, d), k: (1, 4, None, d), v: (1, 4, None, d), attn_mask: (1, 4, None, None)},
    inputs_channel_order=nobuco.ChannelOrder.PYTORCH,
    outputs_channel_order=nobuco.ChannelOrder.PYTORCH,
    trace_shape=True,
)

model_path = 'scaled_dot_product_attention'
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
