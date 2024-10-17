import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
import keras
import numpy as np

class MultiheadAttentionModel(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # Reshape input to (seq_len, batch_size, embed_dim)
        x = x.permute(2, 0, 1)
        
        # Self-attention: use the same tensor for query, key, and value
        attn_output, _ = self.multihead_attn(x, x, x)
        
        # Reshape output back to (batch_size, embed_dim, seq_len)
        return attn_output.permute(1, 2, 0)

# Create the model
model = MultiheadAttentionModel()

# Create a dummy input tensor
dummy_input = torch.randn(1, 128, 4096)  # (batch_size, embed_dim, seq_len)

# Convert PyTorch model to Keras
keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_input]
)

# Save the Keras model
model_path = 'multihead_attention'
keras_model.save(model_path + '.h5')
print('Model saved')

# Define custom objects
custom_objects = {'WeightLayer': WeightLayer}

# Load the Keras model
keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')

# Convert to TFLite
converter = TFLiteConverter.from_keras_model_file(model_path + '.h5', custom_objects=custom_objects)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# Save the TFLite model
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
print('TFLite model saved')

# Test the models
pytorch_output = model(dummy_input)
keras_output = keras_model.predict(dummy_input.numpy())

print("PyTorch output shape:", pytorch_output.shape)
print("Keras output shape:", keras_output.shape)
print("Output difference:", np.abs(pytorch_output.detach().numpy() - keras_output).max())