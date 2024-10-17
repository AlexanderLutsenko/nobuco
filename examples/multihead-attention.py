import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
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

class CustomMultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        
        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

@nobuco.converter(MultiheadAttentionModel, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_MultiheadAttentionModel(self, x):
    custom_mha = CustomMultiHeadAttention(d_model=128, num_heads=8)
    
    def func(x):
        x = tf.transpose(x, perm=[2, 0, 1])  # (seq_len, batch_size, embed_dim)
        output, _ = custom_mha(x, x, x)
        return tf.transpose(output, perm=[1, 2, 0])  # (batch_size, embed_dim, seq_len)
    
    return func

# Create the model
model = MultiheadAttentionModel().eval()

# Create a dummy input tensor
dummy_input = torch.randn(1, 128, 4096)  # (batch_size, embed_dim, seq_len)

# Convert PyTorch model to Keras
keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_input],
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH
)

# Save the Keras model
model_path = 'multihead_attention'
keras_model.save(model_path + '.h5')
print('Model saved')

# Define custom objects
custom_objects = {'WeightLayer': WeightLayer, 'CustomMultiHeadAttention': CustomMultiHeadAttention}

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
with torch.no_grad():
    pytorch_output = model(dummy_input)
keras_output = keras_model.predict(dummy_input.numpy())

print("PyTorch output shape:", pytorch_output.shape)
print("Keras output shape:", keras_output.shape)
print("Output difference:", np.abs(pytorch_output.numpy() - keras_output).max())