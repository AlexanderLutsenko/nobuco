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
        x = x.permute(2, 0, 1)
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output.permute(1, 2, 0)

class FurtherImprovedMultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = keras.layers.Dense(d_model, use_bias=True)
        self.wk = keras.layers.Dense(d_model, use_bias=True)
        self.wv = keras.layers.Dense(d_model, use_bias=True)
        
        self.dense = keras.layers.Dense(d_model)
        
        # Initialize weights to match PyTorch's default initialization
        self.wq.build((None, d_model))
        self.wk.build((None, d_model))
        self.wv.build((None, d_model))
        self.dense.build((None, d_model))
        
        for layer in [self.wq, self.wk, self.wv, self.dense]:
            k = np.sqrt(1.0 / layer.weights[0].shape[0])
            layer.set_weights([
                np.random.uniform(-k, k, layer.weights[0].shape),
                np.zeros(layer.weights[1].shape)
            ])
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v):
        batch_size = tf.shape(q)[1]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(tf.transpose(q, [1, 0, 2]), batch_size)
        k = self.split_heads(tf.transpose(k, [1, 0, 2]), batch_size)
        v = self.split_heads(tf.transpose(v, [1, 0, 2]), batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        output = tf.transpose(output, [1, 0, 2])
        
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
    custom_mha = FurtherImprovedMultiHeadAttention(d_model=128, num_heads=8)
    
    def func(x):
        x = tf.transpose(x, perm=[2, 0, 1])
        output, _ = custom_mha(x, x, x)
        return tf.transpose(output, perm=[1, 2, 0])
    
    return func

# Create the model
model = MultiheadAttentionModel().eval()

# Create a dummy input tensor
dummy_input = torch.randn(1, 128, 4096)

# Convert PyTorch model to Keras
keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_input],
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH
)

# Save the Keras model
model_path = 'multihead_attention'
keras_model.save(model_path + '.keras')
print('Model saved')

# Define custom objects
custom_objects = {'WeightLayer': WeightLayer, 'FurtherImprovedMultiHeadAttention': FurtherImprovedMultiHeadAttention}

# Load the Keras model
keras_model_restored = keras.models.load_model(model_path + '.keras', custom_objects=custom_objects)
print('Model loaded')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model_restored)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
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
print("Relative difference:", np.abs((pytorch_output.numpy() - keras_output) / pytorch_output.numpy()).max())