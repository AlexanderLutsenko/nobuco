import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch.nn.functional as F
from transformers import AutoTokenizer

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
import keras

from einops import einsum

from model import Mamba, ModelArgs, MambaBlock


args = ModelArgs(
    d_model=16,
    n_layer=3,
    vocab_size=50277
)

# https://github.com/johnma2006/mamba-minimal
model = Mamba(args).eval()

dummy_input = "Harry Potter"
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
input_ids = tokenizer(dummy_input, return_tensors='pt').input_ids


@nobuco.converter(F.softplus, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def softplus(input):
    return lambda input: tf.keras.activations.softplus(input)


class SelectiveScanLoop(tf.keras.layers.Layer):
    @tf.function
    def call(self, l, x, deltaA, deltaB_u, C):
        ys = tf.TensorArray(x.dtype, size=l)
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys = ys.write(i, y)
        y = tf.transpose(ys.stack(), (1, 0, 2))  # shape (b, l, d_in)
        return y


@nobuco.converter(MambaBlock.selective_scan_loop, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_selective_scan_loop(self, l, x, deltaA, deltaB_u, C):
    scan_loop = SelectiveScanLoop()

    def func(self, l, x, deltaA, deltaB_u, C):
        return scan_loop(l, x, deltaA, deltaB_u, C)
    return func


keras_model = nobuco.pytorch_to_keras(
    model,
    args=[input_ids], kwargs=None,
    input_shapes={input_ids: (1, None)},  # Annotate dynamic axes with None
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH,
    trace_shape=True,
)

model_path = 'mamba'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer, 'SelectiveScanLoop': SelectiveScanLoop}

from keras.src.saving.object_registration import custom_object_scope

with custom_object_scope(custom_objects):
    keras_model_restored = keras.models.load_model(model_path + '.h5')
    print('Model loaded')

    converter = TFLiteConverter.from_keras_model_file(model_path + '.h5')
    converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(model_path + '.tflite', 'wb') as f:
        f.write(tflite_model)
