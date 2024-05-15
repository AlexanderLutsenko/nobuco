import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.type_cast import dtype_pytorch2keras

import torch
from torch import nn, ops

import tensorflow as tf
import keras
from tensorflow.lite.python.lite import TFLiteConverter


class Identity(nn.Module):
    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.scales = 1.0
        self.zero_points = 128
        # reduce_range=True to avoid occasional overflow
        weight = torch.quantize_per_tensor_dynamic(weight, torch.qint8, reduce_range=True)
        self.packed = torch.ops.quantized.linear_prepack(weight, bias)

    def forward(self, x: torch.Tensor):
        x = torch.quantize_per_tensor(x, self.scales, self.zero_points, torch.quint8)
        x = Identity()(x)
        x = ops.quantized.linear(x, self.packed, self.scales, self.zero_points)
        x = x.dequantize()
        return x


@nobuco.converter(torch.quantize_per_tensor, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_quantize_per_tensor(x: torch.Tensor, scale: float, zero: int, dtype: torch.dtype):
    dtype_tf = dtype_pytorch2keras(dtype)

    min_q = -zero * scale
    max_q = (255 - zero) * scale

    def func(x, *args, **kwargs):
        q = tf.quantization.quantize(x, min_q, max_q, dtype_tf)
        tensor_tf = q.output
        tensor_tf.output_min = q.output_min
        tensor_tf.output_max = q.output_max
        return tensor_tf
    return func


@nobuco.converter(torch.Tensor.dequantize, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_dequantize(x: torch.Tensor):
    scale, zero = x.q_scale(), x.q_zero_point()
    min_q = -zero * scale
    max_q = (255 - zero) * scale

    def func(x):
        return tf.quantization.dequantize(x, min_q, max_q)
    return func


@nobuco.converter(ops.quantized.linear, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_linear_quantized(x: torch.Tensor, packed, out_scale, out_zero):
    weight, bias = ops.quantized.linear_unpack(packed)
    weight = weight.dequantize()
    bias = bias.dequantize()

    out_filters, in_filters = weight.shape

    weights = weight.cpu().detach().numpy()
    weights = weights.transpose(1, 0)
    biases = bias.cpu().detach().numpy()
    params = [weights, biases]
    layer = keras.layers.Dense(out_filters, use_bias=True, weights=params)

    scale, zero = x.q_scale(), x.q_zero_point()
    min_q = -zero * scale
    max_q = (255 - zero) * scale

    min_q_out = -out_zero * out_scale
    max_q_out = (255 - out_zero) * out_scale

    def func(x, packed, out_scale, out_zero):
        x = tf.quantization.dequantize(x, min_q, max_q)

        x = layer(x)

        q = tf.quantization.quantize(x, min_q_out, max_q_out, tf.quint8)
        tensor_tf = q.output
        tensor_tf.output_min = q.output_min
        tensor_tf.output_max = q.output_max
        return tensor_tf
    return func


weight = torch.rand((100, 100))
bias = torch.rand((100,))
model = Model(weight, bias)

x = torch.rand(size=(1, 100)) * 200 - 100

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[x],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'quant'
keras_model.save(model_path + '.h5')

converter = TFLiteConverter.from_keras_model_file(model_path + '.h5')
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
