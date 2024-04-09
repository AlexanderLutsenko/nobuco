import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy

import torch.nn as nn
import torch

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import tensorflow as tf
from tensorflow import keras

import tensorflowjs


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 16

        self.convs_2d = nn.ModuleList()
        for out_channels in [16, 32]:
            for kernel_size in [(3, 3)]:
                for padding in [(1, 1), 'same']:
                    for groups in [1, min(in_channels, out_channels)]:
                        for dilation in [1, 2]:
                            conv = nn.Conv2d(16, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, dilation=dilation)
                            self.convs_2d.append(conv)

        self.convs_transpose_2d = nn.ModuleList()
        for out_channels in [16, 32]:
            for kernel_size in [(3, 3)]:
                for padding in [(1, 1)]:
                    for groups in [1, min(in_channels, out_channels)]:
                        for dilation in [1, 2]:
                            conv = nn.ConvTranspose2d(16, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, dilation=dilation)
                            self.convs_transpose_2d.append(conv)

        self.convs_1d = nn.ModuleList()
        for out_channels in [16, 32]:
            for kernel_size in [3]:
                for padding in [1, 'same']:
                    for groups in [1, min(in_channels, out_channels)]:
                        for dilation in [1, 2]:
                            conv = nn.Conv1d(16, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, dilation=dilation)
                            self.convs_1d.append(conv)

        self.convs_transpose_1d = nn.ModuleList()
        for out_channels in [16, 32]:
            for kernel_size in [3]:
                for padding in [1]:
                    for groups in [1, min(in_channels, out_channels)]:
                        for dilation in [1]:
                            conv = nn.ConvTranspose1d(16, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, dilation=dilation)
                            self.convs_transpose_1d.append(conv)

    def forward(self, x):
        outputs = []

        for conv in self.convs_2d:
            out = conv(x)
            outputs.append(out)

        for conv in self.convs_transpose_2d:
            out = conv(x)
            outputs.append(out)

        b, c, h, w = x.shape
        x = x.view(b, c, h*w)
        x = nobuco.force_tensorflow_order(x)

        for conv in self.convs_1d:
            out = conv(x)
            outputs.append(out)

        for conv in self.convs_transpose_1d:
            out = conv(x)
            outputs.append(out)

        return outputs


model = MyModule()

# Put model in inference mode
model.eval()

x = torch.randn(1, 16, 113, 113, requires_grad=False)

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[x], kwargs=None)

model_name = 'conv_depthwise'
keras_model.save(model_name)
tensorflowjs.converters.convert_tf_saved_model(model_name, model_name + '.js')

# Assuming 'model' is your Keras model
full_model = tf.function(lambda x: keras_model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))

# Convert Keras model to frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

# Print the input and output tensors
print("Frozen model inputs: ", frozen_func.inputs)
print("Frozen model outputs: ", frozen_func.outputs)

# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir='.',
                  name='conv_depthwise_tfjs.pb',
                  as_text=False)
