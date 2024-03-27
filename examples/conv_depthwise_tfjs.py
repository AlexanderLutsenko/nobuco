import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco.commons import ChannelOrder, ChannelOrderingStrategy

import torch.nn as nn
import torch

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import tensorflow as tf
from tensorflow import keras

import tensorflowjs


class ExampleModel(nn.Module):
    def __init__(self,
                 **kwargs):
        super(ExampleModel, self).__init__()
        self.layer1 = nn.Conv2d(16, 16, (3, 3), (1, 1), (0, 0), (1, 1), 16)
        self.layer2 = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


model = ExampleModel()

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
