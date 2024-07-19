import copy

import numpy as np

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverterV2
import keras


class EMAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_in = nn.Linear(8, 4)
        self.linear_out = nn.Linear(4, 2)
        self.alpha = 0.8
        self.register_buffer("x_ema", torch.ones((1, 4)))

    def forward_stateless(self, x, x_ema):
        x = self.linear_in(x)
        x_ema *= self.alpha
        x_ema += (1.0 - self.alpha) * x
        x = self.linear_out(x_ema)
        return x, x_ema

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, self.x_ema = self.forward_stateless(x, self.x_ema)
        return x


class EMAModelKeras(keras.layers.Layer):
    def __init__(self, forward_stateless, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stateful = True  # Mark as stateful, otherwise reset_states() will not work
        self.forward_stateless = forward_stateless
        self.x_ema = tf.Variable(tf.ones((1, 4)), trainable=False)

    def get_config(self):
        config = super().get_config()
        config.update({
            "forward_stateless": self.forward_stateless,
        })
        return config

    @tf.function
    def call(self, x):
        x, x_ema = self.forward_stateless([x, self.x_ema])
        self.x_ema.assign(x_ema)
        return x

    def reset_states(self):
        self.x_ema.assign(tf.ones((1, 4)))


@nobuco.converter(EMAModel, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_EMAModel(self, x):
    order = ChannelOrder.TENSORFLOW
    kwargs = {'inputs_channel_order': order, 'outputs_channel_order': order}
    x_ema = torch.asarray(self.x_ema)
    forward_stateless = nobuco.pytorch_to_keras(self.forward_stateless, [x, x_ema], **kwargs)
    return EMAModelKeras(forward_stateless)


x = torch.zeros(size=(1, 8)) + 3
pytorch_module = EMAModel().eval()

copy.deepcopy(pytorch_module)

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x],
    trace_shape=True,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
)

# Reset x_ema
keras_model.reset_states()

# Save/load Keras model
model_path = 'stateful2'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer, 'EMAModelKeras': EMAModelKeras}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')


# Save TFLite model
converter = TFLiteConverterV2.from_keras_model(keras_model)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)


# Check if the Keras model is indeed stateful
x_tf = tf.zeros((1, 8)) + 3
for _ in range(10):
    y = keras_model_restored(x_tf)
    print(y)


# Check if the TFLite model is indeed stateful

def run_tflite(interpreter, *np_inputs):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for inp, inp_detail in zip(np_inputs, input_details):
        interpreter.set_tensor(inp_detail['index'], inp)

    interpreter.invoke()
    outputs = [interpreter.get_tensor(out_detail['index']) for out_detail in output_details]
    return outputs


interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

x_np = np.zeros((1, 8), dtype=np.float32) + 3
for _ in range(10):
    outputs = run_tflite(interpreter, x_np)
    print(outputs)
