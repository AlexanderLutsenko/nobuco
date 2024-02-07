import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy

import torch

import tensorflow as tf

from transformers import AutoModelForCausalLM, AutoTokenizer


device = 'cpu'

# FIXME: Currently, you have to slightly modify the model's source code for it to convert correctly
# See https://github.com/AlexanderLutsenko/nobuco/issues/22#issuecomment-1930251236
model = AutoModelForCausalLM.from_pretrained('stabilityai/stablelm-2-zephyr-1_6b',trust_remote_code=True).to(device)

tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-2-zephyr-1_6b', trust_remote_code=True)

prompt = [{'role': 'user', 'content': 'Which famous math number begins with 1.6 ...?'}]

inputs = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=False,
    return_tensors='pt'
).to(device)

# Cut inputs to save memory
inputs = inputs[:, :2]

# Create a mask for the original inputs (it should be all 1s since there's no padding)
attention_mask = torch.ones(inputs.shape, dtype=torch.int64)
positional_ids = torch.arange(0, inputs.shape[1]).unsqueeze(0)

past_key_values = []
for i in range(24):
    k = torch.zeros(size=(1, 32, 0, 64))
    v = torch.zeros(size=(1, 32, 0, 64))
    past_key_values.append((k, v))
past_key_values = tuple(past_key_values)


input_shapes = {inputs: (1, None), attention_mask: (1, None), positional_ids: (1, None)}
for (k, v) in past_key_values:
    input_shapes[k] = (1, 32, None, 64)
    input_shapes[v] = (1, 32, None, 64)

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[inputs, attention_mask, positional_ids, past_key_values],
    input_shapes=input_shapes,
    trace_shape=True,
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH,
)

model_path = 'zephyr'

keras_model.save(model_path)
print('Model saved')

keras_model_restored = tf.saved_model.load(model_path)
print('Model loaded')
