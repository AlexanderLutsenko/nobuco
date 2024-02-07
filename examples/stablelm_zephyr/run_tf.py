import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
from transformers import AutoModelForCausalLM, AutoTokenizer

import tensorflow as tf

from examples.stablelm_zephyr.util import prepare_inputs_pt


prompt = 'Which famous math number begins with 1.6 ...?'
model_path = 'zephyr'
max_tokens = 1024
num_steps = 32


def pt2tf(t, dtype=None):
    return tf.convert_to_tensor(t, dtype=dtype)


tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-2-zephyr-1_6b', trust_remote_code=True)

next_token_id, attention_mask, positional_ids, past_key_values_flat, prompt_size = prepare_inputs_pt(prompt, tokenizer, past_kv_flattened=True)
next_token_id = pt2tf(next_token_id, tf.int64)
attention_mask = pt2tf(attention_mask, tf.int64)
positional_ids = pt2tf(positional_ids, tf.int64)
past_key_values_flat = [tf.convert_to_tensor(t.numpy()) for t in past_key_values_flat]

keras_model = tf.saved_model.load(model_path)
print('Model loaded')

generated_tokens = []

# First run, no padding
start = time.time()
outputs = keras_model([next_token_id, attention_mask, positional_ids, *past_key_values_flat])
print('First elapsed:', time.time() - start)

# Pad inputs to `max_tokens`
past_key_values_flat = outputs[1:]
s1, s2, s3, s4 = past_key_values_flat[0].shape
kv_pad = tf.zeros(shape=(s1, s2, max_tokens - s3 - 1, s4))
past_key_values_flat = [tf.concat([kv_pad, t], axis=2) for t in past_key_values_flat]

next_logits = outputs[0]
next_logits = next_logits[:, -1:]
next_token_id = tf.argmax(next_logits, axis=-1)

m1, m2 = attention_mask.shape
mask_pad = tf.zeros(shape=(m1, max_tokens - m2), dtype=tf.int64)
attention_mask = tf.concat([mask_pad, attention_mask, tf.ones((1, 1), dtype=tf.int64)], axis=1)[:, 1:]

positional_ids = tf.convert_to_tensor([[prompt_size + 0]], dtype=tf.int64)

generated_tokens.append(next_token_id.numpy().item())

# Run inference on padded inputs
for cycle in range(1, num_steps):
    start = time.time()
    outputs = keras_model([next_token_id, attention_mask, positional_ids, *past_key_values_flat])
    print('Elapsed:', time.time() - start)

    past_key_values_flat = outputs[1:]
    past_key_values_flat = [t[:, :, 1:] for t in past_key_values_flat]

    next_logits = outputs[0]
    next_logits = next_logits[:, -1:]
    next_token_id = tf.argmax(next_logits, axis=-1)

    attention_mask = tf.concat([attention_mask, tf.ones((1, 1), dtype=tf.int64)], axis=1)[:, 1:]

    positional_ids = tf.convert_to_tensor([[prompt_size + cycle]], dtype=tf.int64)

    generated_tokens.append(next_token_id.numpy().item())

# Decode and print the generated sequence
print(tokenizer.decode(generated_tokens, skip_special_tokens=False))
