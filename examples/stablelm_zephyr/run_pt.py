import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.stablelm_zephyr.util import prepare_inputs_pt


prompt = 'Which famous math number begins with 1.6 ...?'
max_tokens = 1024
num_steps = 32
device = 'cpu'


tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-2-zephyr-1_6b', trust_remote_code=True)
next_token_id, attention_mask, positional_ids, past_key_values_flat, prompt_size = prepare_inputs_pt(prompt, tokenizer, past_kv_flattened=False, device=device)

pytorch_model = AutoModelForCausalLM.from_pretrained('stabilityai/stablelm-2-zephyr-1_6b', trust_remote_code=True).eval().to(device)
print('Model loaded')

generated_tokens = []

# First run, no padding
start = time.time()
with torch.inference_mode():
    outputs = pytorch_model(next_token_id, attention_mask, positional_ids, past_key_values_flat)
print('First elapsed:', time.time() - start)

# Pad inputs to `max_tokens`
past_key_values_flat = outputs['past_key_values']
s1, s2, s3, s4 = past_key_values_flat[0][0].shape
kv_pad = torch.zeros(size=(s1, s2, max_tokens - s3 - 1, s4))
past_key_values_flat = [(torch.cat([kv_pad, k], dim=2), torch.cat([kv_pad, v], dim=2)) for (k, v) in past_key_values_flat]

next_logits = outputs['logits']
next_logits = next_logits[:, -1:]
next_token_id = torch.argmax(next_logits, dim=-1)

m1, m2 = attention_mask.shape
mask_pad = torch.zeros(size=(m1, max_tokens - m2))
attention_mask = torch.cat([mask_pad, attention_mask, torch.ones((1, 1), dtype=torch.int64, device=device)], dim=1)[:, 1:]

positional_ids = torch.asarray([[prompt_size + 0]], dtype=torch.int64, device=device)

generated_tokens.append(next_token_id.detach().cpu().numpy().item())

# Run inference on padded inputs
for cycle in range(1, num_steps):
    start = time.time()
    with torch.inference_mode():
        outputs = pytorch_model(next_token_id, attention_mask, positional_ids, past_key_values_flat)
    print('Elapsed:', time.time() - start)

    past_key_values_flat = outputs['past_key_values']
    past_key_values_flat = [(k[:, :, 1:], v[:, :, 1:]) for (k, v) in past_key_values_flat]

    next_logits = outputs['logits']
    next_logits = next_logits[:, -1:]
    next_token_id = torch.argmax(next_logits, dim=-1)

    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.int64, device=device)], dim=1)[:, 1:]

    positional_ids = torch.asarray([[prompt_size + cycle]], dtype=torch.int64, device=device)

    generated_tokens.append(next_token_id.detach().cpu().numpy().item())

# Decode and print the generated sequence
print(tokenizer.decode(generated_tokens, skip_special_tokens=False))
