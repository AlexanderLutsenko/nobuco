import torch


def prepare_inputs_pt(prompt, tokenizer, past_kv_flattened, device='cpu'):
    prompt = [{'role': 'user', 'content': prompt}]
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors='pt',
        truncation=True
    )
    prompt_size = inputs.shape[1]
    attention_mask = torch.ones(inputs.shape, dtype=torch.int64)
    positional_ids = torch.arange(0, inputs.shape[1])[None]

    past_key_values_flat = []
    for i in range(24):
        k = torch.zeros(size=(1, 32, 0, 64), device=device)
        v = torch.zeros(size=(1, 32, 0, 64), device=device)
        if past_kv_flattened:
            past_key_values_flat += [k, v]
        else:
            past_key_values_flat.append((k, v))
    return inputs.to(device), attention_mask.to(device), positional_ids.to(device), past_key_values_flat, prompt_size
