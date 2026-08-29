from __future__ import annotations

import torch

from apex.model.transformer import ApexForCausalLM


@torch.inference_mode()
def generate(
    model: ApexForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    generated = input_ids
    past = None
    current = input_ids
    for _ in range(max_new_tokens):
        output = model(current, past_key_values=past, use_cache=True)
        logits = output.logits[:, -1, :] / temperature
        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, k).values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        current = torch.multinomial(probabilities, num_samples=1, generator=generator)
        generated = torch.cat((generated, current), dim=1)
        past = output.past_key_values
        if eos_token_id is not None and bool(torch.all(current == eos_token_id)):
            break
    return generated
