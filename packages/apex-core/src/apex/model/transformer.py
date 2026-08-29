from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from apex.model.config import ApexConfig
from apex.model.layers import KVCache, RMSNorm, TransformerBlock


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: tuple[KVCache, ...] | None = None


class ApexForCausalLM(nn.Module):
    def __init__(self, config: ApexConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: tuple[KVCache, ...] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if input_ids.shape[1] + past_length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds configured context length")
        if self.config.gradient_checkpointing and self.training and use_cache:
            raise ValueError("KV cache is incompatible with gradient checkpointing during training")

        hidden_states = self.dropout(self.token_embeddings(input_ids))
        present_values: list[KVCache] = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            if self.config.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    lambda states, current_layer=layer: current_layer(states, None, False)[0],
                    hidden_states,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = layer(hidden_states, past, use_cache)
            if present is not None:
                present_values.append(present)

        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(present_values) if use_cache else None,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
