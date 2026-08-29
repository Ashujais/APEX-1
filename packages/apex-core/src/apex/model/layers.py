from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from apex.model.config import ApexConfig

KVCache = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        input_dtype = value.dtype
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (self.weight * normalized).to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_sequence_length: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[positions], self.sin[positions]


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=query.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=query.dtype)
    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return value
    batch, kv_heads, sequence, head_dim = value.shape
    value = value[:, :, None, :, :].expand(batch, kv_heads, repeats, sequence, head_dim)
    return value.reshape(batch, kv_heads * repeats, sequence, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ApexConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_repeats = self.num_heads // self.num_kv_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.max_sequence_length, config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch, sequence, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(batch, sequence, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch, sequence, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch, sequence, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        past_length = 0 if past_key_value is None else past_key_value[0].shape[2]
        positions = torch.arange(past_length, past_length + sequence, device=hidden_states.device)
        cos, sin = self.rope(positions)
        query, key = apply_rotary(query, key, cos, sin)

        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        present = (key, value) if use_cache else None

        repeated_key = repeat_kv(key, self.kv_repeats)
        repeated_value = repeat_kv(value, self.kv_repeats)
        attention_mask = None
        is_causal = past_length == 0
        if past_length:
            key_positions = torch.arange(repeated_key.shape[2], device=hidden_states.device)
            query_positions = past_length + torch.arange(sequence, device=hidden_states.device)
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.out_proj(attended), present


class SwiGLU(nn.Module):
    def __init__(self, config: ApexConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, config: ApexConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attention_output, present = self.attention(
            self.attention_norm(hidden_states), past_key_value, use_cache
        )
        hidden_states = hidden_states + self.residual_dropout(attention_output)
        hidden_states = hidden_states + self.residual_dropout(
            self.mlp(self.mlp_norm(hidden_states))
        )
        return hidden_states, present
