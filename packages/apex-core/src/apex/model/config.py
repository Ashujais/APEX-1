from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ApexConfig:
    vocab_size: int = 32_000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2_048
    max_sequence_length: int = 2_048
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            self.vocab_size,
            self.hidden_size,
            self.num_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.intermediate_size,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def estimated_parameter_count(self) -> int:
        head_dim = self.head_dim
        key_value_width = self.num_key_value_heads * head_dim
        attention = (
            self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * key_value_width
            + self.hidden_size * self.hidden_size
        )
        feed_forward = 3 * self.hidden_size * self.intermediate_size
        layer_norms = 2 * self.hidden_size
        embeddings = self.vocab_size * self.hidden_size
        output = 0 if self.tie_word_embeddings else embeddings
        return (
            embeddings
            + output
            + self.num_layers * (attention + feed_forward + layer_norms)
            + self.hidden_size
        )

    @classmethod
    def tiny(cls, vocab_size: int = 300) -> ApexConfig:
        return cls(
            vocab_size=vocab_size,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=160,
            max_sequence_length=128,
        )

    @classmethod
    def apex_100m(cls, vocab_size: int = 32_000) -> ApexConfig:
        """Planned Kaggle-first profile; constructing it does not start training."""
        return cls(vocab_size=vocab_size, gradient_checkpointing=True)
