from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from apex.model.transformer import ApexForCausalLM


@dataclass
class TrainerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "none"
    seed: int = 1337


class Trainer:
    def __init__(self, model: ApexForCausalLM, config: TrainerConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.step = 0
        torch.manual_seed(config.seed)

    def train_step(self, input_ids: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(input_ids, labels=input_ids)
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.step += 1
        return float(output.loss.detach())

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "model_config": self.model.config.to_dict(),
                "trainer_config": asdict(self.config),
                "metadata": metadata or {},
            },
            target,
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint_data["model"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer"])
        self.step = int(checkpoint_data["step"])
        return dict(checkpoint_data.get("metadata", {}))
