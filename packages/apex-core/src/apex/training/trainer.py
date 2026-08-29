from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from apex.model.transformer import ApexForCausalLM


@dataclass
class TrainerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "none"
    scheduler: str = "cosine"
    warmup_steps: int = 0
    max_steps: int = 1_000
    minimum_learning_rate_ratio: float = 0.1
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if self.scheduler not in {"constant", "linear", "cosine"}:
            raise ValueError("scheduler must be constant, linear, or cosine")
        if self.max_steps < 1 or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be non-negative and less than max_steps")


class Trainer:
    def __init__(self, model: ApexForCausalLM, config: TrainerConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._learning_rate_multiplier
        )
        self.step = 0
        self.micro_step = 0
        self.device_type = next(model.parameters()).device.type
        if config.mixed_precision != "none" and self.device_type != "cuda":
            raise ValueError("mixed precision training currently requires a CUDA device")
        self.autocast_dtype = {
            "none": None,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[config.mixed_precision]
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.mixed_precision == "fp16" and self.device_type == "cuda"
        )
        torch.manual_seed(config.seed)
        self.optimizer.zero_grad(set_to_none=True)

    def train_step(self, input_ids: torch.Tensor) -> float:
        self.model.train()
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            output = self.model(input_ids, labels=input_ids)
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        scaled_loss = output.loss / self.config.gradient_accumulation_steps
        self.scaler.scale(scaled_loss).backward()
        self.micro_step += 1
        if self.micro_step % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
        return float(output.loss.detach())

    def _learning_rate_multiplier(self, step: int) -> float:
        if self.config.warmup_steps and step < self.config.warmup_steps:
            return max((step + 1) / self.config.warmup_steps, 1e-8)
        if self.config.scheduler == "constant":
            return 1.0
        decay_steps = max(self.config.max_steps - self.config.warmup_steps, 1)
        progress = min(max((step - self.config.warmup_steps) / decay_steps, 0.0), 1.0)
        floor = self.config.minimum_learning_rate_ratio
        if self.config.scheduler == "linear":
            return floor + (1.0 - floor) * (1.0 - progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step,
                "micro_step": self.micro_step,
                "model_config": self.model.config.to_dict(),
                "trainer_config": asdict(self.config),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
                "metadata": metadata or {},
            },
            target,
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint_data["model"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer"])
        self.scheduler.load_state_dict(checkpoint_data["scheduler"])
        self.scaler.load_state_dict(checkpoint_data.get("scaler", {}))
        self.step = int(checkpoint_data["step"])
        self.micro_step = int(checkpoint_data.get("micro_step", self.step))
        torch.set_rng_state(checkpoint_data["torch_rng_state"])
        cuda_rng_state = checkpoint_data.get("cuda_rng_state", [])
        if torch.cuda.is_available() and cuda_rng_state:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        return dict(checkpoint_data.get("metadata", {}))
