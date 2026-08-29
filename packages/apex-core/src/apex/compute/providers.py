from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from apex.hardware import HardwareAdvice, HardwareAdvisor, HardwareReport, detect_hardware


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    available: bool
    capability_status: str
    reason: str
    hardware: HardwareReport
    advice: HardwareAdvice

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hardware"]["gpu_count"] = self.hardware.gpu_count
        return payload


class ComputeProvider(ABC):
    name: str

    @abstractmethod
    def inspect(self, path: str | Path = ".") -> ProviderStatus: ...

    @abstractmethod
    def artifact_root(self) -> Path: ...


class LocalCPUProvider(ComputeProvider):
    name = "local-cpu"

    def inspect(self, path: str | Path = ".") -> ProviderStatus:
        hardware = detect_hardware(path)
        return ProviderStatus(
            provider=self.name,
            available=True,
            capability_status="IMPLEMENTED",
            reason="CPU execution is available for tiny pipeline validation.",
            hardware=hardware,
            advice=HardwareAdvisor().recommend(hardware),
        )

    def artifact_root(self) -> Path:
        return Path("checkpoints")


class LocalGPUProvider(ComputeProvider):
    name = "local-gpu"

    def inspect(self, path: str | Path = ".") -> ProviderStatus:
        hardware = detect_hardware(path)
        available = hardware.cuda_available
        return ProviderStatus(
            provider=self.name,
            available=available,
            capability_status="EXPERIMENTAL" if available else "REQUIRES_GPU",
            reason=(
                "A CUDA device is measurable through PyTorch."
                if available
                else "PyTorch did not report a CUDA compute device."
            ),
            hardware=hardware,
            advice=HardwareAdvisor().recommend(hardware),
        )

    def artifact_root(self) -> Path:
        return Path("checkpoints")


class KaggleProvider(ComputeProvider):
    name = "kaggle"

    @property
    def in_kaggle_runtime(self) -> bool:
        return bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()

    def inspect(self, path: str | Path = ".") -> ProviderStatus:
        hardware = detect_hardware(path)
        available = self.in_kaggle_runtime and hardware.cuda_available
        if not self.in_kaggle_runtime:
            reason = "This process is not running inside a detected Kaggle runtime."
        elif not hardware.cuda_available:
            reason = "Kaggle is detected, but PyTorch does not report an enabled CUDA accelerator."
        else:
            reason = "Kaggle and a CUDA accelerator are both measured."
        return ProviderStatus(
            provider=self.name,
            available=available,
            capability_status="EXPERIMENTAL" if available else "REQUIRES_KAGGLE",
            reason=reason,
            hardware=hardware,
            advice=HardwareAdvisor().recommend(hardware),
        )

    def require_training_runtime(self, path: str | Path = ".") -> ProviderStatus:
        status = self.inspect(path)
        if not status.available:
            raise RuntimeError(status.reason)
        return status

    def artifact_root(self) -> Path:
        if not self.in_kaggle_runtime:
            raise RuntimeError("Kaggle artifact storage is only available inside Kaggle")
        target = Path("/kaggle/working/apex-artifacts")
        target.mkdir(parents=True, exist_ok=True)
        return target
