from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass


@dataclass
class HardwareReport:
    operating_system: str
    architecture: str
    logical_cpus: int
    disk_total_gb: float
    disk_free_gb: float
    torch_available: bool
    cuda_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_hardware(path: str = ".") -> HardwareReport:
    total, _, free = shutil.disk_usage(path)
    torch_available = False
    cuda_available = False
    gpu_name = None
    gpu_vram_gb = None
    try:
        import torch

        torch_available = True
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    except ImportError:
        pass
    recommendation = (
        "Tiny CPU pipeline validation only"
        if not cuda_available
        else "Benchmark available VRAM before selecting a model configuration"
    )
    return HardwareReport(
        operating_system=platform.platform(),
        architecture=platform.machine(),
        logical_cpus=os.cpu_count() or 1,
        disk_total_gb=round(total / 2**30, 2),
        disk_free_gb=round(free / 2**30, 2),
        torch_available=torch_available,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        recommendation=recommendation,
    )
