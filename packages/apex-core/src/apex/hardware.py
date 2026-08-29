from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GPUDevice:
    index: int
    name: str
    vendor: str
    vram_total_gb: float | None
    vram_free_gb: float | None
    compute_capable: bool


@dataclass(frozen=True)
class HardwareReport:
    operating_system: str
    architecture: str
    cpu_model: str
    logical_cpus: int
    ram_total_gb: float | None
    ram_available_gb: float | None
    disk_total_gb: float
    disk_free_gb: float
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    bf16_supported: bool
    gpus: tuple[GPUDevice, ...]
    environment: str

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpu_count"] = self.gpu_count
        return payload


@dataclass(frozen=True)
class HardwareAdvice:
    mode: str
    model_profile: str
    precision: str
    quantization: str
    batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    sequence_length: int
    gradient_checkpointing: bool
    inference_engine: str
    training_supported: bool
    rationale: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HardwareAdvisor:
    """Produce conservative settings from measured resources."""

    def recommend(self, report: HardwareReport) -> HardwareAdvice:
        compute_gpus = [gpu for gpu in report.gpus if gpu.compute_capable]
        known_free_vram = [gpu.vram_free_gb for gpu in compute_gpus if gpu.vram_free_gb]
        available_vram = min(known_free_vram) if known_free_vram else None
        if not report.cuda_available or available_vram is None:
            return HardwareAdvice(
                mode="LOCAL_CPU",
                model_profile="apex-tiny",
                precision="fp32",
                quantization="none",
                batch_size=1,
                micro_batch_size=1,
                gradient_accumulation_steps=8,
                sequence_length=128,
                gradient_checkpointing=True,
                inference_engine="pytorch",
                training_supported=True,
                rationale=(
                    "No measured CUDA compute device with free VRAM is available.",
                    "Only the experimental tiny pipeline is recommended for local validation.",
                ),
            )

        precision = "bf16" if report.bf16_supported else "fp16"
        if available_vram >= 16:
            batch_size, micro_batch, accumulation, sequence = 16, 4, 4, 2048
        elif available_vram >= 8:
            batch_size, micro_batch, accumulation, sequence = 8, 2, 4, 1024
        elif available_vram >= 4:
            batch_size, micro_batch, accumulation, sequence = 4, 1, 4, 512
        else:
            return HardwareAdvice(
                mode="LOCAL_GPU",
                model_profile="apex-tiny",
                precision=precision,
                quantization="none",
                batch_size=1,
                micro_batch_size=1,
                gradient_accumulation_steps=8,
                sequence_length=256,
                gradient_checkpointing=True,
                inference_engine="pytorch",
                training_supported=True,
                rationale=(
                    f"Only {available_vram:.2f} GiB of free CUDA VRAM was measured.",
                    "The larger development configuration is not selected.",
                ),
            )
        return HardwareAdvice(
            mode="KAGGLE_RESEARCH" if report.environment == "kaggle" else "LOCAL_GPU",
            model_profile="apex-100m",
            precision=precision,
            quantization="none",
            batch_size=batch_size,
            micro_batch_size=micro_batch,
            gradient_accumulation_steps=accumulation,
            sequence_length=sequence,
            gradient_checkpointing=True,
            inference_engine="pytorch",
            training_supported=True,
            rationale=(
                f"The least-free CUDA device reports {available_vram:.2f} GiB available VRAM.",
                "Settings reserve headroom and must still pass an allocation smoke test.",
            ),
        )


def detect_hardware(path: str | Path = ".") -> HardwareReport:
    total, _, free = shutil.disk_usage(path)
    ram_total, ram_available = _memory_bytes()
    torch_available = False
    torch_version = None
    cuda_available = False
    cuda_version = None
    bf16_supported = False
    gpus: list[GPUDevice] = []
    try:
        import torch

        torch_available = True
        torch_version = str(torch.__version__)
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        if cuda_available:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                with torch.cuda.device(index):
                    free_bytes, _ = torch.cuda.mem_get_info()
                gpus.append(
                    GPUDevice(
                        index=index,
                        name=properties.name,
                        vendor="NVIDIA",
                        vram_total_gb=_gib(properties.total_memory),
                        vram_free_gb=_gib(free_bytes),
                        compute_capable=True,
                    )
                )
    except (ImportError, RuntimeError, OSError):
        pass

    if not gpus:
        gpus.extend(_display_adapters())

    return HardwareReport(
        operating_system=platform.platform(),
        architecture=platform.machine(),
        cpu_model=_cpu_model(),
        logical_cpus=os.cpu_count() or 1,
        ram_total_gb=_gib(ram_total),
        ram_available_gb=_gib(ram_available),
        disk_total_gb=_gib(total) or 0.0,
        disk_free_gb=_gib(free) or 0.0,
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        bf16_supported=bf16_supported,
        gpus=tuple(gpus),
        environment="kaggle" if _is_kaggle() else "local",
    )


def _gib(value: int | None) -> float | None:
    return round(value / 2**30, 2) if value is not None else None


def _is_kaggle() -> bool:
    return bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()


def _cpu_model() -> str:
    if os.name == "nt":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (OSError, ImportError):
            pass
    name = platform.processor().strip()
    if name:
        return name
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[-1].strip()
    return "unknown"


def _memory_bytes() -> tuple[int | None, int | None]:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical), int(status.available_physical)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = page_size * os.sysconf("SC_PHYS_PAGES")
        available = page_size * os.sysconf("SC_AVPHYS_PAGES")
        return int(total), int(available)
    except (AttributeError, OSError, ValueError):
        return None, None


def _display_adapters() -> list[GPUDevice]:
    if os.name != "nt":
        return []
    powershell = shutil.which("powershell")
    if powershell is None:
        return []
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(  # noqa: S603 - executable and command are fixed
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
        raw = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    items = raw if isinstance(raw, list) else [raw]
    devices = []
    for index, item in enumerate(items):
        name = str(item.get("Name") or "unknown")
        lowered = name.lower()
        vendor = (
            "NVIDIA"
            if "nvidia" in lowered
            else "AMD"
            if "amd" in lowered
            else "Intel"
            if "intel" in lowered
            else "unknown"
        )
        adapter_ram = item.get("AdapterRAM")
        devices.append(
            GPUDevice(
                index=index,
                name=name,
                vendor=vendor,
                vram_total_gb=_gib(int(adapter_ram)) if adapter_ram else None,
                vram_free_gb=None,
                compute_capable=False,
            )
        )
    return devices
