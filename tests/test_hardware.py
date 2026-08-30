from __future__ import annotations

import pytest

from apex.compute import KaggleProvider
from apex.hardware import GPUDevice, HardwareAdvisor, HardwareReport, detect_hardware


def test_hardware_report_and_advice_are_measurement_based(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    report = detect_hardware(tmp_path)
    advice = HardwareAdvisor().recommend(report)
    assert report.logical_cpus >= 1
    assert report.disk_total_gb >= report.disk_free_gb >= 0
    assert report.ram_total_gb is None or report.ram_total_gb > 0
    assert report.gpu_count == len(report.gpus)
    if not report.cuda_available:
        assert advice.mode == "LOCAL_CPU"
        assert advice.model_profile == "apex-tiny"
        assert advice.precision == "fp32"


def test_kaggle_provider_refuses_an_unavailable_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    provider = KaggleProvider()
    if provider.in_kaggle_runtime:
        pytest.skip("test is running inside Kaggle")
    status = provider.inspect(tmp_path)
    assert not status.available
    assert status.capability_status == "REQUIRES_KAGGLE"
    with pytest.raises(RuntimeError):
        provider.require_training_runtime(tmp_path)


def test_kaggle_gpu_vram_cuda_advice_selects_apex_100m(monkeypatch) -> None:
    report = HardwareReport(
        operating_system="Linux",
        architecture="x86_64",
        cpu_model="fixture",
        logical_cpus=4,
        ram_total_gb=16,
        ram_available_gb=12,
        disk_total_gb=100,
        disk_free_gb=80,
        torch_available=True,
        torch_version="fixture",
        cuda_available=True,
        cuda_version="fixture",
        bf16_supported=False,
        gpus=(GPUDevice(0, "fixture GPU", "NVIDIA", 16, 15, True),),
        environment="kaggle",
    )
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr("apex.compute.providers.detect_hardware", lambda _path: report)
    status = KaggleProvider().inspect()
    assert status.available is True
    assert status.hardware.cuda_available is True
    assert status.hardware.gpus[0].vram_free_gb == 15
    assert status.advice.mode == "KAGGLE_RESEARCH"
    assert status.advice.model_profile == "apex-100m"
