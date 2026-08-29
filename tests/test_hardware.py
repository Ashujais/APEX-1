from __future__ import annotations

import pytest

from apex.compute import KaggleProvider
from apex.hardware import HardwareAdvisor, detect_hardware


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
