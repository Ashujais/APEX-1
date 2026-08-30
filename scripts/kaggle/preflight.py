from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apex.compute import KaggleProvider
from apex.model import ApexConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate APEX-100M Kaggle prerequisites")
    parser.add_argument("--config", default="configs/apex-100m-kaggle.json")
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-license", default="unknown")
    return parser


def inspect_preflight(
    config_path: str | Path,
    dataset_path: str | Path | None,
    dataset_license: str,
) -> dict[str, Any]:
    source = Path(config_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Kaggle configuration must contain a model object")
    configured_model = ApexConfig(**payload["model"])
    expected_model = ApexConfig.apex_100m()
    config_ok = (
        payload.get("profile") == "apex-100m"
        and configured_model == expected_model
        and payload.get("expected_parameters") == configured_model.estimated_parameter_count
    )

    provider = KaggleProvider()
    provider_status = provider.inspect()
    dataset = Path(dataset_path) if dataset_path is not None else None
    dataset_ok = dataset is not None and dataset.is_file() and dataset.stat().st_size > 0
    license_ok = bool(dataset_license.strip()) and dataset_license.lower() != "unknown"
    known_vram = [
        gpu.vram_free_gb
        for gpu in provider_status.hardware.gpus
        if gpu.compute_capable and gpu.vram_free_gb is not None
    ]
    free_vram = min(known_vram) if known_vram else None
    required_vram = float(payload.get("minimum_free_vram_gb", 8))
    vram_ok = free_vram is not None and free_vram >= required_vram
    artifact_root: str | None = None
    artifact_ok = False
    if provider_status.available:
        artifact = provider.artifact_root()
        artifact_root = str(artifact)
        artifact_ok = artifact.is_dir()

    if not provider_status.available:
        readiness = "REQUIRES_KAGGLE"
    elif not vram_ok:
        readiness = "BLOCKED_INSUFFICIENT_VRAM"
    elif not dataset_ok or not license_ok:
        readiness = "REQUIRES_DATA"
    elif not config_ok or not artifact_ok:
        readiness = "BLOCKED_CONFIGURATION"
    else:
        readiness = "READY"
    return {
        "status": readiness,
        "capability_status": "PLANNED",
        "profile": "apex-100m",
        "estimated_parameters": configured_model.estimated_parameter_count,
        "checks": {
            "kaggle_runtime": provider.in_kaggle_runtime,
            "cuda": provider_status.hardware.cuda_available,
            "gpu_count": provider_status.hardware.gpu_count,
            "free_vram_gb": free_vram,
            "minimum_free_vram_gb": required_vram,
            "vram": vram_ok,
            "configuration": config_ok,
            "dataset": dataset_ok,
            "dataset_license": license_ok,
            "artifact_storage": artifact_ok,
        },
        "artifact_root": artifact_root,
        "note": "Preflight only; this command never starts training.",
    }


def main() -> int:
    arguments = build_parser().parse_args()
    result = inspect_preflight(
        arguments.config, arguments.dataset, arguments.dataset_license
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
