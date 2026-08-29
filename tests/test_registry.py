from __future__ import annotations

from pathlib import Path

import pytest

from apex.registry import DatasetRecord, ExperimentRecord, ModelRecord, RegistryStore, sha256_file


def test_registry_round_trip_hashes_artifacts_and_rejects_overwrite(tmp_path: Path) -> None:
    artifact = tmp_path / "dataset.txt"
    artifact.write_text("licensed example", encoding="utf-8")
    store = RegistryStore(tmp_path / "registry")
    record = DatasetRecord(
        id="example-data",
        version="1.0.0",
        status="RESEARCH",
        source="project-authored",
        license="test-only",
        creator="test",
    )
    registered = store.datasets.register(record, artifact_path=artifact)
    assert registered.artifact_hash == sha256_file(artifact)
    assert store.datasets.get("example-data", "1.0.0") == registered
    assert store.datasets.list() == [registered]
    with pytest.raises(FileExistsError):
        store.datasets.register(record)
    with pytest.raises(ValueError):
        store.datasets.get("../escape", "1")


def test_registry_validates_model_and_experiment_lifecycle(tmp_path: Path) -> None:
    store = RegistryStore(tmp_path / "registry")
    model = ModelRecord(id="apex-tiny", version="0.1", status="RESEARCH")
    experiment = ExperimentRecord(id="run-1", version="1", status="COMPLETED")
    assert store.models.register(model) == model
    assert store.experiments.register(experiment) == experiment
    with pytest.raises(ValueError):
        ModelRecord(id="bad", version="1", status="DEPLOYED")
