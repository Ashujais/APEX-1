from __future__ import annotations

from pathlib import Path

from apex.registry import RegistryStore
from apex.training import evaluate_small_checkpoint, train_small_text_model


def test_small_training_evaluation_and_registry_pipeline(tmp_path: Path) -> None:
    dataset = tmp_path / "corpus.txt"
    dataset.write_text(
        "APEX validates a real training path.\n"
        "Checkpoints and registries preserve evidence.\n"
        "This corpus is project-authored test data.\n",
        encoding="utf-8",
    )
    output = tmp_path / "run"
    registry_root = tmp_path / "registry"
    result = train_small_text_model(
        dataset,
        output,
        steps=1,
        sequence_length=16,
        vocabulary_size=270,
        checkpoint_interval=1,
        registry_root=registry_root,
        dataset_license="test-only",
        creator="pytest",
    )
    assert result.optimizer_steps == 1
    assert result.final_loss > 0
    assert Path(result.checkpoint).is_file()
    evaluation = evaluate_small_checkpoint(
        result.checkpoint,
        result.tokenizer,
        dataset,
        sequence_length=8,
        max_batches=1,
    )
    assert evaluation["status"] == "EXPERIMENTAL"
    assert evaluation["loss"] > 0

    resumed = train_small_text_model(
        dataset,
        tmp_path / "resumed",
        steps=2,
        sequence_length=16,
        vocabulary_size=270,
        checkpoint_interval=1,
        resume_from=result.checkpoint,
        registry_root=registry_root,
        dataset_license="test-only",
        creator="pytest",
    )
    assert resumed.run_id == result.run_id
    assert resumed.optimizer_steps == 2
    store = RegistryStore(registry_root)
    assert len(store.models.list()) == 1
    assert len(store.datasets.list()) == 1
    assert len(store.experiments.list()) == 1
