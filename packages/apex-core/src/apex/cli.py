from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apex.compute import KaggleProvider, LocalCPUProvider, LocalGPUProvider
from apex.hardware import HardwareAdvisor, detect_hardware
from apex.registry import DatasetRecord, ExperimentRecord, ModelRecord, RegistryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apex", description="APEX-1 research CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    hardware = subcommands.add_parser("hardware", help="Inspect compute and safe settings")
    hardware.add_argument("--path", default=".")
    hardware.add_argument(
        "--provider", choices=("auto", "cpu", "gpu", "kaggle"), default="auto"
    )

    status = subcommands.add_parser("status", help="Inspect local platform/research readiness")
    status.add_argument("--registry-root", default=".apex/registry")

    train = subcommands.add_parser(
        "train-small", help="Run the real experimental tiny text pipeline"
    )
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", default="checkpoints/apex-tiny-run")
    train.add_argument("--steps", type=int, default=2)
    train.add_argument("--sequence-length", type=int, default=64)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--vocabulary-size", type=int, default=300)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--checkpoint-interval", type=int, default=10)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--resume-from")
    train.add_argument("--registry-root", default=".apex/registry")
    train.add_argument("--dataset-license", default="unknown")

    evaluate = subcommands.add_parser("evaluate", help="Measure loss on a real text dataset")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--tokenizer", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--sequence-length", type=int, default=64)
    evaluate.add_argument("--max-batches", type=int, default=8)

    benchmark = subcommands.add_parser("benchmark", help="Time PyTorch fallback generation")
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--tokenizer", required=True)
    benchmark.add_argument("--prompt", default="APEX")
    benchmark.add_argument("--max-new-tokens", type=int, default=16)

    for noun in ("model", "dataset", "experiment"):
        group = subcommands.add_parser(noun, help=f"Manage local {noun} registry records")
        actions = group.add_subparsers(dest="registry_action", required=True)
        listing = actions.add_parser("list")
        listing.add_argument("--registry-root", default=".apex/registry")
        show = actions.add_parser("show")
        show.add_argument("id")
        show.add_argument("version")
        show.add_argument("--registry-root", default=".apex/registry")
        register = actions.add_parser("register")
        register.add_argument("--record", required=True, help="JSON record file")
        register.add_argument("--artifact")
        register.add_argument("--registry-root", default=".apex/registry")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        result = run(arguments)
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


def run(arguments: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    if arguments.command == "hardware":
        if arguments.provider == "cpu":
            return LocalCPUProvider().inspect(arguments.path).to_dict()
        if arguments.provider == "gpu":
            return LocalGPUProvider().inspect(arguments.path).to_dict()
        if arguments.provider == "kaggle":
            return KaggleProvider().inspect(arguments.path).to_dict()
        report = detect_hardware(arguments.path)
        return {
            "hardware": report.to_dict(),
            "advice": HardwareAdvisor().recommend(report).to_dict(),
        }
    if arguments.command == "status":
        store = RegistryStore(arguments.registry_root)
        report = detect_hardware()
        return {
            "capability_status": "EXPERIMENTAL",
            "hardware": report.to_dict(),
            "advice": HardwareAdvisor().recommend(report).to_dict(),
            "providers": {
                "local_cpu": LocalCPUProvider().inspect().capability_status,
                "local_gpu": LocalGPUProvider().inspect().capability_status,
                "kaggle": KaggleProvider().inspect().capability_status,
            },
            "registry": {
                "models": len(store.models.list()),
                "datasets": len(store.datasets.list()),
                "experiments": len(store.experiments.list()),
                "backend": "local-filesystem",
            },
        }
    if arguments.command == "train-small":
        from apex.training import train_small_text_model

        return train_small_text_model(
            arguments.dataset,
            arguments.output,
            steps=arguments.steps,
            sequence_length=arguments.sequence_length,
            batch_size=arguments.batch_size,
            vocabulary_size=arguments.vocabulary_size,
            learning_rate=arguments.learning_rate,
            checkpoint_interval=arguments.checkpoint_interval,
            seed=arguments.seed,
            resume_from=arguments.resume_from,
            registry_root=arguments.registry_root,
            dataset_license=arguments.dataset_license,
        ).to_dict()
    if arguments.command == "evaluate":
        from apex.training import evaluate_small_checkpoint

        return evaluate_small_checkpoint(
            arguments.checkpoint,
            arguments.tokenizer,
            arguments.dataset,
            sequence_length=arguments.sequence_length,
            max_batches=arguments.max_batches,
        )
    if arguments.command == "benchmark":
        from apex.training import benchmark_small_checkpoint

        return benchmark_small_checkpoint(
            arguments.checkpoint,
            arguments.tokenizer,
            arguments.prompt,
            max_new_tokens=arguments.max_new_tokens,
        )
    return _registry_command(arguments)


def _registry_command(arguments: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    store = RegistryStore(arguments.registry_root)
    collection = {
        "model": store.models,
        "dataset": store.datasets,
        "experiment": store.experiments,
    }[arguments.command]
    record_type = {
        "model": ModelRecord,
        "dataset": DatasetRecord,
        "experiment": ExperimentRecord,
    }[arguments.command]
    if arguments.registry_action == "list":
        return [record.to_dict() for record in collection.list()]
    if arguments.registry_action == "show":
        return collection.get(arguments.id, arguments.version).to_dict()
    payload = json.loads(Path(arguments.record).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("registry record JSON must be an object")
    record = record_type.from_dict(payload)
    return collection.register(record, artifact_path=arguments.artifact).to_dict()


if __name__ == "__main__":
    main()
