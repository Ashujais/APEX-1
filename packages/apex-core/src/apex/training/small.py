from __future__ import annotations

import importlib.metadata
import math
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from apex.hardware import HardwareAdvisor, detect_hardware
from apex.inference import generate
from apex.model import ApexConfig, ApexForCausalLM
from apex.registry import DatasetRecord, ExperimentRecord, ModelRecord, RegistryStore, sha256_file
from apex.tokenizer import ByteBPETokenizer
from apex.training.trainer import Trainer, TrainerConfig


@dataclass(frozen=True)
class SmallTrainingResult:
    run_id: str
    checkpoint: str
    tokenizer: str
    optimizer_steps: int
    final_loss: float
    duration_seconds: float
    device: str
    registry_root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_small_text_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    steps: int = 2,
    sequence_length: int = 64,
    batch_size: int = 1,
    vocabulary_size: int = 300,
    learning_rate: float = 3e-4,
    checkpoint_interval: int = 10,
    seed: int = 1337,
    resume_from: str | Path | None = None,
    registry_root: str | Path = ".apex/registry",
    dataset_license: str = "unknown",
    creator: str = "local-user",
) -> SmallTrainingResult:
    if steps < 1 or batch_size < 1 or sequence_length < 2:
        raise ValueError("steps, batch_size, and sequence_length must be positive")
    source = Path(dataset_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("dataset_path must be a text file")
    corpus = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not corpus:
        raise ValueError("the training dataset is empty")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    tokenizer_path = output / "tokenizer.json"
    tokenizer = ByteBPETokenizer()
    tokenizer.train(corpus, vocab_size=vocabulary_size)
    tokenizer.save(tokenizer_path)
    token_ids = tokenizer.encode("\n".join(corpus), add_bos=True, add_eos=True)
    sequence_length = min(sequence_length, ApexConfig.tiny().max_sequence_length)
    minimum_tokens = sequence_length + 1
    if len(token_ids) < minimum_tokens:
        repeats = math.ceil(minimum_tokens / len(token_ids))
        token_ids = (token_ids * repeats)[:minimum_tokens]
    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    hardware = detect_hardware(output)
    advice = HardwareAdvisor().recommend(hardware)
    device = torch.device("cuda" if hardware.cuda_available else "cpu")
    config = ApexConfig.tiny(vocab_size=tokenizer.vocab_size)
    model = ApexForCausalLM(config).to(device)
    trainer_config = TrainerConfig(
        learning_rate=learning_rate,
        gradient_accumulation_steps=advice.gradient_accumulation_steps,
        mixed_precision=advice.precision if advice.precision in {"fp16", "bf16"} else "none",
        scheduler="cosine",
        warmup_steps=min(max(steps // 20, 0), steps - 1),
        max_steps=steps,
        seed=seed,
    )
    trainer = Trainer(model, trainer_config)
    cursor = 0
    if resume_from is not None:
        metadata = trainer.load_checkpoint(Path(resume_from))
        cursor = int(metadata.get("dataset_cursor", 0))
        run_id = str(metadata.get("run_id", run_id))

    started = time.perf_counter()
    final_loss = math.nan
    checkpoint = output / "checkpoint-final.pt"
    last_saved_step = -1
    while trainer.step < steps:
        batch, cursor = _next_batch(token_tensor, cursor, batch_size, sequence_length)
        final_loss = trainer.train_step(batch.to(device))
        should_checkpoint = (
            trainer.step > 0
            and trainer.step != last_saved_step
            and trainer.step % checkpoint_interval == 0
        )
        if should_checkpoint:
            checkpoint = output / f"checkpoint-step-{trainer.step}.pt"
            trainer.save_checkpoint(
                checkpoint,
                metadata=_checkpoint_metadata(run_id, cursor, source, tokenizer_path, hardware),
            )
            last_saved_step = trainer.step
    checkpoint = output / "checkpoint-final.pt"
    trainer.save_checkpoint(
        checkpoint,
        metadata=_checkpoint_metadata(run_id, cursor, source, tokenizer_path, hardware),
    )
    duration = time.perf_counter() - started
    _register_run(
        registry_root=registry_root,
        run_id=run_id,
        source=source,
        dataset_license=dataset_license,
        creator=creator,
        tokenizer_path=tokenizer_path,
        checkpoint=checkpoint,
        model=model,
        trainer_config=trainer_config,
        hardware=hardware.to_dict(),
        final_loss=final_loss,
        duration=duration,
    )
    return SmallTrainingResult(
        run_id=run_id,
        checkpoint=str(checkpoint.resolve()),
        tokenizer=str(tokenizer_path.resolve()),
        optimizer_steps=trainer.step,
        final_loss=final_loss,
        duration_seconds=duration,
        device=str(device),
        registry_root=str(Path(registry_root).resolve()),
    )


@torch.inference_mode()
def evaluate_small_checkpoint(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    dataset_path: str | Path,
    *,
    sequence_length: int = 64,
    max_batches: int = 8,
) -> dict[str, Any]:
    model, tokenizer, device = _load_checkpoint(checkpoint_path, tokenizer_path)
    source = Path(dataset_path).resolve(strict=True)
    tokens = tokenizer.encode(source.read_text(encoding="utf-8"), add_bos=True, add_eos=True)
    sequence_length = min(sequence_length, model.config.max_sequence_length)
    if len(tokens) < sequence_length + 1:
        raise ValueError("evaluation data must contain more tokens than the sequence length")
    tensor = torch.tensor(tokens, dtype=torch.long)
    losses = []
    cursor = 0
    model.eval()
    for _ in range(max_batches):
        batch, cursor = _next_batch(tensor, cursor, 1, sequence_length)
        result = model(batch.to(device), labels=batch.to(device))
        if result.loss is None:
            raise RuntimeError("model did not return evaluation loss")
        losses.append(float(result.loss))
        if cursor == 0:
            break
    mean_loss = sum(losses) / len(losses)
    return {
        "status": "EXPERIMENTAL",
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "batches": len(losses),
        "dataset_hash": sha256_file(source),
        "device": str(device),
    }


@torch.inference_mode()
def benchmark_small_checkpoint(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    model, tokenizer, device = _load_checkpoint(checkpoint_path, tokenizer_path)
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
    if input_ids.shape[1] + max_new_tokens > model.config.max_sequence_length:
        raise ValueError("prompt plus generated tokens exceeds the tested model context")
    model.eval()
    started = time.perf_counter()
    output = generate(model, input_ids, max_new_tokens=max_new_tokens, top_k=20, seed=1337)
    elapsed = time.perf_counter() - started
    produced = output.shape[1] - input_ids.shape[1]
    return {
        "status": "EXPERIMENTAL",
        "engine": "pytorch",
        "device": str(device),
        "generated_tokens": produced,
        "latency_seconds": elapsed,
        "tokens_per_second": produced / elapsed if elapsed else None,
        "hardware": detect_hardware(checkpoint_path).to_dict(),
    }


def _next_batch(
    tokens: torch.Tensor, cursor: int, batch_size: int, sequence_length: int
) -> tuple[torch.Tensor, int]:
    window_count = len(tokens) - sequence_length
    rows = []
    for _ in range(batch_size):
        start = cursor % window_count
        rows.append(tokens[start : start + sequence_length])
        cursor = (cursor + sequence_length) % window_count
    return torch.stack(rows), cursor


def _load_checkpoint(
    checkpoint_path: str | Path, tokenizer_path: str | Path
) -> tuple[ApexForCausalLM, ByteBPETokenizer, torch.device]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ApexConfig(**checkpoint["model_config"])
    model = ApexForCausalLM(config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), ByteBPETokenizer.load(tokenizer_path), device


def _checkpoint_metadata(
    run_id: str,
    cursor: int,
    dataset_path: Path,
    tokenizer_path: Path,
    hardware: Any,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset_cursor": cursor,
        "dataset_hash": sha256_file(dataset_path),
        "tokenizer_hash": sha256_file(tokenizer_path),
        "git_commit": _git_commit(),
        "hardware": hardware.to_dict(),
    }


def _register_run(
    *,
    registry_root: str | Path,
    run_id: str,
    source: Path,
    dataset_license: str,
    creator: str,
    tokenizer_path: Path,
    checkpoint: Path,
    model: ApexForCausalLM,
    trainer_config: TrainerConfig,
    hardware: dict[str, Any],
    final_loss: float,
    duration: float,
) -> None:
    store = RegistryStore(registry_root)
    dataset_hash = sha256_file(source)
    dataset_id = re.sub(r"[^A-Za-z0-9._-]", "-", source.stem)[:80] or "dataset"
    dataset_version = dataset_hash[:12]
    try:
        store.datasets.register(
            DatasetRecord(
                id=dataset_id,
                version=dataset_version,
                status="RESEARCH",
                source=str(source),
                license=dataset_license,
                processing_steps=("UTF-8 text loading",),
                transformations=("byte-level BPE tokenization",),
                creator=creator,
                configuration={"training_run": run_id},
            ),
            artifact_path=source,
        )
    except FileExistsError:
        pass
    model_id = f"apex-tiny-{run_id[:8]}"
    tokenizer_hash = sha256_file(tokenizer_path)
    store.models.register(
        ModelRecord(
            id=model_id,
            version="0.1.0",
            status="RESEARCH",
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            architecture="decoder-only-transformer",
            tokenizer=tokenizer_hash,
            dataset=dataset_id,
            dataset_version=dataset_version,
            training_config=asdict(trainer_config),
            quantization="none",
            context_length=model.config.max_sequence_length,
            capabilities=("experimental-text-generation",),
            hardware_requirements={"profile": "apex-tiny"},
            inference_engine="pytorch",
            git_commit=_git_commit(),
        ),
        artifact_path=checkpoint,
        overwrite=True,
    )
    store.experiments.register(
        ExperimentRecord(
            id=run_id,
            version="1",
            status="COMPLETED",
            model=model_id,
            dataset=dataset_id,
            dataset_version=dataset_version,
            tokenizer=tokenizer_hash,
            configuration=asdict(trainer_config),
            seed=trainer_config.seed,
            git_commit=_git_commit(),
            hardware=hardware,
            environment={"torch": str(torch.__version__)},
            dependencies=_dependencies(),
            metrics={"final_training_loss": final_loss},
            checkpoint_uri=str(checkpoint.resolve()),
            duration_seconds=duration,
        ),
        overwrite=True,
    )


def _dependencies() -> dict[str, str]:
    result = {"python": importlib.metadata.version("pip"), "torch": str(torch.__version__)}
    for package in ("apex-1", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def _git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - executable and arguments are fixed
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
