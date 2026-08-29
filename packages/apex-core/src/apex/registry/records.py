from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Self


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RegistryRecord:
    id: str
    version: str
    status: str
    created_at: str = field(default_factory=utc_timestamp)

    VALID_STATUSES: ClassVar[frozenset[str]] = frozenset()

    def __post_init__(self) -> None:
        if not self.id or not self.version:
            raise ValueError("registry id and version are required")
        if self.status not in self.VALID_STATUSES:
            choices = ", ".join(sorted(self.VALID_STATUSES))
            raise ValueError(f"invalid registry status {self.status!r}; expected one of {choices}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        return cls(**payload)


@dataclass(frozen=True)
class DatasetRecord(RegistryRecord):
    source: str = "unknown"
    license: str = "unknown"
    collection_date: str | None = None
    processing_steps: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()
    transformations: tuple[str, ...] = ()
    parent_dataset: str | None = None
    artifact_uri: str | None = None
    artifact_hash: str | None = None
    creator: str = "unknown"
    configuration: dict[str, Any] = field(default_factory=dict)

    VALID_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"RESEARCH", "VALIDATED", "REJECTED", "RETIRED"}
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        normalized = dict(payload)
        for key in ("processing_steps", "filters", "transformations"):
            normalized[key] = tuple(normalized.get(key, ()))
        return cls(**normalized)


@dataclass(frozen=True)
class ModelRecord(RegistryRecord):
    parameters: int = 0
    architecture: str = "unknown"
    tokenizer: str = "unknown"
    dataset: str = "unknown"
    dataset_version: str = "unknown"
    training_config: dict[str, Any] = field(default_factory=dict)
    checkpoint_uri: str | None = None
    quantization: str = "none"
    context_length: int = 0
    capabilities: tuple[str, ...] = ()
    benchmark_results: dict[str, Any] = field(default_factory=dict)
    safety_results: dict[str, Any] = field(default_factory=dict)
    hardware_requirements: dict[str, Any] = field(default_factory=dict)
    inference_engine: str = "pytorch"
    deployment_status: str = "NOT_DEPLOYED"
    git_commit: str | None = None
    artifact_hash: str | None = None

    VALID_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {
            "RESEARCH",
            "TRAINING",
            "EVALUATION",
            "SAFETY",
            "RED_TEAM",
            "APPROVAL",
            "STAGING",
            "CANARY",
            "PRODUCTION",
            "MONITORING",
            "RETIREMENT",
        }
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        normalized = dict(payload)
        normalized["capabilities"] = tuple(normalized.get("capabilities", ()))
        return cls(**normalized)


@dataclass(frozen=True)
class ExperimentRecord(RegistryRecord):
    model: str = "unknown"
    dataset: str = "unknown"
    dataset_version: str = "unknown"
    tokenizer: str = "unknown"
    configuration: dict[str, Any] = field(default_factory=dict)
    seed: int = 1337
    git_commit: str | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_uri: str | None = None
    duration_seconds: float | None = None

    VALID_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"PLANNED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "PREEMPTED"}
    )
