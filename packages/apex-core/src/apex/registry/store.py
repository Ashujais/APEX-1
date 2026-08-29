from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path

from apex.registry.records import DatasetRecord, ExperimentRecord, ModelRecord, RegistryRecord

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RegistryCollection[RecordT: RegistryRecord]:
    def __init__(self, root: Path, record_type: type[RecordT]) -> None:
        self.root = root
        self.record_type = record_type

    def register(
        self,
        record: RecordT,
        *,
        artifact_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> RecordT:
        target = self._path(record.id, record.version)
        if target.exists() and not overwrite:
            raise FileExistsError(f"registry record already exists: {record.id}@{record.version}")
        if artifact_path is not None:
            artifact = Path(artifact_path).resolve(strict=True)
            if not artifact.is_file():
                raise ValueError("registry artifact must be a file")
            digest = sha256_file(artifact)
            if isinstance(record, ModelRecord):
                record = replace(record, checkpoint_uri=str(artifact), artifact_hash=digest)
            elif isinstance(record, DatasetRecord):
                record = replace(record, artifact_uri=str(artifact), artifact_hash=digest)
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.root, prefix=".pending-", delete=False
        ) as handle:
            handle.write(payload)
            pending = Path(handle.name)
        try:
            os.replace(pending, target)
        finally:
            pending.unlink(missing_ok=True)
        return record

    def get(self, record_id: str, version: str) -> RecordT:
        target = self._path(record_id, version)
        if not target.exists():
            raise KeyError(f"registry record not found: {record_id}@{version}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        return self.record_type.from_dict(payload)

    def list(self) -> list[RecordT]:
        if not self.root.exists():
            return []
        records = []
        for target in sorted(self.root.glob("*.json")):
            payload = json.loads(target.read_text(encoding="utf-8"))
            records.append(self.record_type.from_dict(payload))
        return records

    def _path(self, record_id: str, version: str) -> Path:
        if not SAFE_COMPONENT.fullmatch(record_id) or not SAFE_COMPONENT.fullmatch(version):
            raise ValueError(
                "registry ids and versions may contain letters, numbers, dot, dash, underscore"
            )
        return self.root / f"{record_id}--{version}.json"


class RegistryStore:
    """Filesystem-backed research registry with atomic local writes.

    This implementation is suitable for a single local or notebook process. Shared production
    registry storage remains a separate PostgreSQL/object-storage milestone.
    """

    def __init__(self, root: str | Path = ".apex/registry") -> None:
        base = Path(root)
        self.models = RegistryCollection(base / "models", ModelRecord)
        self.datasets = RegistryCollection(base / "datasets", DatasetRecord)
        self.experiments = RegistryCollection(base / "experiments", ExperimentRecord)
