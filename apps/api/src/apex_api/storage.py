from __future__ import annotations

import hashlib
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

DOCUMENT_EXTENSIONS = {
    ".csv",
    ".docx",
    ".json",
    ".markdown",
    ".md",
    ".pdf",
    ".txt",
    ".xlsx",
}
SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".php",
    ".ps1",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".xml",
    ".yaml",
    ".yml",
}
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
ALLOWED_EXTENSIONS = (
    DOCUMENT_EXTENSIONS
    | SOURCE_EXTENSIONS
    | IMAGE_EXTENSIONS
    | AUDIO_EXTENSIONS
    | VIDEO_EXTENSIONS
)
FORBIDDEN_MIME_TYPES = {
    "application/x-dosexec",
    "application/x-executable",
    "application/x-msdownload",
    "application/x-sharedlib",
}
STREAM_CHUNK_BYTES = 1024 * 1024


class InvalidUpload(ValueError):
    pass


class UploadTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class StoredFile:
    filename: str
    extension: str
    mime_type: str
    size_bytes: int
    checksum_sha256: str
    storage_key: str


class LocalFileStorage:
    """Stream uploads into a tenant-namespaced local storage root."""

    def __init__(self, root: str | Path, max_upload_bytes: int) -> None:
        self.root = Path(root).resolve()
        self.max_upload_bytes = max_upload_bytes
        self.temporary_root = self.root / ".tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)

    async def store(self, upload: UploadFile, tenant_id: str, file_id: str) -> StoredFile:
        filename, extension, mime_type = validate_upload(upload)
        temporary_path = self.temporary_root / f"{file_id}.part"
        relative_key = str(Path(tenant_id) / f"{file_id}{extension}").replace("\\", "/")
        target_path = self.resolve(relative_key)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary_path.open("xb") as target:
                while chunk := await upload.read(STREAM_CHUNK_BYTES):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise UploadTooLarge(
                            f"Upload exceeds the {self.max_upload_bytes}-byte limit"
                        )
                    digest.update(chunk)
                    target.write(chunk)
            if size == 0:
                raise InvalidUpload("Uploaded file is empty")
            os.replace(temporary_path, target_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return StoredFile(
            filename=filename,
            extension=extension,
            mime_type=mime_type,
            size_bytes=size,
            checksum_sha256=digest.hexdigest(),
            storage_key=relative_key,
        )

    def resolve(self, storage_key: str) -> Path:
        candidate = (self.root / storage_key).resolve()
        if not candidate.is_relative_to(self.root):
            raise InvalidUpload("Storage key escapes the configured root")
        return candidate

    def delete(self, storage_key: str) -> None:
        self.resolve(storage_key).unlink(missing_ok=True)


def validate_upload(upload: UploadFile) -> tuple[str, str, str]:
    raw_name = (upload.filename or "").strip()
    filename = Path(raw_name).name
    if not filename or filename in {".", ".."} or filename != raw_name:
        raise InvalidUpload("A safe base filename is required")
    if len(filename) > 255 or "\x00" in filename:
        raise InvalidUpload("Filename is invalid")
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidUpload(f"Unsupported file extension: {extension or '(none)'}")

    declared = (upload.content_type or "").split(";", 1)[0].strip().lower()
    if declared in FORBIDDEN_MIME_TYPES:
        raise InvalidUpload("Executable MIME types are not accepted")
    guessed = mimetypes.guess_type(filename)[0]
    mime_type = declared if declared and declared != "application/octet-stream" else guessed
    return filename, extension, mime_type or "application/octet-stream"
