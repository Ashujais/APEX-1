from __future__ import annotations

import hashlib
import html
import json
import math
import re
import unicodedata
import zipfile
from pathlib import Path

from pypdf import PdfReader

from apex_api.storage import DOCUMENT_EXTENSIONS, SOURCE_EXTENSIONS

TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)
DOCX_TEXT_PATTERN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)
XLSX_TEXT_PATTERN = re.compile(r"<t(?:\s[^>]*)?>(.*?)</t>", re.DOTALL)
XLSX_VALUE_PATTERN = re.compile(r"<c(?:\s[^>]*)?>(?:.*?)<v>(.*?)</v>(?:.*?)</c>", re.DOTALL)


class ExtractionError(ValueError):
    pass


def extract_text(path: Path, extension: str, max_characters: int) -> str:
    if extension in SOURCE_EXTENSIONS or extension in {
        ".csv",
        ".json",
        ".markdown",
        ".md",
        ".txt",
    }:
        text = _read_limited_text(path, max_characters)
        if extension == ".json":
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except json.JSONDecodeError as exc:
                raise ExtractionError("JSON document is not valid") from exc
    elif extension == ".pdf":
        text = _extract_pdf(path, max_characters)
    elif extension == ".docx":
        text = _extract_docx(path, max_characters)
    elif extension == ".xlsx":
        text = _extract_xlsx(path, max_characters)
    elif extension in DOCUMENT_EXTENSIONS:
        raise ExtractionError(f"Document extraction is not available for {extension}")
    else:
        raise ExtractionError("This file type is stored but is not text-extractable")
    cleaned = clean_text(text)
    if not cleaned:
        raise ExtractionError("No extractable text was found")
    return cleaned


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines: list[str] = []
    blank = False
    for raw_line in normalized.splitlines():
        line = re.sub(r"[^\S\n]+", " ", raw_line).strip()
        if line:
            lines.append(line)
            blank = False
        elif lines and not blank:
            lines.append("")
            blank = True
    return "\n".join(lines).strip()


def chunk_text(text: str, chunk_characters: int, overlap: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_characters, len(text))
        end = hard_end
        if hard_end < len(text):
            candidate = text.rfind(" ", start + chunk_characters // 2, hard_end)
            if candidate > start:
                end = candidate
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_text(text: str, dimensions: int) -> list[float]:
    """Create a normalized feature-hashing embedding from lexical tokens."""

    vector = [0.0] * dimensions
    tokens = tokenize(text)
    features = tokens + [
        f"{left}::{right}" for left, right in zip(tokens, tokens[1:], strict=False)
    ]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimensions
        sign = -1.0 if value & 1 else 1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def lexical_relevance(query: str, content: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(content))
    return len(query_tokens & content_tokens) / len(query_tokens)


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


def _read_limited_text(path: Path, max_characters: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as source:
        text = source.read(max_characters + 1)
    if len(text) > max_characters:
        raise ExtractionError("Extracted text exceeds the configured character limit")
    return text


def _extract_pdf(path: Path, max_characters: int) -> str:
    reader = PdfReader(path)
    parts: list[str] = []
    size = 0
    for page in reader.pages:
        content = page.extract_text() or ""
        size += len(content)
        if size > max_characters:
            raise ExtractionError("Extracted PDF text exceeds the configured character limit")
        parts.append(content)
    return "\n\n".join(parts)


def _extract_docx(path: Path, max_characters: int) -> str:
    xml = _read_zip_member(path, "word/document.xml", max_characters * 8)
    parts = [html.unescape(value) for value in DOCX_TEXT_PATTERN.findall(xml)]
    text = "\n".join(parts)
    if len(text) > max_characters:
        raise ExtractionError("Extracted DOCX text exceeds the configured character limit")
    return text


def _extract_xlsx(path: Path, max_characters: int) -> str:
    with zipfile.ZipFile(path) as workbook:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_xml = _decode_zip_member(
                workbook, "xl/sharedStrings.xml", max_characters * 8
            )
            shared = [html.unescape(value) for value in XLSX_TEXT_PATTERN.findall(shared_xml)]
        cells: list[str] = []
        worksheet_names = sorted(
            name
            for name in workbook.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        for name in worksheet_names:
            worksheet = _decode_zip_member(workbook, name, max_characters * 8)
            for raw_value in XLSX_VALUE_PATTERN.findall(worksheet):
                value = html.unescape(raw_value)
                if value.isdigit() and int(value) < len(shared):
                    value = shared[int(value)]
                cells.append(value)
                if sum(map(len, cells)) > max_characters:
                    raise ExtractionError(
                        "Extracted XLSX text exceeds the configured character limit"
                    )
    return "\n".join(cells)


def _read_zip_member(path: Path, name: str, byte_limit: int) -> str:
    with zipfile.ZipFile(path) as archive:
        return _decode_zip_member(archive, name, byte_limit)


def _decode_zip_member(archive: zipfile.ZipFile, name: str, byte_limit: int) -> str:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise ExtractionError(f"Required archive member is missing: {name}") from exc
    if info.file_size > byte_limit:
        raise ExtractionError("Compressed document expands beyond the configured limit")
    with archive.open(info) as member:
        payload = member.read(byte_limit + 1)
    if len(payload) > byte_limit:
        raise ExtractionError("Compressed document expands beyond the configured limit")
    return payload.decode("utf-8", errors="replace")
