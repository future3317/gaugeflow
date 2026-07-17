"""Shared deterministic file and repository provenance helpers."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_hash(value: object) -> str:
    """Hash a JSON-compatible object using one canonical serialization."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_gzip_json(path: Path) -> Any:
    """Read one UTF-8 JSON value from a gzip artifact."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_deterministic_gzip_json(path: Path, value: object) -> None:
    """Write canonical compact JSON with a zero gzip timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text_handle:
                json.dump(value, text_handle, separators=(",", ":"), sort_keys=True)
