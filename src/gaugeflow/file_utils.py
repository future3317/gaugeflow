"""Shared deterministic file and repository provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
