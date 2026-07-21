"""Build immutable benchmark-ID exclusions for continued pretraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pyarrow.parquet as pq

from gaugeflow.file_utils import sha256_file

from .alex_p1_data import PACKED_ALEX_P1_PROTOCOL
from .lemat_data import normalize_external_material_id

BENCHMARK_EXCLUSION_SCHEMA = 1


def build_alex_benchmark_exclusion(
    alex_cache: Path,
    output: Path,
    *,
    splits: Sequence[str] = ("val", "test"),
) -> dict[str, object]:
    """Freeze normalized Alex benchmark IDs without exposing them to model batches."""

    if not splits or len(set(splits)) != len(splits) or any(
        split not in {"val", "test"} for split in splits
    ):
        raise ValueError("benchmark exclusion splits must be unique val/test names")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite benchmark exclusion {output}")
    manifest_path = alex_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PACKED_ALEX_P1_PROTOCOL or not bool(
        manifest.get("qualified")
    ):
        raise ValueError("Alex benchmark cache is not qualified")
    split_manifest = manifest.get("splits")
    if not isinstance(split_manifest, dict):
        raise ValueError("Alex benchmark cache has no split manifest")
    ids: set[str] = set()
    sources: list[dict[str, object]] = []
    for split in splits:
        entry = split_manifest.get(split)
        if not isinstance(entry, dict):
            raise ValueError(f"Alex benchmark cache has no {split} split")
        index_path = alex_cache / str(entry["index_file"])
        digest = sha256_file(index_path)
        if digest != str(entry["index_sha256"]):
            raise ValueError(f"Alex {split} index hash mismatch")
        table = pq.read_table(index_path, columns=["material_id"])
        raw_ids = table.column("material_id").to_pylist()
        if len(raw_ids) != int(entry["rows"]) or not all(
            isinstance(value, str) and value for value in raw_ids
        ):
            raise ValueError(f"Alex {split} material IDs are invalid")
        normalized = {normalize_external_material_id(value) for value in raw_ids}
        ids.update(normalized)
        sources.append(
            {
                "split": split,
                "rows": len(raw_ids),
                "unique_normalized_ids": len(normalized),
                "index_file": str(index_path.resolve()),
                "index_sha256": digest,
            }
        )
    if not ids:
        raise ValueError("Alex benchmark exclusion is empty")
    output.mkdir(parents=True, exist_ok=True)
    ids_path = output / "material_ids.json"
    ids_path.write_text(json.dumps(sorted(ids), indent=2) + "\n", encoding="utf-8")
    result: dict[str, object] = {
        "schema": BENCHMARK_EXCLUSION_SCHEMA,
        "scope": "Alex-MP validation/test IDs excluded from LeMat continued pretraining",
        "alex_cache_manifest": str(manifest_path.resolve()),
        "alex_cache_manifest_sha256": sha256_file(manifest_path),
        "sources": sources,
        "unique_normalized_ids": len(ids),
        "material_ids_file": ids_path.name,
        "material_ids_sha256": sha256_file(ids_path),
        "qualified": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
