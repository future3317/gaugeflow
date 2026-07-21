"""Leakage-safe random-access index over immutable MatPES JSONL artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import torch
from torch.utils.data import Dataset

from gaugeflow.file_utils import load_json_object, sha256_file

from .matpes_data import (
    MatPESEnergyTarget,
    MatPESPhysicalBatch,
    MatPESPhysicalRecord,
    collate_matpes_records,
    matpes_iid_split,
    parse_matpes_row,
)

MATPES_INDEX_SCHEMA = 1
SPLIT_TO_INDEX = {"train": 0, "calibration": 1, "test": 2}


@dataclass(frozen=True)
class MatPESBatchCollator:
    functional_vocabulary: Mapping[str, int]
    teacher_dim: int

    def __call__(self, records: list[MatPESPhysicalRecord]) -> MatPESPhysicalBatch:
        return collate_matpes_records(
            records,
            functional_vocabulary=self.functional_vocabulary,
            teacher_dim=self.teacher_dim,
        )


class IndexedMatPESDataset(Dataset[MatPESPhysicalRecord]):
    """Seek directly to qualified JSONL rows while keeping IDs out of batches."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        verify_hashes: bool = True,
        require_qualified: bool = True,
    ) -> None:
        if split not in SPLIT_TO_INDEX:
            raise ValueError("MatPES split must be train, calibration, or test")
        self.root = Path(root)
        manifest = load_json_object(self.root / "manifest.json")
        if manifest.get("schema") != MATPES_INDEX_SCHEMA:
            raise ValueError("MatPES index manifest schema mismatch")
        if require_qualified and not bool(manifest.get("qualified")):
            raise ValueError("MatPES index is not qualified for training")
        sources = manifest.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("MatPES index manifest has no sources")
        self.source_paths: list[Path] = []
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                raise ValueError("MatPES source manifest is invalid")
            path = Path(source["path"])
            if verify_hashes and source.get("sha256") is not None:
                if sha256_file(path) != source["sha256"]:
                    raise ValueError("MatPES raw source hash mismatch")
            self.source_paths.append(path)
        index_path = self.root / str(manifest["index_file"])
        if verify_hashes and sha256_file(index_path) != manifest["index_sha256"]:
            raise ValueError("MatPES index tensor hash mismatch")
        payload: Any = torch.load(index_path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, dict) or payload.get("schema") != MATPES_INDEX_SCHEMA:
            raise ValueError("MatPES index tensor schema mismatch")
        required = {"source_index", "byte_offset", "node_count", "split_index"}
        if not required.issubset(payload):
            raise ValueError("MatPES index tensor payload is incomplete")
        self.source_index = payload["source_index"]
        self.byte_offset = payload["byte_offset"]
        self.node_count = payload["node_count"]
        self.split_index = payload["split_index"]
        rows = self.byte_offset.numel()
        if self.source_index.dtype != torch.uint8 or self.source_index.shape != (rows,):
            raise ValueError("MatPES source index is invalid")
        if self.byte_offset.dtype != torch.int64 or self.byte_offset.shape != (rows,):
            raise ValueError("MatPES byte offsets are invalid")
        if self.node_count.dtype != torch.uint8 or self.node_count.shape != (rows,):
            raise ValueError("MatPES node counts are invalid")
        if self.split_index.dtype != torch.uint8 or self.split_index.shape != (rows,):
            raise ValueError("MatPES split indices are invalid")
        if rows < 1 or bool((self.node_count < 1).any()):
            raise ValueError("MatPES index is empty or contains empty structures")
        if int(self.source_index.max()) >= len(self.source_paths):
            raise ValueError("MatPES index references an unknown source")
        self.indices = torch.nonzero(
            self.split_index == SPLIT_TO_INDEX[split], as_tuple=False
        ).squeeze(1)
        if self.indices.numel() < 1:
            raise ValueError(f"MatPES index has no {split} rows")
        energy_target = str(manifest["energy_target"])
        if energy_target not in {
            "total_energy_per_atom",
            "cohesive_energy_per_atom",
            "formation_energy_per_atom",
        }:
            raise ValueError("MatPES index energy target is invalid")
        self.energy_target = cast(MatPESEnergyTarget, energy_target)
        self._handles: dict[int, Any] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __len__(self) -> int:
        return self.indices.numel()

    def __getitem__(self, index: int) -> MatPESPhysicalRecord:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        if index < 0:
            index += len(self)
        row = int(self.indices[index])
        source = int(self.source_index[row])
        if source not in self._handles:
            self._handles[source] = self.source_paths[source].open("rb")
        handle = self._handles[source]
        handle.seek(int(self.byte_offset[row]))
        raw = handle.readline()
        if not raw:
            raise ValueError("MatPES indexed row cannot be read")
        record = parse_matpes_row(
            json.loads(raw),
            energy_target=self.energy_target,
        )
        if record.element_tokens.numel() != int(self.node_count[row]):
            raise ValueError("MatPES indexed node count changed after cache construction")
        return record


def build_matpes_index(
    sources: Mapping[str, Sequence[Path]],
    output: Path,
    *,
    energy_target: MatPESEnergyTarget = "cohesive_energy_per_atom",
    maximum_atoms: int = 20,
    seed: int = 5705,
    calibration_fraction: float = 0.05,
    test_fraction: float = 0.05,
    max_rows_per_source: int | None = None,
) -> dict[str, Any]:
    """Build an exact byte-offset index; bounded builds are software smokes only."""

    flattened_sources = [
        (functional, Path(path))
        for functional, paths in sources.items()
        for path in paths
    ]
    if (
        not flattened_sources
        or len(flattened_sources) > 255
        or any(not paths for paths in sources.values())
        or maximum_atoms < 1
        or maximum_atoms > 255
    ):
        raise ValueError("MatPES sources and maximum atom count are invalid")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite MatPES index {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_index: list[int] = []
    byte_offset: list[int] = []
    node_count: list[int] = []
    split_index: list[int] = []
    split_by_material: dict[str, int] = {}
    source_manifest: list[dict[str, Any]] = []
    excluded_large = 0
    invalid: dict[str, str] = {}
    for source_number, (functional, path) in enumerate(flattened_sources):
        rows_seen = 0
        rows_selected = 0
        with path.open("rb") as handle:
            while max_rows_per_source is None or rows_seen < max_rows_per_source:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                rows_seen += 1
                row: Any = {}
                try:
                    row = json.loads(line)
                    if row.get("functional") != functional:
                        raise ValueError("functional label disagrees with source")
                    atoms = int(row.get("nsites", 0))
                    if atoms < 1:
                        raise ValueError("nonpositive site count")
                    if atoms > maximum_atoms:
                        excluded_large += 1
                        continue
                    record = parse_matpes_row(row, energy_target=energy_target)
                    if not (record.energy_present and record.forces_present and record.stress_present):
                        raise ValueError("selected physical target is incomplete")
                    material_id = record.material_id
                    split = SPLIT_TO_INDEX[
                        matpes_iid_split(
                            material_id,
                            seed=seed,
                            calibration_fraction=calibration_fraction,
                            test_fraction=test_fraction,
                        )
                    ]
                    previous = split_by_material.setdefault(material_id, split)
                    if previous != split:
                        raise AssertionError("material identity received inconsistent splits")
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    key = (
                        str(row.get("matpes_id", f"{functional}:{rows_seen}"))
                        if isinstance(row, dict)
                        else f"{functional}:{rows_seen}"
                    )
                    invalid[key] = f"{type(error).__name__}: {error}"
                    continue
                source_index.append(source_number)
                byte_offset.append(offset)
                node_count.append(atoms)
                split_index.append(split)
                rows_selected += 1
        source_manifest.append(
            {
                "functional": functional,
                "path": str(path.resolve()),
                "rows_seen": rows_seen,
                "rows_selected": rows_selected,
                "sha256": sha256_file(path) if max_rows_per_source is None else None,
            }
        )
    payload = {
        "schema": MATPES_INDEX_SCHEMA,
        "source_index": torch.tensor(source_index, dtype=torch.uint8),
        "byte_offset": torch.tensor(byte_offset, dtype=torch.int64),
        "node_count": torch.tensor(node_count, dtype=torch.uint8),
        "split_index": torch.tensor(split_index, dtype=torch.uint8),
    }
    index_path = output / "index.pt"
    torch.save(payload, index_path)
    split_counts = {
        split: int((payload["split_index"] == code).sum())
        for split, code in SPLIT_TO_INDEX.items()
    }
    qualified = max_rows_per_source is None and not invalid and all(split_counts.values())
    manifest = {
        "schema": MATPES_INDEX_SCHEMA,
        "qualified": bool(qualified),
        "scope": "MatPES N<=20 physical-representation pretraining index",
        "sources": source_manifest,
        "index_file": index_path.name,
        "index_sha256": sha256_file(index_path),
        "energy_target": energy_target,
        "maximum_atoms": maximum_atoms,
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "test_fraction": test_fraction,
        "split_counts": split_counts,
        "unique_material_ids": len(split_by_material),
        "excluded_large_cells": excluded_large,
        "invalid_rows": invalid,
        "bounded_smoke": max_rows_per_source is not None,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
