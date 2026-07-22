"""Row-group random-access index for LeMat-BulkUnique parquet shards."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file

from .lemat_data import (
    LeMatPhysicalLabelPolicy,
    lemat_iid_split,
    lemat_split_group,
    normalize_external_material_id,
    parse_lemat_row,
    validate_lemat_physical_label_policy,
)
from .matpes_data import MatPESPhysicalRecord
from .matpes_index import (
    DEFAULT_MAXIMUM_LATTICE_METRIC_CONDITION,
    DEFAULT_MINIMUM_LATTICE_WIDTH_ANGSTROM,
    SPLIT_TO_INDEX,
)

LEMAT_INDEX_SCHEMA = 2
_OVERLAP_COLUMNS = [
    "immutable_id",
    "entalpic_fingerprint",
    "nsites",
    "functional",
    "cross_compatibility",
]
_INDEX_COLUMNS = [
    *_OVERLAP_COLUMNS,
    "lattice_vectors",
]


def _batched_lattice_quality_codes(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_width_angstrom: float,
    maximum_metric_condition: float,
) -> np.ndarray:
    """Return -1 malformed, 0 accepted, or width/condition/volume bits."""

    codes = np.full(len(rows), -1, dtype=np.int8)
    positions: list[int] = []
    lattices: list[np.ndarray] = []
    for position, row in enumerate(rows):
        try:
            lattice = np.asarray(row["lattice_vectors"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        if lattice.shape == (3, 3) and bool(np.isfinite(lattice).all()):
            positions.append(position)
            lattices.append(lattice)
    if not positions:
        return codes
    lattice_batch = np.stack(lattices)
    metric = lattice_batch @ np.swapaxes(lattice_batch, -1, -2)
    eigenvalues = np.linalg.eigvalsh(metric)
    minimum_width = np.sqrt(np.maximum(eigenvalues[:, 0], 0.0))
    condition = eigenvalues[:, -1] / np.maximum(
        eigenvalues[:, 0], np.finfo(np.float64).tiny
    )
    determinant = np.linalg.det(lattice_batch)
    valid_codes = (minimum_width < minimum_width_angstrom).astype(np.int8)
    valid_codes |= (
        (condition > maximum_metric_condition).astype(np.int8) << 1
    )
    valid_codes |= ((determinant <= 0.0).astype(np.int8) << 2)
    codes[np.asarray(positions)] = valid_codes
    return codes


class IndexedLeMatDataset(Dataset[MatPESPhysicalRecord]):
    """Read only requested parquet row groups while keeping audit IDs out of batches."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        verify_hashes: bool = True,
        require_qualified: bool = True,
        cached_row_groups: int = 2,
    ) -> None:
        if split not in SPLIT_TO_INDEX:
            raise ValueError("LeMat split must be train, calibration, or test")
        self.root = Path(root)
        manifest = load_json_object(self.root / "manifest.json")
        if manifest.get("schema") != LEMAT_INDEX_SCHEMA:
            raise ValueError("LeMat index manifest schema mismatch")
        if require_qualified and not bool(manifest.get("qualified")):
            raise ValueError("LeMat index is not qualified")
        if cached_row_groups < 1:
            raise ValueError("LeMat row-group cache size must be positive")
        self.cached_row_groups = cached_row_groups
        self.label_policy = validate_lemat_physical_label_policy(
            str(manifest["physical_label_policy"])
        )
        self.source_paths: list[Path] = []
        source_functionals: list[str] = []
        for source in manifest.get("sources", []):
            path = Path(source["path"])
            if verify_hashes and source.get("sha256") and sha256_file(path) != source["sha256"]:
                raise ValueError("LeMat parquet hash mismatch")
            self.source_paths.append(path)
            source_functionals.append(str(source["functional"]).lower())
        self.functional_names = tuple(dict.fromkeys(source_functionals))
        if not self.functional_names:
            raise ValueError("LeMat index has no functional sources")
        functional_lookup = {name: index for index, name in enumerate(self.functional_names)}
        source_to_functional = torch.tensor(
            [functional_lookup[name] for name in source_functionals], dtype=torch.long
        )
        index_path = self.root / str(manifest["index_file"])
        if verify_hashes and sha256_file(index_path) != manifest["index_sha256"]:
            raise ValueError("LeMat index tensor hash mismatch")
        payload: Any = torch.load(
            index_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(payload, dict) or payload.get("schema") != LEMAT_INDEX_SCHEMA:
            raise ValueError("LeMat index tensor schema mismatch")
        self.source_index = payload["source_index"]
        self.row_group = payload["row_group"]
        self.row_in_group = payload["row_in_group"]
        self.node_count = payload["node_count"]
        self.split_index = payload["split_index"]
        rows = self.source_index.numel()
        tensors = (self.row_group, self.row_in_group, self.node_count, self.split_index)
        if rows < 1 or any(value.shape != (rows,) for value in tensors):
            raise ValueError("LeMat index tensors have inconsistent lengths")
        expected_dtypes = (
            (self.source_index, torch.uint8),
            (self.row_group, torch.int32),
            (self.row_in_group, torch.int32),
            (self.node_count, torch.uint8),
            (self.split_index, torch.uint8),
        )
        if any(value.dtype != dtype for value, dtype in expected_dtypes):
            raise ValueError("LeMat index tensor dtype mismatch")
        if int(self.source_index.max()) >= len(self.source_paths) or bool((self.node_count < 1).any()):
            raise ValueError("LeMat index references an invalid source or node count")
        self.indices = torch.nonzero(
            self.split_index == SPLIT_TO_INDEX[split], as_tuple=False
        ).squeeze(1)
        if self.indices.numel() < 1:
            raise ValueError(f"LeMat index has no {split} rows")
        self._functional_group_index = source_to_functional[
            self.source_index[self.indices].long()
        ].contiguous()
        block_keys = (
            self.source_index[self.indices].long().bitwise_left_shift(32)
            | self.row_group[self.indices].long()
        )
        _, self._sampling_block_index = torch.unique(
            block_keys, sorted=True, return_inverse=True
        )
        self._sampling_block_index = self._sampling_block_index.contiguous()
        self._files: dict[int, pq.ParquetFile] = {}
        self._group_cache: OrderedDict[tuple[int, int], Any] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = {}
        state["_group_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def functional_group_index(self) -> torch.Tensor:
        """Return split-local functional groups for balanced sampling."""

        return self._functional_group_index

    @property
    def sampling_block_index(self) -> torch.Tensor:
        """Return compact split-local parquet blocks for locality-aware sampling."""

        return self._sampling_block_index

    def __getitem__(self, index: int) -> MatPESPhysicalRecord:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        row = int(self.indices[index % len(self)])
        source = int(self.source_index[row])
        group = int(self.row_group[row])
        key = (source, group)
        if key not in self._group_cache:
            parquet = self._files.setdefault(source, pq.ParquetFile(self.source_paths[source]))
            self._group_cache[key] = parquet.read_row_group(group)
            if len(self._group_cache) > self.cached_row_groups:
                self._group_cache.popitem(last=False)
        table = self._group_cache.pop(key)
        self._group_cache[key] = table
        raw = table.slice(int(self.row_in_group[row]), 1).to_pylist()[0]
        record = parse_lemat_row(raw, physical_label_policy=self.label_policy)
        if record.element_tokens.numel() != int(self.node_count[row]):
            raise ValueError("LeMat node count changed after index construction")
        return record


def build_lemat_index(
    sources: Mapping[str, Sequence[Path]],
    output: Path,
    *,
    maximum_atoms: int = 20,
    seed: int = 5705,
    calibration_fraction: float = 0.05,
    test_fraction: float = 0.05,
    physical_label_policy: LeMatPhysicalLabelPolicy = "compatible_only",
    excluded_material_ids: set[str] | None = None,
    excluded_material_ids_artifact_sha256: str | None = None,
    max_row_groups_per_source: int | None = None,
    minimum_lattice_width_angstrom: float = DEFAULT_MINIMUM_LATTICE_WIDTH_ANGSTROM,
    maximum_lattice_metric_condition: float = DEFAULT_MAXIMUM_LATTICE_METRIC_CONDITION,
) -> dict[str, Any]:
    """Build a source-balanced-ready row-group index without copying 5.4M records."""

    policy = validate_lemat_physical_label_policy(physical_label_policy)
    flattened = [(functional, Path(path)) for functional, paths in sources.items() for path in paths]
    if (
        not flattened
        or len(flattened) > 255
        or maximum_atoms < 1
        or maximum_atoms > 255
        or minimum_lattice_width_angstrom <= 0.0
        or maximum_lattice_metric_condition <= 1.0
    ):
        raise ValueError("LeMat source or index-domain bounds are invalid")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite LeMat index {output}")
    output.mkdir(parents=True, exist_ok=True)
    excluded = {normalize_external_material_id(value) for value in (excluded_material_ids or set())}
    exclusion_artifact_bound = not excluded or (
        isinstance(excluded_material_ids_artifact_sha256, str)
        and len(excluded_material_ids_artifact_sha256) == 64
        and all(character in "0123456789abcdef" for character in excluded_material_ids_artifact_sha256)
    )
    values: dict[str, list[int]] = {
        "source_index": [],
        "row_group": [],
        "row_in_group": [],
        "node_count": [],
        "split_index": [],
    }
    manifests: list[dict[str, Any]] = []
    excluded_fingerprints: set[str] = set()
    if excluded:
        for _, path in flattened:
            parquet = pq.ParquetFile(path)
            groups = parquet.num_row_groups
            if max_row_groups_per_source is not None:
                groups = min(groups, max_row_groups_per_source)
            for group in range(groups):
                rows = parquet.read_row_group(group, columns=_OVERLAP_COLUMNS).to_pylist()
                for row in rows:
                    material_id = row.get("immutable_id")
                    if isinstance(material_id, str) and (
                        normalize_external_material_id(material_id) in excluded
                    ):
                        try:
                            excluded_fingerprints.add(lemat_split_group(row))
                        except ValueError:
                            pass
    excluded_large = excluded_overlap = excluded_direct_id = excluded_fingerprint = 0
    excluded_degenerate = excluded_minimum_width = excluded_metric_condition = 0
    excluded_nonpositive_volume = 0
    invalid_rows = 0
    compatible_rows = 0
    split_by_group: dict[str, int] = {}
    for source_number, (declared_functional, path) in enumerate(flattened):
        parquet = pq.ParquetFile(path)
        groups = parquet.num_row_groups
        if max_row_groups_per_source is not None:
            groups = min(groups, max_row_groups_per_source)
        selected = 0
        for group in range(groups):
            rows = parquet.read_row_group(group, columns=_INDEX_COLUMNS).to_pylist()
            lattice_quality_codes = _batched_lattice_quality_codes(
                rows,
                minimum_width_angstrom=minimum_lattice_width_angstrom,
                maximum_metric_condition=maximum_lattice_metric_condition,
            )
            for row_number, row in enumerate(rows):
                try:
                    functional = str(row["functional"]).lower()
                    if functional != declared_functional.lower():
                        raise ValueError("functional disagrees with source")
                    node_count = int(row["nsites"])
                    if node_count < 1:
                        raise ValueError("nonpositive node count")
                    if node_count > maximum_atoms:
                        excluded_large += 1
                        continue
                    normalized_id = normalize_external_material_id(str(row["immutable_id"]))
                    key = lemat_split_group(row)
                    direct_overlap = normalized_id in excluded
                    fingerprint_overlap = key in excluded_fingerprints
                    if direct_overlap or fingerprint_overlap:
                        excluded_overlap += 1
                        excluded_direct_id += int(direct_overlap)
                        excluded_fingerprint += int(fingerprint_overlap and not direct_overlap)
                        continue
                    lattice_quality = int(lattice_quality_codes[row_number])
                    if lattice_quality < 0:
                        raise ValueError("malformed lattice")
                    if lattice_quality:
                        excluded_degenerate += 1
                        excluded_minimum_width += lattice_quality & 1
                        excluded_metric_condition += (lattice_quality >> 1) & 1
                        excluded_nonpositive_volume += (lattice_quality >> 2) & 1
                        continue
                    split = SPLIT_TO_INDEX[
                        lemat_iid_split(
                            key,
                            seed=seed,
                            calibration_fraction=calibration_fraction,
                            test_fraction=test_fraction,
                        )
                    ]
                    if split_by_group.setdefault(key, split) != split:
                        raise AssertionError("LeMat group received inconsistent splits")
                except (KeyError, TypeError, ValueError):
                    invalid_rows += 1
                    continue
                values["source_index"].append(source_number)
                values["row_group"].append(group)
                values["row_in_group"].append(row_number)
                values["node_count"].append(node_count)
                values["split_index"].append(split)
                compatible_rows += int(bool(row["cross_compatibility"]))
                selected += 1
        manifests.append(
            {
                "functional": declared_functional.lower(),
                "path": str(path.resolve()),
                "row_groups_seen": groups,
                "rows_selected": selected,
                "sha256": sha256_file(path) if max_row_groups_per_source is None else None,
            }
        )
    payload = {
        "schema": LEMAT_INDEX_SCHEMA,
        "source_index": torch.tensor(values["source_index"], dtype=torch.uint8),
        "row_group": torch.tensor(values["row_group"], dtype=torch.int32),
        "row_in_group": torch.tensor(values["row_in_group"], dtype=torch.int32),
        "node_count": torch.tensor(values["node_count"], dtype=torch.uint8),
        "split_index": torch.tensor(values["split_index"], dtype=torch.uint8),
    }
    index_path = output / "index.pt"
    torch.save(payload, index_path)
    split_counts = {
        name: int((payload["split_index"] == code).sum())
        for name, code in SPLIT_TO_INDEX.items()
    }
    bounded = max_row_groups_per_source is not None
    qualified = (
        not bounded
        and invalid_rows == 0
        and all(split_counts.values())
        and exclusion_artifact_bound
    )
    manifest = {
        "schema": LEMAT_INDEX_SCHEMA,
        "qualified": qualified,
        "scope": "LeMat-BulkUnique N<=20 structure and masked physical-label index",
        "sources": manifests,
        "index_file": index_path.name,
        "index_sha256": sha256_file(index_path),
        "maximum_atoms": maximum_atoms,
        "minimum_lattice_width_angstrom": minimum_lattice_width_angstrom,
        "maximum_lattice_metric_condition": maximum_lattice_metric_condition,
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "test_fraction": test_fraction,
        "physical_label_policy": policy,
        "split_counts": split_counts,
        "unique_split_groups": len(split_by_group),
        "compatible_selected_rows": compatible_rows,
        "excluded_large_cells": excluded_large,
        "excluded_degenerate_lattice_rows": excluded_degenerate,
        "excluded_minimum_lattice_width_rows": excluded_minimum_width,
        "excluded_lattice_metric_condition_rows": excluded_metric_condition,
        "excluded_nonpositive_lattice_volume_rows": excluded_nonpositive_volume,
        "excluded_external_overlap": excluded_overlap,
        "excluded_direct_id_rows": excluded_direct_id,
        "excluded_cross_id_fingerprint_rows": excluded_fingerprint,
        "excluded_benchmark_fingerprints": len(excluded_fingerprints),
        "excluded_material_ids_count": len(excluded),
        "excluded_material_ids_content_sha256": canonical_json_hash(sorted(excluded)),
        "excluded_material_ids_artifact_sha256": excluded_material_ids_artifact_sha256,
        "exclusion_artifact_bound": exclusion_artifact_bound,
        "invalid_index_rows": invalid_rows,
        "bounded_smoke": bounded,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
