"""Packed, leakage-safe Alex-MP-20 structure data for real-data H1a."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from gaugeflow.file_utils import sha256_file
from gaugeflow.vocabulary import validate_type_tokens

PACKED_ALEX_P1_SCHEMA = 1
PACKED_ALEX_P1_PROTOCOL = "h1a_p1_structure_cache_v1"


@dataclass(frozen=True)
class PackedAlexModelBatch:
    """Vectorized model-only selection from the memory-mapped structure cache."""

    atom_types: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    batch: torch.Tensor


class PackedAlexP1Dataset(Dataset[Data]):
    """Memory-mapped ragged tensors for the H0-A formula/prototype split.

    Only atom tokens, fractional coordinates and lattice are returned by
    default. Source IDs, split labels and Niggli transforms remain in the
    external audit index and never enter a model batch.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        include_material_id: bool = False,
        verify_hashes: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("packed Alex split must be train, val, or test")
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != PACKED_ALEX_P1_PROTOCOL:
            raise ValueError("packed Alex manifest protocol mismatch")
        if not bool(manifest.get("qualified")):
            raise ValueError("packed Alex cache is not qualified")
        split_manifest = manifest.get("splits", {}).get(split)
        if not isinstance(split_manifest, dict):
            raise ValueError(f"packed Alex manifest has no {split!r} split")
        tensor_path = self.root / str(split_manifest["tensor_file"])
        index_path = self.root / str(split_manifest["index_file"])
        if verify_hashes:
            if sha256_file(tensor_path) != str(split_manifest["tensor_sha256"]):
                raise ValueError("packed Alex tensor hash mismatch")
            if sha256_file(index_path) != str(split_manifest["index_sha256"]):
                raise ValueError("packed Alex index hash mismatch")
        payload: Any = torch.load(
            tensor_path, map_location="cpu", weights_only=True, mmap=True
        )
        if not isinstance(payload, dict) or payload.get("schema") != PACKED_ALEX_P1_SCHEMA:
            raise ValueError("packed Alex tensor schema mismatch")
        required = {
            "atom_tokens",
            "fractional_coordinates",
            "lattice",
            "offsets",
            "niggli_transform",
        }
        if not required.issubset(payload):
            raise ValueError("packed Alex tensor payload is incomplete")
        self.atom_tokens = payload["atom_tokens"]
        self.fractional_coordinates = payload["fractional_coordinates"]
        self.lattice = payload["lattice"]
        self.offsets = payload["offsets"]
        self.niggli_transform = payload["niggli_transform"]
        self._validate_tensors(int(split_manifest["rows"]), int(split_manifest["nodes"]))
        self._material_ids: list[str] | None = None
        if include_material_id:
            table = pq.read_table(index_path, columns=["material_id", "cache_row"])
            ids = list(map(str, table.column("material_id").to_pylist()))
            rows = torch.tensor(
                table.column("cache_row").to_pylist(), dtype=torch.long
            )
            if len(ids) != len(self) or not torch.equal(rows, torch.arange(len(self))):
                raise ValueError("packed Alex index order does not match tensor rows")
            self._material_ids = ids

    def _validate_tensors(self, expected_rows: int, expected_nodes: int) -> None:
        rows = self.lattice.shape[0]
        if self.atom_tokens.dtype != torch.uint8 or self.atom_tokens.shape != (
            expected_nodes,
        ):
            raise ValueError("packed Alex atom-token tensor is invalid")
        if self.fractional_coordinates.dtype != torch.float32 or (
            self.fractional_coordinates.shape != (expected_nodes, 3)
        ):
            raise ValueError("packed Alex coordinate tensor is invalid")
        if self.lattice.dtype != torch.float32 or self.lattice.shape != (
            expected_rows,
            3,
            3,
        ):
            raise ValueError("packed Alex lattice tensor is invalid")
        if self.offsets.dtype != torch.int64 or self.offsets.shape != (
            expected_rows + 1,
        ):
            raise ValueError("packed Alex offset tensor is invalid")
        if self.niggli_transform.dtype != torch.int16 or (
            self.niggli_transform.shape != (expected_rows, 3, 3)
        ):
            raise ValueError("packed Alex Niggli certificate tensor is invalid")
        if rows != expected_rows or int(self.offsets[0]) != 0:
            raise ValueError("packed Alex row count or initial offset is invalid")
        if int(self.offsets[-1]) != expected_nodes or bool(
            (self.offsets[1:] <= self.offsets[:-1]).any()
        ):
            raise ValueError("packed Alex ragged offsets are invalid")
        if bool((self.atom_tokens >= 118).any()):
            raise ValueError("packed Alex atom token lies outside the physical vocabulary")
        if not all(
            torch.isfinite(value).all()
            for value in (self.fractional_coordinates, self.lattice)
        ):
            raise ValueError("packed Alex cache contains nonfinite values")
        if bool(
            (
                (self.fractional_coordinates < 0.0)
                | (self.fractional_coordinates >= 1.0)
            ).any()
        ):
            raise ValueError("packed Alex coordinates are not canonical in [0,1)")
        if bool((torch.linalg.det(self.lattice) <= 0.0).any()):
            raise ValueError("packed Alex cache contains a nonpositive lattice volume")

    @property
    def node_counts(self) -> torch.Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    @property
    def material_ids_audit_only(self) -> list[str]:
        """Offline IDs loaded only when explicitly requested by an auditor."""
        if self._material_ids is None:
            raise ValueError("material IDs were not loaded for this dataset")
        return self._material_ids

    def __len__(self) -> int:
        return self.lattice.shape[0]

    def __getitem__(self, index: int) -> Data:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        if index < 0:
            index += len(self)
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        data = Data(
            atom_types=validate_type_tokens(
                self.atom_tokens[start:stop].to(dtype=torch.long)
            ),
            frac_coords=self.fractional_coordinates[start:stop],
            lattice=self.lattice[index].unsqueeze(0),
            num_nodes=stop - start,
        )
        if self._material_ids is not None:
            data.material_id = self._material_ids[index]
        return data

    def select_model_batch(
        self,
        indices: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> PackedAlexModelBatch:
        """Gather ragged rows without exposing audit metadata to the model."""
        if (
            indices.ndim != 1
            or indices.dtype != torch.long
            or indices.numel() < 1
            or int(indices.min()) < 0
            or int(indices.max()) >= len(self)
        ):
            raise ValueError("packed Alex model-batch indices are invalid")
        indices = indices.to(device="cpu")
        starts = self.offsets[indices]
        counts = self.node_counts[indices]
        selected_offset = torch.cumsum(counts, dim=0) - counts
        graph = torch.repeat_interleave(torch.arange(indices.numel()), counts)
        local_node = torch.arange(int(counts.sum())) - torch.repeat_interleave(
            selected_offset,
            counts,
        )
        source_node = starts[graph] + local_node
        return PackedAlexModelBatch(
            atom_types=self.atom_tokens[source_node].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            ),
            fractional_coordinates=self.fractional_coordinates[source_node].to(
                device=device,
                non_blocking=True,
            ),
            lattice=self.lattice[indices].to(device=device, non_blocking=True),
            batch=graph.to(device=device, non_blocking=True),
        )


def collate_packed_alex(records: list[Data]) -> Batch:
    """Collate model-only packed records without audit metadata."""
    if not records:
        raise ValueError("cannot collate an empty packed Alex batch")
    if any(hasattr(record, "material_id") for record in records):
        raise ValueError("training batches must not contain material IDs")
    return Batch.from_data_list(records)
