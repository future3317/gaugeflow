"""Standalone CIF/CSV data path for GaugeFlow using PyG Data/Batch."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .tensor import isotypic_slices, piezo_from_irreps, piezo_to_irreps
from .unit_cell import niggli_reduce_structure_with_transform
from .vocabulary import atomic_numbers_to_tokens, validate_type_tokens

RESPONSE_NORM_BOUNDS = (0.0, 0.05, 0.5, 1.0)
# Schema 2 is the immutable historical v1 cache contract.  Schema 3 records a
# full-O(3) crystal-compatibility projection for the prospective v2 rebuild.
# Keep both readable: accepting a cache is not permission to substitute it for
# a frozen Gate-A split.
SYMMETRY_TARGET_CACHE_SCHEMA = 2
FULL_O3_SYMMETRY_TARGET_CACHE_SCHEMA = 3
SUPPORTED_SYMMETRY_TARGET_CACHE_SCHEMAS = frozenset(
    (SYMMETRY_TARGET_CACHE_SCHEMA, FULL_O3_SYMMETRY_TARGET_CACHE_SCHEMA)
)
PREPROCESSED_CRYSTAL_CACHE_SCHEMA = 2
TENSOR_CONVENTION_VERSION = "gaugeflow-cartesian-ijk=ikj-engineering-shear-v1"


def response_stratum(norm: float) -> int:
    """Map a response norm to the shared versioned magnitude stratum."""
    if norm <= 1.0e-12:
        return 0
    for index, upper in enumerate(RESPONSE_NORM_BOUNDS[1:], start=1):
        if norm < upper:
            return index
    return len(RESPONSE_NORM_BOUNDS)


def load_piezo_frame(source: str | Path) -> pd.DataFrame:
    """Load a piezo CSV, or the three original FlowMM split CSVs as one table.

    The latter is intentionally only an input container.  A protocol split is
    applied afterwards, so the historical random train/val/test files cannot
    leak formula groups into a new formula-grouped evaluation split.
    """
    path = Path(source)
    if path.is_dir():
        files = [path / f"{name}.csv" for name in ("train", "val", "test")]
        files = [file for file in files if file.is_file()]
        if not files:
            raise FileNotFoundError(f"No train.csv, val.csv, or test.csv under {path}")
        frame = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    elif path.is_file():
        frame = pd.read_csv(path)
    else:
        raise FileNotFoundError(path)
    required = {"material_id", "cif"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["material_id"] = frame["material_id"].astype(str)
    if frame.material_id.duplicated().any():
        duplicates = frame.loc[frame.material_id.duplicated(), "material_id"].head().tolist()
        raise ValueError(f"Duplicate material IDs in {path}: {duplicates}")
    return frame


def _select_manifest_split(frame: pd.DataFrame, manifest_path: Path, split: str | None) -> pd.DataFrame:
    if split is None:
        return frame
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split not in manifest or not isinstance(manifest[split], list):
        raise ValueError(f"{manifest_path} has no list-valued '{split}' split")
    ids = [str(material_id) for material_id in manifest[split]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"'{split}' in {manifest_path} contains duplicate material IDs")
    indexed = frame.set_index("material_id", drop=False)
    missing = sorted(set(ids).difference(indexed.index))
    if missing:
        raise ValueError(f"{len(missing)} IDs from '{split}' are absent from {manifest_path.parent}: {missing[:5]}")
    return indexed.loc[ids].reset_index(drop=True)


def _target_cache_file(cache_dir: Path, material_id: str) -> Path:
    digest = hashlib.sha256(material_id.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.pt"


class PiezoCrystalDataset(Dataset):
    """Read paired CIF/full-response conditions without the FlowMM runtime.

    When a TensorOrbit-JARVIS cache is provided, its Reynolds-projected Cartesian
    target is the condition.  Target-CIF stabilizers are deliberately not
    emitted as model inputs: they are unavailable when sampling from a tensor
    orbit and would create a training--inference information mismatch.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        condition_column: str | None = "piezo_irreps_raw",
        split_manifest: str | Path | None = None,
        split: str | None = None,
        target_cache_dir: str | Path | None = None,
        preprocessed_cache: str | Path | None = None,
    ):
        self.path = Path(csv_path)
        self.manifest_path = Path(split_manifest) if split_manifest is not None else None
        if split is not None and self.manifest_path is None:
            raise ValueError("split requires split_manifest")
        self.frame = load_piezo_frame(self.path)
        if split is not None:
            self.frame = _select_manifest_split(self.frame, self.manifest_path, split)
        if target_cache_dir is None and condition_column is not None and condition_column not in self.frame:
            raise ValueError(f"{self.path} does not contain {condition_column}")
        self.condition_column = condition_column
        self.condition_enabled = condition_column is not None or target_cache_dir is not None
        self.target_cache_dir = Path(target_cache_dir) if target_cache_dir is not None else None
        if self.target_cache_dir is not None and not self.target_cache_dir.is_dir():
            raise FileNotFoundError(self.target_cache_dir)
        self._condition_irreps_cache: dict[int, torch.Tensor] = {}
        self._condition_norm_cache: dict[int, float] = {}
        self.preprocessed_cache = Path(preprocessed_cache) if preprocessed_cache is not None else None
        self._preprocessed_records: dict[str, dict[str, Any]] | None = None
        if self.preprocessed_cache is not None:
            payload: Any = torch.load(
                self.preprocessed_cache, map_location="cpu", weights_only=True
            )
            if not isinstance(payload, dict) or payload.get("schema") != PREPROCESSED_CRYSTAL_CACHE_SCHEMA:
                raise ValueError(f"Unexpected preprocessed cache payload in {self.preprocessed_cache}")
            manifest = payload.get("manifest")
            records = payload.get("records")
            if not isinstance(manifest, dict) or manifest.get("tensor_convention_version") != TENSOR_CONVENTION_VERSION:
                raise ValueError("Preprocessed cache tensor convention does not match this runtime")
            if not isinstance(records, dict):
                raise ValueError("Preprocessed cache records must be keyed by material ID")
            selected_ids = set(map(str, self.frame.material_id))
            missing = sorted(selected_ids.difference(records))
            if missing:
                raise ValueError(f"Preprocessed cache is missing selected material IDs: {missing[:5]}")
            self._preprocessed_records = records

    def __len__(self) -> int:
        return len(self.frame)

    def _condition_for_index(self, index: int) -> tuple[torch.Tensor, float]:
        if not self.condition_enabled:
            raise RuntimeError("tensor conditions are disabled for this structure-only dataset")
        cached = self._condition_irreps_cache.get(index)
        if cached is not None:
            return cached, self._condition_norm_cache[index]
        row = self.frame.iloc[index]
        if self._preprocessed_records is not None:
            record = self._preprocessed_records[str(row.material_id)]
            irreps = torch.as_tensor(record["piezo_irreps"], dtype=torch.float32)
            norm = float(record["response_norm"])
            if irreps.shape != (18,) or not torch.isfinite(irreps).all() or not math.isfinite(norm):
                raise ValueError(f"Invalid condition in preprocessed cache for {row.material_id}")
            self._condition_irreps_cache[index] = irreps
            self._condition_norm_cache[index] = norm
            return irreps, norm
        if self.target_cache_dir is None:
            irreps = torch.tensor(json.loads(row[self.condition_column]), dtype=torch.float32)
            if irreps.shape != (18,):
                raise ValueError(f"Expected 18 tensor coordinates for row {index}")
            tensor = piezo_from_irreps(irreps)
        else:
            cache_file = _target_cache_file(self.target_cache_dir, str(row.material_id))
            payload: Any = torch.load(cache_file, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("schema") not in SUPPORTED_SYMMETRY_TARGET_CACHE_SCHEMAS:
                raise ValueError(f"Unexpected TensorOrbit target-cache payload in {cache_file}")
            tensor = torch.as_tensor(payload.get("target"), dtype=torch.float32)
            if tensor.shape != (3, 3, 3) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid projected target in {cache_file}")
            if not torch.allclose(tensor, tensor.transpose(-1, -2), atol=1e-5, rtol=1e-5):
                raise ValueError(f"Projected target is not symmetric in the strain indices: {cache_file}")
            irreps = piezo_to_irreps(tensor)
        if not torch.isfinite(irreps).all():
            raise ValueError(f"Non-finite tensor condition for row {index}")
        self._condition_irreps_cache[index] = irreps
        self._condition_norm_cache[index] = float(torch.linalg.vector_norm(tensor))
        return irreps, self._condition_norm_cache[index]

    def condition_irreps(self) -> torch.Tensor:
        """All selected conditions, using projected cache targets when present."""
        if not self.condition_enabled:
            raise RuntimeError("tensor conditions are disabled for this structure-only dataset")
        return torch.stack([self._condition_for_index(index)[0] for index in range(len(self))])

    def isotypic_scales(self) -> torch.Tensor:
        values = self.condition_irreps()
        return torch.stack(
            [values[:, block].square().mean().sqrt().clamp_min(1e-8) for block in isotypic_slices()]
        )

    def condition_bins(self) -> torch.Tensor:
        """TensorOrbit-JARVIS response-norm strata, with zero as an explicit class."""
        norms = torch.tensor([self._condition_for_index(index)[1] for index in range(len(self))])
        return torch.tensor([response_stratum(float(norm)) for norm in norms], dtype=torch.long)

    def condition_sampling_weights(self, power: float = 0.5) -> torch.Tensor:
        """Inverse-frequency weights over response strata; zero remains physical data."""
        if not 0.0 <= power <= 1.0:
            raise ValueError("condition sampling power must lie in [0, 1]")
        bins = self.condition_bins()
        counts = torch.bincount(bins, minlength=len(RESPONSE_NORM_BOUNDS) + 1).float()
        weights = counts[bins].pow(-power)
        return weights / weights.mean()

    def __getitem__(self, index: int) -> Data:
        row = self.frame.iloc[index]
        if self._preprocessed_records is not None:
            record = self._preprocessed_records[str(row.material_id)]
            if self.condition_enabled:
                condition = torch.as_tensor(record["piezo_irreps"], dtype=torch.float32)
                present = torch.ones((1, 1), dtype=torch.bool)
                stratum = int(record["response_stratum"])
                zero_response = bool(record["zero_response"])
            else:
                condition = torch.zeros((18,), dtype=torch.float32)
                present = torch.zeros((1, 1), dtype=torch.bool)
                stratum = -1
                zero_response = False
            return Data(
                atom_types=validate_type_tokens(torch.as_tensor(record["atom_types"])).clone(),
                frac_coords=torch.as_tensor(record["frac_coords"], dtype=torch.float32).clone(),
                lattice=torch.as_tensor(record["lattice"], dtype=torch.float32).unsqueeze(0).clone(),
                piezo_irreps=condition.unsqueeze(0).clone(),
                condition_present=present,
                niggli_transform=torch.as_tensor(record["niggli_transform"], dtype=torch.int64).unsqueeze(0).clone(),
                response_stratum=torch.tensor([stratum], dtype=torch.long),
                zero_response=torch.tensor([zero_response], dtype=torch.bool),
                material_id=str(row.material_id),
                num_nodes=int(torch.as_tensor(record["atom_types"]).numel()),
            )
        structure = Structure.from_str(row.cif, fmt="cif")
        structure, niggli_transform = niggli_reduce_structure_with_transform(structure)
        if self.condition_enabled:
            irreps, response_norm = self._condition_for_index(index)
            condition_present = torch.ones((1, 1), dtype=torch.bool)
            stratum = response_stratum(response_norm)
            zero_response = response_norm <= 1e-12
        else:
            irreps = torch.zeros((18,), dtype=torch.float32)
            condition_present = torch.zeros((1, 1), dtype=torch.bool)
            stratum = -1
            zero_response = False
        return Data(
            atom_types=atomic_numbers_to_tokens(torch.tensor(structure.atomic_numbers, dtype=torch.long)),
            frac_coords=torch.tensor(structure.frac_coords, dtype=torch.float32),
            lattice=torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0),
            piezo_irreps=irreps.unsqueeze(0),
            condition_present=condition_present,
            niggli_transform=torch.tensor(niggli_transform, dtype=torch.int64).unsqueeze(0),
            response_stratum=torch.tensor([stratum], dtype=torch.long),
            zero_response=torch.tensor([zero_response], dtype=torch.bool),
            material_id=str(row.material_id),
            num_nodes=len(structure),
        )


def collate_crystals(records: list[Data]) -> Batch:
    if not records:
        raise ValueError("Cannot collate an empty crystal batch")
    return Batch.from_data_list(records)
