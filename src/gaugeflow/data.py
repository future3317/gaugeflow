"""Standalone CIF/CSV data path for GaugeFlow using PyG Data/Batch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .stabilizer import proper_stabilizer_rotations
from .tensor import piezo_from_irreps
from .unit_cell import niggli_reduce_structure


class PiezoCrystalDataset(Dataset):
    """Read paired CIF and full-tensor conditions without the FlowMM runtime."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        condition_column: str = "piezo_irreps_raw",
        symmetry_tolerance: float = 1e-3,
    ):
        self.path = Path(csv_path)
        self.frame = pd.read_csv(self.path)
        if condition_column not in self.frame:
            raise ValueError(f"{self.path} does not contain {condition_column}")
        self.condition_column = condition_column
        self.symmetry_tolerance = symmetry_tolerance
        self._stabilizer_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Data:
        row = self.frame.iloc[index]
        structure = Structure.from_str(row.cif, fmt="cif")
        structure = niggli_reduce_structure(structure)
        irreps = torch.tensor(json.loads(row[self.condition_column]), dtype=torch.float32)
        if irreps.shape != (18,):
            raise ValueError(f"Expected 18 tensor coordinates for row {index}")
        # Verify the stored condition is a complete symmetric rank-three tensor.
        _ = piezo_from_irreps(irreps)
        stabilizer = self._stabilizer_cache.get(index)
        if stabilizer is None:
            stabilizer = proper_stabilizer_rotations(
                structure, symprec=self.symmetry_tolerance
            )
            self._stabilizer_cache[index] = stabilizer
        return Data(
            atom_types=torch.tensor(structure.atomic_numbers, dtype=torch.long),
            frac_coords=torch.tensor(structure.frac_coords, dtype=torch.float32),
            lattice=torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0),
            piezo_irreps=irreps.unsqueeze(0),
            condition_present=torch.ones((1, 1), dtype=torch.bool),
            stabilizer_rotations=stabilizer,
            stabilizer_count=torch.tensor([stabilizer.shape[0]], dtype=torch.long),
            num_nodes=len(structure),
        )


def collate_crystals(records: list[Data]) -> Batch:
    if not records:
        raise ValueError("Cannot collate an empty crystal batch")
    return Batch.from_data_list(records)
