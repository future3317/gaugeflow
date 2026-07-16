"""Profile Alex-MP-20 source validity and cross-split formula leakage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def reduced_formula_key(atomic_numbers: list[int]) -> tuple[tuple[int, int], ...]:
    counts = Counter(atomic_numbers)
    divisor = reduce(gcd, counts.values())
    return tuple((atomic_number, count // divisor) for atomic_number, count in sorted(counts.items()))


def profile_split(path: Path, *, batch_size: int = 8192) -> tuple[dict[str, object], set[str], set[tuple]]:
    parquet = pq.ParquetFile(path)
    identifiers: set[str] = set()
    formulas: set[tuple] = set()
    counts: Counter[str] = Counter()
    minimum_atoms: int | None = None
    maximum_atoms = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["positions", "cell", "atomic_numbers", "material_id", "space_group"],
    ):
        for row in batch.to_pylist():
            atomic_numbers = row["atomic_numbers"]
            positions = np.asarray(row["positions"], dtype=np.float64)
            lattice = np.asarray(row["cell"], dtype=np.float64)
            material_id = str(row["material_id"])
            atom_count = len(atomic_numbers)
            minimum_atoms = atom_count if minimum_atoms is None else min(minimum_atoms, atom_count)
            maximum_atoms = max(maximum_atoms, atom_count)
            counts["rows"] += 1
            counts["duplicate_material_ids"] += int(material_id in identifiers)
            identifiers.add(material_id)
            formulas.add(reduced_formula_key(atomic_numbers))
            counts["null_space_group"] += int(row["space_group"] is None)
            counts["bad_atom_range"] += int(
                any(value < 1 or value > 118 for value in atomic_numbers)
            )
            counts["bad_position_shape"] += int(positions.shape != (atom_count, 3))
            counts["bad_cell_shape"] += int(lattice.shape != (3, 3))
            counts["nonfinite"] += int(
                not np.isfinite(positions).all() or not np.isfinite(lattice).all()
            )
            counts["nonpositive_volume"] += int(
                lattice.shape == (3, 3) and np.linalg.det(lattice) <= 1e-8
            )
    profile: dict[str, object] = dict(counts)
    profile.update(
        {
            "path": str(path),
            "parquet_created_by": parquet.metadata.created_by,
            "parquet_format_version": parquet.metadata.format_version,
            "metadata_rows": parquet.metadata.num_rows,
            "unique_material_ids": len(identifiers),
            "reduced_formula_groups": len(formulas),
            "minimum_atoms": minimum_atoms,
            "maximum_atoms": maximum_atoms,
        }
    )
    return profile, identifiers, formulas


def audit_source(root: Path) -> dict[str, object]:
    profiles: dict[str, dict[str, object]] = {}
    identifiers: dict[str, set[str]] = {}
    formulas: dict[str, set[tuple]] = {}
    for split in ("train", "val", "test"):
        profiles[split], identifiers[split], formulas[split] = profile_split(
            root / f"{split}.parquet"
        )
    overlaps: dict[str, dict[str, int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}--{right}"] = {
            "material_ids": len(identifiers[left].intersection(identifiers[right])),
            "reduced_formula_groups": len(formulas[left].intersection(formulas[right])),
        }
    validity_fields = (
        "duplicate_material_ids",
        "null_space_group",
        "bad_atom_range",
        "bad_position_shape",
        "bad_cell_shape",
        "nonfinite",
        "nonpositive_volume",
    )
    source_valid = all(int(profile.get(field, 0)) == 0 for profile in profiles.values() for field in validity_fields)
    formula_disjoint = all(value["reduced_formula_groups"] == 0 for value in overlaps.values())
    return {
        "dataset": "Alex-MP-20",
        "grain": "one child crystal structure per row",
        "profiles": profiles,
        "cross_split_overlap": overlaps,
        "source_structure_validity_passed": source_valid,
        "upstream_split_formula_disjoint": formula_disjoint,
        "gaugeflow_split_reuse_authorized": source_valid and formula_disjoint,
        "decision": (
            "source_valid_but_rebuild_child_split"
            if source_valid and not formula_disjoint
            else "source_and_split_qualified"
            if source_valid
            else "source_invalid"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_source(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"]}))


if __name__ == "__main__":
    main()
