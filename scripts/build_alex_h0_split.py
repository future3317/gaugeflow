"""Build the frozen Alex-MP-20 child split used by GaugeFlow H0-A.

The split is child-first.  Rows are connected when they share either an exact
reduced formula or a composition-anonymous, symmetry-resolved primitive
prototype.  A connected component is assigned atomically to one split, so a
later parent candidate, alternate cell, OPD path or cross-source join can only
inherit the child's assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import spglib

SPLIT_SCHEMA = 1
PROTOTYPE_DEFINITION = "spglib-primitive-orbit-signature-v1"
MATCHER_ENVELOPE_DEFINITION = "anonymous-stoichiometry-plus-primitive-site-count-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reduced_formula_key(atomic_numbers: Iterable[int]) -> str:
    counts = Counter(map(int, atomic_numbers))
    divisor = reduce(gcd, counts.values())
    return "|".join(f"{number}:{count // divisor}" for number, count in sorted(counts.items()))


def anonymous_stoichiometry_key(atomic_numbers: Iterable[int]) -> str:
    counts = Counter(map(int, atomic_numbers))
    divisor = reduce(gcd, counts.values())
    return ":".join(map(str, sorted(count // divisor for count in counts.values())))


def _canonical_orbit_signature(
    numbers: np.ndarray,
    equivalent_atoms: np.ndarray,
    site_symmetry_symbols: tuple[str, ...] | list[str],
) -> tuple[tuple[tuple[int, str], ...], ...]:
    """Remove element names and orbit numbering from a primitive-cell signature."""
    species_blocks: list[tuple[tuple[int, str], ...]] = []
    for atomic_number in np.unique(numbers):
        indices = np.flatnonzero(numbers == atomic_number)
        orbit_blocks: list[tuple[int, str]] = []
        for representative in sorted(set(map(int, equivalent_atoms[indices]))):
            members = np.flatnonzero(equivalent_atoms == representative)
            representative_index = int(members[0])
            orbit_blocks.append((int(members.size), str(site_symmetry_symbols[representative_index])))
        species_blocks.append(tuple(sorted(orbit_blocks)))
    return tuple(sorted(species_blocks))


def prototype_signature(
    positions: list[list[float]],
    cell: list[list[float]],
    atomic_numbers: list[int],
    *,
    symprec: float,
    angle_tolerance: float,
) -> tuple[str, int, int, bool]:
    """Return an anonymous primitive prototype hash and audit metadata."""
    lattice = np.asarray(cell, dtype=np.float64)
    cartesian = np.asarray(positions, dtype=np.float64)
    numbers = np.asarray(atomic_numbers, dtype=np.int32)
    fractional = cartesian @ np.linalg.inv(lattice)
    standardized = spglib.standardize_cell(
        (lattice, fractional, numbers),
        to_primitive=True,
        no_idealize=True,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if standardized is None:
        raise ValueError("spglib failed to standardize a structurally valid Alex row")
    primitive_lattice, primitive_fractional, primitive_numbers = standardized
    dataset = spglib.get_symmetry_dataset(
        (primitive_lattice, primitive_fractional, primitive_numbers),
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if dataset is None:
        raise ValueError("spglib failed to identify the standardized primitive cell")
    orbit_signature = _canonical_orbit_signature(
        np.asarray(primitive_numbers, dtype=np.int32),
        np.asarray(dataset.equivalent_atoms, dtype=np.int64),
        tuple(map(str, dataset.site_symmetry_symbols)),
    )
    payload = {
        "definition": PROTOTYPE_DEFINITION,
        "space_group_number": int(dataset.number),
        "primitive_sites": int(len(primitive_numbers)),
        "anonymous_orbits": orbit_signature,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), int(dataset.number), int(len(primitive_numbers)), True


def _prototype_worker(task: tuple[dict[str, object], float, float]) -> tuple[str, int, int, bool]:
    row, symprec, angle_tolerance = task
    return prototype_signature(
        row["positions"],  # type: ignore[arg-type]
        row["cell"],  # type: ignore[arg-type]
        row["atomic_numbers"],  # type: ignore[arg-type]
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )


class DisjointSet:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.size: list[int] = []
        self.key_to_index: dict[tuple[str, str], int] = {}

    def add(self, key: tuple[str, str]) -> int:
        existing = self.key_to_index.get(key)
        if existing is not None:
            return existing
        index = len(self.parent)
        self.key_to_index[key] = index
        self.parent.append(index)
        self.size.append(1)
        return index

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


@dataclass(frozen=True)
class RowRecord:
    material_id: str
    source_split: str
    formula_key: str
    prototype_key: str
    anonymous_stoichiometry: str
    space_group_number: int
    primitive_sites: int

    @property
    def matcher_envelope_key(self) -> str:
        payload = f"{MATCHER_ENVELOPE_DEFINITION}:{self.anonymous_stoichiometry}:{self.primitive_sites}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_component_id(keys: list[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def assign_components(
    component_sizes: dict[str, int],
    *,
    fractions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Deterministic largest-first assignment with normalized fill balancing."""
    if set(fractions) != {"train", "val", "test"} or not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("split fractions must contain train/val/test and sum to one")
    total = sum(component_sizes.values())
    targets = {split: fractions[split] * total for split in fractions}
    counts = {split: 0 for split in fractions}

    def tie_hash(component_id: str) -> str:
        return hashlib.sha256(f"{seed}:{component_id}".encode()).hexdigest()

    ordered = sorted(component_sizes, key=lambda key: (-component_sizes[key], tie_hash(key)))
    assignment: dict[str, str] = {}
    for component_id in ordered:
        count = component_sizes[component_id]
        candidates = sorted(
            fractions,
            key=lambda split: (
                max(0.0, counts[split] + count - targets[split]) / max(targets[split], 1.0),
                counts[split] / max(targets[split], 1.0),
                hashlib.sha256(f"{seed}:{component_id}:{split}".encode()).hexdigest(),
            ),
        )
        selected = candidates[0]
        assignment[component_id] = selected
        counts[selected] += count
    return assignment


def build_split(
    source_root: Path,
    output_root: Path,
    *,
    workers: int,
    batch_size: int,
    symprec: float,
    angle_tolerance: float,
    seed: int,
    fractions: dict[str, float],
    prototype_cache: Path | None = None,
) -> dict[str, object]:
    rows: list[RowRecord] = []
    disjoint = DisjointSet()
    source_files: list[dict[str, object]] = []
    for source_split in ("train", "val", "test"):
        source_path = source_root / f"{source_split}.parquet"
        source_files.append(
            {
                "path": str(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
        )
    prototype_cache_sha256: str | None = None
    if prototype_cache is not None:
        prototype_cache_sha256 = sha256_file(prototype_cache)
        cached = pq.read_table(
            prototype_cache,
            columns=[
                "material_id",
                "source_split",
                "reduced_formula_key",
                "prototype_key",
                "anonymous_stoichiometry",
                "space_group_number",
                "primitive_sites",
            ],
        ).to_pydict()
        rows = [
            RowRecord(
                material_id=str(values[0]),
                source_split=str(values[1]),
                formula_key=str(values[2]),
                prototype_key=str(values[3]),
                anonymous_stoichiometry=str(values[4]),
                space_group_number=int(values[5]),
                primitive_sites=int(values[6]),
            )
            for values in zip(
                cached["material_id"],
                cached["source_split"],
                cached["reduced_formula_key"],
                cached["prototype_key"],
                cached["anonymous_stoichiometry"],
                cached["space_group_number"],
                cached["primitive_sites"],
                strict=True,
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for source_split in ("train", "val", "test"):
                source_path = source_root / f"{source_split}.parquet"
                parquet = pq.ParquetFile(source_path)
                for batch in parquet.iter_batches(
                    batch_size=batch_size,
                    columns=["positions", "cell", "atomic_numbers", "material_id"],
                ):
                    payload = batch.to_pylist()
                    tasks = ((row, symprec, angle_tolerance) for row in payload)
                    prototypes = executor.map(_prototype_worker, tasks, chunksize=64)
                    for row, (prototype, number, primitive_sites, _) in zip(
                        payload, prototypes, strict=True
                    ):
                        atomic_numbers = list(map(int, row["atomic_numbers"]))
                        formula = reduced_formula_key(atomic_numbers)
                        rows.append(
                            RowRecord(
                                material_id=str(row["material_id"]),
                                source_split=source_split,
                                formula_key=formula,
                                prototype_key=prototype,
                                anonymous_stoichiometry=anonymous_stoichiometry_key(atomic_numbers),
                                space_group_number=number,
                                primitive_sites=primitive_sites,
                            )
                        )

    for row in rows:
        disjoint.union(
            disjoint.add(("formula", row.formula_key)),
            disjoint.add(("matcher_envelope", row.matcher_envelope_key)),
        )

    if len({row.material_id for row in rows}) != len(rows):
        raise ValueError("Alex material IDs are not globally unique")
    component_keys: dict[int, list[str]] = defaultdict(list)
    for key, index in disjoint.key_to_index.items():
        component_keys[disjoint.find(index)].append(f"{key[0]}:{key[1]}")
    root_to_id = {root: _stable_component_id(keys) for root, keys in component_keys.items()}
    row_component_ids = [root_to_id[disjoint.find(disjoint.key_to_index[("formula", row.formula_key)])] for row in rows]
    component_sizes = Counter(row_component_ids)
    component_assignment = assign_components(
        dict(component_sizes), fractions=fractions, seed=seed
    )
    gaugeflow_splits = [component_assignment[component_id] for component_id in row_component_ids]

    output_root.mkdir(parents=True, exist_ok=True)
    assignment_path = output_root / "alex_formula_prototype_assignments.parquet"
    table = pa.table(
        {
            "material_id": [row.material_id for row in rows],
            "source_split": [row.source_split for row in rows],
            "reduced_formula_key": [row.formula_key for row in rows],
            "prototype_key": [row.prototype_key for row in rows],
            "matcher_envelope_key": [row.matcher_envelope_key for row in rows],
            "anonymous_stoichiometry": [row.anonymous_stoichiometry for row in rows],
            "space_group_number": [row.space_group_number for row in rows],
            "primitive_sites": [row.primitive_sites for row in rows],
            "component_id": row_component_ids,
            "gaugeflow_split": gaugeflow_splits,
        }
    )
    pq.write_table(table, assignment_path, compression="zstd", version="2.6")

    split_counts = Counter(gaugeflow_splits)
    split_formulas: dict[str, set[str]] = defaultdict(set)
    split_prototypes: dict[str, set[str]] = defaultdict(set)
    split_envelopes: dict[str, set[str]] = defaultdict(set)
    for row, split in zip(rows, gaugeflow_splits, strict=True):
        split_formulas[split].add(row.formula_key)
        split_prototypes[split].add(row.prototype_key)
        split_envelopes[split].add(row.matcher_envelope_key)
    overlap: dict[str, dict[str, int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{left}--{right}"] = {
            "reduced_formula": len(split_formulas[left] & split_formulas[right]),
            "prototype": len(split_prototypes[left] & split_prototypes[right]),
            "matcher_envelope": len(split_envelopes[left] & split_envelopes[right]),
        }
    observed_fractions = {split: split_counts[split] / len(rows) for split in fractions}
    maximum_fraction_deviation = max(
        abs(observed_fractions[split] - fractions[split]) for split in fractions
    )
    manifest: dict[str, object] = {
        "schema": SPLIT_SCHEMA,
        "protocol": "h0_a_alex_formula_prototype_split_v1",
        "source": "Alex-MP-20",
        "source_files": source_files,
        "rows": len(rows),
        "seed": seed,
        "target_fractions": fractions,
        "observed_fractions": observed_fractions,
        "split_counts": dict(split_counts),
        "component_count": len(component_sizes),
        "largest_component_rows": max(component_sizes.values()),
        "maximum_fraction_deviation": maximum_fraction_deviation,
        "prototype_definition": {
            "name": PROTOTYPE_DEFINITION,
            "symprec_angstrom": symprec,
            "angle_tolerance_degree": angle_tolerance,
            "primitive_standardization": "spglib.standardize_cell(to_primitive=True,no_idealize=True)",
            "anonymous_signature": (
                "space-group number, primitive site count, species-anonymized orbit "
                "multiplicities and site symmetries"
            ),
        },
        "matcher_envelope_definition": {
            "name": MATCHER_ENVELOPE_DEFINITION,
            "necessary_condition": (
                "StructureMatcher.fit_anonymous with primitive_cell=True and attempt_supercell=False "
                "cannot match structures in different envelopes"
            ),
        },
        "component_rule": (
            "connected components of exact reduced formula and StructureMatcher necessary-condition "
            "envelope bipartite graph"
        ),
        "split_inheritance_rule": (
            "all parent candidates, alternate cells, OPD paths, mode scans and cross-source "
            "joins inherit the child assignment"
        ),
        "cross_split_overlap": overlap,
        "assignment_path": assignment_path.name,
        "assignment_bytes": assignment_path.stat().st_size,
        "assignment_sha256": sha256_file(assignment_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "prototype_cache_input_sha256": prototype_cache_sha256,
        "qualification": {
            "all_rows_assigned_once": len(rows) == table.num_rows,
            "material_ids_unique": len({row.material_id for row in rows}) == len(rows),
            "formula_overlap_zero": all(value["reduced_formula"] == 0 for value in overlap.values()),
            "prototype_overlap_zero": all(value["prototype"] == 0 for value in overlap.values()),
            "matcher_envelope_overlap_zero": all(
                value["matcher_envelope"] == 0 for value in overlap.values()
            ),
            "fraction_deviation_at_most_0_02": maximum_fraction_deviation <= 0.02,
            "structure_matcher_audit": "pending_separate_frozen_audit",
        },
    }
    manifest_path = output_root / "alex_formula_prototype_split.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--prototype-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_split(
        args.source_root,
        args.output_root,
        workers=args.workers,
        batch_size=args.batch_size,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
        seed=args.seed,
        fractions={"train": 0.8, "val": 0.1, "test": 0.1},
        prototype_cache=args.prototype_cache,
    )
    print(json.dumps({"rows": manifest["rows"], "split_counts": manifest["split_counts"]}))


if __name__ == "__main__":
    main()
