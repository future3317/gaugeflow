"""Qualify the frozen Alex-MP-20 formula/prototype-disjoint H0-A split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from scripts.build_alex_h0_split import SPLIT_SCHEMA, sha256_file


def _pair_hash(seed: int, left: str, right: str) -> str:
    ordered = sorted((left, right))
    return hashlib.sha256(f"{seed}:{ordered[0]}:{ordered[1]}".encode()).hexdigest()


def _select_candidates(
    table: dict[str, list[Any]],
    *,
    seed: int,
    representatives_per_bucket_split: int,
    maximum_pairs: int,
) -> list[tuple[str, str]]:
    buckets: dict[tuple[str, int], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for material_id, stoichiometry, primitive_sites, split in zip(
        table["material_id"],
        table["anonymous_stoichiometry"],
        table["primitive_sites"],
        table["gaugeflow_split"],
        strict=True,
    ):
        buckets[(str(stoichiometry), int(primitive_sites))][str(split)].append(str(material_id))
    pairs: list[tuple[str, str]] = []
    for split_groups in buckets.values():
        selected: dict[str, list[str]] = {}
        for split, identifiers in split_groups.items():
            selected[split] = sorted(
                identifiers,
                key=lambda material_id: hashlib.sha256(f"{seed}:{material_id}".encode()).hexdigest(),
            )[:representatives_per_bucket_split]
        for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
            for left in selected.get(left_split, []):
                for right in selected.get(right_split, []):
                    pairs.append((left, right))
    pairs.sort(key=lambda pair: _pair_hash(seed, *pair))
    return pairs[:maximum_pairs]


def _load_selected_structures(source_root: Path, identifiers: set[str]) -> dict[str, Structure]:
    if not identifiers:
        return {}
    selected: dict[str, Structure] = {}
    for split in ("train", "val", "test"):
        parquet = pq.ParquetFile(source_root / f"{split}.parquet")
        for batch in parquet.iter_batches(
            batch_size=8192,
            columns=["positions", "cell", "atomic_numbers", "material_id"],
        ):
            for row in batch.to_pylist():
                material_id = str(row["material_id"])
                if material_id not in identifiers:
                    continue
                selected[material_id] = Structure(
                    Lattice(row["cell"]),
                    row["atomic_numbers"],
                    row["positions"],
                    coords_are_cartesian=True,
                    to_unit_cell=True,
                )
    missing = identifiers.difference(selected)
    if missing:
        raise ValueError(f"assignment references {len(missing)} IDs absent from Alex sources")
    return selected


def audit_split(
    source_root: Path,
    split_manifest_path: Path,
    *,
    seed: int,
    representatives_per_bucket_split: int,
    maximum_pairs: int,
    ltol: float,
    stol: float,
    angle_tol: float,
) -> dict[str, object]:
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SPLIT_SCHEMA:
        raise ValueError("unsupported Alex H0 split schema")
    assignment_path = split_manifest_path.parent / str(manifest["assignment_path"])
    assignment_hash_matches = sha256_file(assignment_path) == manifest["assignment_sha256"]
    assignments = pq.read_table(assignment_path).to_pydict()
    rows = len(assignments["material_id"])
    unique_ids = len(set(map(str, assignments["material_id"])))
    split_counts = Counter(map(str, assignments["gaugeflow_split"]))

    formula_sets: dict[str, set[str]] = defaultdict(set)
    prototype_sets: dict[str, set[str]] = defaultdict(set)
    envelope_sets: dict[str, set[str]] = defaultdict(set)
    component_sets: dict[str, set[str]] = defaultdict(set)
    for formula, prototype, envelope, component, split in zip(
        assignments["reduced_formula_key"],
        assignments["prototype_key"],
        assignments["matcher_envelope_key"],
        assignments["component_id"],
        assignments["gaugeflow_split"],
        strict=True,
    ):
        split = str(split)
        formula_sets[split].add(str(formula))
        prototype_sets[split].add(str(prototype))
        envelope_sets[split].add(str(envelope))
        component_sets[split].add(str(component))
    overlap: dict[str, dict[str, int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{left}--{right}"] = {
            "reduced_formula": len(formula_sets[left] & formula_sets[right]),
            "prototype": len(prototype_sets[left] & prototype_sets[right]),
            "matcher_envelope": len(envelope_sets[left] & envelope_sets[right]),
            "component": len(component_sets[left] & component_sets[right]),
        }

    envelope_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for envelope, split in zip(
        assignments["matcher_envelope_key"], assignments["gaugeflow_split"], strict=True
    ):
        envelope_counts[str(envelope)][str(split)] += 1
    possible_cross_split_matcher_pairs = sum(
        counts["train"] * counts["val"]
        + counts["train"] * counts["test"]
        + counts["val"] * counts["test"]
        for counts in envelope_counts.values()
    )

    pairs = _select_candidates(
        assignments,
        seed=seed,
        representatives_per_bucket_split=representatives_per_bucket_split,
        maximum_pairs=maximum_pairs,
    )
    selected_ids = {material_id for pair in pairs for material_id in pair}
    structures = _load_selected_structures(source_root, selected_ids)
    matcher = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )
    matches: list[dict[str, str]] = []
    for left, right in pairs:
        if matcher.fit_anonymous(structures[left], structures[right]):
            matches.append({"left": left, "right": right})

    target_fractions = {key: float(value) for key, value in manifest["target_fractions"].items()}
    observed_fractions = {split: split_counts[split] / rows for split in target_fractions}
    maximum_fraction_deviation = max(
        abs(observed_fractions[split] - target_fractions[split]) for split in target_fractions
    )
    source_rows = sum(
        pq.ParquetFile(source_root / f"{split}.parquet").metadata.num_rows
        for split in ("train", "val", "test")
    )
    checks = {
        "assignment_hash_matches": assignment_hash_matches,
        "all_source_rows_assigned_once": rows == source_rows == unique_ids,
        "split_names_exact": set(split_counts) == {"train", "val", "test"},
        "formula_overlap_zero": all(value["reduced_formula"] == 0 for value in overlap.values()),
        "prototype_overlap_zero": all(value["prototype"] == 0 for value in overlap.values()),
        "matcher_envelope_overlap_zero": all(
            value["matcher_envelope"] == 0 for value in overlap.values()
        ),
        "component_overlap_zero": all(value["component"] == 0 for value in overlap.values()),
        "fraction_deviation_at_most_0_02": maximum_fraction_deviation <= 0.02,
        "structure_matcher_cross_split_matches_zero": len(matches) == 0,
        "structure_matcher_candidate_universe_empty": possible_cross_split_matcher_pairs == 0,
    }
    return {
        "protocol": "h0_a_alex_formula_prototype_split_audit_v1",
        "qualified": all(checks.values()),
        "checks": checks,
        "rows": rows,
        "split_counts": dict(split_counts),
        "observed_fractions": observed_fractions,
        "maximum_fraction_deviation": maximum_fraction_deviation,
        "cross_split_overlap": overlap,
        "structure_matcher": {
            "ltol": ltol,
            "stol": stol,
            "angle_tol": angle_tol,
            "primitive_cell": True,
            "scale": True,
            "attempt_supercell": False,
            "candidate_bucket": "anonymous_stoichiometry plus standardized primitive site count",
            "representatives_per_bucket_split": representatives_per_bucket_split,
            "pairs_tested": len(pairs),
            "possible_cross_split_candidate_pairs": possible_cross_split_matcher_pairs,
            "matches": matches,
        },
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "assignment_path": str(assignment_path),
        "assignment_sha256": sha256_file(assignment_path),
        "auditor_sha256": sha256_file(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--representatives-per-bucket-split", type=int, default=4)
    parser.add_argument("--maximum-pairs", type=int, default=20000)
    parser.add_argument("--ltol", type=float, default=0.2)
    parser.add_argument("--stol", type=float, default=0.3)
    parser.add_argument("--angle-tol", type=float, default=5.0)
    args = parser.parse_args()
    result = audit_split(
        args.source_root,
        args.split_manifest,
        seed=args.seed,
        representatives_per_bucket_split=args.representatives_per_bucket_split,
        maximum_pairs=args.maximum_pairs,
        ltol=args.ltol,
        stol=args.stol,
        angle_tol=args.angle_tol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"qualified": result["qualified"], "pairs": result["structure_matcher"]["pairs_tested"]}))
    if not result["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
