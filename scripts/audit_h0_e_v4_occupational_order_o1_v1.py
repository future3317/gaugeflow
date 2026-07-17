"""Independently rebuild the frozen H0-E-v4 O1-v1 held-out census."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from gaugeflow.catalogue import (
    OccupationalParentOccurrence,
    search_occupational_maximal_k_parents,
    search_occupational_maximal_t_parents,
    standardize_child_to_e0_setting,
)
from gaugeflow.file_utils import load_gzip_json, sha256_file

_AUDIT_SOURCE: dict[str, dict[str, object]] = {}
_AUDIT_T: dict[int, list[dict[str, object]]] = {}
_AUDIT_K: dict[int, list[dict[str, object]]] = {}
_AUDIT_SETTINGS: dict[str, object] = {}


def _ordered_hash(values: object) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _crystal_system(space_group: int) -> str:
    boundaries = ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"))
    boundaries += ((142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"))
    for upper, label in boundaries:
        if space_group <= upper:
            return label
    return "cubic"


def _site_bin(site_count: int) -> str:
    for upper, label in ((4, "le4"), (8, "5_8"), (16, "9_16")):
        if site_count <= upper:
            return label
    if site_count <= 32:
        return "17_32"
    if site_count <= 64:
        return "33_64"
    return "gt64"


def _source_rows(
    material_ids: set[str], data_root: Path
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    columns = ["material_id", "positions", "cell", "atomic_numbers"]
    for split in ("test", "val", "train"):
        path = data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet"
        table = pds.dataset(path, format="parquet").to_table(
            filter=pds.field("material_id").isin(material_ids), columns=columns
        )
        for row in table.to_pylist():
            material_id = str(row["material_id"])
            if material_id in rows:
                raise ValueError("independent source join found a duplicate material ID")
            row["source_split_observed"] = split
            rows[material_id] = row
    if set(rows) != material_ids:
        raise ValueError("independent source join did not recover the O1 census")
    return rows


def _selection(
    config: dict[str, Any], data_root: Path, repo_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dependencies = config["dependencies"]
    selection_path = (
        data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/selection.parquet"
    )
    decomposition_path = (
        data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/decompositions.parquet"
    )
    ordered = pq.read_table(selection_path).to_pylist()
    decomposition_rows = pq.read_table(decomposition_path).to_pylist()
    by_id = {str(row["material_id"]): row for row in decomposition_rows}
    if len(by_id) != len(decomposition_rows):
        raise ValueError("independent decomposition join found duplicate IDs")
    cleaning = json.loads(
        (repo_root / str(dependencies["data_cleaning"])).read_text(encoding="utf-8")
    )
    o0_source = json.loads(
        (repo_root / str(dependencies["o0_source_panel_config"])).read_text(
            encoding="utf-8"
        )
    )
    excluded = {str(row["material_id"]) for row in cleaning["material_exclusions"]}
    o0_ids = {str(value) for value in o0_source["selection"]["material_ids"]}
    clean = [row for row in ordered if str(row["material_id"]) not in excluded]
    positives = [
        by_id[str(row["material_id"])]
        for row in clean
        if int(by_id[str(row["material_id"])]["candidate_count"]) > 0
    ]
    o0_clean = [row for row in clean if str(row["material_id"]) in o0_ids]
    selected = [
        by_id[str(row["material_id"])]
        for row in clean
        if int(by_id[str(row["material_id"])]["candidate_count"]) == 0
        and not bool(by_id[str(row["material_id"])]["processing_failure"])
        and str(row["material_id"]) not in o0_ids
    ]
    expected = config["selection"]
    id_groups = (
        [str(row["material_id"]) for row in clean],
        [str(row["material_id"]) for row in positives],
        [str(row["material_id"]) for row in o0_clean],
        [str(row["material_id"]) for row in selected],
    )
    hash_keys = (
        "clean_universe_ordered_ids_sha256",
        "v1_qualified_ordered_ids_sha256",
        "o0_clean_ordered_ids_sha256",
        "o1_ordered_ids_sha256",
    )
    if any(_ordered_hash(ids) != expected[key] for ids, key in zip(id_groups, hash_keys)):
        raise ValueError("independent O1 partition hash mismatch")
    selected_ids = set(id_groups[-1])
    if selected_ids & set(id_groups[2]):
        raise ValueError("independent O1 selection overlaps O0")
    if set(id_groups[1]) | set(id_groups[2]) | selected_ids != set(id_groups[0]):
        raise ValueError("independent partitions do not cover the clean universe")
    strata = [
        {
            "material_id": str(row["material_id"]),
            "gaugeflow_split": str(row["gaugeflow_split"]),
            "child_crystal_system": _crystal_system(int(row["space_group_number"])),
            "primitive_site_bin": _site_bin(int(row["primitive_sites"])),
        }
        for row in selected
    ]
    if _ordered_hash(strata) != expected["o1_strata_records_sha256"]:
        raise ValueError("independent O1 stratum hash mismatch")
    if Counter(str(row["gaugeflow_split"]) for row in selected) != Counter(
        expected["gaugeflow_split_counts"]
    ):
        raise ValueError("independent O1 split counts mismatch")
    return selected, positives


def _serialize(value: OccupationalParentOccurrence) -> dict[str, object]:
    projection = value.projection
    pattern = value.occupational_pattern
    return {
        "embedding_key": value.embedding_key,
        "parent_space_group": value.parent_space_group,
        "cell_index": value.cell_index,
        "full_action_order": value.full_action_order,
        "parent_site_count": value.parent_site_count,
        "child_operation_order": value.child_operation_order,
        "occupational_stabilizer_order": int(
            value.occupational_stabilizer_indices.size
        ),
        "occupational_stabilizer_indices": (
            value.occupational_stabilizer_indices.tolist()
        ),
        "occupational_site_classes": pattern.site_classes.tolist(),
        "occupational_species_by_class_tokens": pattern.species_by_class.tolist(),
        "child_atomic_numbers": value.child_atomic_numbers.tolist(),
        "exact_coloring_reconstruction": value.exact_coloring_reconstruction,
        "stabilizer_order_matches_child": value.stabilizer_order_matches_child,
        "occupationally_nontrivial": (
            value.occupational_stabilizer_indices.size < value.full_action_order
        ),
        "projection_source_max_displacement_angstrom": (
            projection.source_max_displacement_angstrom
        ),
        "projection_source_rms_displacement_angstrom": (
            projection.source_rms_displacement_angstrom
        ),
        "projection_source_hencky_norm": projection.source_hencky_norm,
        "projected_group_max_error_angstrom": (
            projection.projected_group_max_error_angstrom
        ),
        "parent_lattice": projection.lattice.tolist(),
        "parent_fractional": projection.fractional.tolist(),
        "parent_action_permutations": projection.permutations.tolist(),
    }


def _key(value: dict[str, object]) -> tuple[str, int, int]:
    return (
        str(value["embedding_key"]),
        int(value["parent_space_group"]),
        int(value["cell_index"]),
    )


def _audit_material(selected: dict[str, object]) -> dict[str, object]:
    material_id = str(selected["material_id"])
    raw = _AUDIT_SOURCE[material_id]
    child = standardize_child_to_e0_setting(
        np.asarray(raw["cell"], dtype=np.float64),
        np.asarray(raw["positions"], dtype=np.float64),
        np.asarray(raw["atomic_numbers"], dtype=np.int64),
        expected_space_group=int(selected["space_group_number"]),
        expected_primitive_sites=int(selected["primitive_sites"]),
        symprec=float(_AUDIT_SETTINGS["child_symprec_angstrom"]),
        angle_tolerance=float(_AUDIT_SETTINGS["angle_tolerance_degree"]),
    )
    records_t = _AUDIT_T.get(child.space_group, [])
    records_k = _AUDIT_K.get(child.space_group, [])
    common = {
        "maximum_source_displacement_angstrom": float(
            _AUDIT_SETTINGS["maximum_source_displacement_angstrom"]
        ),
        "maximum_source_hencky_norm": float(
            _AUDIT_SETTINGS["source_hencky_norm_max"]
        ),
        "angle_tolerance": float(_AUDIT_SETTINGS["angle_tolerance_degree"]),
    }
    rebuilt = (
        *search_occupational_maximal_k_parents(child, records_k, **common),
        *search_occupational_maximal_t_parents(child, records_t, **common),
    )
    serialized = [_serialize(value) for value in rebuilt]
    return {
        "material_id": material_id,
        "source_split_matches": str(raw["source_split_observed"])
        == str(selected["source_split"]),
        "candidate_edges": len(records_t) + len(records_k),
        "candidates": serialized,
    }


def _candidate_matches(left: dict[str, object], right: dict[str, object]) -> bool:
    exact_arrays = (
        "occupational_stabilizer_indices",
        "occupational_site_classes",
        "occupational_species_by_class_tokens",
        "child_atomic_numbers",
        "parent_action_permutations",
    )
    float_arrays = ("parent_lattice", "parent_fractional")
    scalars = (
        "projection_source_max_displacement_angstrom",
        "projection_source_rms_displacement_angstrom",
        "projection_source_hencky_norm",
        "projected_group_max_error_angstrom",
    )
    contracts = (
        "full_action_order",
        "parent_site_count",
        "child_operation_order",
        "occupational_stabilizer_order",
        "exact_coloring_reconstruction",
        "stabilizer_order_matches_child",
        "occupationally_nontrivial",
    )
    return (
        all(np.array_equal(np.asarray(left[key]), np.asarray(right[key])) for key in exact_arrays)
        and all(
            np.allclose(
                np.asarray(left[key], dtype=np.float64),
                np.asarray(right[key], dtype=np.float64),
                atol=1e-12,
                rtol=0.0,
            )
            for key in float_arrays
        )
        and all(
            np.isclose(float(left[key]), float(right[key]), atol=1e-12, rtol=0.0)
            for key in scalars
        )
        and all(left[key] == right[key] for key in contracts)
    )


def audit(
    config: dict[str, Any], config_path: Path, data_root: Path, repo_root: Path
) -> dict[str, object]:
    global _AUDIT_K, _AUDIT_SETTINGS, _AUDIT_SOURCE, _AUDIT_T
    outputs = config["required_outputs"]
    dependencies = config["dependencies"]
    results_path = data_root / outputs["results"]
    manifest_path = data_root / outputs["manifest"]
    results = load_gzip_json(results_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected, positives = _selection(config, data_root, repo_root)
    selected_ids = {str(row["material_id"]) for row in selected}
    result_by_id = {str(row["material_id"]): row for row in results}
    source = _source_rows(selected_ids, data_root)
    records_path = (
        data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_records.json.gz"
    )
    by_child_t: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_child_k: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in reversed(load_gzip_json(records_path)):
        if str(record["kind"]) == "t" and int(record["cell_index"]) == 1:
            by_child_t[int(record["child_space_group"])].append(record)
        elif str(record["kind"]) == "k" and 2 <= int(record["cell_index"]) <= 4:
            by_child_k[int(record["child_space_group"])].append(record)
    _AUDIT_SOURCE = source
    _AUDIT_T = dict(by_child_t)
    _AUDIT_K = dict(by_child_k)
    _AUDIT_SETTINGS = config["setting_and_search"]
    failures: list[str] = []
    rows_rebuilt = 0
    edge_count = 0
    candidate_materials = 0
    occupational_materials = 0
    occurrence_count = 0
    numeric_matches = 0
    set_matches = 0
    worker_count = min(4, os.cpu_count() or 1)
    context = multiprocessing.get_context("fork")
    reverse_selection = list(reversed(selected))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        iterator = executor.map(_audit_material, reverse_selection, chunksize=1)
        for rebuilt in iterator:
            material_id = str(rebuilt["material_id"])
            try:
                stored = result_by_id[material_id]
                if not bool(rebuilt["source_split_matches"]):
                    raise ValueError("independent source split mismatch")
                edge_count += int(rebuilt["candidate_edges"])
                if int(stored["candidate_edges_evaluated"]) != int(
                    rebuilt["candidate_edges"]
                ):
                    raise ValueError("per-row candidate edge count mismatch")
                expected = {_key(value): value for value in rebuilt["candidates"]}
                observed = {_key(value): value for value in stored["candidates"]}
                if set(expected) != set(observed):
                    raise ValueError("independent embedding set differs")
                set_matches += 1
                numeric_matches += sum(
                    _candidate_matches(value, observed[key])
                    for key, value in expected.items()
                )
                candidate_materials += bool(expected)
                occupational_materials += any(
                    bool(value["occupationally_nontrivial"])
                    for value in expected.values()
                )
                occurrence_count += len(expected)
                rows_rebuilt += 1
            except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                failures.append(f"{material_id}: {type(error).__name__}: {error}")
    o0_results_path = (
        data_root / "processed/gaugeflow_h0_v7/occupational_order_o0_v2/results.json.gz"
    )
    o0_results = load_gzip_json(o0_results_path)
    o0_candidate_materials = sum(bool(row["candidates"]) for row in o0_results)
    aggregate_materials = len(positives) + o0_candidate_materials + candidate_materials
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checks = {
        "results_hash_matches_manifest": sha256_file(results_path)
        == manifest["results_sha256"],
        "protocol_hash_matches_manifest": sha256_file(config_path)
        == manifest["protocol_config_sha256"],
        "implementation_commit_matches_head": manifest["implementation_commit"] == head,
        "e0_hash_matches_protocol": sha256_file(records_path)
        == dependencies["e0_records_sha256"],
        "o0_results_hash_matches_protocol": sha256_file(o0_results_path)
        == dependencies["o0_results_sha256"],
        "selected_and_result_ids_exact": len(selected) == len(result_by_id)
        and set(result_by_id) == selected_ids,
        "all_rows_rebuilt": rows_rebuilt == len(selected) and not failures,
        "candidate_edge_count_reproduced": edge_count
        == int(manifest["metrics"]["candidate_edges_evaluated"]),
        "candidate_sets_reproduced": set_matches == len(selected),
        "candidate_material_count_reproduced": candidate_materials
        == int(manifest["metrics"]["new_candidate_materials"]),
        "occupational_material_count_reproduced": occupational_materials
        == int(manifest["metrics"]["occupationally_nontrivial_materials"]),
        "occurrence_count_reproduced": occurrence_count
        == int(manifest["metrics"]["certified_embedding_occurrences"]),
        "candidate_numerics_reproduced": numeric_matches == occurrence_count,
        "aggregate_material_count_reproduced": aggregate_materials
        == int(manifest["metrics"]["aggregate_qualified_materials"]),
        "stored_decision_matches_checks": bool(manifest["qualified"])
        == all(bool(value) for value in manifest["checks"].values()),
    }
    passed = all(checks.values())
    return {
        "protocol": config["protocol"] + "_independent_reverse_order_audit",
        "audit_passed": passed,
        "gate_qualified": bool(manifest["qualified"]),
        "worker_processes": worker_count,
        "checks": checks,
        "rebuild": {
            "rows": rows_rebuilt,
            "candidate_edges": edge_count,
            "candidate_materials": candidate_materials,
            "occupationally_nontrivial_materials": occupational_materials,
            "occurrences": occurrence_count,
            "failures": failures,
        },
        "decision": (
            "independent_audit_verified_frozen_O1-v1_result"
            if passed
            else "independent_audit_failed_O1-v1_result_untrusted"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    result = audit(config, config_path, arguments.data_root, repo_root)
    output = arguments.data_root / config["required_outputs"]["independent_audit"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if bool(result["audit_passed"]) else 2)


if __name__ == "__main__":
    main()
