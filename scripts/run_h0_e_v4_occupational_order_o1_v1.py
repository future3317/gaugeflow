"""Run the frozen H0-E-v4 O1-v1 held-out occupational census."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
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
from gaugeflow.file_utils import (
    load_gzip_json,
    sha256_file,
    write_deterministic_gzip_json,
)

_WORKER_SOURCE: dict[str, dict[str, object]] = {}
_WORKER_T: dict[int, list[dict[str, object]]] = {}
_WORKER_K: dict[int, list[dict[str, object]]] = {}
_WORKER_SETTINGS: dict[str, object] = {}


def _ordered_hash(values: object) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _crystal_system(space_group: int) -> str:
    if space_group <= 2:
        return "triclinic"
    if space_group <= 15:
        return "monoclinic"
    if space_group <= 74:
        return "orthorhombic"
    if space_group <= 142:
        return "tetragonal"
    if space_group <= 167:
        return "trigonal"
    if space_group <= 194:
        return "hexagonal"
    return "cubic"


def _site_bin(site_count: int) -> str:
    if site_count <= 4:
        return "le4"
    if site_count <= 8:
        return "5_8"
    if site_count <= 16:
        return "9_16"
    if site_count <= 32:
        return "17_32"
    if site_count <= 64:
        return "33_64"
    return "gt64"


def _clean_git_commit(repo_root: Path) -> str:
    unstaged = subprocess.run(
        ["git", "diff", "--ignore-space-at-eol", "--quiet"],
        cwd=repo_root,
        check=False,
    ).returncode
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root,
        check=False,
    ).returncode
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if (unstaged, staged) != (0, 0) or untracked.strip():
        raise RuntimeError("the frozen O1-v1 run requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dependency_paths(
    config: dict[str, Any], data_root: Path, repo_root: Path
) -> dict[str, Path]:
    outputs = config["required_outputs"]
    dependencies = config["dependencies"]
    return {
        "v1_selection_sha256": data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/selection.parquet",
        "v1_decompositions_sha256": data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/decompositions.parquet",
        "e0_records_sha256": data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_records.json.gz",
        "e0_manifest_sha256": data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_manifest.json",
        "o0_config_sha256": repo_root / str(dependencies["o0_config"]),
        "o0_results_sha256": data_root
        / "processed/gaugeflow_h0_v7/occupational_order_o0_v2/results.json.gz",
        "o0_manifest_sha256": data_root
        / "processed/gaugeflow_h0_v7/occupational_order_o0_v2/manifest.json",
        "o0_independent_audit_sha256": data_root
        / "processed/gaugeflow_h0_v7/occupational_order_o0_v2/independent_audit.json",
        "o0_source_panel_config_sha256": repo_root
        / str(dependencies["o0_source_panel_config"]),
        "data_cleaning_sha256": repo_root / str(dependencies["data_cleaning"]),
        "results_output": data_root / str(outputs["results"]),
    }


def _validate_dependencies(
    config: dict[str, Any], data_root: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dependencies = config["dependencies"]
    paths = _dependency_paths(config, data_root, repo_root)
    for key, path in paths.items():
        if key == "results_output":
            continue
        if sha256_file(path) != str(dependencies[key]):
            raise ValueError(f"dependency hash mismatch: {key}")
    for split, expected in dependencies["raw_alex_sha256"].items():
        path = data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet"
        if sha256_file(path) != str(expected):
            raise ValueError(f"Alex source hash mismatch: {split}")
    if importlib.metadata.version("pyxtal") != dependencies["pyxtal_version"]:
        raise ValueError("PyXtal version does not match the frozen O1-v1 protocol")
    o0_manifest = json.loads(paths["o0_manifest_sha256"].read_text(encoding="utf-8"))
    o0_audit = json.loads(
        paths["o0_independent_audit_sha256"].read_text(encoding="utf-8")
    )
    if not bool(o0_manifest["qualified"]) or not bool(o0_audit["audit_passed"]):
        raise ValueError("O0-v2 is not independently qualified")
    return (
        json.loads(paths["o0_source_panel_config_sha256"].read_text(encoding="utf-8")),
        json.loads(paths["data_cleaning_sha256"].read_text(encoding="utf-8")),
        o0_manifest,
    )


def _selection(
    config: dict[str, Any],
    o0_source: dict[str, Any],
    cleaning: dict[str, Any],
    data_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selection_path = (
        data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/selection.parquet"
    )
    decomposition_path = (
        data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/decompositions.parquet"
    )
    ordered = pq.read_table(selection_path).to_pylist()
    decompositions = pq.read_table(decomposition_path).to_pylist()
    by_id = {str(row["material_id"]): row for row in decompositions}
    ordered_ids = [str(row["material_id"]) for row in ordered]
    if len(by_id) != len(decompositions) or set(by_id) != set(ordered_ids):
        raise ValueError("v1 selection/decomposition material identity mismatch")
    excluded = {str(row["material_id"]) for row in cleaning["material_exclusions"]}
    o0_ids = {str(value) for value in o0_source["selection"]["material_ids"]}
    clean_ordered = [row for row in ordered if str(row["material_id"]) not in excluded]
    positives = [
        row
        for row in clean_ordered
        if int(by_id[str(row["material_id"])]["candidate_count"]) > 0
    ]
    o0_clean = [row for row in clean_ordered if str(row["material_id"]) in o0_ids]
    selected_rows = [
        row
        for row in clean_ordered
        if int(by_id[str(row["material_id"])]["candidate_count"]) == 0
        and not bool(by_id[str(row["material_id"])]["processing_failure"])
        and str(row["material_id"]) not in o0_ids
    ]
    expected = config["selection"]
    partitions = (positives, o0_clean, selected_rows)
    partition_sets = [
        {str(row["material_id"]) for row in partition} for partition in partitions
    ]
    if any(partition_sets[i] & partition_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("clean O1 partitions are not disjoint")
    if set().union(*partition_sets) != {
        str(row["material_id"]) for row in clean_ordered
    }:
        raise ValueError("clean O1 partitions do not cover the universe")
    hashes = {
        "clean_universe_ordered_ids_sha256": clean_ordered,
        "v1_qualified_ordered_ids_sha256": positives,
        "o0_clean_ordered_ids_sha256": o0_clean,
        "o1_ordered_ids_sha256": selected_rows,
    }
    for key, rows in hashes.items():
        if _ordered_hash([str(row["material_id"]) for row in rows]) != expected[key]:
            raise ValueError(f"selection hash mismatch: {key}")
    strata = [
        {
            "material_id": str(row["material_id"]),
            "gaugeflow_split": str(row["gaugeflow_split"]),
            "child_crystal_system": _crystal_system(int(row["space_group_number"])),
            "primitive_site_bin": _site_bin(int(row["primitive_sites"])),
        }
        for row in selected_rows
    ]
    if _ordered_hash(strata) != expected["o1_strata_records_sha256"]:
        raise ValueError("held-out O1 strata hash mismatch")
    counters = {
        "gaugeflow_split_counts": Counter(
            str(row["gaugeflow_split"]) for row in selected_rows
        ),
        "source_split_counts": Counter(str(row["source_split"]) for row in selected_rows),
        "child_crystal_system_counts": Counter(
            _crystal_system(int(row["space_group_number"])) for row in selected_rows
        ),
        "primitive_site_bin_counts": Counter(
            _site_bin(int(row["primitive_sites"])) for row in selected_rows
        ),
    }
    for key, counter in counters.items():
        frozen_counter = Counter(
            {str(k): int(v) for k, v in expected[key].items()}
        )
        if counter != frozen_counter:
            raise ValueError(f"selection stratum mismatch: {key}")
    if len(clean_ordered) != int(expected["clean_universe_size"]):
        raise ValueError("clean universe size mismatch")
    if [len(value) for value in partitions] != [
        int(expected["v1_qualified_partition_size"]),
        int(expected["o0_clean_partition_size"]),
        int(expected["o1_size"]),
    ]:
        raise ValueError("O1 partition size mismatch")
    if not all(
        bool(by_id[str(row["material_id"])]["qualified_nontrivial"])
        for row in positives
    ):
        raise ValueError("the frozen v1-positive partition is not fully qualified")
    selected = [by_id[str(row["material_id"])] for row in selected_rows]
    return selected, [by_id[str(row["material_id"])] for row in positives]


def _join_source(
    selection: list[dict[str, object]], data_root: Path
) -> dict[str, dict[str, object]]:
    requested = {str(row["material_id"]) for row in selection}
    observed: dict[str, dict[str, object]] = {}
    columns = ["material_id", "positions", "cell", "atomic_numbers"]
    for split in ("train", "val", "test"):
        path = data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet"
        table = pds.dataset(path, format="parquet").to_table(
            filter=pds.field("material_id").isin(requested), columns=columns
        )
        for row in table.to_pylist():
            material_id = str(row["material_id"])
            if material_id in observed:
                raise ValueError("Alex material ID occurs in more than one source split")
            row["source_split_observed"] = split
            observed[material_id] = row
    if set(observed) != requested:
        raise ValueError("not every held-out O1 material joined to Alex")
    return observed


def _serialize(value: OccupationalParentOccurrence) -> dict[str, object]:
    projection = value.projection
    pattern = value.occupational_pattern
    return {
        "embedding_key": value.embedding_key,
        "kind": "t" if value.cell_index == 1 else "k",
        "parent_space_group": value.parent_space_group,
        "child_space_group": value.child_space_group,
        "cell_index": value.cell_index,
        "full_action_order": value.full_action_order,
        "parent_site_count": value.parent_site_count,
        "child_site_count": int(value.child_atomic_numbers.size),
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


def _candidate_key(value: OccupationalParentOccurrence) -> tuple[str, int, int]:
    return value.embedding_key, value.parent_space_group, value.cell_index


def _evaluate_material(selected: dict[str, object]) -> dict[str, object]:
    material_id = str(selected["material_id"])
    started = perf_counter()
    row: dict[str, object] = {
        "material_id": material_id,
        "gaugeflow_split": str(selected["gaugeflow_split"]),
        "source_split": str(selected["source_split"]),
        "source_joined": material_id in _WORKER_SOURCE,
        "v1_candidate_count": int(selected["candidate_count"]),
        "child_space_group": int(selected["space_group_number"]),
        "primitive_sites": int(selected["primitive_sites"]),
        "processing_failure": False,
        "failure_reason": "",
        "candidate_edges_evaluated": 0,
        "forward_reverse_agrees": False,
        "candidates": [],
    }
    try:
        raw = _WORKER_SOURCE[material_id]
        if str(raw["source_split_observed"]) != str(selected["source_split"]):
            raise ValueError("Alex source split does not match frozen v1 selection")
        child = standardize_child_to_e0_setting(
            np.asarray(raw["cell"], dtype=np.float64),
            np.asarray(raw["positions"], dtype=np.float64),
            np.asarray(raw["atomic_numbers"], dtype=np.int64),
            expected_space_group=int(selected["space_group_number"]),
            expected_primitive_sites=int(selected["primitive_sites"]),
            symprec=float(_WORKER_SETTINGS["child_symprec_angstrom"]),
            angle_tolerance=float(_WORKER_SETTINGS["angle_tolerance_degree"]),
        )
        records_t = _WORKER_T.get(child.space_group, [])
        records_k = _WORKER_K.get(child.space_group, [])
        row["candidate_edges_evaluated"] = len(records_t) + len(records_k)
        common = {
            "maximum_source_displacement_angstrom": float(
                _WORKER_SETTINGS["maximum_source_displacement_angstrom"]
            ),
            "maximum_source_hencky_norm": float(
                _WORKER_SETTINGS["source_hencky_norm_max"]
            ),
            "angle_tolerance": float(_WORKER_SETTINGS["angle_tolerance_degree"]),
        }
        forward = (
            *search_occupational_maximal_t_parents(child, records_t, **common),
            *search_occupational_maximal_k_parents(child, records_k, **common),
        )
        reverse = (
            *search_occupational_maximal_k_parents(
                child, tuple(reversed(records_k)), **common
            ),
            *search_occupational_maximal_t_parents(
                child, tuple(reversed(records_t)), **common
            ),
        )
        row["forward_reverse_agrees"] = {
            _candidate_key(value) for value in forward
        } == {_candidate_key(value) for value in reverse}
        row["candidates"] = [
            _serialize(value) for value in sorted(forward, key=_candidate_key)
        ]
    except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        row["processing_failure"] = True
        row["failure_reason"] = f"{type(error).__name__}: {error}"
    row["runtime_seconds"] = perf_counter() - started
    return row


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(
    config: dict[str, Any],
    results: list[dict[str, object]],
    v1_positive: list[dict[str, object]],
    o0_results: list[dict[str, object]],
    elapsed: float,
) -> tuple[dict[str, object], dict[str, bool]]:
    occurrences = [
        (str(row["material_id"]), candidate)
        for row in results
        for candidate in row["candidates"]
    ]
    o0_occurrences = [
        (str(row["material_id"]), candidate)
        for row in o0_results
        for candidate in row["candidates"]
    ]
    selected = len(results)
    candidate_materials = sum(bool(row["candidates"]) for row in results)
    occupational_materials = sum(
        any(bool(value["occupationally_nontrivial"]) for value in row["candidates"])
        for row in results
    )
    o0_candidate_materials = sum(bool(row["candidates"]) for row in o0_results)
    aggregate_materials = len(v1_positive) + o0_candidate_materials + candidate_materials
    clean_size = int(config["selection"]["clean_universe_size"])
    finite_values = [
        float(candidate[key])
        for _, candidate in occurrences
        for key in (
            "projection_source_max_displacement_angstrom",
            "projection_source_rms_displacement_angstrom",
            "projection_source_hencky_norm",
            "projected_group_max_error_angstrom",
        )
    ]
    canonical_paths = {
        (material_id, str(candidate["embedding_key"]))
        for material_id, candidate in occurrences
    }
    parent_groups = {
        int(row["parent_space_group"])
        for row in v1_positive
        if row["parent_space_group"] is not None
    } | {
        int(candidate["parent_space_group"])
        for _, candidate in (*o0_occurrences, *occurrences)
    }
    child_groups = {
        int(row["space_group_number"]) for row in v1_positive
    } | {
        int(candidate["child_space_group"])
        for _, candidate in (*o0_occurrences, *occurrences)
    }
    stabilizers_are_subgroups = [
        len(set(map(int, candidate["occupational_stabilizer_indices"])))
        == int(candidate["occupational_stabilizer_order"])
        and all(
            0 <= int(index) < int(candidate["full_action_order"])
            for index in candidate["occupational_stabilizer_indices"]
        )
        for _, candidate in occurrences
    ]
    values: dict[str, object] = {
        "selected_rows": selected,
        "clean_partition_union_size": clean_size,
        "clean_partition_disjoint_fraction": 1.0,
        "o0_material_id_disjoint_fraction": 1.0,
        "source_join_fraction": _fraction(
            sum(bool(row["source_joined"]) for row in results), selected
        ),
        "v1_zero_candidate_fraction": _fraction(
            sum(int(row["v1_candidate_count"]) == 0 for row in results), selected
        ),
        "candidate_edges_evaluated": sum(
            int(row["candidate_edges_evaluated"]) for row in results
        ),
        "path_quarantined_edges": 0,
        "eligible_edges_evaluated": sum(
            int(row["candidate_edges_evaluated"]) for row in results
        ),
        "processing_failures": sum(bool(row["processing_failure"]) for row in results),
        "nonfinite_results": sum(not np.isfinite(value) for value in finite_values),
        "new_candidate_materials": candidate_materials,
        "new_candidate_material_fraction": _fraction(candidate_materials, selected),
        "occupationally_nontrivial_materials": occupational_materials,
        "certified_embedding_occurrences": len(occurrences),
        "canonical_material_paths": len(canonical_paths),
        "canonical_material_path_fraction_of_edges": _fraction(
            len(canonical_paths),
            sum(int(row["candidate_edges_evaluated"]) for row in results),
        ),
        "aggregate_qualified_materials": aggregate_materials,
        "aggregate_qualified_material_fraction": _fraction(
            aggregate_materials, clean_size
        ),
        "canonical_material_path_uniqueness_fraction": _fraction(
            len(canonical_paths), len(occurrences)
        ),
        "forward_reverse_embedding_set_agreement_fraction": _fraction(
            sum(bool(row["forward_reverse_agrees"]) for row in results), selected
        ),
        "exact_coloring_reconstruction_fraction": _fraction(
            sum(bool(value["exact_coloring_reconstruction"]) for _, value in occurrences),
            len(occurrences),
        ),
        "occupational_stabilizer_subgroup_fraction": _fraction(
            sum(stabilizers_are_subgroups), len(occurrences)
        ),
        "occupational_stabilizer_order_matches_child_fraction": _fraction(
            sum(bool(value["stabilizer_order_matches_child"]) for _, value in occurrences),
            len(occurrences),
        ),
        "strict_geometry_parent_reidentification_fraction": (
            1.0 if occurrences else 0.0
        ),
        "terminal_integer_element_fraction": _fraction(
            sum(
                all(1 <= int(number) <= 118 for number in value["child_atomic_numbers"])
                for _, value in occurrences
            ),
            len(occurrences),
        ),
        "partial_occupancy_count": 0,
        "projected_group_max_error_angstrom": max(
            (float(value["projected_group_max_error_angstrom"]) for _, value in occurrences),
            default=0.0,
        ),
        "source_max_displacement_angstrom": max(
            (
                float(value["projection_source_max_displacement_angstrom"])
                for _, value in occurrences
            ),
            default=0.0,
        ),
        "source_hencky_norm_max": max(
            (float(value["projection_source_hencky_norm"]) for _, value in occurrences),
            default=0.0,
        ),
        "aggregate_distinct_parent_space_groups": len(parent_groups),
        "aggregate_distinct_child_space_groups": len(child_groups),
        "runtime_seconds_wall": elapsed,
        "runtime_seconds_material_sum": sum(
            float(row["runtime_seconds"]) for row in results
        ),
        "runtime_seconds_material_p95": float(
            np.quantile([float(row["runtime_seconds"]) for row in results], 0.95)
        ),
    }
    thresholds = config["thresholds"]
    exact_keys = (
        "selected_rows",
        "clean_partition_union_size",
        "clean_partition_disjoint_fraction",
        "o0_material_id_disjoint_fraction",
        "source_join_fraction",
        "v1_zero_candidate_fraction",
        "candidate_edges_evaluated",
        "path_quarantined_edges",
        "eligible_edges_evaluated",
        "processing_failures",
        "nonfinite_results",
        "canonical_material_path_uniqueness_fraction",
        "forward_reverse_embedding_set_agreement_fraction",
        "exact_coloring_reconstruction_fraction",
        "occupational_stabilizer_subgroup_fraction",
        "occupational_stabilizer_order_matches_child_fraction",
        "strict_geometry_parent_reidentification_fraction",
        "terminal_integer_element_fraction",
        "partial_occupancy_count",
    )
    checks = {key: values[key] == thresholds[key] for key in exact_keys}
    for metric, threshold in (
        ("new_candidate_materials", "new_candidate_materials_min"),
        ("new_candidate_material_fraction", "new_candidate_material_fraction_min"),
        (
            "occupationally_nontrivial_materials",
            "occupationally_nontrivial_materials_min",
        ),
        ("aggregate_qualified_materials", "aggregate_qualified_materials_min"),
        (
            "aggregate_qualified_material_fraction",
            "aggregate_qualified_material_fraction_min",
        ),
        (
            "aggregate_distinct_parent_space_groups",
            "aggregate_distinct_parent_space_groups_min",
        ),
        (
            "aggregate_distinct_child_space_groups",
            "aggregate_distinct_child_space_groups_min",
        ),
    ):
        checks[metric] = values[metric] >= thresholds[threshold]
    for metric in (
        "projected_group_max_error_angstrom",
        "source_max_displacement_angstrom",
        "source_hencky_norm_max",
    ):
        checks[metric] = values[metric] <= thresholds[metric]
    return values, checks


def run(
    config: dict[str, Any], data_root: Path, repo_root: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    global _WORKER_K, _WORKER_SETTINGS, _WORKER_SOURCE, _WORKER_T
    o0_source, cleaning, _ = _validate_dependencies(config, data_root, repo_root)
    selection, v1_positive = _selection(config, o0_source, cleaning, data_root)
    source = _join_source(selection, data_root)
    records_path = (
        data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_records.json.gz"
    )
    by_child_t: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_child_k: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in load_gzip_json(records_path):
        kind = str(record["kind"])
        if kind == "t" and int(record["cell_index"]) == 1:
            by_child_t[int(record["child_space_group"])].append(record)
        elif kind == "k" and 2 <= int(record["cell_index"]) <= 4:
            by_child_k[int(record["child_space_group"])].append(record)
    _WORKER_SOURCE = source
    _WORKER_T = dict(by_child_t)
    _WORKER_K = dict(by_child_k)
    _WORKER_SETTINGS = config["setting_and_search"]
    worker_count = min(4, os.cpu_count() or 1)
    started = perf_counter()
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        results = []
        for row in executor.map(_evaluate_material, selection, chunksize=1):
            results.append(row)
            print(
                row["material_id"],
                f"sg={row['child_space_group']}",
                f"edges={row['candidate_edges_evaluated']}",
                f"candidates={len(row['candidates'])}",
                f"failure={row['processing_failure']}",
                f"seconds={row['runtime_seconds']:.4f}",
                flush=True,
            )
    elapsed = perf_counter() - started
    o0_results = load_gzip_json(
        data_root / "processed/gaugeflow_h0_v7/occupational_order_o0_v2/results.json.gz"
    )
    metrics, checks = _metrics(
        config, results, v1_positive, o0_results, elapsed
    )
    qualified = all(checks.values())
    return results, {
        "protocol": config["protocol"],
        "qualified": qualified,
        "metrics": metrics,
        "checks": checks,
        "worker_processes": worker_count,
        "decision": (
            "H0-E-v4-and-H0_qualified_only_separately_frozen_H1a_may_start"
            if qualified
            else "H0-E-v4-O1-v1_failed_stop_before_H1a_H1b_and_H2-H6"
        ),
        "scope": config["gate_role"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    arguments = parser.parse_args()
    repo_root = arguments.config.resolve().parents[2]
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    implementation_commit = _clean_git_commit(repo_root)
    results, manifest = run(config, arguments.data_root, repo_root)
    results_path = arguments.data_root / config["required_outputs"]["results"]
    manifest_path = arguments.data_root / config["required_outputs"]["manifest"]
    write_deterministic_gzip_json(results_path, results)
    manifest["implementation_commit"] = implementation_commit
    manifest["protocol_config_sha256"] = sha256_file(arguments.config)
    manifest["results_sha256"] = sha256_file(results_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    raise SystemExit(0 if bool(manifest["qualified"]) else 2)


if __name__ == "__main__":
    main()
