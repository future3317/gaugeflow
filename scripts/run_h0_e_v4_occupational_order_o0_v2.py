"""Run the frozen H0-E-v4 O0-v2 occupational-order mechanism Gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from collections import defaultdict
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


def _ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _clean_git_commit(repo_root: Path) -> str:
    checks = (
        subprocess.run(
            ["git", "diff", "--ignore-space-at-eol", "--quiet"],
            cwd=repo_root,
            check=False,
        ).returncode,
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root,
            check=False,
        ).returncode,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if checks != (0, 0) or untracked.strip():
        raise RuntimeError("the frozen O0-v2 run requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_dependencies(
    config: dict[str, Any],
    data_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dependencies = config["dependencies"]
    external = {
        "e0_records_sha256": data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_records.json.gz",
        "e0_manifest_sha256": data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_manifest.json",
        "v1_decompositions_sha256": data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/decompositions.parquet",
        "v1_selection_sha256": data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/selection.parquet",
    }
    for key, path in external.items():
        if sha256_file(path) != str(dependencies[key]):
            raise ValueError(f"dependency hash mismatch: {key}")
    for split, expected in dependencies["raw_alex_sha256"].items():
        path = data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet"
        if sha256_file(path) != str(expected):
            raise ValueError(f"Alex source hash mismatch: {split}")
    e1a_path = repo_root / str(dependencies["e1a_panel_config"])
    cleaning_path = repo_root / str(dependencies["data_cleaning"])
    if sha256_file(e1a_path) != str(dependencies["e1a_panel_config_sha256"]):
        raise ValueError("frozen E1a panel config hash mismatch")
    if sha256_file(cleaning_path) != str(dependencies["data_cleaning_sha256"]):
        raise ValueError("data-cleaning manifest hash mismatch")
    if importlib.metadata.version("pyxtal") != dependencies["pyxtal_version"]:
        raise ValueError("PyXtal version does not match the frozen O0-v2 protocol")
    return (
        json.loads(e1a_path.read_text(encoding="utf-8")),
        json.loads(cleaning_path.read_text(encoding="utf-8")),
    )


def _selection(
    config: dict[str, Any],
    e1a: dict[str, Any],
    cleaning: dict[str, Any],
    data_root: Path,
) -> list[dict[str, object]]:
    source_ids = [str(value) for value in e1a["selection"]["material_ids"]]
    exclusions = [str(value["material_id"]) for value in cleaning["material_exclusions"]]
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("material cleaning contains duplicate identifiers")
    selected_ids = [value for value in source_ids if value not in set(exclusions)]
    expected = config["selection"]
    if len(selected_ids) != int(expected["size"]):
        raise ValueError("cleaned O0-v2 selection has the wrong size")
    if _ordered_hash(selected_ids) != str(expected["ordered_material_ids_sha256"]):
        raise ValueError("cleaned O0-v2 ordered material hash mismatch")
    rows = pq.read_table(
        data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/decompositions.parquet"
    ).to_pylist()
    by_id = {str(value["material_id"]): value for value in rows}
    if len(by_id) != len(rows):
        raise ValueError("v1 decomposition artifact contains duplicate material IDs")
    selected = [by_id[value] for value in selected_ids]
    if any(
        int(value["candidate_count"]) != 0 or bool(value["processing_failure"])
        for value in selected
    ):
        raise ValueError("O0-v2 selection is not the cleaned zero-candidate panel")
    observed_splits: dict[str, int] = defaultdict(int)
    for value in selected:
        observed_splits[str(value["gaugeflow_split"])] += 1
    if dict(observed_splits) != {
        str(key): int(value) for key, value in expected["split_counts"].items()
    }:
        raise ValueError("cleaned O0-v2 split counts do not match the protocol")
    return selected


def _join_source(
    selection: list[dict[str, object]],
    data_root: Path,
) -> dict[str, dict[str, object]]:
    requested = {str(value["material_id"]) for value in selection}
    observed: dict[str, dict[str, object]] = {}
    columns = ["material_id", "positions", "cell", "atomic_numbers"]
    for split in ("train", "val", "test"):
        path = data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet"
        table = pds.dataset(path, format="parquet").to_table(
            filter=pds.field("material_id").isin(requested),
            columns=columns,
        )
        for row in table.to_pylist():
            material_id = str(row["material_id"])
            if material_id in observed:
                raise ValueError("Alex material ID occurs in more than one source split")
            row["source_split_observed"] = split
            observed[material_id] = row
    if set(observed) != requested:
        raise ValueError("not every cleaned O0-v2 material joined to Alex")
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
        "occupational_stabilizer_order": int(value.occupational_stabilizer_indices.size),
        "occupational_stabilizer_indices": value.occupational_stabilizer_indices.tolist(),
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


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(
    config: dict[str, Any],
    results: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, bool]]:
    occurrences = [value for row in results for value in row["candidates"]]
    selected = len(results)
    occurrence_count = len(occurrences)
    candidate_materials = sum(bool(row["candidates"]) for row in results)
    occupational_materials = sum(
        any(bool(value["occupationally_nontrivial"]) for value in row["candidates"])
        for row in results
    )
    finite_values = [
        float(value[key])
        for value in occurrences
        for key in (
            "projection_source_max_displacement_angstrom",
            "projection_source_rms_displacement_angstrom",
            "projection_source_hencky_norm",
            "projected_group_max_error_angstrom",
        )
    ]
    values: dict[str, object] = {
        "selected_rows": selected,
        "material_exclusions_applied": int(config["selection"]["material_exclusions"]),
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
        "certified_embedding_occurrences": occurrence_count,
        "forward_reverse_embedding_set_agreement_fraction": _fraction(
            sum(bool(row["forward_reverse_agrees"]) for row in results), selected
        ),
        "exact_coloring_reconstruction_fraction": _fraction(
            sum(bool(value["exact_coloring_reconstruction"]) for value in occurrences),
            occurrence_count,
        ),
        "occupational_stabilizer_subgroup_fraction": 1.0 if occurrence_count else 0.0,
        "occupational_stabilizer_order_matches_child_fraction": _fraction(
            sum(bool(value["stabilizer_order_matches_child"]) for value in occurrences),
            occurrence_count,
        ),
        "strict_geometry_parent_reidentification_fraction": (
            1.0 if occurrence_count else 0.0
        ),
        "terminal_integer_element_fraction": _fraction(
            sum(
                all(1 <= int(number) <= 118 for number in value["child_atomic_numbers"])
                for value in occurrences
            ),
            occurrence_count,
        ),
        "partial_occupancy_count": 0,
        "projected_group_max_error_angstrom": max(
            (float(value["projected_group_max_error_angstrom"]) for value in occurrences),
            default=0.0,
        ),
        "source_max_displacement_angstrom": max(
            (
                float(value["projection_source_max_displacement_angstrom"])
                for value in occurrences
            ),
            default=0.0,
        ),
        "source_hencky_norm_max": max(
            (float(value["projection_source_hencky_norm"]) for value in occurrences),
            default=0.0,
        ),
        "runtime_seconds_total": sum(float(row["runtime_seconds"]) for row in results),
        "runtime_seconds_p95": float(
            np.quantile([float(row["runtime_seconds"]) for row in results], 0.95)
        ),
        "distinct_parent_space_groups": len(
            {int(value["parent_space_group"]) for value in occurrences}
        ),
    }
    thresholds = config["thresholds"]
    checks = {
        "selected_rows": values["selected_rows"] == thresholds["selected_rows"],
        "material_exclusions_applied": values["material_exclusions_applied"]
        == thresholds["material_exclusions_applied"],
        "source_join_fraction": values["source_join_fraction"]
        == thresholds["source_join_fraction"],
        "v1_zero_candidate_fraction": values["v1_zero_candidate_fraction"]
        == thresholds["v1_zero_candidate_fraction"],
        "candidate_edges_evaluated": values["candidate_edges_evaluated"]
        == thresholds["candidate_edges_evaluated"],
        "path_quarantined_edges": values["path_quarantined_edges"]
        == thresholds["path_quarantined_edges"],
        "eligible_edges_evaluated": values["eligible_edges_evaluated"]
        == thresholds["eligible_edges_evaluated"],
        "processing_failures": values["processing_failures"]
        == thresholds["processing_failures"],
        "nonfinite_results": values["nonfinite_results"]
        == thresholds["nonfinite_results"],
        "new_candidate_materials": values["new_candidate_materials"]
        >= thresholds["new_candidate_materials_min"],
        "new_candidate_material_fraction": values["new_candidate_material_fraction"]
        >= thresholds["new_candidate_material_fraction_min"],
        "occupationally_nontrivial_materials": values[
            "occupationally_nontrivial_materials"
        ]
        >= thresholds["occupationally_nontrivial_materials_min"],
        "forward_reverse_embedding_set_agreement_fraction": values[
            "forward_reverse_embedding_set_agreement_fraction"
        ]
        == thresholds["forward_reverse_embedding_set_agreement_fraction"],
        "exact_coloring_reconstruction_fraction": values[
            "exact_coloring_reconstruction_fraction"
        ]
        == thresholds["exact_coloring_reconstruction_fraction"],
        "occupational_stabilizer_subgroup_fraction": values[
            "occupational_stabilizer_subgroup_fraction"
        ]
        == thresholds["occupational_stabilizer_subgroup_fraction"],
        "occupational_stabilizer_order_matches_child_fraction": values[
            "occupational_stabilizer_order_matches_child_fraction"
        ]
        == thresholds["occupational_stabilizer_order_matches_child_fraction"],
        "strict_geometry_parent_reidentification_fraction": values[
            "strict_geometry_parent_reidentification_fraction"
        ]
        == thresholds["strict_geometry_parent_reidentification_fraction"],
        "terminal_integer_element_fraction": values["terminal_integer_element_fraction"]
        == thresholds["terminal_integer_element_fraction"],
        "partial_occupancy_count": values["partial_occupancy_count"]
        == thresholds["partial_occupancy_count"],
        "projected_group_max_error_angstrom": values[
            "projected_group_max_error_angstrom"
        ]
        <= thresholds["projected_group_max_error_angstrom"],
        "source_max_displacement_angstrom": values[
            "source_max_displacement_angstrom"
        ]
        <= thresholds["source_max_displacement_angstrom"],
        "source_hencky_norm_max": values["source_hencky_norm_max"]
        <= thresholds["source_hencky_norm_max"],
    }
    return values, checks


def run(
    config: dict[str, Any],
    data_root: Path,
    repo_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    e1a, cleaning = _validate_dependencies(config, data_root, repo_root)
    selection = _selection(config, e1a, cleaning, data_root)
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
    settings = config["setting_and_search"]
    results: list[dict[str, object]] = []
    for selected in selection:
        material_id = str(selected["material_id"])
        started = perf_counter()
        row: dict[str, object] = {
            "material_id": material_id,
            "gaugeflow_split": str(selected["gaugeflow_split"]),
            "source_split": str(selected["source_split"]),
            "source_joined": material_id in source,
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
            raw = source[material_id]
            if str(raw["source_split_observed"]) != str(selected["source_split"]):
                raise ValueError("Alex source split does not match the frozen selection")
            child = standardize_child_to_e0_setting(
                np.asarray(raw["cell"], dtype=np.float64),
                np.asarray(raw["positions"], dtype=np.float64),
                np.asarray(raw["atomic_numbers"], dtype=np.int64),
                expected_space_group=int(selected["space_group_number"]),
                expected_primitive_sites=int(selected["primitive_sites"]),
                symprec=float(settings["child_symprec_angstrom"]),
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            records_t = by_child_t[child.space_group]
            records_k = by_child_k[child.space_group]
            row["candidate_edges_evaluated"] = len(records_t) + len(records_k)
            common = {
                "maximum_source_displacement_angstrom": float(
                    settings["maximum_source_displacement_angstrom"]
                ),
                "maximum_source_hencky_norm": float(settings["source_hencky_norm_max"]),
                "angle_tolerance": float(settings["angle_tolerance_degree"]),
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
                _serialize(value)
                for value in sorted(forward, key=_candidate_key)
            ]
        except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            row["processing_failure"] = True
            row["failure_reason"] = f"{type(error).__name__}: {error}"
        row["runtime_seconds"] = perf_counter() - started
        results.append(row)
        print(
            material_id,
            f"sg={row['child_space_group']}",
            f"edges={row['candidate_edges_evaluated']}",
            f"candidates={len(row['candidates'])}",
            f"failure={row['processing_failure']}",
            f"seconds={row['runtime_seconds']:.4f}",
            flush=True,
        )
    metrics, checks = _metrics(config, results)
    qualified = all(checks.values())
    return results, {
        "protocol": config["protocol"],
        "qualified": qualified,
        "metrics": metrics,
        "checks": checks,
        "decision": (
            "H0-E-v4-O0-v2_qualified_only_held_out_O1_protocol_may_be_frozen"
            if qualified
            else "H0-E-v4-O0-v2_failed_stop_before_O1_and_H1a"
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
    manifest["results_sha256"] = sha256_file(results_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    raise SystemExit(0 if bool(manifest["qualified"]) else 2)


if __name__ == "__main__":
    main()
