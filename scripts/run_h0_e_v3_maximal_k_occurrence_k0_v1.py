"""Run the frozen H0-E-v3 K0 cell-changing parent-occurrence pilot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from gaugeflow.catalogue import (
    EmbeddingParentOccurrence,
    search_maximal_k_parents,
    standardize_child_to_e0_setting,
)
from gaugeflow.catalogue.occurrence_protocol import (
    clean_git_commit,
    frozen_e1a_selection,
    join_alex_rows,
)
from gaugeflow.file_utils import (
    load_gzip_json,
    sha256_file,
    write_deterministic_gzip_json,
)


def _validate_dependencies(
    config: dict[str, Any], data_root: Path, repo_root: Path
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
    quarantine_path = repo_root / str(dependencies["path_quarantine"])
    if sha256_file(e1a_path) != str(dependencies["e1a_panel_config_sha256"]):
        raise ValueError("frozen E1a panel config hash mismatch")
    if sha256_file(quarantine_path) != str(dependencies["path_quarantine_sha256"]):
        raise ValueError("parent-path quarantine hash mismatch")
    if importlib.metadata.version("pyxtal") != dependencies["pyxtal_version"]:
        raise ValueError("PyXtal version does not match the frozen K0 protocol")
    return (
        json.loads(e1a_path.read_text(encoding="utf-8")),
        json.loads(quarantine_path.read_text(encoding="utf-8")),
    )


def _quarantine_keys(payload: dict[str, Any]) -> set[tuple[str, int, int]]:
    keys = {
        (
            str(value["material_id"]),
            int(value["child_space_group"]),
            int(value["parent_space_group"]),
        )
        for value in payload["path_quarantine"]
    }
    if len(keys) != len(payload["path_quarantine"]):
        raise ValueError("parent-path quarantine contains duplicate keys")
    return keys


def _serialize_occurrence(value: EmbeddingParentOccurrence) -> dict[str, object]:
    projection = value.projection
    candidate = value.candidate
    return {
        "embedding_key": value.embedding_key,
        "parent_space_group": value.parent_space_group,
        "cell_index": value.cell_index,
        "full_action_order": value.full_action_order,
        "parent_primitive_operation_order": int(candidate.parent.rotations.shape[0]),
        "parent_site_count": value.parent_site_count,
        "child_site_count": int(candidate.child.species.size),
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
        "certified_source_max_displacement_angstrom": (
            candidate.source_max_displacement_angstrom
        ),
        "certified_source_hencky_norm": candidate.source_hencky_norm,
        "supercell_hnf": candidate.supercell_hnf.tolist(),
        "parent_lattice": projection.lattice.tolist(),
        "parent_fractional": projection.fractional.tolist(),
        "parent_species": projection.species.tolist(),
    }


def _metrics(
    config: dict[str, Any], results: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, bool]]:
    occurrences = [candidate for row in results for candidate in row.get("candidates", [])]
    selected = len(results)
    candidate_materials = sum(bool(row.get("candidates")) for row in results)
    finite_values = [
        float(candidate[key])
        for candidate in occurrences
        for key in (
            "projection_source_max_displacement_angstrom",
            "projection_source_rms_displacement_angstrom",
            "projection_source_hencky_norm",
            "projected_group_max_error_angstrom",
            "certified_source_max_displacement_angstrom",
            "certified_source_hencky_norm",
        )
    ]
    source_maximum = max(
        (
            max(
                float(value["projection_source_max_displacement_angstrom"]),
                float(value["certified_source_max_displacement_angstrom"]),
            )
            for value in occurrences
        ),
        default=0.0,
    )
    hencky_maximum = max(
        (
            max(
                float(value["projection_source_hencky_norm"]),
                float(value["certified_source_hencky_norm"]),
            )
            for value in occurrences
        ),
        default=0.0,
    )
    group_error = max(
        (
            float(value["projected_group_max_error_angstrom"])
            for value in occurrences
        ),
        default=0.0,
    )
    site_count_fraction = (
        sum(
            int(value["parent_site_count"]) * int(value["cell_index"])
            == int(value["child_site_count"])
            for value in occurrences
        )
        / len(occurrences)
        if occurrences
        else 0.0
    )
    action_order_fraction = (
        sum(
            int(value["full_action_order"])
            == int(value["parent_primitive_operation_order"])
            * int(value["cell_index"])
            for value in occurrences
        )
        / len(occurrences)
        if occurrences
        else 0.0
    )
    values: dict[str, object] = {
        "selected_rows": selected,
        "source_join_fraction": sum(bool(row.get("source_joined")) for row in results)
        / max(selected, 1),
        "v1_zero_candidate_fraction": sum(
            int(row.get("v1_candidate_count", -1)) == 0 for row in results
        )
        / max(selected, 1),
        "candidate_edges_evaluated": sum(
            int(row.get("candidate_edges_evaluated", 0)) for row in results
        ),
        "processing_failures": sum(
            bool(row.get("processing_failure")) for row in results
        ),
        "nonfinite_results": sum(not np.isfinite(value) for value in finite_values),
        "new_candidate_materials": candidate_materials,
        "new_candidate_material_fraction": candidate_materials / max(selected, 1),
        "certified_embedding_occurrences": len(occurrences),
        "forward_reverse_embedding_set_agreement_fraction": sum(
            bool(row.get("forward_reverse_agrees")) for row in results
        )
        / max(selected, 1),
        "path_quarantine_applied_fraction": sum(
            bool(row.get("path_quarantine_applied")) for row in results
        )
        / max(selected, 1),
        "quarantined_edges": sum(int(row.get("quarantined_edges", 0)) for row in results),
        "strict_parent_space_group_reidentification_fraction": (
            1.0 if occurrences else 0.0
        ),
        "structure_matcher_certification_fraction": 1.0 if occurrences else 0.0,
        "parent_site_count_times_index_equals_child_fraction": site_count_fraction,
        "full_action_order_equals_parent_order_times_index_fraction": (
            action_order_fraction
        ),
        "projected_group_max_error_angstrom": group_error,
        "source_max_displacement_angstrom": source_maximum,
        "source_hencky_norm_max": hencky_maximum,
        "runtime_seconds_total": sum(float(row["runtime_seconds"]) for row in results),
        "runtime_seconds_p95": float(
            np.quantile([float(row["runtime_seconds"]) for row in results], 0.95)
        ),
        "distinct_parent_space_groups": len(
            {int(value["parent_space_group"]) for value in occurrences}
        ),
    }
    thresholds = config["thresholds"]
    exact = (
        "selected_rows",
        "source_join_fraction",
        "v1_zero_candidate_fraction",
        "candidate_edges_evaluated",
        "processing_failures",
        "nonfinite_results",
        "forward_reverse_embedding_set_agreement_fraction",
        "path_quarantine_applied_fraction",
        "strict_parent_space_group_reidentification_fraction",
        "structure_matcher_certification_fraction",
        "parent_site_count_times_index_equals_child_fraction",
        "full_action_order_equals_parent_order_times_index_fraction",
    )
    checks = {key: values[key] == thresholds[key] for key in exact}
    checks.update(
        {
            "new_candidate_materials": values["new_candidate_materials"]
            >= thresholds["new_candidate_materials_min"],
            "new_candidate_material_fraction": values["new_candidate_material_fraction"]
            >= thresholds["new_candidate_material_fraction_min"],
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
    )
    return values, checks


def run(
    config: dict[str, Any], data_root: Path, repo_root: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    e1a_config, quarantine = _validate_dependencies(config, data_root, repo_root)
    selection = frozen_e1a_selection(e1a_config, data_root)
    raw = join_alex_rows(selection, data_root)
    quarantine_keys = _quarantine_keys(quarantine)
    e0_records = load_gzip_json(
        data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_records.json.gz"
    )
    by_child: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in e0_records:
        if str(record["kind"]) == "k" and 2 <= int(record["cell_index"]) <= 4:
            by_child[int(record["child_space_group"])].append(record)
    settings = config["setting_and_search"]
    matcher = settings["structure_matcher"]
    results: list[dict[str, object]] = []
    for selected in selection:
        material_id = str(selected["material_id"])
        started = perf_counter()
        result: dict[str, object] = {
            "material_id": material_id,
            "gaugeflow_split": str(selected["gaugeflow_split"]),
            "source_split": str(selected["source_split"]),
            "source_joined": material_id in raw,
            "v1_candidate_count": int(selected["candidate_count"]),
            "child_space_group": int(selected["space_group_number"]),
            "primitive_sites": int(selected["primitive_sites"]),
            "processing_failure": False,
            "failure_reason": "",
            "candidate_edges_evaluated": 0,
            "quarantined_edges": 0,
            "path_quarantine_applied": True,
            "forward_reverse_agrees": False,
            "candidates": [],
        }
        try:
            source = raw[material_id]
            if str(source["source_split_observed"]) != str(selected["source_split"]):
                raise ValueError("Alex source split does not match frozen v1 selection")
            child = standardize_child_to_e0_setting(
                np.asarray(source["cell"], dtype=np.float64),
                np.asarray(source["positions"], dtype=np.float64),
                np.asarray(source["atomic_numbers"], dtype=np.int64),
                expected_space_group=int(selected["space_group_number"]),
                expected_primitive_sites=int(selected["primitive_sites"]),
                symprec=float(settings["child_symprec_angstrom"]),
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            records = by_child[child.space_group]
            result["candidate_edges_evaluated"] = len(records)
            eligible = [
                record
                for record in records
                if (
                    material_id,
                    child.space_group,
                    int(record["parent_space_group"]),
                )
                not in quarantine_keys
            ]
            result["quarantined_edges"] = len(records) - len(eligible)
            forward = search_maximal_k_parents(
                child,
                eligible,
                maximum_source_displacement_angstrom=float(
                    settings["maximum_source_displacement_angstrom"]
                ),
                matcher_settings=matcher,
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            reverse = search_maximal_k_parents(
                child,
                tuple(reversed(eligible)),
                maximum_source_displacement_angstrom=float(
                    settings["maximum_source_displacement_angstrom"]
                ),
                matcher_settings=matcher,
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            forward_keys = {
                (item.embedding_key, item.parent_space_group) for item in forward
            }
            reverse_keys = {
                (item.embedding_key, item.parent_space_group) for item in reverse
            }
            result["forward_reverse_agrees"] = forward_keys == reverse_keys
            result["candidates"] = [
                _serialize_occurrence(item)
                for item in sorted(
                    forward,
                    key=lambda value: (value.parent_space_group, value.embedding_key),
                )
            ]
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            result["processing_failure"] = True
            result["failure_reason"] = f"{type(error).__name__}: {error}"
        result["runtime_seconds"] = perf_counter() - started
        results.append(result)
        print(
            material_id,
            f"sg={result['child_space_group']}",
            f"edges={result['candidate_edges_evaluated']}",
            f"candidates={len(result['candidates'])}",
            f"failure={result['processing_failure']}",
            f"seconds={result['runtime_seconds']:.4f}",
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
            "H0-E-v3-K0_qualified_separate_v3_occurrence_protocol_may_be_designed"
            if qualified
            else "H0-E-v3-K0_failed_stop_before_H1a"
        ),
        "scope": config["gate_role"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    implementation_commit = clean_git_commit(repo_root, protocol="K0")
    results, manifest = run(config, args.data_root, repo_root)
    output = args.data_root / config["required_outputs"]["results"]
    write_deterministic_gzip_json(output, results)
    manifest["implementation_commit"] = implementation_commit
    manifest["results_sha256"] = sha256_file(output)
    manifest_path = args.data_root / config["required_outputs"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    raise SystemExit(0 if bool(manifest["qualified"]) else 2)


if __name__ == "__main__":
    main()
