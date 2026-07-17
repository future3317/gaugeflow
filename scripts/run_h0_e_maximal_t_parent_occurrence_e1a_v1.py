"""Run the frozen H0-E-v2 E1a maximal-t parent-occurrence pilot."""

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
    EmbeddingParentOccurrence,
    balanced_selection,
    search_maximal_t_parents,
    standardize_child_to_e0_setting,
)
from gaugeflow.file_utils import (
    load_gzip_json,
    sha256_file,
    write_deterministic_gzip_json,
)


def _ordered_id_hash(values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_commit(repo_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("the frozen E1a run requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_dependencies(config: dict[str, Any], data_root: Path) -> None:
    dependencies = config["dependencies"]
    paths = {
        "e0_records_sha256": data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_records.json.gz",
        "e0_manifest_sha256": data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_manifest.json",
        "v1_decompositions_sha256": data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/decompositions.parquet",
        "v1_selection_sha256": data_root / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/selection.parquet",
    }
    for key, path in paths.items():
        if sha256_file(path) != str(dependencies[key]):
            raise ValueError(f"dependency hash mismatch: {key}")
    for split, expected in dependencies["raw_alex_sha256"].items():
        path = data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet"
        if sha256_file(path) != str(expected):
            raise ValueError(f"Alex source hash mismatch: {split}")
    if importlib.metadata.version("pyxtal") != dependencies["pyxtal_version"]:
        raise ValueError("PyXtal version does not match the frozen E1a protocol")


def _frozen_selection(config: dict[str, Any], data_root: Path) -> list[dict[str, object]]:
    decompositions = pq.read_table(
        data_root / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/decompositions.parquet"
    ).to_pylist()
    eligible = [
        row for row in decompositions if int(row["candidate_count"]) == 0 and not bool(row["processing_failure"])
    ]
    selection = config["selection"]
    reproduced = list(
        balanced_selection(
            eligible,
            split_counts={str(key): int(value) for key, value in selection["split_counts"].items()},
            seed=int(selection["seed"]),
            site_boundaries=selection["site_bins"],
        )
    )
    observed_ids = [str(row["material_id"]) for row in reproduced]
    if observed_ids != list(selection["material_ids"]):
        raise ValueError("frozen E1a material IDs do not reproduce from v1")
    if _ordered_id_hash(observed_ids) != selection["ordered_material_ids_sha256"]:
        raise ValueError("frozen E1a ordered material-ID hash does not match")
    return reproduced


def _join_raw_rows(selection: list[dict[str, object]], data_root: Path) -> dict[str, dict[str, object]]:
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
        raise ValueError("not every frozen E1a material joined to Alex")
    return observed


def _serialize_occurrence(value: EmbeddingParentOccurrence) -> dict[str, object]:
    projection = value.projection
    candidate = value.candidate
    return {
        "embedding_key": value.embedding_key,
        "parent_space_group": value.parent_space_group,
        "projection_source_max_displacement_angstrom": (projection.source_max_displacement_angstrom),
        "projection_source_rms_displacement_angstrom": (projection.source_rms_displacement_angstrom),
        "projection_source_hencky_norm": projection.source_hencky_norm,
        "projected_group_max_error_angstrom": (projection.projected_group_max_error_angstrom),
        "certified_source_max_displacement_angstrom": (candidate.source_max_displacement_angstrom),
        "certified_source_hencky_norm": candidate.source_hencky_norm,
        "parent_lattice": projection.lattice.tolist(),
        "parent_fractional": projection.fractional.tolist(),
        "parent_species": projection.species.tolist(),
    }


def _metrics(config: dict[str, Any], results: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, bool]]:
    occurrences = [candidate for row in results for candidate in row.get("candidates", [])]
    candidate_materials = sum(bool(row.get("candidates")) for row in results)
    selected = len(results)
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
    group_error_maximum = max(
        (float(value["projected_group_max_error_angstrom"]) for value in occurrences),
        default=0.0,
    )
    processing_failures = sum(bool(row.get("processing_failure")) for row in results)
    reverse_agreement = sum(bool(row.get("forward_reverse_agrees")) for row in results) / max(selected, 1)
    values: dict[str, object] = {
        "selected_rows": selected,
        "source_join_fraction": sum(bool(row.get("source_joined")) for row in results) / max(selected, 1),
        "v1_zero_candidate_fraction": sum(int(row.get("v1_candidate_count", -1)) == 0 for row in results)
        / max(selected, 1),
        "processing_failures": processing_failures,
        "nonfinite_results": sum(not np.isfinite(value) for value in finite_values),
        "new_candidate_materials": candidate_materials,
        "new_candidate_material_fraction": candidate_materials / max(selected, 1),
        "certified_embedding_occurrences": len(occurrences),
        "forward_reverse_embedding_set_agreement_fraction": reverse_agreement,
        "strict_parent_space_group_reidentification_fraction": (1.0 if occurrences else 0.0),
        "structure_matcher_certification_fraction": (1.0 if occurrences else 0.0),
        "projected_group_max_error_angstrom": group_error_maximum,
        "source_max_displacement_angstrom": source_maximum,
        "source_hencky_norm_max": hencky_maximum,
        "runtime_seconds_total": sum(float(row["runtime_seconds"]) for row in results),
        "runtime_seconds_p95": float(np.quantile([float(row["runtime_seconds"]) for row in results], 0.95)),
        "distinct_parent_space_groups": len({int(value["parent_space_group"]) for value in occurrences}),
        "distinct_child_space_groups_with_new_parent": len(
            {int(row["child_space_group"]) for row in results if row.get("candidates")}
        ),
    }
    thresholds = config["thresholds"]
    checks = {
        "selected_rows": values["selected_rows"] == thresholds["selected_rows"],
        "source_join_fraction": values["source_join_fraction"] == thresholds["source_join_fraction"],
        "v1_zero_candidate_fraction": values["v1_zero_candidate_fraction"] == thresholds["v1_zero_candidate_fraction"],
        "processing_failures": values["processing_failures"] == thresholds["processing_failures"],
        "nonfinite_results": values["nonfinite_results"] == thresholds["nonfinite_results"],
        "new_candidate_materials": values["new_candidate_materials"] >= thresholds["new_candidate_materials_min"],
        "new_candidate_material_fraction": values["new_candidate_material_fraction"]
        >= thresholds["new_candidate_material_fraction_min"],
        "forward_reverse_embedding_set_agreement_fraction": values["forward_reverse_embedding_set_agreement_fraction"]
        == thresholds["forward_reverse_embedding_set_agreement_fraction"],
        "strict_parent_space_group_reidentification_fraction": values[
            "strict_parent_space_group_reidentification_fraction"
        ]
        == thresholds["strict_parent_space_group_reidentification_fraction"],
        "structure_matcher_certification_fraction": values["structure_matcher_certification_fraction"]
        == thresholds["structure_matcher_certification_fraction"],
        "projected_group_max_error_angstrom": values["projected_group_max_error_angstrom"]
        <= thresholds["projected_group_max_error_angstrom"],
        "source_max_displacement_angstrom": values["source_max_displacement_angstrom"]
        <= thresholds["source_max_displacement_angstrom"],
        "source_hencky_norm_max": values["source_hencky_norm_max"] <= thresholds["source_hencky_norm_max"],
    }
    return values, checks


def run(config: dict[str, Any], data_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    _validate_dependencies(config, data_root)
    selection = _frozen_selection(config, data_root)
    raw = _join_raw_rows(selection, data_root)
    e0_records = load_gzip_json(
        data_root / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_records.json.gz"
    )
    by_child: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in e0_records:
        if str(record["kind"]) == "t":
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
            forward = search_maximal_t_parents(
                child,
                records,
                maximum_source_displacement_angstrom=float(settings["maximum_source_displacement_angstrom"]),
                matcher_settings=matcher,
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            reverse = search_maximal_t_parents(
                child,
                tuple(reversed(records)),
                maximum_source_displacement_angstrom=float(settings["maximum_source_displacement_angstrom"]),
                matcher_settings=matcher,
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            forward_keys = {(item.embedding_key, item.parent_space_group) for item in forward}
            reverse_keys = {(item.embedding_key, item.parent_space_group) for item in reverse}
            result["forward_reverse_agrees"] = forward_keys == reverse_keys
            result["candidates"] = [
                _serialize_occurrence(item)
                for item in sorted(
                    forward,
                    key=lambda value: (
                        value.parent_space_group,
                        value.embedding_key,
                    ),
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
    manifest: dict[str, object] = {
        "protocol": config["protocol"],
        "qualified": qualified,
        "metrics": metrics,
        "checks": checks,
        "decision": (
            "H0-E-v2-E1a_qualified_E1b_protocol_may_be_designed"
            if qualified
            else "H0-E-v2-E1a_failed_stop_before_E1b_and_H1a"
        ),
        "scope": config["gate_role"],
    }
    return results, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.config.resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    implementation_commit = _git_commit(repo_root)
    results, manifest = run(config, args.data_root)
    results_path = args.data_root / config["required_outputs"]["results"]
    manifest_path = args.data_root / config["required_outputs"]["manifest"]
    write_deterministic_gzip_json(results_path, results)
    manifest["implementation_commit"] = implementation_commit
    manifest["results_sha256"] = sha256_file(results_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    raise SystemExit(0 if bool(manifest["qualified"]) else 2)


if __name__ == "__main__":
    main()
