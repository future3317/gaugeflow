"""Build the frozen geometry-complete assignment carrier v2."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pds
import torch

from gaugeflow.catalogue import (
    CanonicalAssignmentCarrier,
    canonicalize_assignment_carrier,
    project_geometry_complete_occupational_embedding,
    standardize_child_to_e0_setting,
)
from gaugeflow.file_utils import sha256_file, write_deterministic_gzip_json
from gaugeflow.production.blueprint import OccupationalPattern


def _git_identity(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("assignment carrier compilation requires a clean Git tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_gzip_list(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON record list: {path}")
    return value


def _validate_sources(
    config: dict[str, Any],
    repository: Path,
    data_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    source = config["source"]
    o1_config_path = repository / source["o1_config"]
    if sha256_file(o1_config_path) != source["o1_config_sha256"]:
        raise ValueError("archived O1 protocol identity changed")
    o1_config = _load_json(o1_config_path)
    for split, expected in o1_config["dependencies"]["raw_alex_sha256"].items():
        raw_path = data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet"
        if sha256_file(raw_path) != expected:
            raise ValueError(f"raw Alex source identity changed: {split}")
    if importlib.metadata.version("pyxtal") != o1_config["dependencies"]["pyxtal_version"]:
        raise ValueError("PyXtal version differs from the archived O1 compiler")
    o1_root = data_root / source["o1_root"]
    for name, expected in source["o1_artifact_sha256"].items():
        if sha256_file(o1_root / name) != expected:
            raise ValueError(f"archived O1 artifact identity changed: {name}")
    manifest = _load_json(o1_root / "manifest.json")
    independent = _load_json(o1_root / "independent_audit.json")
    if manifest.get("qualified") is not True or not all(manifest["checks"].values()):
        raise ValueError("archived O1 manifest is not qualified")
    if independent.get("audit_passed") is not True:
        raise ValueError("archived O1 independent audit is not qualified")
    records = _load_gzip_list(o1_root / "results.json.gz")

    catalogue_path = data_root / source["embedding_catalogue"]
    if sha256_file(catalogue_path) != source["embedding_catalogue_sha256"]:
        raise ValueError("embedding catalogue identity changed")
    catalogue = _load_gzip_list(catalogue_path)
    by_key = {str(row["embedding_key"]): row for row in catalogue}
    if len(by_key) != len(catalogue):
        raise ValueError("embedding catalogue contains duplicate keys")
    return o1_config, records, by_key


def _join_raw_sources(
    records: list[dict[str, Any]],
    data_root: Path,
) -> dict[str, dict[str, Any]]:
    requested = {str(record["material_id"]) for record in records if bool(record.get("candidates"))}
    observed: dict[str, dict[str, Any]] = {}
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
                raise ValueError("material occurs in more than one raw source split")
            row["source_split_observed"] = split
            observed[material_id] = row
    if set(observed) != requested:
        raise ValueError("not every assignment carrier material joined to raw Alex")
    return observed


def _archived_identity(
    archived: dict[str, Any],
    complete: Any,
) -> bool:
    occurrence = complete.occurrence
    projection = occurrence.projection
    scalar_pairs = (
        (occurrence.embedding_key, archived["embedding_key"]),
        (occurrence.parent_space_group, archived["parent_space_group"]),
        (occurrence.child_space_group, archived["child_space_group"]),
        (occurrence.cell_index, archived["cell_index"]),
        (occurrence.full_action_order, archived["full_action_order"]),
        (occurrence.parent_site_count, archived["parent_site_count"]),
        (occurrence.child_operation_order, archived["child_operation_order"]),
    )
    if not all(left == right for left, right in scalar_pairs):
        return False
    exact_arrays = (
        (projection.permutations, archived["parent_action_permutations"]),
        (occurrence.child_atomic_numbers, archived["child_atomic_numbers"]),
        (
            occurrence.occupational_pattern.site_classes.detach().cpu().numpy(),
            archived["occupational_site_classes"],
        ),
        (
            occurrence.occupational_pattern.species_by_class.detach().cpu().numpy(),
            archived["occupational_species_by_class_tokens"],
        ),
        (
            occurrence.occupational_stabilizer_indices,
            archived["occupational_stabilizer_indices"],
        ),
    )
    if not all(np.array_equal(np.asarray(left), np.asarray(right)) for left, right in exact_arrays):
        return False
    float_arrays = (
        (projection.lattice, archived["parent_lattice"]),
        (projection.fractional, archived["parent_fractional"]),
    )
    return all(np.allclose(np.asarray(left), np.asarray(right), atol=1e-10, rtol=1e-10) for left, right in float_arrays)


def _carrier_equal(
    left: CanonicalAssignmentCarrier,
    right: CanonicalAssignmentCarrier,
) -> bool:
    exact_names = (
        "supercell_hnf",
        "translation_cosets",
        "node_parent_site_indices",
        "node_translation_coset_indices",
        "parent_action_permutations",
    )
    float_names = (
        "primitive_parent_lattice",
        "primitive_parent_fractional",
        "expanded_parent_lattice",
        "expanded_parent_fractional",
    )
    return all(np.array_equal(getattr(left, name), getattr(right, name)) for name in exact_names) and all(
        np.allclose(
            getattr(left, name),
            getattr(right, name),
            atol=1e-10,
            rtol=1e-10,
        )
        for name in float_names
    )


def _relabel_consistent(complete: Any, reference: CanonicalAssignmentCarrier) -> bool:
    occurrence = complete.occurrence
    nodes = complete.expanded_fractional.shape[0]
    digest = hashlib.sha256(occurrence.embedding_key.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    order = rng.permutation(nodes)
    inverse = np.empty(nodes, dtype=np.int64)
    inverse[order] = np.arange(nodes, dtype=np.int64)
    old_permutations = occurrence.projection.permutations
    new_permutations = inverse[old_permutations[:, order]]
    source_atomic_numbers = occurrence.child_atomic_numbers[order]
    source_tokens = torch.from_numpy(source_atomic_numbers.copy()) - 1
    relabeled_projection = replace(
        occurrence.projection,
        permutations=new_permutations,
    )
    relabeled_occurrence = replace(
        occurrence,
        projection=relabeled_projection,
        occupational_pattern=OccupationalPattern.from_tokens(source_tokens),
        child_atomic_numbers=source_atomic_numbers,
    )
    relabeled_complete = replace(
        complete,
        occurrence=relabeled_occurrence,
        expanded_fractional=complete.expanded_fractional[order],
    )
    rebuilt = canonicalize_assignment_carrier(relabeled_complete).carrier
    return _carrier_equal(reference, rebuilt)


def _serialize_carrier(carrier: CanonicalAssignmentCarrier) -> dict[str, Any]:
    return {
        "primitive_parent_lattice": carrier.primitive_parent_lattice.tolist(),
        "primitive_parent_fractional": carrier.primitive_parent_fractional.tolist(),
        "expanded_parent_lattice": carrier.expanded_parent_lattice.tolist(),
        "expanded_parent_fractional": carrier.expanded_parent_fractional.tolist(),
        "supercell_hnf": carrier.supercell_hnf.tolist(),
        "translation_cosets": carrier.translation_cosets.tolist(),
        "node_parent_site_indices": carrier.node_parent_site_indices.tolist(),
        "node_translation_coset_indices": (carrier.node_translation_coset_indices.tolist()),
        "parent_action_permutations": carrier.parent_action_permutations.tolist(),
    }


def _rebuild_candidate(
    child: Any,
    archived: dict[str, Any],
    embedding: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    complete = project_geometry_complete_occupational_embedding(
        child,
        embedding,
        maximum_source_displacement_angstrom=float(settings["maximum_source_displacement_angstrom"]),
        maximum_source_hencky_norm=float(settings["source_hencky_norm_max"]),
        angle_tolerance=float(settings["angle_tolerance_degree"]),
    )
    if complete is None:
        raise RuntimeError("certified archived occurrence did not rebuild")
    identity = _archived_identity(archived, complete)
    aligned = canonicalize_assignment_carrier(complete)
    carrier = aligned.carrier
    target_source = np.asarray(archived["child_atomic_numbers"], dtype=np.int64) - 1
    target = target_source[aligned.source_node_by_carrier_node]
    reconstructed = np.empty_like(target_source)
    reconstructed[aligned.source_node_by_carrier_node] = target
    target_reconstruction = bool(np.array_equal(reconstructed, target_source))
    species, counts = np.unique(target, return_counts=True)
    carrier_payload = _serialize_carrier(carrier)
    target_payload = {
        "assignment_tokens": target.tolist(),
        "active_species_tokens": species.tolist(),
        "active_species_counts": counts.tolist(),
    }
    output = {
        "embedding_key": str(archived["embedding_key"]),
        "parent_space_group": int(archived["parent_space_group"]),
        "child_space_group_audit_only": int(archived["child_space_group"]),
        "cell_index": int(archived["cell_index"]),
        "carrier": carrier_payload,
        "target": target_payload,
        "alignment_audit": {
            "source_node_by_carrier_node": (aligned.source_node_by_carrier_node.tolist()),
            "maximum_periodic_alignment_error_angstrom": (aligned.maximum_periodic_alignment_error_angstrom),
        },
    }
    nodes = target.size
    hnf_index = abs(int(round(np.linalg.det(carrier.supercell_hnf))))
    diagnostics = {
        "archived_identity": identity,
        "hnf_index_closure": hnf_index == int(archived["cell_index"]),
        "expanded_node_closure": (
            carrier.expanded_parent_fractional.shape == (nodes, 3)
            and carrier.node_parent_site_indices.shape == (nodes,)
            and carrier.node_translation_coset_indices.shape == (nodes,)
        ),
        "action_node_alignment": carrier.parent_action_permutations.shape[1] == nodes,
        "target_reconstruction": target_reconstruction,
        "source_relabel_consistency": _relabel_consistent(complete, carrier),
        "carrier_target_key_overlap_count": len(set(carrier_payload) & set(target_payload)),
        "maximum_periodic_alignment_error_angstrom": (aligned.maximum_periodic_alignment_error_angstrom),
        "nonfinite_value_count": sum(
            int(np.size(value) - np.isfinite(value).sum())
            for value in (
                carrier.primitive_parent_lattice,
                carrier.primitive_parent_fractional,
                carrier.expanded_parent_lattice,
                carrier.expanded_parent_fractional,
            )
        ),
    }
    return output, diagnostics


def _report_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    checks = result["checks"]
    rows = [
        ("source candidates", metrics["source_candidate_count"], "454"),
        ("candidate rebuild fraction", metrics["candidate_rebuild_fraction"], "1"),
        ("archived identity fraction", metrics["archived_identity_fraction"], "1"),
        ("HNF index closure", metrics["hnf_index_closure_fraction"], "1"),
        ("expanded node closure", metrics["expanded_node_closure_fraction"], "1"),
        ("action-node alignment", metrics["action_node_alignment_fraction"], "1"),
        ("target reconstruction", metrics["target_reconstruction_fraction"], "1"),
        ("source relabel consistency", metrics["source_relabel_consistency_fraction"], "1"),
        (
            "maximum periodic alignment error (A)",
            metrics["maximum_periodic_alignment_error_angstrom"],
            "<=1e-6",
        ),
        ("processing failures", metrics["processing_failure_count"], "0"),
    ]
    table = "\n".join(f"| {name} | {value} | {threshold} |" for name, value, threshold in rows)
    return f"""# Geometry-complete assignment carrier v2

Decision: **{"PASS" if result["qualified"] else "FAIL"}**.

This Gate recompiles the unchanged qualified O1 occurrences into a canonical
row-HNF carrier with one species-free coordinate per finite-action node.  The
terminal coloring is stored in a separate target object and never participates
in HNF conversion, periodic node alignment, or action conjugation.

| metric | observed | frozen requirement |
|---|---:|---:|
{table}

All checks: `{all(checks.values())}`.  Archived O1 and failed Q1 artifacts were
read-only dependencies and were not overwritten.  Passing this Gate permits
only a geometry-aware zero-training assignment expressivity audit.
"""


def run(
    config: dict[str, Any],
    repository: Path,
    data_root: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    o1_config, records, embeddings = _validate_sources(
        config,
        repository,
        data_root,
    )
    raw = _join_raw_sources(records, data_root)
    settings = o1_config["setting_and_search"]
    rebuilt_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    source_candidates = sum(len(row["candidates"]) for row in records)
    for row in records:
        if not row["candidates"]:
            continue
        material_id = str(row["material_id"])
        source = raw[material_id]
        try:
            if str(source["source_split_observed"]) != str(row["source_split"]):
                raise ValueError("raw Alex source split differs from archived O1")
            child = standardize_child_to_e0_setting(
                np.asarray(source["cell"], dtype=np.float64),
                np.asarray(source["positions"], dtype=np.float64),
                np.asarray(source["atomic_numbers"], dtype=np.int64),
                expected_space_group=int(row["child_space_group"]),
                expected_primitive_sites=int(row["primitive_sites"]),
                symprec=float(settings["child_symprec_angstrom"]),
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            candidates = []
            for archived in row["candidates"]:
                key = str(archived["embedding_key"])
                candidate, diagnostic = _rebuild_candidate(
                    child,
                    archived,
                    embeddings[key],
                    settings,
                )
                candidates.append(candidate)
                diagnostics.append(diagnostic)
            rebuilt_records.append(
                {
                    "material_id_audit_only": material_id,
                    "gaugeflow_split_audit_only": str(row["gaugeflow_split"]),
                    "source_split_audit_only": str(row["source_split"]),
                    "candidates": candidates,
                }
            )
        except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failures.append(
                {
                    "material_id": material_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    rebuilt = len(diagnostics)

    def fraction(key: str) -> float:
        if not source_candidates:
            return 0.0
        return sum(bool(value[key]) for value in diagnostics) / source_candidates

    metrics = {
        "source_candidate_count": source_candidates,
        "rebuilt_candidate_count": rebuilt,
        "candidate_rebuild_fraction": rebuilt / source_candidates if source_candidates else 0.0,
        "archived_identity_fraction": fraction("archived_identity"),
        "hnf_index_closure_fraction": fraction("hnf_index_closure"),
        "expanded_node_closure_fraction": fraction("expanded_node_closure"),
        "action_node_alignment_fraction": fraction("action_node_alignment"),
        "target_reconstruction_fraction": fraction("target_reconstruction"),
        "source_relabel_consistency_fraction": fraction("source_relabel_consistency"),
        "carrier_target_key_overlap_count": sum(
            int(value["carrier_target_key_overlap_count"]) for value in diagnostics
        ),
        "maximum_periodic_alignment_error_angstrom": max(
            (float(value["maximum_periodic_alignment_error_angstrom"]) for value in diagnostics),
            default=float("inf"),
        ),
        "nonfinite_value_count": sum(int(value["nonfinite_value_count"]) for value in diagnostics),
        "processing_failure_count": len(failures),
        "cell_index_counts": {
            str(index): sum(
                int(candidate["cell_index"]) == index
                for record in rebuilt_records
                for candidate in record["candidates"]
            )
            for index in range(1, 5)
        },
    }
    thresholds = config["thresholds"]
    checks = {
        "source_candidate_count": metrics["source_candidate_count"] == thresholds["source_candidate_count"],
        "candidate_rebuild_fraction": metrics["candidate_rebuild_fraction"] == thresholds["candidate_rebuild_fraction"],
        "archived_identity_fraction": metrics["archived_identity_fraction"] == thresholds["archived_identity_fraction"],
        "hnf_index_closure_fraction": metrics["hnf_index_closure_fraction"] == thresholds["hnf_index_closure_fraction"],
        "expanded_node_closure_fraction": metrics["expanded_node_closure_fraction"]
        == thresholds["expanded_node_closure_fraction"],
        "action_node_alignment_fraction": metrics["action_node_alignment_fraction"]
        == thresholds["action_node_alignment_fraction"],
        "target_reconstruction_fraction": metrics["target_reconstruction_fraction"]
        == thresholds["target_reconstruction_fraction"],
        "source_relabel_consistency_fraction": metrics["source_relabel_consistency_fraction"]
        == thresholds["source_relabel_consistency_fraction"],
        "carrier_target_key_overlap_count": metrics["carrier_target_key_overlap_count"]
        == thresholds["carrier_target_key_overlap_count"],
        "maximum_periodic_alignment_error_angstrom": metrics["maximum_periodic_alignment_error_angstrom"]
        <= thresholds["maximum_periodic_alignment_error_angstrom"],
        "nonfinite_value_count": metrics["nonfinite_value_count"] == thresholds["nonfinite_value_count"],
        "processing_failure_count": metrics["processing_failure_count"] == thresholds["processing_failure_count"],
    }
    output_root = data_root / config["outputs"]["data_root"]
    records_path = output_root / config["outputs"]["records"]
    write_deterministic_gzip_json(records_path, rebuilt_records)
    result = {
        "protocol": config["protocol"],
        "qualified": all(checks.values()),
        "implementation_commit": implementation_commit,
        "protocol_config_sha256": sha256_file(repository / "configs/gates/h1a_assignment_geometry_carrier_v2.json"),
        "records_sha256": sha256_file(records_path),
        "metrics": metrics,
        "checks": checks,
        "failures": failures,
        "decision": (
            "geometry_aware_zero_training_audit_authorized"
            if all(checks.values())
            else "stop_assignment_work_and_repair_carrier_interface"
        ),
    }
    manifest_path = output_root / config["outputs"]["manifest"]
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_root = repository / config["outputs"]["report_root"]
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (report_root / "README.md").write_text(
        _report_markdown(result),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gates/h1a_assignment_geometry_carrier_v2.json"),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    repository = args.config.resolve().parents[2]
    config = _load_json(args.config)
    if (
        config.get("protocol") != "h1a_assignment_geometry_carrier_v2"
        or config.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen assignment carrier protocol")
    identity = _git_identity(repository)
    result = run(config, repository, args.data_root, identity)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["qualified"] else 2)


if __name__ == "__main__":
    main()
