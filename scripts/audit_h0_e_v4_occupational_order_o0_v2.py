"""Independently rebuild and audit the frozen H0-E-v4 O0-v2 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
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


def _ordered_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(values, separators=(",", ":")).encode())
    return digest.hexdigest()


def _source_rows(
    material_ids: set[str],
    data_root: Path,
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    columns = ["material_id", "positions", "cell", "atomic_numbers"]
    for split in ("test", "val", "train"):
        path = data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet"
        table = pds.dataset(path, format="parquet").to_table(
            filter=pds.field("material_id").isin(material_ids),
            columns=columns,
        )
        for row in table.to_pylist():
            material_id = str(row["material_id"])
            if material_id in rows:
                raise ValueError("independent source join found a duplicate material ID")
            row["source_split_observed"] = split
            rows[material_id] = row
    if set(rows) != material_ids:
        raise ValueError("independent source join did not recover the selected panel")
    return rows


def _key(value: OccupationalParentOccurrence) -> str:
    return f"{value.embedding_key}\x1f{value.parent_space_group}\x1f{value.cell_index}"


def _stored_key(value: dict[str, object]) -> str:
    return "\x1f".join(
        (
            str(value["embedding_key"]),
            str(int(value["parent_space_group"])),
            str(int(value["cell_index"])),
        )
    )


def _candidate_matches(
    rebuilt: OccupationalParentOccurrence,
    stored: dict[str, object],
) -> bool:
    projection = rebuilt.projection
    pattern = rebuilt.occupational_pattern
    arrays_match = (
        np.allclose(
            projection.lattice,
            np.asarray(stored["parent_lattice"], dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )
        and np.allclose(
            projection.fractional,
            np.asarray(stored["parent_fractional"], dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )
        and np.array_equal(
            projection.permutations,
            np.asarray(stored["parent_action_permutations"], dtype=np.int64),
        )
        and np.array_equal(
            rebuilt.occupational_stabilizer_indices,
            np.asarray(stored["occupational_stabilizer_indices"], dtype=np.int64),
        )
        and np.array_equal(
            pattern.site_classes.detach().cpu().numpy(),
            np.asarray(stored["occupational_site_classes"], dtype=np.int64),
        )
        and np.array_equal(
            pattern.species_by_class.detach().cpu().numpy(),
            np.asarray(stored["occupational_species_by_class_tokens"], dtype=np.int64),
        )
        and np.array_equal(
            rebuilt.child_atomic_numbers,
            np.asarray(stored["child_atomic_numbers"], dtype=np.int64),
        )
    )
    scalar_pairs = (
        (
            projection.source_max_displacement_angstrom,
            stored["projection_source_max_displacement_angstrom"],
        ),
        (
            projection.source_rms_displacement_angstrom,
            stored["projection_source_rms_displacement_angstrom"],
        ),
        (
            projection.source_hencky_norm,
            stored["projection_source_hencky_norm"],
        ),
        (
            projection.projected_group_max_error_angstrom,
            stored["projected_group_max_error_angstrom"],
        ),
    )
    scalars_match = all(
        np.isclose(float(left), float(right), atol=1e-12, rtol=0.0)
        for left, right in scalar_pairs
    )
    contracts_match = (
        rebuilt.full_action_order == int(stored["full_action_order"])
        and rebuilt.parent_site_count == int(stored["parent_site_count"])
        and rebuilt.child_operation_order == int(stored["child_operation_order"])
        and rebuilt.occupational_stabilizer_indices.size
        == int(stored["occupational_stabilizer_order"])
        and rebuilt.exact_coloring_reconstruction
        and bool(stored["exact_coloring_reconstruction"])
        and rebuilt.stabilizer_order_matches_child
        and bool(stored["stabilizer_order_matches_child"])
    )
    return arrays_match and scalars_match and contracts_match


def audit(
    config: dict[str, Any],
    data_root: Path,
    repo_root: Path,
) -> dict[str, object]:
    outputs = config["required_outputs"]
    results_path = data_root / outputs["results"]
    manifest_path = data_root / outputs["manifest"]
    results = load_gzip_json(results_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dependencies = config["dependencies"]
    e1a_path = repo_root / str(dependencies["e1a_panel_config"])
    cleaning_path = repo_root / str(dependencies["data_cleaning"])
    e1a = json.loads(e1a_path.read_text(encoding="utf-8"))
    cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
    excluded = {str(value["material_id"]) for value in cleaning["material_exclusions"]}
    source_ids = [str(value) for value in e1a["selection"]["material_ids"]]
    ids = [value for value in source_ids if value not in excluded]
    result_by_id = {str(value["material_id"]): value for value in results}
    decomposition_path = (
        data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/decompositions.parquet"
    )
    decompositions = {
        str(value["material_id"]): value
        for value in pq.read_table(decomposition_path).to_pylist()
        if str(value["material_id"]) in set(ids)
    }
    source = _source_rows(set(ids), data_root)
    records_path = (
        data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_records.json.gz"
    )
    by_child_t: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_child_k: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in reversed(load_gzip_json(records_path)):
        kind = str(record["kind"])
        if kind == "t" and int(record["cell_index"]) == 1:
            by_child_t[int(record["child_space_group"])].append(record)
        elif kind == "k" and 2 <= int(record["cell_index"]) <= 4:
            by_child_k[int(record["child_space_group"])].append(record)

    settings = config["setting_and_search"]
    common = {
        "maximum_source_displacement_angstrom": float(
            settings["maximum_source_displacement_angstrom"]
        ),
        "maximum_source_hencky_norm": float(settings["source_hencky_norm_max"]),
        "angle_tolerance": float(settings["angle_tolerance_degree"]),
    }
    rebuilt_rows = 0
    rebuilt_edges = 0
    rebuilt_candidate_materials = 0
    rebuilt_occupational_materials = 0
    rebuilt_occurrences = 0
    candidate_set_matches = 0
    candidate_numeric_matches = 0
    failures: list[str] = []
    for material_id in ids:
        try:
            frozen = decompositions[material_id]
            raw = source[material_id]
            stored = result_by_id[material_id]
            if int(frozen["candidate_count"]) != 0 or bool(frozen["processing_failure"]):
                raise ValueError("independent selection is not a v1 zero-candidate row")
            if str(raw["source_split_observed"]) != str(frozen["source_split"]):
                raise ValueError("independent source split does not match v1")
            child = standardize_child_to_e0_setting(
                np.asarray(raw["cell"], dtype=np.float64),
                np.asarray(raw["positions"], dtype=np.float64),
                np.asarray(raw["atomic_numbers"], dtype=np.int64),
                expected_space_group=int(frozen["space_group_number"]),
                expected_primitive_sites=int(frozen["primitive_sites"]),
                symprec=float(settings["child_symprec_angstrom"]),
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            records_t = by_child_t[child.space_group]
            records_k = by_child_k[child.space_group]
            rebuilt_edges += len(records_t) + len(records_k)
            rebuilt = (
                *search_occupational_maximal_k_parents(child, records_k, **common),
                *search_occupational_maximal_t_parents(child, records_t, **common),
            )
            rebuilt_rows += 1
            rebuilt_candidate_materials += bool(rebuilt)
            rebuilt_occupational_materials += any(
                value.occupational_stabilizer_indices.size < value.full_action_order
                for value in rebuilt
            )
            rebuilt_occurrences += len(rebuilt)
            expected = {_key(value): value for value in rebuilt}
            observed = {_stored_key(value): value for value in stored["candidates"]}
            if set(expected) != set(observed):
                raise ValueError("independent embedding set differs from stored result")
            candidate_set_matches += 1
            candidate_numeric_matches += sum(
                _candidate_matches(value, observed[key])
                for key, value in expected.items()
            )
            if int(stored["candidate_edges_evaluated"]) != len(records_t) + len(records_k):
                raise ValueError("stored per-row edge count is inconsistent")
        except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failures.append(f"{material_id}: {type(error).__name__}: {error}")

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
        "implementation_commit_matches_head": manifest["implementation_commit"] == head,
        "e0_hash_matches_protocol": sha256_file(records_path)
        == dependencies["e0_records_sha256"],
        "e1a_hash_matches_protocol": sha256_file(e1a_path)
        == dependencies["e1a_panel_config_sha256"],
        "cleaning_hash_matches_protocol": sha256_file(cleaning_path)
        == dependencies["data_cleaning_sha256"],
        "cleaned_ordered_panel_hash_matches": _ordered_hash(ids)
        == config["selection"]["ordered_material_ids_sha256"],
        "excluded_material_absent": excluded.isdisjoint(result_by_id),
        "selected_and_result_ids_exact": len(ids) == int(config["selection"]["size"])
        and len(set(ids)) == len(ids)
        and set(result_by_id) == set(ids),
        "all_rows_rebuilt": rebuilt_rows == len(ids) and not failures,
        "candidate_edge_count_reproduced": rebuilt_edges
        == int(manifest["metrics"]["candidate_edges_evaluated"]),
        "candidate_sets_reproduced": candidate_set_matches == len(ids),
        "candidate_material_count_reproduced": rebuilt_candidate_materials
        == int(manifest["metrics"]["new_candidate_materials"]),
        "occupational_material_count_reproduced": rebuilt_occupational_materials
        == int(manifest["metrics"]["occupationally_nontrivial_materials"]),
        "occurrence_count_reproduced": rebuilt_occurrences
        == int(manifest["metrics"]["certified_embedding_occurrences"]),
        "candidate_numerics_reproduced": candidate_numeric_matches
        == rebuilt_occurrences,
        "stored_decision_matches_checks": bool(manifest["qualified"])
        == all(bool(value) for value in manifest["checks"].values()),
    }
    passed = all(checks.values())
    return {
        "protocol": config["protocol"] + "_independent_reverse_order_audit",
        "audit_passed": passed,
        "gate_qualified": bool(manifest["qualified"]),
        "checks": checks,
        "rebuild": {
            "rows": rebuilt_rows,
            "candidate_edges": rebuilt_edges,
            "candidate_materials": rebuilt_candidate_materials,
            "occupationally_nontrivial_materials": rebuilt_occupational_materials,
            "occurrences": rebuilt_occurrences,
            "failures": failures,
        },
        "decision": (
            "independent_audit_verified_frozen_O0_v2_result"
            if passed
            else "independent_audit_failed_O0_v2_result_untrusted"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    result = audit(config, arguments.data_root, repo_root)
    output = arguments.data_root / config["required_outputs"]["independent_audit"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if bool(result["audit_passed"]) else 2)


if __name__ == "__main__":
    main()
