"""Independently rebuild and audit the frozen H0-E-v3 K0 result."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from gaugeflow.catalogue import search_maximal_k_parents, standardize_child_to_e0_setting
from gaugeflow.file_utils import load_gzip_json, sha256_file


def _source_rows(
    material_ids: set[str], data_root: Path
) -> dict[str, dict[str, object]]:
    requested = frozenset(material_ids)
    rows: dict[str, dict[str, object]] = {}
    columns = ["material_id", "positions", "cell", "atomic_numbers"]
    paths = tuple(
        (
            split,
            data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet",
        )
        for split in ("test", "val", "train")
    )
    for split, path in paths:
        table = pds.dataset(path, format="parquet").to_table(
            filter=pds.field("material_id").isin(requested), columns=columns
        )
        for row in table.to_pylist():
            material_id = str(row["material_id"])
            if material_id in rows:
                raise ValueError("independent raw join found a duplicate material ID")
            row["source_split_observed"] = split
            rows[material_id] = row
    missing = requested.difference(rows)
    if missing:
        raise ValueError(f"independent raw join missed {len(missing)} material IDs")
    return rows


def _ordered_hash(values: list[str]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit(
    config: dict[str, Any], data_root: Path, repo_root: Path
) -> dict[str, object]:
    outputs = config["required_outputs"]
    results_path = data_root / outputs["results"]
    manifest_path = data_root / outputs["manifest"]
    results = load_gzip_json(results_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dependencies = config["dependencies"]
    e1a_path = repo_root / str(dependencies["e1a_panel_config"])
    quarantine_path = repo_root / str(dependencies["path_quarantine"])
    e1a = json.loads(e1a_path.read_text(encoding="utf-8"))
    quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    ids = [str(value) for value in e1a["selection"]["material_ids"]]
    quarantine_keys = {
        (
            str(value["material_id"]),
            int(value["child_space_group"]),
            int(value["parent_space_group"]),
        )
        for value in quarantine["path_quarantine"]
    }
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
    e0_path = (
        data_root
        / "processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/catalogue_records.json.gz"
    )
    by_child: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in reversed(load_gzip_json(e0_path)):
        if str(record["kind"]) == "k" and 2 <= int(record["cell_index"]) <= 4:
            by_child[int(record["child_space_group"])].append(record)
    settings = config["setting_and_search"]
    matcher = settings["structure_matcher"]
    rebuilt_rows = 0
    rebuilt_candidate_materials = 0
    rebuilt_occurrences = 0
    rebuilt_edges = 0
    candidate_set_matches = 0
    numeric_matches = 0
    structural_contract_matches = 0
    rebuild_failures: list[str] = []
    for material_id in ids:
        try:
            frozen = decompositions[material_id]
            raw = source[material_id]
            stored = result_by_id[material_id]
            if int(frozen["candidate_count"]) != 0 or bool(frozen["processing_failure"]):
                raise ValueError("audit selection is not a frozen v1 no-candidate row")
            if str(raw["source_split_observed"]) != str(frozen["source_split"]):
                raise ValueError("audit source split does not match v1")
            child = standardize_child_to_e0_setting(
                np.asarray(raw["cell"], dtype=np.float64),
                np.asarray(raw["positions"], dtype=np.float64),
                np.asarray(raw["atomic_numbers"], dtype=np.int64),
                expected_space_group=int(frozen["space_group_number"]),
                expected_primitive_sites=int(frozen["primitive_sites"]),
                symprec=float(settings["child_symprec_angstrom"]),
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            records = by_child[child.space_group]
            rebuilt_edges += len(records)
            records = [
                record
                for record in records
                if (
                    material_id,
                    child.space_group,
                    int(record["parent_space_group"]),
                )
                not in quarantine_keys
            ]
            rebuilt = search_maximal_k_parents(
                child,
                records,
                maximum_source_displacement_angstrom=float(
                    settings["maximum_source_displacement_angstrom"]
                ),
                matcher_settings=matcher,
                angle_tolerance=float(settings["angle_tolerance_degree"]),
            )
            rebuilt_rows += 1
            rebuilt_candidate_materials += bool(rebuilt)
            rebuilt_occurrences += len(rebuilt)
            expected = {
                (value.embedding_key, value.parent_space_group): value
                for value in rebuilt
            }
            observed = {
                (str(value["embedding_key"]), int(value["parent_space_group"])): value
                for value in stored["candidates"]
            }
            if set(expected) != set(observed):
                raise ValueError("rebuilt embedding set differs from stored result")
            candidate_set_matches += 1
            for key, occurrence in expected.items():
                saved = observed[key]
                arrays_match = (
                    np.allclose(
                        occurrence.projection.lattice,
                        np.asarray(saved["parent_lattice"], dtype=np.float64),
                        atol=1e-12,
                        rtol=0.0,
                    )
                    and np.allclose(
                        occurrence.projection.fractional,
                        np.asarray(saved["parent_fractional"], dtype=np.float64),
                        atol=1e-12,
                        rtol=0.0,
                    )
                    and np.array_equal(
                        occurrence.projection.species,
                        np.asarray(saved["parent_species"], dtype=np.int64),
                    )
                )
                scalars = (
                    (
                        occurrence.projection.source_max_displacement_angstrom,
                        saved["projection_source_max_displacement_angstrom"],
                    ),
                    (
                        occurrence.projection.source_rms_displacement_angstrom,
                        saved["projection_source_rms_displacement_angstrom"],
                    ),
                    (
                        occurrence.projection.source_hencky_norm,
                        saved["projection_source_hencky_norm"],
                    ),
                    (
                        occurrence.projection.projected_group_max_error_angstrom,
                        saved["projected_group_max_error_angstrom"],
                    ),
                )
                scalars_match = all(
                    np.isclose(float(left), float(right), atol=1e-12, rtol=0.0)
                    for left, right in scalars
                )
                numeric_matches += arrays_match and scalars_match
                structural_contract_matches += (
                    occurrence.parent_site_count * occurrence.cell_index
                    == occurrence.candidate.child.species.size
                    and occurrence.full_action_order
                    == occurrence.candidate.parent.rotations.shape[0]
                    * occurrence.cell_index
                    and occurrence.cell_index == int(saved["cell_index"])
                )
        except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            rebuild_failures.append(f"{material_id}: {type(error).__name__}: {error}")
    checks = {
        "results_hash_matches_manifest": sha256_file(results_path)
        == manifest["results_sha256"],
        "e0_hash_matches_protocol": sha256_file(e0_path)
        == dependencies["e0_records_sha256"],
        "e1a_config_hash_matches_protocol": sha256_file(e1a_path)
        == dependencies["e1a_panel_config_sha256"],
        "quarantine_hash_matches_protocol": sha256_file(quarantine_path)
        == dependencies["path_quarantine_sha256"],
        "ordered_panel_hash_matches": _ordered_hash(ids)
        == config["selection"]["ordered_material_ids_sha256"],
        "selected_and_result_ids_exact": len(ids) == 64
        and len(set(ids)) == 64
        and set(result_by_id) == set(ids),
        "all_rows_rebuilt": rebuilt_rows == len(ids) and not rebuild_failures,
        "candidate_edge_count_reproduced": rebuilt_edges
        == int(manifest["metrics"]["candidate_edges_evaluated"]),
        "candidate_sets_reproduced": candidate_set_matches == len(ids),
        "candidate_material_count_reproduced": rebuilt_candidate_materials
        == int(manifest["metrics"]["new_candidate_materials"]),
        "occurrence_count_reproduced": rebuilt_occurrences
        == int(manifest["metrics"]["certified_embedding_occurrences"]),
        "candidate_numerics_reproduced": numeric_matches == rebuilt_occurrences,
        "cell_and_action_contracts_reproduced": structural_contract_matches
        == rebuilt_occurrences,
        "stored_decision_matches_checks": bool(manifest["qualified"])
        == all(bool(value) for value in manifest["checks"].values()),
    }
    passed = all(checks.values())
    return {
        "protocol": config["protocol"] + "_independent_audit",
        "audit_passed": passed,
        "gate_qualified": bool(manifest["qualified"]),
        "checks": checks,
        "rebuild": {
            "rows": rebuilt_rows,
            "candidate_edges": rebuilt_edges,
            "candidate_materials": rebuilt_candidate_materials,
            "occurrences": rebuilt_occurrences,
            "failures": rebuild_failures,
        },
        "decision": (
            "independent_audit_verified_frozen_K0_result"
            if passed
            else "independent_audit_failed_K0_result_untrusted"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    result = audit(config, args.data_root, repo_root)
    output = args.data_root / config["required_outputs"]["independent_audit"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if bool(result["audit_passed"]) else 2)


if __name__ == "__main__":
    main()
