"""Independent fail-closed audit of the frozen H0-E parent pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from gaugeflow.catalogue import RealizedPathClass, allocate_reference_measure
from gaugeflow.catalogue.parent_decomposition import (
    decompose_parent_candidate,
    find_parent_candidates,
)
from gaugeflow.file_utils import sha256_file
from scripts.build_h0_e_parent_decomposition_pilot import _load_source_rows


def _qualified(record: dict[str, Any], config: dict[str, Any]) -> bool:
    contract = config["decomposition"]
    return bool(
        record.get("qualified_nontrivial") is True
        and float(record["periodic_rms_angstrom"])
        <= float(contract["residual_rms_limit_angstrom"])
        and record.get("structure_matcher_agrees") is True
        and record.get("occurrence_integral") is True
        and record.get("opd_mapping_complete") is True
        and 1
        <= int(record["active_branch_count"])
        <= int(contract["maximum_active_irrep_components"])
    )


def _recompute_metrics(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, float | int]:
    candidates = [value for value in records if int(value.get("candidate_count", 0)) > 0]
    qualified = [value for value in records if _qualified(value, config)]
    residual = np.asarray(
        [float(value["periodic_rms_angstrom"]) for value in qualified], dtype=np.float64
    )
    energy = np.asarray(
        [float(value["top2_energy_fraction"]) for value in qualified], dtype=np.float64
    )
    return {
        "selected": len(records),
        "selection_and_split_join_fraction": float(np.mean(
            [value.get("source_split") == value.get("source_split_observed") for value in records]
        )),
        "exact_branch_reconstruction_fraction": float(np.mean(
            [
                value.get("exact_branch_success") is True
                and float(value.get("exact_branch_periodic_rms_angstrom", math.inf)) <= 1e-12
                for value in records
            ]
        )),
        "sampling_failures": sum(value.get("processing_failure") is True for value in records),
        "nonfinite_results": sum(
            not math.isfinite(float(value[key]))
            for value in records
            for key in ("periodic_rms_angstrom", "top2_energy_fraction")
            if value.get(key) is not None
        ),
        "nontrivial_parent_candidate_fraction": len(candidates) / len(records),
        "qualified_nontrivial_fraction_of_candidates": (
            len(qualified) / len(candidates) if candidates else 0.0
        ),
        "qualified_nontrivial_count": len(qualified),
        "qualified_nontrivial_periodic_rms_max_angstrom": (
            float(residual.max()) if residual.size else math.inf
        ),
        "qualified_nontrivial_periodic_rms_p95_angstrom": (
            float(np.quantile(residual, 0.95)) if residual.size else math.inf
        ),
        "qualified_nontrivial_top2_energy_fraction_median": (
            float(np.median(energy)) if energy.size else 0.0
        ),
        "terminal_space_group_agreement_fraction": (
            float(np.mean([value["terminal_space_group_agrees"] is True for value in qualified]))
            if qualified
            else 0.0
        ),
        "occurrence_integrality_fraction": (
            float(np.mean([value["occurrence_integral"] is True for value in qualified]))
            if qualified
            else 0.0
        ),
        "distinct_nontrivial_parent_space_groups": len(
            {int(value["parent_space_group"]) for value in qualified}
        ),
        "distinct_nontrivial_child_space_groups": len(
            {int(value["selected_child_space_group"]) for value in qualified}
        ),
    }


def _threshold_checks(
    metrics: dict[str, float | int], config: dict[str, Any]
) -> dict[str, bool]:
    threshold = config["thresholds"]
    return {
        "pilot_size": metrics["selected"] == int(config["pilot"]["size"]),
        "selection_and_split_join": metrics["selection_and_split_join_fraction"]
        >= threshold["selection_and_split_join_fraction"],
        "exact_branch_reconstruction": metrics["exact_branch_reconstruction_fraction"]
        >= threshold["exact_branch_reconstruction_fraction"],
        "sampling_failures": metrics["sampling_failures"] <= threshold["sampling_failures"],
        "nonfinite_results": metrics["nonfinite_results"] <= threshold["nonfinite_results"],
        "nontrivial_parent_candidate_fraction": metrics[
            "nontrivial_parent_candidate_fraction"
        ]
        >= threshold["nontrivial_parent_candidate_fraction_min"],
        "qualified_nontrivial_fraction_of_candidates": metrics[
            "qualified_nontrivial_fraction_of_candidates"
        ]
        >= threshold["qualified_nontrivial_fraction_of_candidates_min"],
        "periodic_rms_max": metrics["qualified_nontrivial_periodic_rms_max_angstrom"]
        <= threshold["qualified_nontrivial_periodic_rms_max_angstrom"],
        "periodic_rms_p95": metrics["qualified_nontrivial_periodic_rms_p95_angstrom"]
        <= threshold["qualified_nontrivial_periodic_rms_p95_max_angstrom"],
        "top2_energy": metrics["qualified_nontrivial_top2_energy_fraction_median"]
        >= threshold["qualified_nontrivial_top2_energy_fraction_median_min"],
        "terminal_space_group": metrics["terminal_space_group_agreement_fraction"]
        >= threshold["terminal_space_group_agreement_fraction_min"],
        "occurrence_integrality": metrics["occurrence_integrality_fraction"]
        >= threshold["occurrence_integrality_fraction"],
        "parent_diversity": metrics["distinct_nontrivial_parent_space_groups"]
        >= threshold["distinct_nontrivial_parent_space_groups_min"],
        "child_diversity": metrics["distinct_nontrivial_child_space_groups"]
        >= threshold["distinct_nontrivial_child_space_groups_min"],
    }


def _measure_checks(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, bool]:
    by_parent: dict[int, list[tuple[RealizedPathClass, float]]] = defaultdict(list)
    for record in records:
        if not _qualified(record, config):
            continue
        path_class = RealizedPathClass(
            int(record["supercell_index"]),
            int(record["active_branch_count"]),
            str(record["physical_class_key"]),
        )
        by_parent[int(record["parent_space_group"])].append(
            (path_class, float(record["distorted_class_reference_mass"]))
        )
    assignment_ok = True
    normalized = True
    duplicate_ok = True
    order_ok = True
    for values in by_parent.values():
        classes = [value[0] for value in values]
        expected = dict(allocate_reference_measure(classes, exact_mass=0.5))
        assignment_ok &= all(np.isclose(mass, expected[path], atol=1e-12) for path, mass in values)
        normalized &= np.isclose(sum(expected.values()), 0.5, atol=1e-12)
        duplicate_ok &= expected == dict(
            allocate_reference_measure([*classes, *classes], exact_mass=0.5)
        )
        order_ok &= expected == dict(
            allocate_reference_measure(list(reversed(classes)), exact_mass=0.5)
        )
    return {
        "physical_measure_assignments_recomputed": bool(assignment_ok),
        "physical_measure_normalized": bool(normalized),
        "physical_measure_duplicate_invariant": bool(duplicate_ok),
        "physical_measure_order_invariant": bool(order_ok),
    }


def _panel_rebuild(
    selection: list[dict[str, Any]],
    stored: dict[str, dict[str, Any]],
    source_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    panel = sorted(
        selection,
        key=lambda value: hashlib.sha256(
            f"h0-e-independent-32:{value['material_id']}".encode()
        ).hexdigest(),
    )[:32]
    sources, missing = _load_source_rows(panel, source_root)
    agreements: list[bool] = []
    candidate_rows = 0
    for selected in panel:
        material_id = str(selected["material_id"])
        source = sources[material_id]
        lattice = np.asarray(source["cell"], dtype=np.float64)
        cartesian = np.asarray(source["positions"], dtype=np.float64)
        species = np.asarray(source["atomic_numbers"], dtype=np.int64)
        search = config["parent_search"]
        child_contract = config["child_standardization"]
        _, candidates = find_parent_candidates(
            lattice,
            cartesian @ np.linalg.inv(lattice),
            species,
            child_symprec=float(child_contract["symprec_angstrom"]),
            symprec_ladder=search["symprec_ladder_angstrom"],
            angle_tolerance=float(search["angle_tolerance_degree"]),
            matcher_settings=search["structure_matcher"],
        )
        candidate_rows += bool(candidates)
        contract = config["decomposition"]
        results = [
            decompose_parent_candidate(
                candidate,
                residual_rms_limit=float(contract["residual_rms_limit_angstrom"]),
                stabilizer_rms_tolerance=float(
                    contract["realized_stabilizer_displacement_rms_tolerance_angstrom"]
                ),
                stabilizer_metric_tolerance=float(
                    contract["realized_stabilizer_metric_relative_tolerance"]
                ),
                displacement_energy_floor=float(
                    contract["component_energy_floors"][
                        "displacement_mass_weighted_amu_angstrom2"
                    ]
                ),
                strain_energy_floor=float(
                    contract["component_energy_floors"]["strain_kelvin_norm2"]
                ),
                terminal_symprec=float(child_contract["symprec_angstrom"]),
                angle_tolerance=float(search["angle_tolerance_degree"]),
                matcher_settings=search["structure_matcher"],
            )
            for candidate in candidates
        ]
        qualified = [
            value
            for value in results
            if value.periodic_rms_angstrom <= float(contract["residual_rms_limit_angstrom"])
            and value.structure_matcher_agrees
            and value.occurrence_integral
            and value.opd_mapping_complete
            and 1 <= len(value.active_components) <= 2
        ]
        observed = stored[material_id]
        agreement = (
            len(candidates) == int(observed["candidate_count"])
            and len(results) == int(observed["decomposed_candidate_count"])
            and bool(qualified) is (observed["qualified_nontrivial"] is True)
        )
        if qualified and agreement:
            result = min(
                qualified,
                key=lambda value: (
                    value.periodic_rms_angstrom,
                    value.symprec,
                    value.supercell_index,
                    value.parent_space_group,
                ),
            )
            agreement &= (
                result.parent_space_group == int(observed["parent_space_group"])
                and result.child_space_group == int(observed["selected_child_space_group"])
                and len(result.active_components) == int(observed["active_branch_count"])
                and [value.sector for value in result.active_components]
                == json.loads(observed["active_sectors_json"])
                and np.isclose(
                    result.periodic_rms_angstrom,
                    float(observed["periodic_rms_angstrom"]),
                    atol=1e-12,
                    rtol=1e-10,
                )
            )
        agreements.append(bool(agreement))
    return {
        "panel_size": len(panel),
        "missing_source_rows": missing,
        "candidate_rows": candidate_rows,
        "exact_record_agreements": sum(agreements),
        "passed": not missing and all(agreements),
    }


def audit(config: dict[str, Any], data_root: Path, source_root: Path) -> dict[str, Any]:
    required = config["required_outputs"]
    selection_path = data_root / required["selection_records"]
    records_path = data_root / required["decomposition_records"]
    manifest_path = data_root / required["pilot_manifest"]
    selection_manifest_path = data_root / required["selection_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    selection = pq.read_table(selection_path).to_pylist()
    records = pq.read_table(records_path).to_pylist()
    stored = {str(value["material_id"]): value for value in records}
    metrics = _recompute_metrics(records, config)
    threshold_checks = {
        key: bool(value) for key, value in _threshold_checks(metrics, config).items()
    }
    measure_checks = _measure_checks(records, config)

    h0_d_path = data_root / "processed/gaugeflow_h0_v4/opd_catalogue_v2/catalogue_records.json.gz"
    structural_checks = {
        "protocol_matches": manifest.get("protocol") == config["protocol"],
        "selection_protocol_matches": selection_manifest.get("protocol") == config["protocol"],
        "selection_hash_matches": selection_manifest.get("selection_records_sha256")
        == sha256_file(selection_path),
        "records_hash_matches": manifest.get("decomposition_records_sha256")
        == sha256_file(records_path),
        "h0_d_hash_matches": sha256_file(h0_d_path)
        == config["depends_on"]["h0_d_records_sha256"],
        "selection_ids_unique": len(selection)
        == len({str(value["material_id"]) for value in selection}),
        "record_ids_exactly_selection": set(stored)
        == {str(value["material_id"]) for value in selection},
        "active_sectors_are_declared": all(
            set(json.loads(value["active_sectors_json"])).issubset(
                {"displacement", "strain"}
            )
            for value in records
            if _qualified(value, config)
        ),
        "tensor_condition_absent": not any(
            "tensor" in name.lower() or "piezo" in name.lower()
            for name in pq.read_schema(records_path).names
        ),
    }
    panel = _panel_rebuild(selection, stored, source_root, config)
    gate_qualified = all(threshold_checks.values()) and all(measure_checks.values())
    manifest_consistent = (
        manifest.get("qualified") is gate_qualified
        and manifest.get("decision")
        == (
            "H0-E-v1_qualified_H0_complete_H1a_may_start"
            if gate_qualified
            else "H0-E-v1_failed_stop_before_H1"
        )
    )
    audit_passed = (
        all(structural_checks.values())
        and all(measure_checks.values())
        and panel["passed"]
        and manifest_consistent
    )
    return {
        "protocol": f"{config['protocol']}_independent_audit",
        "audit_passed": audit_passed,
        "gate_qualified": gate_qualified,
        "decision": (
            "H0-E-v1_qualified_H0_complete_H1a_may_start"
            if gate_qualified
            else "H0-E-v1_failed_stop_before_H1"
        ),
        "failed_frozen_thresholds": [
            name for name, passed in threshold_checks.items() if not passed
        ],
        "structural_checks": structural_checks,
        "measure_checks": measure_checks,
        "threshold_checks": threshold_checks,
        "manifest_consistent": manifest_consistent,
        "metrics_recomputed": metrics,
        "deterministic_32_record_rebuild": panel,
        "records_sha256": sha256_file(records_path),
        "selection_sha256": sha256_file(selection_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit(config, args.data_root, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["audit_passed"] else 2)


if __name__ == "__main__":
    main()
