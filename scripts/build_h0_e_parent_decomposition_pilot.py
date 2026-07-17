"""Build the frozen H0-E concrete parent-decomposition pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gaugeflow.catalogue import RealizedPathClass, allocate_reference_measure
from gaugeflow.catalogue.parent_decomposition import (
    balanced_selection,
    decompose_parent_candidate,
    find_parent_candidates,
    hnf_key,
)
from gaugeflow.file_utils import sha256_file


def _load_assignment(config: dict[str, Any], data_root: Path) -> tuple[Path, list[dict[str, object]]]:
    split_manifest_path = data_root / "processed/gaugeflow_h0_v2/alex_formula_prototype_split.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    assignment_path = split_manifest_path.parent / split_manifest["assignment_path"]
    expected = config["depends_on"]["h0_a_assignment_sha256"]
    if sha256_file(assignment_path) != expected:
        raise ValueError("H0-A assignment hash does not match the frozen H0-E dependency")
    columns = [
        "material_id",
        "source_split",
        "gaugeflow_split",
        "space_group_number",
        "primitive_sites",
        "reduced_formula_key",
        "prototype_key",
        "component_id",
    ]
    return assignment_path, pq.read_table(assignment_path, columns=columns).to_pylist()


def _select(config: dict[str, Any], data_root: Path) -> tuple[Path, list[dict[str, object]], dict[str, Any]]:
    assignment_path, assignment = _load_assignment(config, data_root)
    pilot = config["pilot"]
    selected = list(
        balanced_selection(
            assignment,
            split_counts={key: int(value) for key, value in pilot["gaugeflow_split_counts"].items()},
            seed=int(pilot["seed"]),
            site_boundaries=pilot["primitive_site_bins"],
        )
    )
    output_path = data_root / config["required_outputs"]["selection_records"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), output_path, compression="zstd", version="2.6")
    manifest = {
        "protocol": config["protocol"],
        "selection_rule": config["pilot"],
        "assignment_path": str(assignment_path),
        "assignment_sha256": sha256_file(assignment_path),
        "selection_records": str(output_path),
        "selection_records_sha256": sha256_file(output_path),
        "selected": len(selected),
        "gaugeflow_split_counts": dict(Counter(str(value["gaugeflow_split"]) for value in selected)),
        "source_split_counts": dict(Counter(str(value["source_split"]) for value in selected)),
        "material_ids_unique": len({str(value["material_id"]) for value in selected}) == len(selected),
    }
    manifest_path = data_root / config["required_outputs"]["selection_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path, selected, manifest


def _load_source_rows(
    selected: list[dict[str, object]], source_root: Path
) -> tuple[dict[str, dict[str, object]], list[str]]:
    wanted = {str(value["material_id"]) for value in selected}
    rows: dict[str, dict[str, object]] = {}
    for source_split in ("train", "val", "test"):
        parquet = pq.ParquetFile(source_root / f"{source_split}.parquet")
        for batch in parquet.iter_batches(
            batch_size=8192,
            columns=["material_id", "positions", "cell", "atomic_numbers"],
        ):
            ids = batch.column("material_id").to_pylist()
            keep = [index for index, material_id in enumerate(ids) if str(material_id) in wanted]
            if not keep:
                continue
            for index in keep:
                row = {
                    name: batch.column(name)[index].as_py()
                    for name in ("material_id", "positions", "cell", "atomic_numbers")
                }
                material_id = str(row["material_id"])
                if material_id in rows:
                    raise ValueError("Alex source contains a duplicate selected material ID")
                row["source_split_observed"] = source_split
                rows[material_id] = row
    return rows, sorted(wanted - set(rows))


def _process_one(task: tuple[dict[str, object], dict[str, object], dict[str, Any]]) -> dict[str, object]:
    selection, source, config = task
    material_id = str(selection["material_id"])
    output: dict[str, object] = {
        **selection,
        "material_id": material_id,
        "source_split_observed": str(source["source_split_observed"]),
        "exact_branch_success": True,
        "exact_branch_periodic_rms_angstrom": 0.0,
        "candidate_count": 0,
        "decomposed_candidate_count": 0,
        "qualified_nontrivial": False,
        "processing_failure": False,
        "failure_reason": "",
    }
    try:
        lattice = np.asarray(source["cell"], dtype=np.float64)
        cartesian = np.asarray(source["positions"], dtype=np.float64)
        species = np.asarray(source["atomic_numbers"], dtype=np.int64)
        fractional = cartesian @ np.linalg.inv(lattice)
        search = config["parent_search"]
        child_contract = config["child_standardization"]
        child, candidates = find_parent_candidates(
            lattice,
            fractional,
            species,
            child_symprec=float(child_contract["symprec_angstrom"]),
            symprec_ladder=search["symprec_ladder_angstrom"],
            angle_tolerance=float(search["angle_tolerance_degree"]),
            matcher_settings=search["structure_matcher"],
        )
        output["child_space_group_recomputed"] = child.space_group
        output["child_primitive_sites_recomputed"] = int(child.species.size)
        output["candidate_count"] = len(candidates)
        decomposition_contract = config["decomposition"]
        results = []
        errors = []
        for candidate in candidates:
            try:
                result = decompose_parent_candidate(
                    candidate,
                    residual_rms_limit=float(
                        decomposition_contract["residual_rms_limit_angstrom"]
                    ),
                    stabilizer_rms_tolerance=float(
                        decomposition_contract[
                            "realized_stabilizer_displacement_rms_tolerance_angstrom"
                        ]
                    ),
                    stabilizer_metric_tolerance=float(
                        decomposition_contract[
                            "realized_stabilizer_metric_relative_tolerance"
                        ]
                    ),
                    displacement_energy_floor=float(
                        decomposition_contract["component_energy_floors"][
                            "displacement_mass_weighted_amu_angstrom2"
                        ]
                    ),
                    strain_energy_floor=float(
                        decomposition_contract["component_energy_floors"][
                            "strain_kelvin_norm2"
                        ]
                    ),
                    terminal_symprec=float(child_contract["symprec_angstrom"]),
                    angle_tolerance=float(search["angle_tolerance_degree"]),
                    matcher_settings=search["structure_matcher"],
                )
                results.append(result)
            except Exception as exc:  # fail per candidate, preserve the pilot row
                errors.append(f"{type(exc).__name__}: {exc}")
        output["decomposed_candidate_count"] = len(results)
        output["candidate_errors_json"] = json.dumps(errors, separators=(",", ":"))
        output["decomposed_candidates_json"] = json.dumps(
            [
                {
                    "parent_space_group": value.parent_space_group,
                    "child_space_group": value.child_space_group,
                    "supercell_index": value.supercell_index,
                    "symprec": value.symprec,
                    "periodic_rms_angstrom": value.periodic_rms_angstrom,
                    "top2_energy_fraction": value.top2_energy_fraction,
                    "terminal_space_group_agrees": value.terminal_space_group_agrees,
                    "structure_matcher_agrees": value.structure_matcher_agrees,
                    "occurrence_integral": value.occurrence_integral,
                    "opd_mapping_complete": value.opd_mapping_complete,
                    "active_branch_count": len(value.active_components),
                }
                for value in results
            ],
            separators=(",", ":"),
        )
        qualified = [
            value
            for value in results
            if value.periodic_rms_angstrom
            <= float(decomposition_contract["residual_rms_limit_angstrom"])
            and value.structure_matcher_agrees
            and value.occurrence_integral
            and value.opd_mapping_complete
            and 1 <= len(value.active_components) <= int(
                decomposition_contract["maximum_active_irrep_components"]
            )
        ]
        if qualified:
            selected_result = min(
                qualified,
                key=lambda value: (
                    value.periodic_rms_angstrom,
                    value.symprec,
                    value.supercell_index,
                    value.parent_space_group,
                ),
            )
            output.update(
                {
                    "qualified_nontrivial": True,
                    "parent_space_group": selected_result.parent_space_group,
                    "selected_child_space_group": selected_result.child_space_group,
                    "supercell_hnf": hnf_key(selected_result.supercell_hnf),
                    "supercell_index": selected_result.supercell_index,
                    "parent_search_symprec": selected_result.symprec,
                    "periodic_rms_angstrom": selected_result.periodic_rms_angstrom,
                    "top2_energy_fraction": selected_result.top2_energy_fraction,
                    "terminal_space_group": selected_result.terminal_space_group,
                    "terminal_space_group_agrees": selected_result.terminal_space_group_agrees,
                    "structure_matcher_agrees": selected_result.structure_matcher_agrees,
                    "occurrence_integral": selected_result.occurrence_integral,
                    "opd_mapping_complete": selected_result.opd_mapping_complete,
                    "stabilizer_size": selected_result.stabilizer_size,
                    "physical_class_key": selected_result.physical_class_key,
                    "active_branch_count": len(selected_result.active_components),
                    "active_irrep_keys_json": json.dumps(
                        [value.irrep_key for value in selected_result.active_components],
                        separators=(",", ":"),
                    ),
                    "active_sectors_json": json.dumps(
                        [value.sector for value in selected_result.active_components],
                        separators=(",", ":"),
                    ),
                    "active_branch_keys_json": json.dumps(
                        [value.branch_key for value in selected_result.active_components],
                        separators=(",", ":"),
                    ),
                    "active_multiplicities_json": json.dumps(
                        [value.multiplicity for value in selected_result.active_components],
                        separators=(",", ":"),
                    ),
                }
            )
    except Exception as exc:  # preserve every selected ID and fail the gate
        output["processing_failure"] = True
        output["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return output


def _attach_measure(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, Any]]:
    by_parent: dict[int, list[RealizedPathClass]] = defaultdict(list)
    for record in records:
        if record.get("qualified_nontrivial") is not True:
            continue
        by_parent[int(record["parent_space_group"])].append(
            RealizedPathClass(
                int(record["supercell_index"]),
                int(record["active_branch_count"]),
                str(record["physical_class_key"]),
            )
        )
    masses: dict[tuple[int, RealizedPathClass], float] = {}
    maximum_error = 0.0
    maximum_duplicate_change = 0.0
    for parent, classes in by_parent.items():
        measure = allocate_reference_measure(classes, exact_mass=0.5)
        duplicate = allocate_reference_measure([*classes, *classes], exact_mass=0.5)
        maximum_error = max(
            maximum_error, abs(sum(value for _, value in measure) + 0.5 - 1.0)
        )
        base = dict(measure)
        changed = dict(duplicate)
        maximum_duplicate_change = max(
            maximum_duplicate_change,
            max((abs(base[key] - changed[key]) for key in base), default=0.0),
        )
        for path_class, mass in measure:
            masses[(parent, path_class)] = mass
    for record in records:
        if record.get("qualified_nontrivial") is not True:
            record["distorted_class_reference_mass"] = None
            continue
        path_class = RealizedPathClass(
            int(record["supercell_index"]),
            int(record["active_branch_count"]),
            str(record["physical_class_key"]),
        )
        record["distorted_class_reference_mass"] = masses[
            (int(record["parent_space_group"]), path_class)
        ]
    return records, {
        "parents_with_realized_distortions": len(by_parent),
        "unique_realized_physical_classes": len(masses),
        "maximum_parent_mass_sum_abs_error": maximum_error,
        "maximum_duplicate_expansion_mass_change": maximum_duplicate_change,
    }


def _finite_count(records: list[dict[str, object]]) -> int:
    failures = 0
    for record in records:
        for key in ("periodic_rms_angstrom", "top2_energy_fraction"):
            value = record.get(key)
            if value is not None and not math.isfinite(float(value)):
                failures += 1
    return failures


def _records_table(records: list[dict[str, object]]) -> pa.Table:
    """Build a union-schema table for sparse qualified-result dictionaries.

    ``Table.from_pylist`` otherwise infers its schema from the first mapping and
    silently drops fields that appear only on later qualified rows.
    """
    columns = sorted({key for record in records for key in record})
    normalized = [{key: record.get(key) for key in columns} for record in records]
    return pa.Table.from_pylist(normalized)


def _manifest(
    config: dict[str, Any],
    records: list[dict[str, object]],
    selection_manifest: dict[str, Any],
    measure: dict[str, Any],
    records_path: Path,
) -> dict[str, Any]:
    selected = len(records)
    candidates = [value for value in records if int(value.get("candidate_count", 0)) > 0]
    qualified = [value for value in records if value.get("qualified_nontrivial") is True]
    residuals = np.asarray(
        [float(value["periodic_rms_angstrom"]) for value in qualified], dtype=np.float64
    )
    energy = np.asarray(
        [float(value["top2_energy_fraction"]) for value in qualified], dtype=np.float64
    )
    thresholds = config["thresholds"]
    metrics = {
        "selected": selected,
        "selection_and_split_join_fraction": sum(
            str(value.get("source_split")) == str(value.get("source_split_observed"))
            for value in records
        )
        / selected,
        "exact_branch_reconstruction_fraction": sum(
            value.get("exact_branch_success") is True
            and float(value.get("exact_branch_periodic_rms_angstrom", math.inf)) <= 1e-12
            for value in records
        )
        / selected,
        "sampling_failures": sum(value.get("processing_failure") is True for value in records),
        "nonfinite_results": _finite_count(records),
        "nontrivial_parent_candidate_fraction": len(candidates) / selected,
        "qualified_nontrivial_fraction_of_candidates": (
            len(qualified) / len(candidates) if candidates else 0.0
        ),
        "qualified_nontrivial_count": len(qualified),
        "qualified_nontrivial_periodic_rms_max_angstrom": (
            float(residuals.max()) if residuals.size else math.inf
        ),
        "qualified_nontrivial_periodic_rms_p95_angstrom": (
            float(np.quantile(residuals, 0.95)) if residuals.size else math.inf
        ),
        "qualified_nontrivial_top2_energy_fraction_median": (
            float(np.median(energy)) if energy.size else 0.0
        ),
        "terminal_space_group_agreement_fraction": (
            sum(value.get("terminal_space_group_agrees") is True for value in qualified)
            / len(qualified)
            if qualified
            else 0.0
        ),
        "occurrence_integrality_fraction": (
            sum(value.get("occurrence_integral") is True for value in qualified)
            / len(qualified)
            if qualified
            else 0.0
        ),
        "distinct_nontrivial_parent_space_groups": len(
            {int(value["parent_space_group"]) for value in qualified}
        ),
        "distinct_nontrivial_child_space_groups": len(
            {int(value["selected_child_space_group"]) for value in qualified}
        ),
        **measure,
    }
    checks = {
        "pilot_size": selected == int(config["pilot"]["size"]),
        "selection_and_split_join": metrics["selection_and_split_join_fraction"]
        >= thresholds["selection_and_split_join_fraction"],
        "exact_branch_reconstruction": metrics["exact_branch_reconstruction_fraction"]
        >= thresholds["exact_branch_reconstruction_fraction"],
        "sampling_failures": metrics["sampling_failures"] <= thresholds["sampling_failures"],
        "nonfinite_results": metrics["nonfinite_results"] <= thresholds["nonfinite_results"],
        "nontrivial_parent_candidate_fraction": metrics[
            "nontrivial_parent_candidate_fraction"
        ]
        >= thresholds["nontrivial_parent_candidate_fraction_min"],
        "qualified_nontrivial_fraction_of_candidates": metrics[
            "qualified_nontrivial_fraction_of_candidates"
        ]
        >= thresholds["qualified_nontrivial_fraction_of_candidates_min"],
        "periodic_rms_max": metrics["qualified_nontrivial_periodic_rms_max_angstrom"]
        <= thresholds["qualified_nontrivial_periodic_rms_max_angstrom"],
        "periodic_rms_p95": metrics["qualified_nontrivial_periodic_rms_p95_angstrom"]
        <= thresholds["qualified_nontrivial_periodic_rms_p95_max_angstrom"],
        "top2_energy": metrics["qualified_nontrivial_top2_energy_fraction_median"]
        >= thresholds["qualified_nontrivial_top2_energy_fraction_median_min"],
        "terminal_space_group": metrics["terminal_space_group_agreement_fraction"]
        >= thresholds["terminal_space_group_agreement_fraction_min"],
        "occurrence_integrality": metrics["occurrence_integrality_fraction"]
        >= thresholds["occurrence_integrality_fraction"],
        "measure_normalization": measure["maximum_parent_mass_sum_abs_error"]
        <= thresholds["physical_measure_normalization_abs_error_max"],
        "duplicate_measure": measure["maximum_duplicate_expansion_mass_change"]
        <= thresholds["duplicate_measure_max_abs_change"],
        "parent_diversity": metrics["distinct_nontrivial_parent_space_groups"]
        >= thresholds["distinct_nontrivial_parent_space_groups_min"],
        "child_diversity": metrics["distinct_nontrivial_child_space_groups"]
        >= thresholds["distinct_nontrivial_child_space_groups_min"],
    }
    qualified_gate = all(checks.values())
    return {
        "protocol": config["protocol"],
        "qualified": qualified_gate,
        "decision": (
            "H0-E-v1_qualified_H0_complete_H1a_may_start"
            if qualified_gate
            else "H0-E-v1_failed_stop_before_H1"
        ),
        "checks": checks,
        "metrics": metrics,
        "thresholds": thresholds,
        "selection_manifest": selection_manifest,
        "decomposition_records": str(records_path),
        "decomposition_records_sha256": sha256_file(records_path),
        "h0_d_records_sha256": config["depends_on"]["h0_d_records_sha256"],
    }


def build(
    config: dict[str, Any],
    data_root: Path,
    source_root: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    _, selected, selection_manifest = _select(config, data_root)
    source_rows, missing = _load_source_rows(selected, source_root)
    if missing:
        raise ValueError(f"selected Alex IDs missing from raw source: {missing[:8]}")
    tasks = [
        (selection, source_rows[str(selection["material_id"])], config)
        for selection in selected
    ]
    if workers == 1:
        records = [_process_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_process_one, tasks, chunksize=1))
    records.sort(key=lambda value: str(value["material_id"]))
    records, measure = _attach_measure(records)
    records_path = data_root / config["required_outputs"]["decomposition_records"]
    pq.write_table(_records_table(records), records_path, compression="zstd", version="2.6")
    manifest = _manifest(config, records, selection_manifest, measure, records_path)
    manifest_path = data_root / config["required_outputs"]["pilot_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("H0-E builder permits 1..4 workers")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = build(
        config,
        args.data_root,
        args.source_root,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["qualified"] else 2)


if __name__ == "__main__":
    main()
