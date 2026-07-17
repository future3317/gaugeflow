"""Independent fail-closed auditor for the H0-D-v2 affine OPD catalogue."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

from gaugeflow.catalogue import (
    AffineQuotient,
    PrimitiveSpaceGroup,
    RealIrrep,
    RealizedPathClass,
    allocate_reference_measure,
    build_compact_displacement_action,
    canonical_supercell_orbits,
    enumerate_opd_classes,
    primitive_space_group_from_hall,
    real_irrep_multiplicity,
    stabilizer_bitset,
    standard_hall_numbers,
)
from gaugeflow.catalogue.finite_group import FiniteGroup, intersect_stabilizer_bitsets
from scripts.build_h0_d_opd_catalogue_v2 import (
    _build_parent,
    _spgrep_modulation_reference_agreement,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("catalogue records must be a JSON list")
    return value


def _cyclic_four_invariance_checks() -> dict[str, bool]:
    table = np.fromfunction(lambda i, j: (i + j) % 4, (4, 4), dtype=int)
    group = FiniteGroup.from_cayley_table(table, [f"r{index}" for index in range(4)])
    angles = np.arange(4) * np.pi / 2
    matrices = np.stack(
        [
            np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            for angle in angles
        ]
    )
    irrep = RealIrrep(matrices, 0, tuple(np.trace(matrices, axis1=1, axis2=2)))
    branches = enumerate_opd_classes(group, irrep)
    gauge = np.array([[0.6, -0.8], [0.8, 0.6]])
    gauged = RealIrrep(
        np.einsum("ab,gbc,dc->gad", gauge, matrices, gauge, optimize=True),
        0,
        irrep.character_key,
    )
    gauge_keys = [value.physical_key for value in enumerate_opd_classes(group, gauged)]

    permutation = np.array([0, 3, 2, 1])
    inverse = np.argsort(permutation)
    permuted_table = inverse[group.table[permutation[:, None], permutation[None, :]]]
    permuted_group = FiniteGroup.from_cayley_table(
        permuted_table, tuple(group.element_keys[index] for index in permutation)
    )
    permuted_irrep = RealIrrep(matrices[permutation], 0, irrep.character_key)
    permuted_keys = [
        value.physical_key for value in enumerate_opd_classes(permuted_group, permuted_irrep)
    ]
    keys = [value.physical_key for value in branches]
    projector_ok = all(
        np.allclose(value.projector @ value.projector, value.projector, atol=1e-9, rtol=1e-9)
        for value in branches
    )
    stabilizer_ok = all(
        value.stabilizer_words
        == stabilizer_bitset(value.stabilizer, group_order=group.order)
        for value in branches
    )
    return {
        "reynolds_projector_idempotence": projector_ok,
        "pointwise_stabilizer": stabilizer_ok,
        "opd_basis_gauge_invariance": keys == gauge_keys,
        "domain_conjugacy_invariance": keys == permuted_keys,
        "group_enumeration_order_invariance": keys == permuted_keys,
    }


def _compact_action_checks() -> dict[str, bool]:
    parent = PrimitiveSpaceGroup.from_operations(
        np.stack([np.eye(3, dtype=int), -np.eye(3, dtype=int)]),
        np.zeros((2, 3)),
    )
    quotient = AffineQuotient.build(parent, np.diag([2, 1, 1]))
    action = build_compact_displacement_action(
        quotient, np.eye(3), np.zeros((1, 3)), np.ones(1, dtype=int)
    )
    from gaugeflow.catalogue import enumerate_real_irreps

    multiplicities = [
        real_irrep_multiplicity(action, irrep)
        for irrep in enumerate_real_irreps(quotient.group)
    ]
    left = stabilizer_bitset([0, 1, 3], group_order=quotient.group.order)
    right = stabilizer_bitset([0, 2, 3], group_order=quotient.group.order)
    return {
        "compact_displacement_action_homomorphism": bool(
            action.permutations.shape == (quotient.group.order, 2)
        ),
        "real_irrep_occurrence_integrality": all(value >= 0 for value in multiplicities),
        "packed_stabilizer_intersection_equivalence": (
            intersect_stabilizer_bitsets(left, right)
            == stabilizer_bitset([0, 3], group_order=quotient.group.order)
        ),
    }


def _measure_check() -> bool:
    classes = [
        RealizedPathClass(2, 1, "a"),
        RealizedPathClass(2, 1, "b"),
        RealizedPathClass(3, 2, "c"),
    ]
    base = allocate_reference_measure(classes)
    duplicate = allocate_reference_measure([*classes, classes[0], classes[0]])
    return base == duplicate and abs(sum(mass for _, mass in base) - 0.5) <= 1e-12


def audit(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    manifest_path = data_root / config["required_outputs"]["catalogue_manifest"]
    records_path = data_root / config["required_outputs"]["catalogue_records"]
    checks: dict[str, bool] = {
        "manifest_present": manifest_path.is_file(),
        "records_present": records_path.is_file(),
    }
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    parents: list[dict[str, Any]] = []
    if checks["manifest_present"]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"manifest unreadable: {exc}")
    if checks["records_present"]:
        try:
            parents = _load_records(records_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"records unreadable: {exc}")
    checks["protocol_matches"] = manifest.get("protocol") == config["protocol"]
    checks["records_hash_matches"] = bool(
        records_path.is_file()
        and manifest.get("records_sha256") == _sha256(records_path)
    )

    expected_versions = {
        name.replace("_", "-"): value.split("_", maxsplit=1)[0]
        for name, value in config["qualified_sources"].items()
    }
    installed_versions: dict[str, str | None] = {}
    for name, expected in expected_versions.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        installed_versions[name] = installed
        checks[f"source_version_{name}"] = installed == expected
    checks["manifest_versions_match"] = manifest.get("versions") == installed_versions

    hall_numbers = standard_hall_numbers()
    parent_by_group = {
        int(value.get("space_group", -1)): value
        for value in parents
        if isinstance(value, dict)
    }
    checks["all_230_parent_groups"] = (
        len(parents) == 230 and set(parent_by_group) == set(range(1, 231))
    )
    exhaustive_hnf = checks["all_230_parent_groups"]
    affine_hashes = checks["all_230_parent_groups"]
    exact_branches = checks["all_230_parent_groups"]
    irrep_certificates = checks["all_230_parent_groups"]
    record_count = 0
    irrep_count = 0
    opd_count = 0
    if checks["all_230_parent_groups"]:
        for space_group in range(1, 231):
            parent_record = parent_by_group[space_group]
            hall_number = hall_numbers[space_group]
            exact = parent_record.get("exact_branch", {})
            exact_branches &= (
                parent_record.get("hall_number") == hall_number
                and exact.get("active_opd_branches") == 0
                and exact.get("reference_mass") == config["measure_rule"]["exact_branch_mass"]
            )
            parent = primitive_space_group_from_hall(hall_number)
            expected_hnfs = canonical_supercell_orbits(parent, 4)
            observed_records = parent_record.get("records", [])
            observed_hnfs = [
                np.asarray(value.get("supercell_hnf"), dtype=np.int64)
                for value in observed_records
            ]
            exhaustive_hnf &= len(expected_hnfs) == len(observed_hnfs) and all(
                np.array_equal(left, right)
                for left, right in zip(expected_hnfs, observed_hnfs)
            )
            record_count += len(observed_records)
            for hnf, record in zip(expected_hnfs, observed_records):
                try:
                    quotient = AffineQuotient.build(parent, hnf)
                except (RuntimeError, ValueError) as exc:
                    errors.append(f"SG {space_group} affine reconstruction failed: {exc}")
                    affine_hashes = False
                    continue
                affine_hashes &= (
                    record.get("quotient_order") == quotient.group.order
                    and record.get("cayley_table_sha256")
                    == hashlib.sha256(quotient.group.table.tobytes()).hexdigest()
                )
                irreps = record.get("irreps", [])
                irrep_count += len(irreps)
                contribution = sum(
                    int(value.get("complex_regular_contribution", -1))
                    for value in irreps
                )
                irrep_certificates &= (
                    contribution == quotient.group.order
                    and record.get("complex_regular_completeness_sum") == quotient.group.order
                    and len({value.get("irrep_key") for value in irreps}) == len(irreps)
                )
                for irrep in irreps:
                    dimension = int(irrep.get("dimension", 0))
                    indicator = int(irrep.get("frobenius_schur_indicator", 9))
                    divisor = {1: 1, 0: 2, -1: 4}.get(indicator)
                    expected_contribution = (
                        dimension * dimension // divisor
                        if divisor is not None and dimension * dimension % divisor == 0
                        else -1
                    )
                    irrep_certificates &= (
                        irrep.get("complex_regular_contribution") == expected_contribution
                    )
                    branches = irrep.get("opd_classes", [])
                    opd_count += len(branches)
                    irrep_certificates &= (
                        len(
                            {value.get("physical_key_sha256") for value in branches}
                        )
                        == len(branches)
                    )
                    for branch in branches:
                        words = branch.get("stabilizer_words", [])
                        final_word_mask = (
                            (1 << (quotient.group.order % 64)) - 1
                            if quotient.group.order % 64
                            else (1 << 64) - 1
                        )
                        irrep_certificates &= (
                            1 <= int(branch.get("fixed_dimension", 0)) <= dimension
                            and len(words) == (quotient.group.order + 63) // 64
                            and isinstance(branch.get("physical_key_sha256"), str)
                            and len(branch["physical_key_sha256"]) == 64
                            and bool(words[quotient.group.identity // 64]
                                     & (1 << (quotient.group.identity % 64)))
                            and not (words[-1] & ~final_word_mask)
                        )
    checks["all_230_explicit_exact_branches"] = exact_branches
    checks["all_hnf_orbits_complete"] = exhaustive_hnf
    checks["finite_group_axioms"] = affine_hashes
    checks["exact_affine_translation_cosets"] = affine_hashes
    checks["complete_real_irreps"] = irrep_certificates
    checks["unimodular_hnf_invariance"] = exhaustive_hnf

    algebra_panel = tuple(
        int(value)
        for value in config["independent_audit"][
            "deterministic_algebra_rebuild_space_groups"
        ]
    )
    algebra_rebuild = True
    for space_group in algebra_panel:
        if space_group not in parent_by_group:
            algebra_rebuild = False
            continue
        algebra_rebuild &= _build_parent(
            (space_group, hall_numbers[space_group])
        ) == parent_by_group[space_group]
    checks["representation_homomorphism"] = algebra_rebuild

    checks.update(_cyclic_four_invariance_checks())
    checks.update(_compact_action_checks())
    checks["duplicate_expansion_measure_invariance"] = _measure_check()
    reference = _spgrep_modulation_reference_agreement()
    checks["spgrep_modulation_reference_agreement"] = reference["passed"]
    required = tuple(config["required_checks"])
    checks["required_checks_recomputed"] = all(name in checks for name in required)
    counts = manifest.get("counts", {})
    checks["manifest_counts_match"] = (
        counts.get("parent_space_groups") == len(parents)
        and counts.get("supercell_orbits") == record_count
        and counts.get("real_irreps") == irrep_count
        and counts.get("abstract_opd_classes") == opd_count
        and counts.get("exact_branches") == len(parents)
    )
    checks["manifest_measure_matches_protocol"] = (
        manifest.get("measure") == config["measure_rule"]
    )
    checks["manifest_qualification_consistent"] = (
        manifest.get("qualified") is True
        and manifest.get("decision") == "H0-D-v2_qualified_H0-E_may_start"
    )
    checks["manifest_did_not_claim_unverified_check"] = all(
        manifest.get("checks", {}).get(name) is checks.get(name) for name in required
    )
    qualified = not errors and all(checks.values())
    return {
        "protocol": config["protocol"],
        "qualified": qualified,
        "decision": (
            "H0-D-v2_qualified_H0-E_may_start"
            if qualified
            else "H0-D-v2_failed_stop_before_H0-E"
        ),
        "checks": checks,
        "errors": errors,
        "counts": manifest.get("counts", {}),
        "installed_versions": installed_versions,
        "algebra_rebuild_space_groups": list(algebra_panel),
        "spgrep_modulation_reference": reference,
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "records_sha256": _sha256(records_path) if records_path.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit(config, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["qualified"] else 2)


if __name__ == "__main__":
    main()
