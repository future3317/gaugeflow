"""Build an assignment IID calibration split without consuming the OOD panels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from scripts.audit_h1a_assignment_global_interactions import (
    action_pair_descriptors,
    action_site_signatures,
)


def composition_partition(candidate: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    tokens = list(map(int, candidate["child_atomic_numbers"]))
    return len(tokens), tuple(sorted(Counter(tokens).values(), reverse=True))


def assignment_action_signature(candidate: dict[str, Any]) -> str:
    """Hash a target-free relabeling-invariant action/carrier feature signature."""
    action = torch.tensor(candidate["parent_action_permutations"], dtype=torch.long)
    site = sorted(action_site_signatures(action))
    pair, _, _ = action_pair_descriptors(action)
    payload = {
        "parent_space_group": int(candidate["parent_space_group"]),
        "cell_index": int(candidate["cell_index"]),
        "site_signatures": site,
        "pair_descriptors": sorted(pair),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tie_hash(seed: int, partition: tuple[int, tuple[int, ...]], material_id: str) -> str:
    return hashlib.sha256(f"{seed}|{partition}|{material_id}".encode()).hexdigest()


def assign_iid_roles(
    material_partition: dict[str, tuple[int, tuple[int, ...]]],
    *,
    seed: int,
    holdout_fraction: float,
    minimum_partition_materials: int,
) -> dict[str, str]:
    """Stratify original-train materials with exact fit support for both IID panels."""
    if not 0.0 < holdout_fraction < 0.5 or minimum_partition_materials < 3:
        raise ValueError("invalid IID split bounds")
    grouped: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for material_id, partition in material_partition.items():
        grouped[partition].append(material_id)
    roles: dict[str, str] = {}
    for partition, values in grouped.items():
        ordered = sorted(values, key=lambda value: _tie_hash(seed, partition, value))
        if len(ordered) < minimum_partition_materials:
            roles.update({value: "iid_fit_rare" for value in ordered})
            continue
        holdout = max(1, round(holdout_fraction * len(ordered)))
        holdout = min(holdout, (len(ordered) - 1) // 2)
        roles.update({value: "iid_test" for value in ordered[:holdout]})
        roles.update({value: "iid_calibration" for value in ordered[holdout : 2 * holdout]})
        roles.update({value: "iid_fit" for value in ordered[2 * holdout :]})
    return roles


def duplicate_role_overlap(
    records: Sequence[dict[str, Any]],
    material_roles: dict[str, str],
) -> int:
    grouped: dict[tuple[str, tuple[int, ...]], set[str]] = defaultdict(set)
    for record in records:
        for candidate in record["candidates"]:
            key = (
                str(candidate["embedding_key"]),
                tuple(map(int, candidate["child_atomic_numbers"])),
            )
            grouped[key].add(material_roles[str(record["material_id"])])
    return sum(len(roles) > 1 for roles in grouped.values())


def _support_fraction(
    rows: Iterable[dict[str, Any]],
    fit_partitions: set[tuple[int, tuple[int, ...]]],
) -> float:
    values = list(rows)
    if not values:
        return 0.0
    return sum(
        (int(row["partition"][0]), tuple(map(int, row["partition"][1]))) in fit_partitions
        for row in values
    ) / len(values)


def _validate_protocol(protocol: dict[str, Any], o1_root: Path) -> None:
    if (
        protocol.get("protocol") != "h1a_assignment_iid_calibration_split_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen assignment IID split protocol")
    for name, expected in protocol["source"]["artifact_sha256"].items():
        if sha256_file(o1_root / name) != expected:
            raise ValueError(f"assignment source identity changed: {name}")


def _write_readme(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    lines = [
        "# H1a assignment IID calibration split v1",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "The IID calibration/test rows are drawn only from the original GaugeFlow",
        "train partition. Original validation and test remain untouched OOD stress",
        "panels, so one checkpoint can report IID calibration and OOD generalization",
        "without mixing their scientific meanings.",
        "",
        "| role | materials | carriers |",
        "|---|---:|---:|",
    ]
    for role in (
        "iid_fit",
        "iid_fit_rare",
        "iid_calibration",
        "iid_test",
        "ood_validation",
        "ood_test",
    ):
        lines.append(
            f"| {role} | {metrics['materials_by_role'].get(role, 0)} | "
            f"{metrics['carriers_by_role'].get(role, 0)} |"
        )
    lines.extend(
        [
            "",
            f"IID calibration partition support: `{metrics['iid_calibration_partition_fit_support']:.6f}`.",
            f"IID test partition support: `{metrics['iid_test_partition_fit_support']:.6f}`.",
            f"Exact input-output duplicate role overlap: `{metrics['exact_input_output_duplicate_role_overlap']}`.",
            "",
            "Formula/prototype overlap inside these IID roles is intentional; it tests",
            "calibration on supported contexts. Formula/prototype-disjoint evidence is",
            "reported only on the untouched original OOD panels.",
            "",
            f"Boundary: {result['boundary']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--o1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    _validate_protocol(protocol, args.o1_root)
    with gzip.open(args.o1_root / "results.json.gz", "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    original_counts = Counter(
        str(record["gaugeflow_split"])
        for record in records
        for _ in record["candidates"]
    )
    if dict(original_counts) != protocol["source"]["original_carriers_by_split"]:
        raise ValueError("original assignment split carrier counts changed")
    material_partition: dict[str, tuple[int, tuple[int, ...]]] = {}
    for record in records:
        if record["gaugeflow_split"] != "train" or not record["candidates"]:
            continue
        partitions = {composition_partition(candidate) for candidate in record["candidates"]}
        if len(partitions) != 1:
            raise ValueError("one material spans multiple assignment composition partitions")
        material_partition[str(record["material_id"])] = partitions.pop()
    split = protocol["split"]
    material_roles = assign_iid_roles(
        material_partition,
        seed=int(split["seed"]),
        holdout_fraction=float(split["iid_holdout_fraction_per_supported_partition"]),
        minimum_partition_materials=int(split["minimum_materials_per_iid_partition"]),
    )
    for record in records:
        if not record["candidates"]:
            continue
        material_id = str(record["material_id"])
        original = str(record["gaugeflow_split"])
        if original == "val":
            material_roles[material_id] = "ood_validation"
        elif original == "test":
            material_roles[material_id] = "ood_test"
    carrier_rows: list[dict[str, Any]] = []
    for record in records:
        if not record["candidates"]:
            continue
        material_id = str(record["material_id"])
        role = material_roles[material_id]
        for candidate_index, candidate in enumerate(record["candidates"]):
            partition = composition_partition(candidate)
            carrier_rows.append(
                {
                    "material_id": material_id,
                    "candidate_index": candidate_index,
                    "role": role,
                    "original_split": str(record["gaugeflow_split"]),
                    "partition": [partition[0], list(partition[1])],
                    "embedding_key": str(candidate["embedding_key"]),
                    "action_signature": assignment_action_signature(candidate),
                }
            )
    fit_roles = {"iid_fit", "iid_fit_rare"}
    fit_partitions = {
        (int(row["partition"][0]), tuple(map(int, row["partition"][1])))
        for row in carrier_rows
        if row["role"] in fit_roles
    }
    by_role = Counter(row["role"] for row in carrier_rows)
    material_by_role = Counter(material_roles.values())
    calibration = [row for row in carrier_rows if row["role"] == "iid_calibration"]
    iid_test = [row for row in carrier_rows if row["role"] == "iid_test"]
    calibration_support = _support_fraction(calibration, fit_partitions)
    test_support = _support_fraction(iid_test, fit_partitions)
    duplicate_overlap = duplicate_role_overlap(records, material_roles)
    fit_action = {
        row["action_signature"] for row in carrier_rows if row["role"] in fit_roles
    }
    metrics = {
        "materials_by_role": dict(sorted(material_by_role.items())),
        "carriers_by_role": dict(sorted(by_role.items())),
        "iid_calibration_partition_fit_support": calibration_support,
        "iid_test_partition_fit_support": test_support,
        "iid_calibration_action_signature_fit_support": sum(
            row["action_signature"] in fit_action for row in calibration
        )
        / len(calibration),
        "iid_test_action_signature_fit_support": sum(
            row["action_signature"] in fit_action for row in iid_test
        )
        / len(iid_test),
        "exact_input_output_duplicate_role_overlap": duplicate_overlap,
        "original_ood_rows_used_for_iid_fit": sum(
            row["original_split"] != "train" and row["role"] in fit_roles
            for row in carrier_rows
        ),
    }
    acceptance = protocol["acceptance"]
    checks = {
        "all_source_carriers_assigned_once": len(carrier_rows)
        == int(protocol["source"]["candidate_carriers"]),
        "material_role_overlap": len(material_roles)
        == int(protocol["source"]["materials"]),
        "exact_input_output_duplicate_disjoint": duplicate_overlap
        == int(acceptance["exact_input_output_duplicate_role_overlap"]),
        "iid_calibration_partition_support": calibration_support
        == float(acceptance["iid_calibration_partition_fit_support"]),
        "iid_test_partition_support": test_support
        == float(acceptance["iid_test_partition_fit_support"]),
        "iid_calibration_size": material_by_role["iid_calibration"]
        >= int(acceptance["iid_calibration_materials_min"]),
        "iid_test_size": material_by_role["iid_test"]
        >= int(acceptance["iid_test_materials_min"]),
        "ood_panels_untouched": (
            by_role["ood_validation"] == int(acceptance["ood_validation_carriers"])
            and by_role["ood_test"] == int(acceptance["ood_test_carriers"])
            and metrics["original_ood_rows_used_for_iid_fit"]
            == int(acceptance["original_ood_rows_used_for_iid_fit"])
        ),
    }
    qualified = all(checks.values())
    assignments = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "material_roles": dict(sorted(material_roles.items())),
    }
    payload = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "qualified": qualified,
        "checks": checks,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
        "metrics": metrics,
        "carrier_rows": carrier_rows,
        "source_hashes": protocol["source"]["artifact_sha256"],
        "assignments_sha256": canonical_json_hash(assignments),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "assignments.json").write_text(
        json.dumps(assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_readme(args.output_dir / "README.md", payload)
    print(json.dumps({key: payload[key] for key in ("qualified", "checks", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
