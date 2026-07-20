"""Audit whether the qualified H0 occupational records can support assignment Q1."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _normalized_source_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _composition(tokens: torch.Tensor) -> torch.Tensor:
    return torch.bincount(tokens, minlength=CHEMICAL_ELEMENT_COUNT)


def _action_orbits(permutations: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    unseen = set(range(permutations.shape[1]))
    orbits: list[tuple[int, ...]] = []
    while unseen:
        seed = min(unseen)
        orbit = tuple(sorted(set(map(int, permutations[:, seed].tolist()))))
        unseen.difference_update(orbit)
        orbits.append(orbit)
    return tuple(orbits)


def _validate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    tokens = torch.tensor(candidate["child_atomic_numbers"], dtype=torch.long) - 1
    if bool(((tokens < 0) | (tokens >= CHEMICAL_ELEMENT_COUNT)).any()):
        raise ValueError("candidate contains a nonphysical element")
    site_class = torch.tensor(candidate["occupational_site_classes"], dtype=torch.long)
    species_by_class = torch.tensor(
        candidate["occupational_species_by_class_tokens"], dtype=torch.long
    )
    if not torch.equal(species_by_class[site_class], tokens):
        raise ValueError("occupational class reconstruction changed the target assignment")
    nodes = tokens.numel()
    permutations = torch.tensor(candidate["parent_action_permutations"], dtype=torch.long)
    if permutations.ndim != 2 or permutations.shape[1] != nodes:
        raise ValueError("parent site action has the wrong shape")
    expected = torch.arange(nodes)
    if not torch.equal(torch.sort(permutations, dim=1).values, expected.expand_as(permutations)):
        raise ValueError("parent action contains a non-permutation row")
    if not bool(torch.all(permutations == expected.unsqueeze(0), dim=1).any()):
        raise ValueError("parent action lost the identity")
    if torch.unique(permutations, dim=0).shape[0] != permutations.shape[0]:
        raise ValueError("parent action catalogue contains duplicate group elements")
    if int(candidate["child_site_count"]) != nodes:
        raise ValueError("child site count and target assignment disagree")
    if int(candidate["parent_site_count"]) * int(candidate["cell_index"]) != nodes:
        raise ValueError("parent carrier expansion does not close on the child site count")
    if not bool(candidate["exact_coloring_reconstruction"]):
        raise ValueError("H0 record did not exactly reconstruct the occupational coloring")

    counts = _composition(tokens)
    active_counts = counts[counts > 0]
    assignments = math.factorial(nodes)
    for value in active_counts.tolist():
        assignments //= math.factorial(int(value))
    orbit_labelings = torch.unique(tokens[permutations], dim=0)
    if orbit_labelings.shape[0] > assignments:
        raise RuntimeError("target quotient orbit exceeds the exact assignment space")
    action_orbits = _action_orbits(permutations)
    mixed = sum(torch.unique(tokens[list(orbit)]).numel() > 1 for orbit in action_orbits)
    state_count = int(torch.prod(active_counts + 1))
    return {
        "nodes": nodes,
        "species": int(active_counts.numel()),
        "assignment_count": assignments,
        "target_quotient_size": int(orbit_labelings.shape[0]),
        "uniform_target_quotient_probability": orbit_labelings.shape[0] / assignments,
        "uniform_quotient_nll": math.log(assignments / orbit_labelings.shape[0]),
        "carrier_orbits": len(action_orbits),
        "mixed_carrier_orbits": mixed,
        "dynamic_program_states": state_count,
        "parent_action_order": int(permutations.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--o1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_assignment_carrier_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen assignment-carrier protocol")
    source = protocol["source"]
    identities = {
        args.o1_root / "manifest.json": source["manifest_sha256"],
        args.o1_root / "results.json.gz": source["results_sha256"],
        args.o1_root / "independent_audit.json": source["independent_audit_sha256"],
    }
    for path, expected in identities.items():
        if sha256_file(path) != expected:
            raise ValueError(f"assignment-carrier source identity changed: {path}")
    if _normalized_source_sha256(Path(__file__)) != source["auditor_sha256"]:
        raise ValueError("assignment-carrier auditor source changed")
    manifest = load_json_object(args.o1_root / "manifest.json")
    independent = load_json_object(args.o1_root / "independent_audit.json")
    if not manifest.get("qualified") or not independent.get("qualified"):
        raise ValueError("H0 occupational source did not pass both audits")
    with gzip.open(args.o1_root / "results.json.gz", "rt", encoding="utf-8") as handle:
        records = json.load(handle)

    rows: list[dict[str, Any]] = []
    material_by_split: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    for record in records:
        split = str(record["gaugeflow_split"])
        if split not in material_by_split:
            raise ValueError("occupational record uses an unknown GaugeFlow split")
        for candidate in record["candidates"]:
            row = _validate_candidate(candidate)
            row["split"] = split
            row["material_id"] = str(record["material_id"])
            rows.append(row)
            material_by_split[split].add(str(record["material_id"]))
    if not rows:
        raise ValueError("occupational source contains no certified carrier candidates")

    split_candidates = Counter(str(row["split"]) for row in rows)
    material_sets = list(material_by_split.values())
    disjoint = all(
        material_sets[left].isdisjoint(material_sets[right])
        for left in range(len(material_sets))
        for right in range(left + 1, len(material_sets))
    )
    quotient_probability = torch.tensor(
        [row["uniform_target_quotient_probability"] for row in rows], dtype=torch.float64
    )
    quotient_nll = torch.tensor([row["uniform_quotient_nll"] for row in rows])
    metrics: dict[str, Any] = {
        "records": len(records),
        "candidate_carriers": len(rows),
        "candidate_carriers_by_split": dict(sorted(split_candidates.items())),
        "materials_by_split": {
            name: len(values) for name, values in material_by_split.items()
        },
        "node_count_min": min(int(row["nodes"]) for row in rows),
        "node_count_max": max(int(row["nodes"]) for row in rows),
        "species_count_max": max(int(row["species"]) for row in rows),
        "dynamic_program_states_max": max(
            int(row["dynamic_program_states"]) for row in rows
        ),
        "parent_action_order_max": max(int(row["parent_action_order"]) for row in rows),
        "target_quotient_size_max": max(int(row["target_quotient_size"]) for row in rows),
        "uniform_target_quotient_probability_median": float(quotient_probability.median()),
        "uniform_target_quotient_probability_max": float(quotient_probability.max()),
        "uniform_quotient_nll_median": float(quotient_nll.median()),
        "occupational_symmetry_breaking_fraction": sum(
            int(row["mixed_carrier_orbits"]) > 0 for row in rows
        )
        / len(rows),
        "material_split_disjoint": disjoint,
    }
    acceptance = protocol["acceptance"]
    checks = {
        "candidate_count": metrics["candidate_carriers"]
        == int(acceptance["candidate_carriers"]),
        "split_counts": metrics["candidate_carriers_by_split"]
        == acceptance["candidate_carriers_by_split"],
        "material_split_disjoint": disjoint,
        "node_bound": metrics["node_count_max"] <= int(acceptance["maximum_atoms"]),
        "species_bound": metrics["species_count_max"]
        <= int(acceptance["maximum_species"]),
        "dp_bound": metrics["dynamic_program_states_max"]
        <= int(acceptance["maximum_dynamic_program_states"]),
        "nontrivial_quotient": metrics["uniform_target_quotient_probability_median"]
        < float(acceptance["uniform_target_quotient_probability_median_max"]),
        "symmetry_breaking_present": metrics["occupational_symmetry_breaking_fraction"]
        >= float(acceptance["occupational_symmetry_breaking_fraction_min"]),
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "source_hashes": {path.name: sha256_file(path) for path in identities},
        "metrics": metrics,
        "checks": checks,
        "qualified": qualified,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "leakage_contract": {
            "allowed_inputs": [
                "predicted_or_oracle-labelled_composition_counts_with_role_flag",
                "species_free_parent_geometry",
                "parent_space_group_prior",
                "sampled_supercell_index",
                "parent_site_action",
            ],
            "target_only": [
                "child_atomic_numbers",
                "occupational_site_classes",
                "occupational_species_by_class_tokens",
                "child_space_group",
                "occupational_stabilizer_indices",
            ],
        },
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
