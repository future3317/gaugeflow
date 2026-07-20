"""Audit a relabeling-invariant DeepSet of exact parent pair orbitals."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.assignment_scorer import faithful_parent_action
from scripts.audit_h1a_assignment_global_interactions import (
    _assignment_chunks,
    _canonical_local_species,
    _relabel_action,
    _relation_histograms,
    _target_orbit,
    _unordered_pairs,
    action_pair_descriptors,
    action_pair_orbit_labels,
    action_site_signatures,
    unary_collision_class_size,
)


def _orbital_descriptor_groups(
    permutations: torch.Tensor,
    *,
    maximum_sites: int,
) -> tuple[np.ndarray, int, tuple[tuple[int, ...], ...], tuple[np.ndarray, ...]]:
    orbital_labels, orbital_count = action_pair_orbit_labels(permutations)
    pair_descriptors, _, _ = action_pair_descriptors(
        permutations, maximum_sites=maximum_sites
    )
    descriptor_by_orbital: list[tuple[int, ...]] = []
    for orbital in range(orbital_count):
        indices = np.flatnonzero(orbital_labels == orbital)
        values = {pair_descriptors[int(index)] for index in indices}
        if len(values) != 1:
            raise ValueError("target-free pair descriptor is not constant on a pair orbit")
        descriptor_by_orbital.append(values.pop())
    grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for orbital, descriptor in enumerate(descriptor_by_orbital):
        grouped[descriptor].append(orbital)
    ordered = sorted(grouped)
    groups = tuple(np.asarray(grouped[key], dtype=np.int64) for key in ordered)
    return orbital_labels, orbital_count, tuple(ordered), groups


def _void_rows(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * values.shape[-1]))).reshape(
        values.shape[:-1]
    )


def _orbital_set_matches(
    histograms: np.ndarray,
    target: np.ndarray,
    groups: Sequence[np.ndarray],
) -> np.ndarray:
    matches = np.ones(histograms.shape[0], dtype=bool)
    for group in groups:
        observed = np.sort(_void_rows(histograms[:, group, :]), axis=1)
        expected = np.sort(_void_rows(target[group, :]), axis=0)
        matches &= np.all(observed == expected[None, :], axis=1)
    return matches


def orbital_set_signature(
    assignment: Sequence[int],
    permutations: torch.Tensor,
    *,
    maximum_sites: int = 20,
) -> tuple[tuple[tuple[int, ...], tuple[bytes, ...]], ...]:
    """Return the exact relabeling-invariant colored-orbital multiset signature."""
    target = _canonical_local_species(assignment)
    pair_indices = np.asarray(_unordered_pairs(target.size), dtype=np.int64)
    orbital_labels, orbital_count, descriptors, groups = _orbital_descriptor_groups(
        permutations, maximum_sites=maximum_sites
    )
    species = int(np.unique(target).size)
    histogram = _relation_histograms(
        target[None, :], pair_indices, orbital_labels, orbital_count, species
    ).reshape(orbital_count, species * species)
    output: list[tuple[tuple[int, ...], tuple[bytes, ...]]] = []
    for descriptor, group in zip(descriptors, groups, strict=True):
        rows = sorted(np.ascontiguousarray(histogram[index]).tobytes() for index in group)
        output.append((descriptor, tuple(rows)))
    return tuple(output)


def audit_orbital_set_carrier(
    assignment: Sequence[int],
    permutations: torch.Tensor,
    *,
    maximum_sites: int,
    maximum_collision_class: int,
    chunk_size: int,
) -> dict[str, Any]:
    action = faithful_parent_action(permutations).detach().cpu()
    target = _canonical_local_species(assignment)
    site_signatures = action_site_signatures(action, maximum_sites=maximum_sites)
    unary_size = unary_collision_class_size(target.tolist(), site_signatures)
    target_orbit = _target_orbit(target, action)
    base = {
        "unary_collision_class_size": unary_size,
        "target_orbit_size": len(target_orbit),
        "unary_non_orbit_collision": unary_size > len(target_orbit),
        "exact_enumerated": unary_size <= maximum_collision_class,
    }
    if not base["exact_enumerated"]:
        return base
    pair_indices = np.asarray(_unordered_pairs(target.size), dtype=np.int64)
    orbital_labels, orbital_count, _, groups = _orbital_descriptor_groups(
        action, maximum_sites=maximum_sites
    )
    species = int(np.unique(target).size)
    target_histogram = _relation_histograms(
        target[None, :], pair_indices, orbital_labels, orbital_count, species
    ).reshape(orbital_count, species * species)
    enumerated = 0
    matches = 0
    target_members = 0
    for chunk in _assignment_chunks(target, site_signatures, chunk_size=chunk_size):
        enumerated += chunk.shape[0]
        histograms = _relation_histograms(
            chunk, pair_indices, orbital_labels, orbital_count, species
        ).reshape(chunk.shape[0], orbital_count, species * species)
        equal = _orbital_set_matches(histograms, target_histogram, groups)
        matches += int(equal.sum())
        for value in chunk[equal]:
            target_members += int(np.ascontiguousarray(value).tobytes() in target_orbit)
    if enumerated != unary_size:
        raise AssertionError("unary collision enumeration was incomplete")
    return {
        **base,
        "enumerated_assignments": enumerated,
        "pair_orbit_count": orbital_count,
        "orbital_set_collision_class_size": matches,
        "orbital_set_target_ceiling": len(target_orbit) / matches,
        "orbital_set_resolved": matches == len(target_orbit),
        "target_orbit_containment_failure": target_members != len(target_orbit),
    }


def relabel_orbital_set_check(
    assignment: Sequence[int],
    permutations: torch.Tensor,
    *,
    seed: int,
) -> bool:
    generator = torch.Generator().manual_seed(seed)
    relabel = torch.randperm(len(assignment), generator=generator)
    transformed_action = _relabel_action(permutations, relabel)
    transformed_assignment = torch.as_tensor(assignment, dtype=torch.long)[relabel].tolist()
    return orbital_set_signature(assignment, permutations) == orbital_set_signature(
        transformed_assignment, transformed_action
    )


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exact = [row for row in rows if row["exact_enumerated"]]
    collisions = [row for row in exact if row["unary_non_orbit_collision"]]
    return {
        "carriers": len(rows),
        "materials": len({row["material_id"] for row in rows}),
        "exact_enumerated_carrier_fraction": len(exact) / len(rows),
        "unary_collision_carriers_exact": len(collisions),
        "orbital_set_resolved_unary_collision_fraction": (
            sum(bool(row["orbital_set_resolved"]) for row in collisions) / len(collisions)
            if collisions
            else 1.0
        ),
        "orbital_set_mean_target_ceiling": (
            sum(float(row["orbital_set_target_ceiling"]) for row in collisions)
            / len(collisions)
            if collisions
            else 1.0
        ),
        "relabel_failures": sum(not bool(row["relabel_invariant"]) for row in rows),
        "target_orbit_containment_failures": sum(
            bool(row.get("target_orbit_containment_failure", False)) for row in exact
        ),
    }


def _validate_protocol(protocol: dict[str, Any], repository: Path, o1_root: Path) -> None:
    if (
        protocol.get("protocol") != "h1a_assignment_orbital_set_expressivity_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
        or protocol["scientific_contract"].get("no_training") is not True
    ):
        raise ValueError("unexpected or unfrozen orbital-set protocol")
    prerequisite = repository / "reports/h1a_assignment_global_interaction_audit_v1/result.json"
    if sha256_file(prerequisite).lower() != str(
        protocol["prerequisites"]["global_interaction_result_sha256"]
    ).lower():
        raise ValueError("global-interaction prerequisite identity changed")
    result = load_json_object(prerequisite)
    if result.get("decision_class") != protocol["prerequisites"]["required_decision_class"]:
        raise ValueError("orbital-set audit lacks the required orbital-only result")
    for name, expected in protocol["source"]["artifact_sha256"].items():
        if sha256_file(o1_root / name) != expected:
            raise ValueError(f"assignment source identity changed: {name}")


def _write_readme(path: Path, result: dict[str, Any]) -> None:
    metric = result["metrics"]["all"]
    lines = [
        "# H1a assignment orbital-set expressivity audit v1",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| exact enumeration coverage | {metric['exact_enumerated_carrier_fraction']:.6f} |",
        f"| exact unary-collision carriers | {metric['unary_collision_carriers_exact']} |",
        "| orbital-set resolved fraction | "
        f"{metric['orbital_set_resolved_unary_collision_fraction']:.6f} |",
        f"| orbital-set mean target ceiling | {metric['orbital_set_mean_target_ceiling']:.6f} |",
        f"| relabel failures | {metric['relabel_failures']} |",
        "",
        "The representation keeps exact pair orbitals as an unordered set and",
        "uses no orbit index. This audit is about identifiability only; pairwise",
        "global normalization remains a separate blocker before training.",
        "",
        f"Boundary: {result['boundary']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--o1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    _validate_protocol(protocol, repository, args.o1_root)
    with gzip.open(args.o1_root / "results.json.gz", "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    audit = protocol["audit"]
    rows: list[dict[str, Any]] = []
    for record in source:
        for candidate_index, candidate in enumerate(record["candidates"]):
            action = torch.tensor(candidate["parent_action_permutations"], dtype=torch.long)
            result = audit_orbital_set_carrier(
                candidate["child_atomic_numbers"],
                action,
                maximum_sites=20,
                maximum_collision_class=int(audit["maximum_exact_unary_collision_class"]),
                chunk_size=int(audit["assignment_chunk_size"]),
            )
            result.update(
                {
                    "material_id": str(record["material_id"]),
                    "original_split": str(record["gaugeflow_split"]),
                    "candidate_index": candidate_index,
                    "embedding_key": str(candidate["embedding_key"]),
                    "relabel_invariant": relabel_orbital_set_check(
                        candidate["child_atomic_numbers"],
                        action,
                        seed=int(audit["relabel_seed"]) + len(rows),
                    ),
                }
            )
            rows.append(result)
    observed = Counter(row["original_split"] for row in rows)
    if (
        len(rows) != int(protocol["source"]["candidate_carriers"])
        or dict(observed) != protocol["source"]["candidate_carriers_by_split"]
    ):
        raise ValueError("assignment carrier source counts changed")
    metrics = {"all": _summarize(rows)}
    for split in ("train", "val", "test"):
        metrics[split] = _summarize([row for row in rows if row["original_split"] == split])
    all_metrics = metrics["all"]
    acceptance = protocol["acceptance"]
    checks = {
        "exact_enumeration_coverage": all_metrics["exact_enumerated_carrier_fraction"]
        >= float(acceptance["exact_enumerated_carrier_fraction_min"]),
        "relabel_invariance": all_metrics["relabel_failures"]
        == int(acceptance["relabel_failures"]),
        "target_orbit_containment": all_metrics["target_orbit_containment_failures"]
        == int(acceptance["target_orbit_containment_failures"]),
        "orbital_set_resolution": all_metrics[
            "orbital_set_resolved_unary_collision_fraction"
        ]
        >= float(acceptance["orbital_set_resolved_unary_collision_fraction_min"]),
        "orbital_set_ceiling": all_metrics["orbital_set_mean_target_ceiling"]
        >= float(acceptance["orbital_set_mean_target_ceiling_min"]),
    }
    qualified = all(checks.values())
    payload = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "qualified": qualified,
        "checks": checks,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
        "metrics": metrics,
        "carrier_rows": rows,
        "source_hashes": protocol["source"]["artifact_sha256"],
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_readme(args.output_dir / "README.md", payload)
    print(json.dumps({key: payload[key] for key in ("qualified", "checks", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
