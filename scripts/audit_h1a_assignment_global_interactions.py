"""Audit whether target-free pair interactions break the failed Q1 unary collisions."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from gaugeflow.file_utils import (
    canonical_json_hash,
    load_json_object,
    sha256_file,
)
from gaugeflow.production.assignment_scorer import faithful_parent_action

Signature = tuple[int, ...]


def _cycle_mass_histogram(permutations: torch.Tensor, maximum_sites: int) -> tuple[int, ...]:
    histogram = [0] * maximum_sites
    for row in permutations.tolist():
        unseen = set(range(len(row)))
        while unseen:
            current = min(unseen)
            length = 0
            while current in unseen:
                unseen.remove(current)
                length += 1
                current = row[current]
            histogram[length - 1] += length
    return tuple(histogram)


def action_site_signatures(
    permutations: torch.Tensor,
    *,
    maximum_sites: int = 20,
) -> tuple[Signature, ...]:
    """Return exact integer counterparts of the active Q1 unary site features."""
    image = faithful_parent_action(permutations).detach().cpu()
    operations, sites = image.shape
    if sites > maximum_sites:
        raise ValueError("parent action exceeds the qualified site bound")
    site_ids = torch.arange(sites)
    signatures: list[Signature] = []
    for site in range(sites):
        orbit_size = int(torch.unique(image[:, site]).numel())
        stabilizer = image[image[:, site] == site]
        if stabilizer.shape[0] * orbit_size != operations:
            raise ValueError("parent action violates orbit-stabilizer closure")
        suborbit = Counter(int(torch.unique(stabilizer[:, other]).numel()) for other in range(sites))
        fixed = Counter(int((row == site_ids).sum()) for row in stabilizer)
        signatures.append(
            (
                orbit_size,
                int(stabilizer.shape[0]),
                operations,
                sites,
                *(suborbit.get(size, 0) for size in range(1, maximum_sites + 1)),
                *(fixed.get(size, 0) for size in range(1, maximum_sites + 1)),
                *_cycle_mass_histogram(stabilizer, maximum_sites),
            )
        )
    return tuple(signatures)


def _unordered_pairs(sites: int) -> tuple[tuple[int, int], ...]:
    return tuple(itertools.combinations(range(sites), 2))


def _pair_image(pair: tuple[int, int], image: torch.Tensor) -> frozenset[tuple[int, int]]:
    left = image[:, pair[0]].tolist()
    right = image[:, pair[1]].tolist()
    return frozenset((min(i, j), max(i, j)) for i, j in zip(left, right, strict=True))


def action_pair_orbit_labels(permutations: torch.Tensor) -> tuple[np.ndarray, int]:
    """Label exact unordered-pair orbits; labels are used only as an upper bound."""
    image = faithful_parent_action(permutations).detach().cpu()
    pairs = _unordered_pairs(image.shape[1])
    orbit_to_label: dict[frozenset[tuple[int, int]], int] = {}
    labels: list[int] = []
    for pair in pairs:
        orbit = _pair_image(pair, image)
        if orbit not in orbit_to_label:
            orbit_to_label[orbit] = len(orbit_to_label)
        labels.append(orbit_to_label[orbit])
    return np.asarray(labels, dtype=np.int64), len(orbit_to_label)


def action_pair_descriptors(
    permutations: torch.Tensor,
    *,
    maximum_sites: int = 20,
) -> tuple[tuple[Signature, ...], np.ndarray, int]:
    """Construct target-free relabeling-invariant descriptors of unordered pairs."""
    image = faithful_parent_action(permutations).detach().cpu()
    operations, sites = image.shape
    site_signatures = action_site_signatures(image, maximum_sites=maximum_sites)
    site_ids = torch.arange(sites)
    descriptors: list[Signature] = []
    for left, right in _unordered_pairs(sites):
        pair_orbit = _pair_image((left, right), image)
        maps_left = image[:, left]
        maps_right = image[:, right]
        setwise = image[((maps_left == left) & (maps_right == right)) | ((maps_left == right) & (maps_right == left))]
        pointwise = int(((maps_left == left) & (maps_right == right)).sum())
        swaps = int(((maps_left == right) & (maps_right == left)).sum())
        if setwise.shape[0] * len(pair_orbit) != operations:
            raise ValueError("unordered-pair action violates orbit-stabilizer closure")
        suborbit = Counter(int(torch.unique(setwise[:, other]).numel()) for other in range(sites))
        fixed = Counter(int((row == site_ids).sum()) for row in setwise)
        endpoints = sorted((site_signatures[left], site_signatures[right]))
        same_site_orbit = int(right in torch.unique(image[:, left]).tolist())
        descriptors.append(
            (
                *endpoints[0],
                *endpoints[1],
                same_site_orbit,
                len(pair_orbit),
                int(setwise.shape[0]),
                pointwise,
                swaps,
                *(suborbit.get(size, 0) for size in range(1, maximum_sites + 1)),
                *(fixed.get(size, 0) for size in range(0, maximum_sites + 1)),
                *_cycle_mass_histogram(setwise, maximum_sites),
            )
        )
    unique = {value: index for index, value in enumerate(sorted(set(descriptors)))}
    labels = np.asarray([unique[value] for value in descriptors], dtype=np.int64)
    return tuple(descriptors), labels, len(unique)


def _multinomial_count(values: Sequence[int]) -> int:
    result = math.factorial(len(values))
    for count in Counter(values).values():
        result //= math.factorial(count)
    return result


def _unique_multiset_permutations(values: Sequence[int]) -> Iterator[tuple[int, ...]]:
    counts = Counter(map(int, values))
    ordered = sorted(counts)
    output = [0] * len(values)

    def visit(position: int) -> Iterator[tuple[int, ...]]:
        if position == len(output):
            yield tuple(output)
            return
        for value in ordered:
            if counts[value] == 0:
                continue
            counts[value] -= 1
            output[position] = value
            yield from visit(position + 1)
            counts[value] += 1

    yield from visit(0)


def unary_collision_class_size(
    assignment: Sequence[int],
    site_signatures: Sequence[Signature],
) -> int:
    grouped: dict[Signature, list[int]] = defaultdict(list)
    for token, signature in zip(assignment, site_signatures, strict=True):
        grouped[signature].append(int(token))
    return math.prod(_multinomial_count(values) for values in grouped.values())


def _assignment_chunks(
    assignment: np.ndarray,
    site_signatures: Sequence[Signature],
    *,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    grouped: dict[Signature, list[int]] = defaultdict(list)
    for site, signature in enumerate(site_signatures):
        grouped[signature].append(site)
    groups = [np.asarray(grouped[key], dtype=np.int64) for key in sorted(grouped)]
    options = [tuple(_unique_multiset_permutations(assignment[index].tolist())) for index in groups]
    buffer: list[np.ndarray] = []
    for combination in itertools.product(*options):
        value = assignment.copy()
        for index, tokens in zip(groups, combination, strict=True):
            value[index] = tokens
        buffer.append(value)
        if len(buffer) == chunk_size:
            yield np.stack(buffer)
            buffer.clear()
    if buffer:
        yield np.stack(buffer)


def _relation_histograms(
    assignments: np.ndarray,
    pair_indices: np.ndarray,
    relation_labels: np.ndarray,
    relation_count: int,
    species_count: int,
) -> np.ndarray:
    left = assignments[:, pair_indices[:, 0]]
    right = assignments[:, pair_indices[:, 1]]
    code = np.minimum(left, right) * species_count + np.maximum(left, right)
    combined = relation_labels[None, :] * (species_count * species_count) + code
    width = relation_count * species_count * species_count
    histogram = np.zeros((assignments.shape[0], width), dtype=np.int16)
    row = np.repeat(np.arange(assignments.shape[0]), pair_indices.shape[0])
    np.add.at(histogram, (row, combined.reshape(-1)), 1)
    return histogram


def _relation_signature(
    assignment: np.ndarray,
    pair_indices: np.ndarray,
    relation_labels: np.ndarray,
    relation_count: int,
    species_count: int,
) -> np.ndarray:
    return _relation_histograms(
        assignment[None, :],
        pair_indices,
        relation_labels,
        relation_count,
        species_count,
    )[0]


def _canonical_local_species(assignment: Sequence[int]) -> np.ndarray:
    active = {value: index for index, value in enumerate(sorted(set(map(int, assignment))))}
    return np.asarray([active[int(value)] for value in assignment], dtype=np.int16)


def _target_orbit(assignment: np.ndarray, permutations: torch.Tensor) -> set[bytes]:
    action = faithful_parent_action(permutations).detach().cpu().numpy()
    return {np.ascontiguousarray(assignment[row]).tobytes() for row in action}


def audit_carrier(
    assignment: Sequence[int],
    permutations: torch.Tensor,
    *,
    maximum_sites: int,
    maximum_collision_class: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Exactly audit one target's current-unary collision class."""
    action = faithful_parent_action(permutations).detach().cpu()
    target = _canonical_local_species(assignment)
    sites = target.size
    pair_indices = np.asarray(_unordered_pairs(sites), dtype=np.int64)
    site_signatures = action_site_signatures(action, maximum_sites=maximum_sites)
    unary_size = unary_collision_class_size(target.tolist(), site_signatures)
    target_orbit = _target_orbit(target, action)
    if unary_size < len(target_orbit):
        raise ValueError("target orbit escaped the current unary collision class")
    base = {
        "site_count": sites,
        "species_count": int(np.unique(target).size),
        "action_order": int(action.shape[0]),
        "unary_collision_class_size": unary_size,
        "target_orbit_size": len(target_orbit),
        "unary_target_ceiling": len(target_orbit) / unary_size,
        "unary_non_orbit_collision": unary_size > len(target_orbit),
        "exact_enumerated": unary_size <= maximum_collision_class,
    }
    if not base["exact_enumerated"]:
        return base

    orbital_labels, orbital_count = action_pair_orbit_labels(action)
    _, descriptor_labels, descriptor_count = action_pair_descriptors(action, maximum_sites=maximum_sites)
    species = int(np.unique(target).size)
    orbital_target = _relation_signature(target, pair_indices, orbital_labels, orbital_count, species)
    descriptor_target = _relation_signature(target, pair_indices, descriptor_labels, descriptor_count, species)
    enumerated = 0
    orbital_matches = 0
    descriptor_matches = 0
    orbital_target_members = 0
    descriptor_target_members = 0
    for chunk in _assignment_chunks(target, site_signatures, chunk_size=chunk_size):
        enumerated += chunk.shape[0]
        orbital_equal = np.all(
            _relation_histograms(chunk, pair_indices, orbital_labels, orbital_count, species) == orbital_target,
            axis=1,
        )
        descriptor_equal = np.all(
            _relation_histograms(chunk, pair_indices, descriptor_labels, descriptor_count, species)
            == descriptor_target,
            axis=1,
        )
        orbital_matches += int(orbital_equal.sum())
        descriptor_matches += int(descriptor_equal.sum())
        for value, is_orbital, is_descriptor in zip(chunk, orbital_equal, descriptor_equal, strict=True):
            if not (is_orbital or is_descriptor):
                continue
            is_target = np.ascontiguousarray(value).tobytes() in target_orbit
            orbital_target_members += int(is_orbital and is_target)
            descriptor_target_members += int(is_descriptor and is_target)
    if enumerated != unary_size:
        raise AssertionError("unary collision enumeration was incomplete")
    containment_failure = orbital_target_members != len(target_orbit) or descriptor_target_members != len(target_orbit)
    return {
        **base,
        "enumerated_assignments": enumerated,
        "pair_orbit_count": orbital_count,
        "pair_descriptor_count": descriptor_count,
        "orbital_pair_collision_class_size": orbital_matches,
        "orbital_pair_target_ceiling": len(target_orbit) / orbital_matches,
        "orbital_pair_resolved": orbital_matches == len(target_orbit),
        "transferable_pair_collision_class_size": descriptor_matches,
        "transferable_pair_target_ceiling": len(target_orbit) / descriptor_matches,
        "transferable_pair_resolved": descriptor_matches == len(target_orbit),
        "target_orbit_containment_failure": containment_failure,
    }


def _relabel_action(action: torch.Tensor, relabel: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(relabel)
    inverse[relabel] = torch.arange(relabel.numel())
    return inverse[action[:, relabel]]


def relabel_invariance_check(
    assignment: Sequence[int],
    permutations: torch.Tensor,
    *,
    seed: int,
    maximum_sites: int,
) -> bool:
    """Check the target-free descriptor relation under an arbitrary node relabeling."""
    generator = torch.Generator().manual_seed(seed)
    relabel = torch.randperm(len(assignment), generator=generator)
    transformed_action = _relabel_action(permutations, relabel)
    original_descriptors, _, _ = action_pair_descriptors(permutations, maximum_sites=maximum_sites)
    transformed_descriptors, _, _ = action_pair_descriptors(transformed_action, maximum_sites=maximum_sites)
    pairs = _unordered_pairs(len(assignment))
    original = {(pair, descriptor) for pair, descriptor in zip(pairs, original_descriptors, strict=True)}
    inverse = torch.empty_like(relabel)
    inverse[relabel] = torch.arange(relabel.numel())
    mapped: set[tuple[tuple[int, int], Signature]] = set()
    for pair, descriptor in zip(pairs, transformed_descriptors, strict=True):
        old = sorted((int(relabel[pair[0]]), int(relabel[pair[1]])))
        mapped.add(((old[0], old[1]), descriptor))
    del inverse
    return mapped == original


def _validate_protocol(protocol: dict[str, Any], repository: Path, o1_root: Path) -> None:
    if (
        protocol.get("protocol") != "h1a_assignment_global_interaction_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
        or protocol["scientific_contract"].get("no_training") is not True
    ):
        raise ValueError("unexpected or unfrozen global-interaction audit protocol")
    prerequisites = {
        "q1_result_sha256": repository / "reports/h1a_oracle_c_assignment_q1_v1/result.json",
        "q1_failure_analysis_sha256": repository
        / "reports/h1a_oracle_c_assignment_q1_v1/independent_failure_analysis.json",
    }
    for key, path in prerequisites.items():
        if sha256_file(path) != protocol["prerequisites"][key]:
            raise ValueError(f"assignment audit prerequisite changed: {path}")
    for name, expected in protocol["source"]["artifact_sha256"].items():
        if sha256_file(o1_root / name) != expected:
            raise ValueError(f"assignment source identity changed: {name}")


def _material_mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["material_id"])].append(float(row[key]))
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exact = [row for row in rows if row["exact_enumerated"]]
    collisions = [row for row in exact if row["unary_non_orbit_collision"]]
    return {
        "carriers": len(rows),
        "materials": len({row["material_id"] for row in rows}),
        "exact_enumerated_carriers": len(exact),
        "exact_enumerated_carrier_fraction": len(exact) / len(rows),
        "unary_collision_carriers_exact": len(collisions),
        "orbital_pair_resolved_unary_collision_fraction": (
            sum(bool(row["orbital_pair_resolved"]) for row in collisions) / len(collisions) if collisions else 1.0
        ),
        "transferable_pair_resolved_unary_collision_fraction": (
            sum(bool(row["transferable_pair_resolved"]) for row in collisions) / len(collisions) if collisions else 1.0
        ),
        "transferable_pair_mean_target_ceiling": (
            sum(float(row["transferable_pair_target_ceiling"]) for row in collisions) / len(collisions)
            if collisions
            else 1.0
        ),
        "transferable_pair_material_mean_target_ceiling": (
            _material_mean(collisions, "transferable_pair_target_ceiling") if collisions else 1.0
        ),
        "target_orbit_containment_failures": sum(
            bool(row.get("target_orbit_containment_failure", False)) for row in exact
        ),
        "relabel_failures": sum(not bool(row["relabel_invariant"]) for row in rows),
    }


def _write_readme(path: Path, result: dict[str, Any]) -> None:
    metric = result["metrics"]["all"]
    lines = [
        "# H1a assignment global-interaction audit v1",
        "",
        "This is a zero-training expressivity audit of the frozen failed Q1 unary",
        "assignment family. It does not qualify assignment or authorize a successor",
        "training run by itself.",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| exact enumeration coverage | {metric['exact_enumerated_carrier_fraction']:.6f} |",
        f"| exact unary-collision carriers | {metric['unary_collision_carriers_exact']} |",
        f"| exact pair-orbital resolved fraction | {metric['orbital_pair_resolved_unary_collision_fraction']:.6f} |",
        "| transferable pair resolved fraction | "
        f"{metric['transferable_pair_resolved_unary_collision_fraction']:.6f} |",
        f"| transferable pair mean target ceiling | {metric['transferable_pair_mean_target_ceiling']:.6f} |",
        f"| relabel failures | {metric['relabel_failures']} |",
        f"| target-orbit containment failures | {metric['target_orbit_containment_failures']} |",
        "",
        "The exact pair-orbital statistic is explicitly an upper bound. It may use",
        "the carrier's complete pair-orbit partition but never a prototype ID. The",
        "transferable statistic merges pair orbits whenever their target-free action",
        "descriptors coincide; only that family is eligible to motivate a shared scorer.",
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
    relabel_seed = int(audit["relabel_seed"])
    for record in source:
        for candidate_index, candidate in enumerate(record["candidates"]):
            action = torch.tensor(candidate["parent_action_permutations"], dtype=torch.long)
            result = audit_carrier(
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
                    "relabel_invariant": relabel_invariance_check(
                        candidate["child_atomic_numbers"],
                        action,
                        seed=relabel_seed + len(rows),
                        maximum_sites=20,
                    ),
                }
            )
            rows.append(result)
    expected = protocol["source"]["candidate_carriers_by_split"]
    observed = Counter(row["original_split"] for row in rows)
    if len(rows) != int(protocol["source"]["candidate_carriers"]) or dict(observed) != expected:
        raise ValueError("assignment carrier source counts changed")
    metrics = {"all": _summarize(rows)}
    for split in ("train", "val", "test"):
        metrics[split] = _summarize([row for row in rows if row["original_split"] == split])
    acceptance = protocol["acceptance"]
    all_metrics = metrics["all"]
    checks = {
        "exact_enumeration_coverage": all_metrics["exact_enumerated_carrier_fraction"]
        >= float(acceptance["exact_enumerated_carrier_fraction_min"]),
        "relabel_invariance": all_metrics["relabel_failures"] == int(acceptance["invariant_relabel_failures"]),
        "target_orbit_containment": all_metrics["target_orbit_containment_failures"]
        == int(acceptance["target_orbit_containment_failures"]),
        "orbital_pair_resolution": all_metrics["orbital_pair_resolved_unary_collision_fraction"]
        >= float(acceptance["orbital_pair_resolved_unary_collision_fraction_min"]),
        "transferable_pair_resolution": all_metrics["transferable_pair_resolved_unary_collision_fraction"]
        >= float(acceptance["transferable_pair_resolved_unary_collision_fraction_min"]),
        "transferable_pair_ceiling": all_metrics["transferable_pair_mean_target_ceiling"]
        >= float(acceptance["transferable_pair_mean_target_ceiling_min"]),
    }
    if all(checks.values()):
        decision_key = "pass"
    elif (
        checks["exact_enumeration_coverage"]
        and checks["relabel_invariance"]
        and checks["target_orbit_containment"]
        and checks["orbital_pair_resolution"]
    ):
        decision_key = "orbital_only"
    else:
        decision_key = "fail"
    payload = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "qualified": all(checks.values()),
        "checks": checks,
        "decision": protocol["decision_rule"][decision_key],
        "decision_class": decision_key,
        "boundary": protocol["decision_rule"]["boundary"],
        "metrics": metrics,
        "carrier_rows": rows,
        "source_hashes": protocol["source"]["artifact_sha256"],
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_readme(args.output_dir / "README.md", payload)
    print(json.dumps({key: payload[key] for key in ("qualified", "checks", "decision_class", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
