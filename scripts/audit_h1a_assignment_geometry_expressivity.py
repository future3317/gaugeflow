"""Audit geometry-resolved occupation identifiability without training."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.geometry import closest_image_displacements_numpy
from gaugeflow.production.assignment_scorer import faithful_parent_action
from scripts.audit_h1a_assignment_global_interactions import (
    Signature,
    _assignment_chunks,
    _canonical_local_species,
    _relation_histograms,
    _target_orbit,
    _unordered_pairs,
    action_site_signatures,
    unary_collision_class_size,
)


def periodic_distance_matrix(
    fractional: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Return exact-CVP pair distances in a row-vector periodic cell."""
    positions = np.asarray(fractional, dtype=np.float64) % 1.0
    cell = np.asarray(lattice, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or cell.shape != (3, 3) or np.linalg.det(cell) <= 0.0:
        raise ValueError("expanded carrier geometry has invalid shapes or volume")
    delta = (positions[:, None, :] - positions[None, :, :]).reshape(-1, 3)
    cartesian, _ = closest_image_displacements_numpy(delta, cell)
    distances = np.linalg.norm(cartesian, axis=1).reshape(
        positions.shape[0],
        positions.shape[0],
    )
    if not np.allclose(distances, distances.T, atol=1e-10, rtol=1e-10):
        raise RuntimeError("periodic distance matrix is not symmetric")
    return distances


def _quantized_distances(distances: np.ndarray, resolution: float) -> np.ndarray:
    if resolution <= 0.0:
        raise ValueError("distance quantization resolution must be positive")
    return np.rint(np.asarray(distances, dtype=np.float64) / resolution).astype(np.int64)


def geometry_site_signatures(
    permutations: torch.Tensor,
    fractional: np.ndarray,
    lattice: np.ndarray,
    *,
    maximum_sites: int = 20,
    distance_resolution_angstrom: float = 1e-6,
) -> tuple[Signature, ...]:
    """Combine exact action signatures with all-pair radial environments."""
    action = action_site_signatures(permutations, maximum_sites=maximum_sites)
    quantized = _quantized_distances(
        periodic_distance_matrix(fractional, lattice),
        distance_resolution_angstrom,
    )
    if quantized.shape[0] != len(action):
        raise ValueError("geometry and parent action use different node counts")
    return tuple(
        (*signature, *sorted(map(int, np.delete(quantized[site], site)))) for site, signature in enumerate(action)
    )


def geometry_pair_descriptors(
    site_signatures: Sequence[Signature],
    fractional: np.ndarray,
    lattice: np.ndarray,
    *,
    distance_resolution_angstrom: float = 1e-6,
) -> tuple[tuple[Signature, ...], np.ndarray, int]:
    """Describe each unordered pair by its complete two-point distance view.

    The descriptor contains no carrier-local orbit ID.  Endpoint exchange is
    quotiented by choosing the lexicographically smaller orientation of the
    two-point distances to every third site.
    """
    quantized = _quantized_distances(
        periodic_distance_matrix(fractional, lattice),
        distance_resolution_angstrom,
    )
    sites = quantized.shape[0]
    if len(site_signatures) != sites:
        raise ValueError("site signatures and geometry use different node counts")
    descriptors: list[Signature] = []
    for left, right in _unordered_pairs(sites):
        endpoints = sorted((site_signatures[left], site_signatures[right]))
        others = [index for index in range(sites) if index not in (left, right)]
        forward = tuple(
            sorted(
                (
                    int(quantized[left, other]),
                    int(quantized[right, other]),
                )
                for other in others
            )
        )
        reverse = tuple((right_distance, left_distance) for left_distance, right_distance in forward)
        reverse = tuple(sorted(reverse))
        if site_signatures[left] < site_signatures[right]:
            view = forward
        elif site_signatures[right] < site_signatures[left]:
            view = reverse
        else:
            view = min(forward, reverse)
        descriptors.append(
            (
                *endpoints[0],
                *endpoints[1],
                int(quantized[left, right]),
                *(value for pair in view for value in pair),
            )
        )
    unique = {value: index for index, value in enumerate(sorted(set(descriptors)))}
    labels = np.asarray([unique[value] for value in descriptors], dtype=np.int64)
    return tuple(descriptors), labels, len(unique)


def audit_geometry_carrier(
    assignment: Sequence[int],
    permutations: torch.Tensor,
    fractional: np.ndarray,
    lattice: np.ndarray,
    *,
    maximum_sites: int,
    maximum_collision_class: int,
    chunk_size: int,
    distance_resolution_angstrom: float,
) -> dict[str, Any]:
    """Exactly audit one geometry-unary collision class."""
    action = faithful_parent_action(permutations).detach().cpu()
    target = _canonical_local_species(assignment)
    signatures = geometry_site_signatures(
        action,
        fractional,
        lattice,
        maximum_sites=maximum_sites,
        distance_resolution_angstrom=distance_resolution_angstrom,
    )
    unary_size = unary_collision_class_size(target.tolist(), signatures)
    target_orbit = _target_orbit(target, action)
    if unary_size < len(target_orbit):
        raise ValueError("target orbit escaped the geometry-unary collision class")
    base = {
        "site_count": int(target.size),
        "species_count": int(np.unique(target).size),
        "action_order": int(action.shape[0]),
        "geometry_unary_collision_class_size": unary_size,
        "target_orbit_size": len(target_orbit),
        "geometry_unary_target_ceiling": len(target_orbit) / unary_size,
        "geometry_unary_non_orbit_collision": unary_size > len(target_orbit),
        "geometry_unary_resolved": unary_size == len(target_orbit),
        "exact_enumerated": unary_size <= maximum_collision_class,
    }
    if not base["exact_enumerated"]:
        return base

    pair_indices = np.asarray(_unordered_pairs(target.size), dtype=np.int64)
    _, pair_labels, pair_count = geometry_pair_descriptors(
        signatures,
        fractional,
        lattice,
        distance_resolution_angstrom=distance_resolution_angstrom,
    )
    species = int(np.unique(target).size)
    target_signature = _relation_histograms(
        target[None, :],
        pair_indices,
        pair_labels,
        pair_count,
        species,
    )[0]
    enumerated = 0
    matches = 0
    target_members = 0
    for chunk in _assignment_chunks(target, signatures, chunk_size=chunk_size):
        enumerated += chunk.shape[0]
        equal = np.all(
            _relation_histograms(
                chunk,
                pair_indices,
                pair_labels,
                pair_count,
                species,
            )
            == target_signature,
            axis=1,
        )
        matches += int(equal.sum())
        for value in chunk[equal]:
            target_members += int(np.ascontiguousarray(value).tobytes() in target_orbit)
    if enumerated != unary_size:
        raise AssertionError("geometry-unary collision enumeration was incomplete")
    return {
        **base,
        "enumerated_assignments": enumerated,
        "geometry_pair_descriptor_count": pair_count,
        "geometry_pair_collision_class_size": matches,
        "geometry_pair_target_ceiling": len(target_orbit) / matches,
        "geometry_pair_resolved": matches == len(target_orbit),
        "target_orbit_containment_failure": target_members != len(target_orbit),
    }


def _relabel_action(action: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    return inverse[action[:, order]]


def representation_invariance_checks(
    permutations: torch.Tensor,
    fractional: np.ndarray,
    lattice: np.ndarray,
    *,
    seed: int,
    maximum_sites: int,
    distance_resolution_angstrom: float,
) -> tuple[bool, bool]:
    """Check node relabeling and unimodular cell-basis invariance."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(permutations.shape[1], generator=generator)
    original_sites = geometry_site_signatures(
        permutations,
        fractional,
        lattice,
        maximum_sites=maximum_sites,
        distance_resolution_angstrom=distance_resolution_angstrom,
    )
    original_pairs, _, _ = geometry_pair_descriptors(
        original_sites,
        fractional,
        lattice,
        distance_resolution_angstrom=distance_resolution_angstrom,
    )
    relabeled_action = _relabel_action(permutations, order)
    relabeled_fractional = np.asarray(fractional)[order.numpy()]
    relabeled_sites = geometry_site_signatures(
        relabeled_action,
        relabeled_fractional,
        lattice,
        maximum_sites=maximum_sites,
        distance_resolution_angstrom=distance_resolution_angstrom,
    )
    site_consistent = all(relabeled_sites[new] == original_sites[int(old)] for new, old in enumerate(order.tolist()))
    pair_lookup = {
        pair: descriptor
        for pair, descriptor in zip(
            _unordered_pairs(len(original_sites)),
            original_pairs,
            strict=True,
        )
    }
    relabeled_pairs, _, _ = geometry_pair_descriptors(
        relabeled_sites,
        relabeled_fractional,
        lattice,
        distance_resolution_angstrom=distance_resolution_angstrom,
    )
    pair_consistent = True
    for new_pair, descriptor in zip(
        _unordered_pairs(len(original_sites)),
        relabeled_pairs,
        strict=True,
    ):
        old_pair = tuple(sorted((int(order[new_pair[0]]), int(order[new_pair[1]]))))
        pair_consistent &= descriptor == pair_lookup[old_pair]

    unimodular = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    changed_lattice = unimodular @ np.asarray(lattice, dtype=np.float64)
    changed_fractional = (np.asarray(fractional, dtype=np.float64) @ np.linalg.inv(unimodular.astype(np.float64))) % 1.0
    changed_sites = geometry_site_signatures(
        permutations,
        changed_fractional,
        changed_lattice,
        maximum_sites=maximum_sites,
        distance_resolution_angstrom=distance_resolution_angstrom,
    )
    changed_pairs, _, _ = geometry_pair_descriptors(
        changed_sites,
        changed_fractional,
        changed_lattice,
        distance_resolution_angstrom=distance_resolution_angstrom,
    )
    return site_consistent and pair_consistent, (changed_sites == original_sites and changed_pairs == original_pairs)


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exact = [row for row in rows if row["exact_enumerated"]]
    collisions = [row for row in exact if row["geometry_unary_non_orbit_collision"]]
    return {
        "carriers": len(rows),
        "materials": len({row["material_id"] for row in rows}),
        "exact_enumerated_carrier_fraction": len(exact) / len(rows),
        "geometry_unary_resolved_fraction": sum(bool(row["geometry_unary_resolved"]) for row in exact) / len(exact),
        "geometry_unary_collision_carriers_exact": len(collisions),
        "geometry_pair_resolved_collision_fraction": (
            sum(bool(row["geometry_pair_resolved"]) for row in collisions) / len(collisions) if collisions else 1.0
        ),
        "geometry_pair_mean_target_ceiling": (
            sum(float(row["geometry_pair_target_ceiling"]) for row in collisions) / len(collisions)
            if collisions
            else 1.0
        ),
        "relabel_failures": sum(not bool(row["relabel_invariant"]) for row in rows),
        "cell_basis_failures": sum(not bool(row["cell_basis_invariant"]) for row in rows),
        "target_orbit_containment_failures": sum(
            bool(row.get("target_orbit_containment_failure", False)) for row in exact
        ),
    }


def _write_readme(path: Path, result: dict[str, Any]) -> None:
    metric = result["metrics"]["all"]
    path.write_text(
        "\n".join(
            (
                "# Geometry-aware assignment expressivity audit",
                "",
                f"Decision: **{'PASS' if result['qualified'] else 'FAIL'}**.",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| exact enumeration coverage | {metric['exact_enumerated_carrier_fraction']:.6f} |",
                f"| geometry-unary resolved fraction | {metric['geometry_unary_resolved_fraction']:.6f} |",
                f"| remaining collision carriers | {metric['geometry_unary_collision_carriers_exact']} |",
                f"| geometry-pair resolved fraction | {metric['geometry_pair_resolved_collision_fraction']:.6f} |",
                f"| geometry-pair mean target ceiling | {metric['geometry_pair_mean_target_ceiling']:.6f} |",
                f"| relabel failures | {metric['relabel_failures']} |",
                f"| GL(3,Z) basis failures | {metric['cell_basis_failures']} |",
                "",
                "The observed coloring selects the collision class only. Site and pair",
                "features contain the species-free HNF carrier, exact periodic distances",
                "and the faithful parent action; no target species enters a descriptor.",
                "This is an identifiability result, not assignment qualification.",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--carrier-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_assignment_geometry_expressivity_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen geometry-expressivity protocol")
    prerequisite = repository / protocol["prerequisite"]["result"]
    if sha256_file(prerequisite) != protocol["prerequisite"]["result_sha256"]:
        raise ValueError("geometry-complete carrier result identity changed")
    prerequisite_result = load_json_object(prerequisite)
    if prerequisite_result.get("qualified") is not True:
        raise ValueError("geometry-complete carrier is not qualified")
    for name, expected in protocol["source"]["artifact_sha256"].items():
        if sha256_file(args.carrier_root / name) != expected:
            raise ValueError(f"assignment carrier v2 identity changed: {name}")
    iid_roles_path = repository / protocol["source"]["iid_roles"]
    if sha256_file(iid_roles_path) != protocol["source"]["iid_roles_sha256"]:
        raise ValueError("assignment IID/OOD role identity changed")
    iid_roles = load_json_object(iid_roles_path)["material_roles"]
    with gzip.open(args.carrier_root / "records.json.gz", "rt", encoding="utf-8") as handle:
        source = json.load(handle)

    audit = protocol["audit"]
    rows: list[dict[str, Any]] = []
    for record in source:
        for candidate_index, candidate in enumerate(record["candidates"]):
            carrier = candidate["carrier"]
            target = candidate["target"]["assignment_tokens"]
            action = torch.tensor(carrier["parent_action_permutations"], dtype=torch.long)
            fractional = np.asarray(carrier["expanded_parent_fractional"], dtype=np.float64)
            lattice = np.asarray(carrier["expanded_parent_lattice"], dtype=np.float64)
            result = audit_geometry_carrier(
                target,
                action,
                fractional,
                lattice,
                maximum_sites=20,
                maximum_collision_class=int(audit["maximum_exact_geometry_unary_collision_class"]),
                chunk_size=int(audit["assignment_chunk_size"]),
                distance_resolution_angstrom=float(audit["distance_resolution_angstrom"]),
            )
            relabel, cell_basis = representation_invariance_checks(
                action,
                fractional,
                lattice,
                seed=int(audit["relabel_seed"]) + len(rows),
                maximum_sites=20,
                distance_resolution_angstrom=float(audit["distance_resolution_angstrom"]),
            )
            result.update(
                {
                    "material_id": str(record["material_id_audit_only"]),
                    "original_split": str(record["gaugeflow_split_audit_only"]),
                    "evidence_role": str(iid_roles[str(record["material_id_audit_only"])]),
                    "candidate_index": candidate_index,
                    "embedding_key": str(candidate["embedding_key"]),
                    "relabel_invariant": relabel,
                    "cell_basis_invariant": cell_basis,
                }
            )
            rows.append(result)
    observed = Counter(row["original_split"] for row in rows)
    if (
        len(rows) != int(protocol["source"]["candidate_carriers"])
        or dict(observed) != protocol["source"]["candidate_carriers_by_split"]
    ):
        raise ValueError("assignment carrier v2 counts changed")
    metrics = {"all": _summarize(rows)}
    for split in ("train", "val", "test"):
        metrics[split] = _summarize([row for row in rows if row["original_split"] == split])
    for role in (
        "iid_fit",
        "iid_fit_rare",
        "iid_calibration",
        "iid_test",
        "ood_validation",
        "ood_test",
    ):
        metrics[role] = _summarize([row for row in rows if row["evidence_role"] == role])
    total = metrics["all"]
    acceptance = protocol["acceptance"]
    checks = {
        "exact_enumeration_coverage": total["exact_enumerated_carrier_fraction"]
        >= float(acceptance["exact_enumerated_carrier_fraction_min"]),
        "relabel_invariance": total["relabel_failures"] == int(acceptance["relabel_failures"]),
        "cell_basis_invariance": total["cell_basis_failures"] == int(acceptance["cell_basis_failures"]),
        "target_orbit_containment": total["target_orbit_containment_failures"]
        == int(acceptance["target_orbit_containment_failures"]),
        "geometry_pair_resolution": total["geometry_pair_resolved_collision_fraction"]
        >= float(acceptance["geometry_pair_resolved_collision_fraction_min"]),
        "geometry_pair_ceiling": total["geometry_pair_mean_target_ceiling"]
        >= float(acceptance["geometry_pair_mean_target_ceiling_min"]),
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "qualified": qualified,
        "checks": checks,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "metrics": metrics,
        "carrier_rows": rows,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(args.output_dir / "README.md", result)
    print(json.dumps({key: result[key] for key in ("qualified", "checks", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
