"""Target-free feature compilation for parent-conditioned assignment."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from gaugeflow.file_utils import load_json_object
from gaugeflow.geometry import closest_image_displacements_numpy
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT

from .assignment_scorer import (
    faithful_parent_action,
    parent_action_site_features,
    parent_carrier_graph_features,
)
from .assignment_training import AssignmentCarrierBatch
from .autoregressive_assignment import complete_pair_context_features, complete_pair_rbf


@dataclass(frozen=True)
class AssignmentCarrierExample:
    """One species-free carrier paired with an occupational training label."""

    embedding_key: str
    material_id_audit_only: str
    evidence_role_audit_only: str
    site_features: torch.Tensor
    graph_features: torch.Tensor
    edge_source: torch.Tensor
    edge_target: torch.Tensor
    edge_rbf: torch.Tensor
    composition_counts: torch.Tensor
    target_assignment: torch.Tensor
    parent_permutations: torch.Tensor
    parent_space_group: int
    cell_index: int


def load_assignment_carrier_examples(
    carrier_root: Path,
    role_result_path: Path,
    *,
    maximum_sites: int,
    radial_channels: int,
) -> list[AssignmentCarrierExample]:
    """Load geometry-complete carriers with frozen audit-only evidence roles."""

    role_result = load_json_object(role_result_path)
    if role_result.get("qualified") is not True or not all(role_result["checks"].values()):
        raise ValueError("assignment IID role split is not qualified")
    roles = {
        (str(row["material_id"]), int(row["candidate_index"]), str(row["embedding_key"])): str(
            row["role"]
        )
        for row in role_result["carrier_rows"]
    }
    with gzip.open(carrier_root / "records.json.gz", "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    examples: list[AssignmentCarrierExample] = []
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        material_id = str(record["material_id_audit_only"])
        for candidate_index, candidate in enumerate(record["candidates"]):
            embedding_key = str(candidate["embedding_key"])
            key = (material_id, candidate_index, embedding_key)
            if key not in roles or key in seen:
                raise ValueError(f"carrier role identity is missing or duplicated: {key}")
            seen.add(key)
            examples.append(
                prepare_assignment_carrier_example(
                    candidate,
                    embedding_key=embedding_key,
                    material_id_audit_only=material_id,
                    evidence_role_audit_only=roles[key],
                    maximum_sites=maximum_sites,
                    radial_channels=radial_channels,
                )
            )
    if seen != set(roles):
        raise ValueError("geometry-complete carriers and frozen IID roles differ")
    return examples


def _exact_complete_pair_distances(
    fractional: torch.Tensor,
    lattice: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sites = fractional.shape[0]
    # Target-major ordering is a production invariant.  It permits a
    # deterministic linear-time segment reduction in every message block.
    target, source = torch.nonzero(~torch.eye(sites, dtype=torch.bool), as_tuple=True)
    delta = (
        fractional[target].to(dtype=torch.float64, device="cpu")
        - fractional[source].to(dtype=torch.float64, device="cpu")
    ).numpy()
    cartesian, _ = closest_image_displacements_numpy(
        delta,
        lattice.to(dtype=torch.float64, device="cpu").numpy(),
    )
    distance = torch.from_numpy(np.linalg.norm(cartesian, axis=1))
    return source, target, distance


def prepare_assignment_carrier_example(
    candidate: dict[str, Any],
    *,
    embedding_key: str,
    material_id_audit_only: str,
    evidence_role_audit_only: str,
    maximum_sites: int = 20,
    radial_channels: int = 16,
) -> AssignmentCarrierExample:
    """Compile one geometry-complete v2 record without target leakage."""
    carrier = candidate["carrier"]
    target = candidate["target"]
    assignment = torch.tensor(target["assignment_tokens"], dtype=torch.long)
    if assignment.ndim != 1 or assignment.numel() < 1 or assignment.numel() > maximum_sites:
        raise ValueError("assignment carrier lies outside the qualified node support")
    if bool(((assignment < 0) | (assignment >= CHEMICAL_ELEMENT_COUNT)).any()):
        raise ValueError("assignment carrier contains an unsupported element token")
    counts = torch.bincount(assignment, minlength=CHEMICAL_ELEMENT_COUNT)
    active_token = torch.tensor(target["active_species_tokens"], dtype=torch.long)
    active_count = torch.tensor(target["active_species_counts"], dtype=torch.long)
    if not torch.equal(counts[active_token], active_count) or int(active_count.sum()) != assignment.numel():
        raise ValueError("assignment target and active composition disagree")

    fractional = torch.tensor(carrier["expanded_parent_fractional"], dtype=torch.float64)
    lattice = torch.tensor(carrier["expanded_parent_lattice"], dtype=torch.float64)
    if fractional.shape != (assignment.numel(), 3) or lattice.shape != (3, 3):
        raise ValueError("expanded parent geometry does not cover the assignment")
    if not torch.isfinite(fractional).all() or not torch.isfinite(lattice).all():
        raise ValueError("expanded parent geometry contains a nonfinite value")
    volume = torch.linalg.det(lattice)
    if float(volume) <= 0.0:
        raise ValueError("expanded parent lattice must have positive volume")
    permutations = faithful_parent_action(
        torch.tensor(carrier["parent_action_permutations"], dtype=torch.long)
    )
    if permutations.shape[1] != assignment.numel():
        raise ValueError("parent action does not cover the expanded carrier")
    cell_index = int(candidate["cell_index"])
    hnf = torch.tensor(carrier["supercell_hnf"], dtype=torch.float64)
    if round(float(torch.linalg.det(hnf))) != cell_index:
        raise ValueError("carrier HNF determinant and cell index disagree")

    edge_source, edge_target, distance = _exact_complete_pair_distances(fractional, lattice)
    normalized_distance = distance / volume.pow(1.0 / 3.0)
    edge_rbf = complete_pair_rbf(
        normalized_distance.to(torch.float32),
        radial_channels=radial_channels,
    )
    edge_features = complete_pair_context_features(
        edge_source,
        edge_target,
        edge_rbf,
        node_count=assignment.numel(),
    )
    return AssignmentCarrierExample(
        embedding_key=embedding_key,
        material_id_audit_only=material_id_audit_only,
        evidence_role_audit_only=evidence_role_audit_only,
        site_features=parent_action_site_features(permutations, maximum_sites=maximum_sites),
        graph_features=parent_carrier_graph_features(
            fractional,
            lattice,
            permutations,
            cell_index=cell_index,
            maximum_sites=maximum_sites,
            radial_channels=radial_channels,
        ),
        edge_source=edge_source,
        edge_target=edge_target,
        edge_rbf=edge_features,
        composition_counts=counts,
        target_assignment=assignment,
        parent_permutations=permutations,
        parent_space_group=int(candidate["parent_space_group"]),
        cell_index=cell_index,
    )


def pack_assignment_carriers(
    examples: Sequence[AssignmentCarrierExample],
    *,
    device: torch.device | str,
) -> AssignmentCarrierBatch:
    """Pack variable-size carriers for one vectorized reveal-path objective."""
    if not examples:
        raise ValueError("cannot pack an empty assignment carrier batch")
    node_counts = torch.tensor([value.target_assignment.numel() for value in examples], dtype=torch.long)
    offsets = torch.cumsum(node_counts, dim=0) - node_counts
    return AssignmentCarrierBatch(
        site_features=torch.cat([value.site_features for value in examples]).to(device),
        graph_features=torch.stack([value.graph_features for value in examples]).to(device),
        batch=torch.repeat_interleave(torch.arange(len(examples)), node_counts).to(device),
        edge_source=torch.cat(
            [value.edge_source + offsets[index] for index, value in enumerate(examples)]
        ).to(device),
        edge_target=torch.cat(
            [value.edge_target + offsets[index] for index, value in enumerate(examples)]
        ).to(device),
        edge_rbf=torch.cat([value.edge_rbf for value in examples]).to(device),
        composition_counts=torch.stack([value.composition_counts for value in examples]).to(device),
        target_assignment=torch.cat([value.target_assignment for value in examples]).to(device),
        parent_space_group=torch.tensor(
            [value.parent_space_group for value in examples], dtype=torch.long, device=device
        ),
        cell_index=torch.tensor(
            [value.cell_index for value in examples], dtype=torch.long, device=device
        ),
    )
