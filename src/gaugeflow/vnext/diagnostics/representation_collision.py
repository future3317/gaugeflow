"""Exact collisions, near pairs, and explicit representation-alias witnesses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CollisionAudit:
    """Disjointly named exact, near, and target-alias pair statistics."""

    pair_count: int
    exact_collision_count: int
    near_distance_threshold: torch.Tensor
    near_pair_count: int
    alias_collision_count: int
    max_local_target_ratio: torch.Tensor
    median_local_target_ratio: torch.Tensor


@dataclass(frozen=True)
class CollisionWitness:
    """One source pair whose representation is exact or near but target differs."""

    source_left: int
    source_right: int
    representation_distance: float
    target_distance: float
    lift_difference: float
    exact_representation_collision: bool


def _positive_median(value: torch.Tensor, floor: float) -> torch.Tensor:
    positive = value[value > floor]
    if positive.numel() == 0:
        return value.new_tensor(1.0)
    return positive.median()


def audit_representation_collisions(
    representation: torch.Tensor,
    target: torch.Tensor,
    *,
    exact_absolute_tolerance: float,
    near_quantile: float,
    alias_target_distance_min: float,
    distance_floor: float,
    lift_representation: torch.Tensor | None = None,
) -> tuple[CollisionAudit, list[CollisionWitness]]:
    """Audit exact equality separately from quantile-defined proximity."""
    if representation.ndim != 2 or target.ndim != 2:
        raise ValueError("representation and target must be rank-two tensors")
    if representation.shape[0] != target.shape[0] or representation.shape[0] < 2:
        raise ValueError("collision audit requires at least two paired observations")
    if lift_representation is not None and lift_representation.shape[0] != representation.shape[0]:
        raise ValueError("lift representation must contain the same observations")
    if exact_absolute_tolerance <= 0.0 or not 0.0 < near_quantile <= 1.0:
        raise ValueError("exact tolerance and near quantile are invalid")
    if alias_target_distance_min <= 0.0 or distance_floor <= 0.0:
        raise ValueError("alias threshold and distance floor must be positive")
    if not torch.isfinite(representation).all() or not torch.isfinite(target).all():
        raise ValueError("collision-audit inputs must be finite")
    indices = torch.triu_indices(representation.shape[0], representation.shape[0], offset=1)
    representation_distance = torch.linalg.vector_norm(representation[indices[0]] - representation[indices[1]], dim=-1)
    target_distance = torch.linalg.vector_norm(target[indices[0]] - target[indices[1]], dim=-1)
    exact = representation_distance <= exact_absolute_tolerance
    representation_scale = _positive_median(representation_distance, distance_floor)
    target_scale = _positive_median(target_distance, distance_floor)
    normalized_representation = representation_distance / representation_scale
    normalized_target = target_distance / target_scale
    near_threshold = torch.quantile(normalized_representation, near_quantile)
    near = normalized_representation <= near_threshold
    alias = near & (target_distance >= alias_target_distance_min)
    ratios = normalized_target / normalized_representation.clamp_min(distance_floor)
    selected = ratios[near]
    maximum = selected.max() if selected.numel() else ratios.new_tensor(0.0)
    median = selected.median() if selected.numel() else ratios.new_tensor(0.0)
    if lift_representation is None:
        lift_difference = torch.zeros_like(target_distance)
    else:
        lift_difference = torch.linalg.vector_norm(
            lift_representation[indices[0]] - lift_representation[indices[1]], dim=-1
        )
    witnesses = [
        CollisionWitness(
            source_left=int(indices[0, pair]),
            source_right=int(indices[1, pair]),
            representation_distance=float(representation_distance[pair]),
            target_distance=float(target_distance[pair]),
            lift_difference=float(lift_difference[pair]),
            exact_representation_collision=bool(exact[pair]),
        )
        for pair in torch.nonzero(alias, as_tuple=False).flatten().tolist()
    ]
    return (
        CollisionAudit(
            pair_count=representation_distance.numel(),
            exact_collision_count=int(exact.sum()),
            near_distance_threshold=near_threshold,
            near_pair_count=int(near.sum()),
            alias_collision_count=int(alias.sum()),
            max_local_target_ratio=maximum,
            median_local_target_ratio=median,
        ),
        witnesses,
    )
