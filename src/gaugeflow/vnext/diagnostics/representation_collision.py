"""Near-collision and local target-Lipschitz diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CollisionAudit:
    """Pairwise collision statistics after dimensionless distance scaling."""

    pair_count: int
    distance_threshold: torch.Tensor
    near_pair_count: int
    collision_count: int
    max_target_lipschitz_ratio: torch.Tensor
    median_target_lipschitz_ratio: torch.Tensor


def _positive_median(value: torch.Tensor, floor: float) -> torch.Tensor:
    positive = value[value > floor]
    if positive.numel() == 0:
        return value.new_tensor(1.0)
    return positive.median()


def audit_representation_collisions(
    representation: torch.Tensor,
    target: torch.Tensor,
    *,
    near_quantile: float,
    target_ratio_min: float,
    distance_floor: float,
) -> CollisionAudit:
    """Count small representation distances paired with large target changes.

    Representation and target distances are divided by their respective
    positive median pairwise distance before taking the local Lipschitz ratio.
    This makes the registered ratio dimensionless across heterogeneous
    coordinate and feature representations.
    """
    if representation.ndim != 2 or target.ndim != 2:
        raise ValueError("representation and target must be rank-two tensors")
    if representation.shape[0] != target.shape[0] or representation.shape[0] < 2:
        raise ValueError("collision audit requires at least two paired observations")
    if not 0.0 < near_quantile <= 1.0:
        raise ValueError("near_quantile must lie in (0, 1]")
    if target_ratio_min <= 0.0 or distance_floor <= 0.0:
        raise ValueError("ratio threshold and distance floor must be positive")
    if not torch.isfinite(representation).all() or not torch.isfinite(target).all():
        raise ValueError("collision-audit inputs must be finite")
    representation_distance = torch.pdist(representation)
    target_distance = torch.pdist(target)
    representation_scale = _positive_median(representation_distance, distance_floor)
    target_scale = _positive_median(target_distance, distance_floor)
    normalized_representation = representation_distance / representation_scale
    normalized_target = target_distance / target_scale
    threshold = torch.quantile(normalized_representation, near_quantile)
    near = normalized_representation <= threshold
    ratios = normalized_target / normalized_representation.clamp_min(distance_floor)
    collision = near & (ratios >= target_ratio_min)
    selected = ratios[near]
    maximum = selected.max() if selected.numel() else ratios.new_tensor(0.0)
    median = selected.median() if selected.numel() else ratios.new_tensor(0.0)
    return CollisionAudit(
        pair_count=representation_distance.numel(),
        distance_threshold=threshold,
        near_pair_count=int(near.sum()),
        collision_count=int(collision.sum()),
        max_target_lipschitz_ratio=maximum,
        median_target_lipschitz_ratio=median,
    )
