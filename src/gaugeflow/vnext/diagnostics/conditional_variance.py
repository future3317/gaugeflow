"""Finite-sample dispersion and exact representation-aliasing risks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalTargetDispersion:
    """Local target dispersion, explicitly not a conditional variance."""

    neighbors: int
    trace_dispersion: torch.Tensor
    target_trace_variance: torch.Tensor
    normalized_trace_dispersion: torch.Tensor


@dataclass(frozen=True)
class ExactEquivalenceRisk:
    """Discrete conditional-variance risk over exact representation classes."""

    equivalence_class_count: int
    nontrivial_class_count: int
    exact_collision_count: int
    trace_risk: torch.Tensor
    target_trace_variance: torch.Tensor
    normalized_trace_risk: torch.Tensor


def _standardize(representation: torch.Tensor) -> torch.Tensor:
    centered = representation - representation.mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=0, keepdim=True).sqrt()
    active = scale > 64.0 * torch.finfo(representation.dtype).eps
    return torch.where(active, centered / scale.clamp_min(torch.finfo(representation.dtype).eps), centered)


def _target_trace_variance(target: torch.Tensor) -> torch.Tensor:
    centered = target - target.mean(dim=0, keepdim=True)
    return centered.square().mean(dim=0).sum()


def knn_local_target_dispersion(
    representation: torch.Tensor,
    target: torch.Tensor,
    *,
    neighbors: int,
    standardize: bool = True,
) -> LocalTargetDispersion:
    """Measure finite-neighbourhood target variation.

    The quantity includes ordinary target-field variation, finite bandwidth,
    and sampling effects. It must never be labelled as irreducible
    ``Var(target | representation)``.
    """
    if representation.ndim != 2 or target.ndim != 2:
        raise ValueError("representation and target must be rank-two tensors")
    if representation.shape[0] != target.shape[0]:
        raise ValueError("representation and target must contain the same observations")
    observations = representation.shape[0]
    if not 1 <= neighbors < observations:
        raise ValueError("neighbors must lie in [1, observations - 1]")
    if not torch.isfinite(representation).all() or not torch.isfinite(target).all():
        raise ValueError("local-dispersion inputs must be finite")
    features = _standardize(representation) if standardize else representation
    distances = torch.cdist(features, features)
    distances.fill_diagonal_(torch.inf)
    nearest = torch.topk(distances, k=neighbors, largest=False).indices
    query = torch.arange(observations, device=representation.device).unsqueeze(-1)
    indices = torch.cat((query, nearest), dim=-1)
    local_targets = target[indices]
    local_centered = local_targets - local_targets.mean(dim=1, keepdim=True)
    local_trace = local_centered.square().mean(dim=1).sum(dim=-1)
    trace = local_trace.mean()
    target_trace = _target_trace_variance(target)
    normalized = trace / target_trace.clamp_min(torch.finfo(target.dtype).eps)
    return LocalTargetDispersion(neighbors, trace, target_trace, normalized)


def exact_equivalence_risk(
    representation: torch.Tensor,
    target: torch.Tensor,
    *,
    absolute_tolerance: float,
) -> ExactEquivalenceRisk:
    """Compute exact finite-sample aliasing risk over tolerance classes.

    Connected components of the fixed-tolerance equality graph define the
    representation classes. With the preregistered tiny numerical tolerance,
    singleton continuous states contribute exactly zero while a collapsed
    endpoint class contributes its population target variance.
    """
    if representation.ndim != 2 or target.ndim != 2:
        raise ValueError("representation and target must be rank-two tensors")
    if representation.shape[0] != target.shape[0] or representation.shape[0] < 1:
        raise ValueError("exact risk requires paired observations")
    if absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be positive")
    if not torch.isfinite(representation).all() or not torch.isfinite(target).all():
        raise ValueError("exact-risk inputs must be finite")
    observations = representation.shape[0]
    parent = list(range(observations))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    distances = torch.cdist(representation, representation)
    pairs = torch.nonzero(torch.triu(distances <= absolute_tolerance, diagonal=1), as_tuple=False)
    for left, right in pairs.tolist():
        union(left, right)
    classes: dict[int, list[int]] = {}
    for index in range(observations):
        classes.setdefault(find(index), []).append(index)
    risk = target.new_zeros(())
    nontrivial = 0
    collision_count = 0
    for members in classes.values():
        values = target[torch.tensor(members, device=target.device)]
        centered = values - values.mean(dim=0, keepdim=True)
        risk = risk + (len(members) / observations) * centered.square().mean(dim=0).sum()
        if len(members) > 1:
            nontrivial += 1
            collision_count += len(members) * (len(members) - 1) // 2
    target_trace = _target_trace_variance(target)
    return ExactEquivalenceRisk(
        equivalence_class_count=len(classes),
        nontrivial_class_count=nontrivial,
        exact_collision_count=collision_count,
        trace_risk=risk,
        target_trace_variance=target_trace,
        normalized_trace_risk=risk / target_trace.clamp_min(torch.finfo(target.dtype).eps),
    )
