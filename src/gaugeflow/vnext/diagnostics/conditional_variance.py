"""Finite-sample conditional-variance estimators for path identifiability."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConditionalVarianceEstimate:
    """Trace conditional variance averaged over query neighbourhoods."""

    neighbors: int
    trace_variance: torch.Tensor
    target_trace_variance: torch.Tensor
    normalized_trace_variance: torch.Tensor


def _standardize(representation: torch.Tensor) -> torch.Tensor:
    centered = representation - representation.mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=0, keepdim=True).sqrt()
    active = scale > 64.0 * torch.finfo(representation.dtype).eps
    return torch.where(active, centered / scale.clamp_min(torch.finfo(representation.dtype).eps), centered)


def knn_conditional_variance(
    representation: torch.Tensor,
    target: torch.Tensor,
    *,
    neighbors: int,
    standardize: bool = True,
) -> ConditionalVarianceEstimate:
    """Estimate ``E tr Cov(target | representation)`` using local neighbours.

    Every query neighbourhood contains the query and its ``neighbors`` nearest
    *other* observations.  Constant representation channels are ignored by the
    optional per-time standardization.  The estimator is diagnostic rather
    than a hypothesis test and therefore reports its normalization explicitly.
    """
    if representation.ndim != 2 or target.ndim != 2:
        raise ValueError("representation and target must be rank-two tensors")
    if representation.shape[0] != target.shape[0]:
        raise ValueError("representation and target must contain the same observations")
    observations = representation.shape[0]
    if not 1 <= neighbors < observations:
        raise ValueError("neighbors must lie in [1, observations - 1]")
    if not torch.isfinite(representation).all() or not torch.isfinite(target).all():
        raise ValueError("conditional-variance inputs must be finite")
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
    global_centered = target - target.mean(dim=0, keepdim=True)
    target_trace = global_centered.square().mean(dim=0).sum()
    normalized = trace / target_trace.clamp_min(torch.finfo(target.dtype).eps)
    return ConditionalVarianceEstimate(neighbors, trace, target_trace, normalized)
