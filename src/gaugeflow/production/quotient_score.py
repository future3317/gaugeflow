"""Rao--Blackwellized heat-kernel scores on the translation quotient."""

from __future__ import annotations

import math

import torch

from .schedules import wrapped_normal_log_density_and_score
from .state_projection import graph_mean, project_translation_state


def factorized_translation_quotient_scaled_score(
    displacement: torch.Tensor,
    sigma: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    *,
    quadrature_points: int = 32,
) -> torch.Tensor:
    """Return the common-translation-marginalized scaled torus score.

    The production heat kernel is isotropic in fractional coordinates, so its
    three common-translation integrals factorize into one-dimensional periodic
    integrals.  A circular-mean-shifted trapezoidal rule resolves each analytic
    periodic integrand with ``O(nodes * quadrature_points)`` batched work.

    This is the conditional expectation of the projected per-site denoising
    target given the visible translation-quotient state.  It preserves the
    same probability path while removing nuisance target variance.
    """
    if displacement.ndim != 2 or displacement.shape[1] != 3:
        raise ValueError("quotient score displacement must have shape [nodes,3]")
    if sigma.shape != (graph_count,) or batch.shape != displacement.shape[:1]:
        raise ValueError("quotient score graph tensors have incompatible shapes")
    if batch.dtype != torch.long or quadrature_points < 4:
        raise ValueError("quotient score needs int64 batches and at least four nodes")
    if not torch.isfinite(displacement).all() or bool((sigma <= 0.0).any()):
        raise ValueError("quotient score inputs must be finite with positive scales")
    angle = 2.0 * math.pi * displacement
    circular_sine = graph_mean(angle.sin(), batch, graph_count)
    circular_cosine = graph_mean(angle.cos(), batch, graph_count)
    center = torch.atan2(circular_sine, circular_cosine) / (2.0 * math.pi)
    grid = torch.arange(
        quadrature_points,
        dtype=displacement.dtype,
        device=displacement.device,
    ) / quadrature_points
    translation = center[:, None, :] + grid[None, :, None]
    residual = displacement[:, None, :] - translation[batch]
    node_sigma = sigma[batch, None, None]
    log_kernel, site_score = wrapped_normal_log_density_and_score(
        residual, node_sigma
    )
    posterior_log = displacement.new_zeros(
        (graph_count, quadrature_points, 3)
    )
    posterior_log.index_add_(0, batch, log_kernel.to(posterior_log.dtype))
    posterior = torch.softmax(posterior_log, dim=1)
    quotient_score = (site_score * posterior[batch]).sum(dim=1)
    quotient_score = project_translation_state(
        quotient_score, batch, graph_count
    )
    return sigma[batch].unsqueeze(-1) * quotient_score
