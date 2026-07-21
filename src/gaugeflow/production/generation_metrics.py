"""Shared distribution metrics for tensor-free crystal generation."""

from __future__ import annotations

import math

import torch
from torch_geometric.utils import scatter

from gaugeflow.geometry import periodic_radius_multigraph


def minimum_periodic_distances(
    fractional_coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    *,
    cutoff: float = 8.0,
) -> torch.Tensor:
    """Return the shortest non-self periodic distance in each graph."""

    edges = periodic_radius_multigraph(
        fractional_coordinates,
        lattice,
        batch,
        cutoff=cutoff,
    )
    if edges.target.numel() == 0:
        return lattice.new_full((lattice.shape[0],), math.inf)
    return scatter(
        edges.distance,
        batch[edges.target],
        dim=0,
        dim_size=lattice.shape[0],
        reduce="min",
    )


def element_histogram(tokens: torch.Tensor, classes: int = 118) -> torch.Tensor:
    """Count element tokens in a fixed vocabulary."""

    return torch.bincount(tokens.long(), minlength=classes).double()


def formula_keys(
    tokens: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    *,
    classes: int = 118,
) -> list[str]:
    """Serialize exact integer compositions independently of site order."""

    counts = torch.zeros(
        (graph_count, classes),
        dtype=torch.int32,
        device=tokens.device,
    )
    counts.index_put_(
        (batch, tokens),
        torch.ones_like(tokens, dtype=torch.int32),
        accumulate=True,
    )
    return [
        ";".join(f"{index + 1}:{count}" for index, count in enumerate(row) if count)
        for row in counts.cpu().tolist()
    ]


def jensen_shannon(first: torch.Tensor, second: torch.Tensor) -> float:
    """Jensen--Shannon divergence for nonnegative count vectors."""

    first_probability = first.double() / first.sum()
    second_probability = second.double() / second.sum()
    midpoint = 0.5 * (first_probability + second_probability)
    first_term = torch.where(
        first_probability > 0.0,
        first_probability * (first_probability / midpoint).log(),
        0.0,
    )
    second_term = torch.where(
        second_probability > 0.0,
        second_probability * (second_probability / midpoint).log(),
        0.0,
    )
    return float(0.5 * (first_term.sum() + second_term.sum()))


def quantile_wasserstein(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    points: int,
) -> float:
    """One-dimensional W1 estimated on a fixed uniform quantile grid."""

    if points < 2 or first.numel() < 1 or second.numel() < 1:
        raise ValueError("quantile Wasserstein requires samples and at least two points")
    probabilities = torch.linspace(0.0, 1.0, points, dtype=torch.float64)
    return float(
        (
            torch.quantile(first.double(), probabilities)
            - torch.quantile(second.double(), probabilities)
        )
        .abs()
        .mean()
    )


def robust_scale(reference: torch.Tensor) -> float:
    """Return a strictly positive reference interquartile range."""

    scale = torch.quantile(reference.double(), 0.75) - torch.quantile(
        reference.double(), 0.25
    )
    if not torch.isfinite(scale) or float(scale) <= 0.0:
        raise ValueError("reference distribution has no finite positive IQR")
    return float(scale)
