"""Allowed-subspace projection shared by every future reverse-process step."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lattice_volume_shape import project_lattice_state


def sorted_segment_sum(
    value: torch.Tensor,
    index: torch.Tensor,
    segment_count: int,
) -> torch.Tensor:
    """Linearly reduce a production-contiguous segment vector without atomics."""
    if index.shape != value.shape[:1] or index.dtype != torch.long:
        raise ValueError("segment index must provide one int64 value per row")
    if segment_count < 0:
        raise ValueError("segment count must be nonnegative")
    # Batch contiguity and edge target ordering are construction invariants.
    # Re-sorting here would add O(E log E) work to every message block.
    lengths = torch.bincount(index, minlength=segment_count)
    return torch.segment_reduce(value, "sum", lengths=lengths)


@dataclass(frozen=True)
class ProjectedContinuousState:
    fractional_coordinates: torch.Tensor
    log_shape: torch.Tensor


def graph_mean(value: torch.Tensor, batch: torch.Tensor, graph_count: int) -> torch.Tensor:
    """Return one mean per graph for a node-leading tensor."""
    if value.shape[:1] != batch.shape:
        raise ValueError("batch must provide one graph index per node")
    counts = torch.bincount(batch, minlength=graph_count).clamp_min(1)
    total = sorted_segment_sum(value, batch, graph_count)
    return total / counts.to(value).reshape(
        (graph_count,) + (1,) * (value.ndim - 1)
    )


def graph_sum(value: torch.Tensor, batch: torch.Tensor, graph_count: int) -> torch.Tensor:
    """Return one sum per graph for a node-leading tensor."""
    if value.shape[:1] != batch.shape:
        raise ValueError("batch must provide one graph index per node")
    return sorted_segment_sum(value, batch, graph_count)


def fractional_tangent_to_cartesian(
    fractional_tangent: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Push a fractional tangent vector into the Cartesian row-basis chart.

    For row coordinates ``r = f L``, tangent vectors obey ``v_r = v_f L``.
    The production coordinate path uses the fractional Brownian score through
    its mobility as a reverse-drift tangent, so this is the physical chart in
    which endpoint displacement errors are compared.
    """
    if fractional_tangent.ndim != 2 or fractional_tangent.shape[-1] != 3:
        raise ValueError("fractional tangent must have shape [sites,3]")
    if batch.shape != fractional_tangent.shape[:1] or batch.dtype != torch.long:
        raise ValueError("batch must provide one graph index per tangent")
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError("lattice must have shape [graphs,3,3]")
    if batch.numel() and int(batch.max()) >= lattice.shape[0]:
        raise ValueError("tangent batch index exceeds the lattice batch")
    return torch.einsum("ni,nij->nj", fractional_tangent, lattice[batch])


def cartesian_tangent_to_fractional(
    cartesian_tangent: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Pull a Cartesian tangent vector back to fractional row coordinates.

    This is the inverse of :func:`fractional_tangent_to_cartesian`. Solving
    ``L^T v_f^T = v_r^T`` is batched and avoids explicitly materializing
    ``L^-1``.
    """
    if cartesian_tangent.ndim != 2 or cartesian_tangent.shape[-1] != 3:
        raise ValueError("Cartesian tangent must have shape [sites,3]")
    if batch.shape != cartesian_tangent.shape[:1] or batch.dtype != torch.long:
        raise ValueError("batch must provide one graph index per tangent")
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError("lattice must have shape [graphs,3,3]")
    if batch.numel() and int(batch.max()) >= lattice.shape[0]:
        raise ValueError("tangent batch index exceeds the lattice batch")
    return torch.linalg.solve(
        lattice[batch].transpose(-1, -2), cartesian_tangent.unsqueeze(-1)
    ).squeeze(-1)


def project_translation_state(
    fractional_coordinates: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    """Choose the zero-mean representative of each translation quotient."""
    if fractional_coordinates.ndim != 2 or fractional_coordinates.shape[-1] != 3:
        raise ValueError("fractional coordinates must have shape [sites,3]")
    if batch.shape != fractional_coordinates.shape[:1]:
        raise ValueError("batch must provide one graph index per site")
    mean = graph_mean(fractional_coordinates, batch, graph_count)
    return fractional_coordinates - mean[batch]


def project_hybrid_reverse_state(
    fractional_coordinates: torch.Tensor,
    log_shape: torch.Tensor,
    batch: torch.Tensor,
    shape_projector: torch.Tensor,
) -> ProjectedContinuousState:
    """Project coordinates and lattice shape after one reverse update.

    The operation deliberately does not wrap fractional coordinates.  The
    reverse trajectory remains in a continuous translation-horizontal lift;
    wrapping to the torus is a terminal decoding operation.
    """
    graphs = log_shape.shape[0]
    return ProjectedContinuousState(
        fractional_coordinates=project_translation_state(
            fractional_coordinates, batch, graphs
        ),
        log_shape=project_lattice_state(log_shape, shape_projector),
    )
