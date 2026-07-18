"""Allowed-subspace projection shared by every future reverse-process step."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.utils import scatter

from .lattice_volume_shape import project_lattice_state


@dataclass(frozen=True)
class ProjectedContinuousState:
    fractional_coordinates: torch.Tensor
    log_shape: torch.Tensor


def graph_mean(value: torch.Tensor, batch: torch.Tensor, graph_count: int) -> torch.Tensor:
    """Return one mean per graph for a node-leading tensor."""
    if value.shape[:1] != batch.shape:
        raise ValueError("batch must provide one graph index per node")
    return scatter(value, batch, dim=0, dim_size=graph_count, reduce="mean")


def graph_sum(value: torch.Tensor, batch: torch.Tensor, graph_count: int) -> torch.Tensor:
    """Return one sum per graph for a node-leading tensor."""
    if value.shape[:1] != batch.shape:
        raise ValueError("batch must provide one graph index per node")
    return scatter(value, batch, dim=0, dim_size=graph_count, reduce="sum")


def fractional_covector_to_cartesian(
    fractional_covector: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Express a fractional covector in the orthonormal Cartesian chart.

    For row coordinates ``r = f L``, covectors obey
    ``s_f = s_r L^T``. Solving ``L s_r^T = s_f^T`` avoids an explicit inverse
    and gives the unique Cartesian covector used by the equivalent physical
    loss metric. The reverse process still consumes the exact fractional
    covector; this function changes only the chart in which errors are
    compared.
    """
    if fractional_covector.ndim != 2 or fractional_covector.shape[-1] != 3:
        raise ValueError("fractional covector must have shape [sites,3]")
    if batch.shape != fractional_covector.shape[:1] or batch.dtype != torch.long:
        raise ValueError("batch must provide one graph index per covector")
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError("lattice must have shape [graphs,3,3]")
    if batch.numel() and int(batch.max()) >= lattice.shape[0]:
        raise ValueError("covector batch index exceeds the lattice batch")
    return torch.linalg.solve(
        lattice[batch], fractional_covector.unsqueeze(-1)
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
