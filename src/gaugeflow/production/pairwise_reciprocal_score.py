"""Pairwise reciprocal-torus score covectors for the production denoiser."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from gaugeflow.geometry import GaussianRadialBasis


@dataclass(frozen=True)
class ProjectiveReciprocalBall:
    """Padded complete physical reciprocal balls modulo ``k ~ -k``."""

    integer_modes: torch.Tensor
    cartesian_covectors: torch.Tensor
    norms: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class UnorderedNodePairs:
    """All unordered within-graph node pairs in contiguous packed graphs."""

    first: torch.Tensor
    second: torch.Tensor
    graph: torch.Tensor


def projective_reciprocal_ball(
    lattice: torch.Tensor,
    cutoff: float,
) -> ProjectiveReciprocalBall:
    """Enumerate every physical reciprocal covector below ``cutoff`` once.

    For row-style ``r=fL``, reciprocal covectors are
    ``q_k=2*pi*k*L^-T``.  Cauchy--Schwarz gives a complete finite integer
    search bound for each graph.  Padding is only across graphs; candidate
    construction and selection are tensorized.
    """
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError("reciprocal modes require lattice shape [graphs,3,3]")
    if cutoff <= 0.0 or not math.isfinite(cutoff):
        raise ValueError("reciprocal cutoff must be finite and positive")
    if not torch.isfinite(lattice).all():
        raise ValueError("reciprocal lattice must be finite")
    graphs = lattice.shape[0]
    if graphs < 1:
        raise ValueError("reciprocal modes require at least one graph")

    # q=2*pi*k*L^-T implies k=q*L^T/(2*pi).  The norm of row j
    # of L therefore bounds integer component k_j for every |q|<cutoff.
    with torch.autocast(device_type=lattice.device.type, enabled=False):
        bounds = torch.ceil(
            cutoff
            * torch.linalg.vector_norm(lattice, dim=-1)
            / (2.0 * math.pi)
            + 1.0
        ).to(torch.long)
        maximum = bounds.amax(dim=0)
        axis_0 = torch.arange(
            -int(maximum[0]), int(maximum[0]) + 1, device=lattice.device
        )
        axis_1 = torch.arange(
            -int(maximum[1]), int(maximum[1]) + 1, device=lattice.device
        )
        axis_2 = torch.arange(
            -int(maximum[2]), int(maximum[2]) + 1, device=lattice.device
        )
        grid_0, grid_1, grid_2 = torch.meshgrid(
            axis_0, axis_1, axis_2, indexing="ij"
        )
        integer_grid = torch.stack((grid_0, grid_1, grid_2), dim=-1).reshape(-1, 3)
        within_bound = (
            integer_grid.abs().unsqueeze(0) <= bounds.unsqueeze(1)
        ).all(dim=-1)
        reciprocal_basis = (
            2.0 * math.pi * torch.linalg.inv(lattice).transpose(-1, -2)
        )
        cartesian_grid = torch.einsum(
            "ki,gij->gkj", integer_grid.to(lattice), reciprocal_basis
        )
        norms = torch.linalg.vector_norm(cartesian_grid, dim=-1)

    # One deterministic representative of each exact {k,-k} pair.
    projective = (integer_grid[:, 0] > 0) | (
        (integer_grid[:, 0] == 0)
        & (
            (integer_grid[:, 1] > 0)
            | ((integer_grid[:, 1] == 0) & (integer_grid[:, 2] > 0))
        )
    )
    valid = within_bound & projective.unsqueeze(0) & (norms < cutoff)
    graph, column = torch.nonzero(valid, as_tuple=True)
    counts = torch.bincount(graph, minlength=graphs)
    maximum_count = int(counts.max()) if graph.numel() else 0
    integer_modes = integer_grid.new_zeros((graphs, maximum_count, 3))
    cartesian_covectors = lattice.new_zeros((graphs, maximum_count, 3))
    padded_norms = lattice.new_zeros((graphs, maximum_count))
    mask = torch.zeros(
        (graphs, maximum_count), dtype=torch.bool, device=lattice.device
    )
    if graph.numel():
        starts = torch.repeat_interleave(
            torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1])), counts
        )
        slot = torch.arange(graph.numel(), device=lattice.device) - starts
        integer_modes[graph, slot] = integer_grid[column]
        cartesian_covectors[graph, slot] = cartesian_grid[graph, column]
        padded_norms[graph, slot] = norms[graph, column]
        mask[graph, slot] = True
    return ProjectiveReciprocalBall(
        integer_modes=integer_modes,
        cartesian_covectors=cartesian_covectors,
        norms=padded_norms,
        mask=mask,
    )


def complete_unordered_node_pairs(
    batch: torch.Tensor,
    graph_count: int,
) -> UnorderedNodePairs:
    """Construct all ``i<j`` pairs without a graph or pair Python loop."""
    if batch.ndim != 1 or batch.dtype != torch.long:
        raise ValueError("pair construction requires a rank-one int64 batch")
    if graph_count < 1 or batch.numel() < graph_count:
        raise ValueError("every packed graph must contain at least one node")
    if not torch.equal(batch, torch.sort(batch).values):
        raise ValueError("pair construction requires graph-contiguous nodes")
    counts = torch.bincount(batch, minlength=graph_count)
    if counts.shape != (graph_count,) or bool((counts < 1).any()):
        raise ValueError("batch does not cover every declared graph")
    maximum_count = int(counts.max())
    local_first, local_second = torch.triu_indices(
        maximum_count,
        maximum_count,
        offset=1,
        device=batch.device,
    )
    graph_grid = torch.arange(graph_count, device=batch.device).unsqueeze(1)
    valid = local_second.unsqueeze(0) < counts.unsqueeze(1)
    offsets = counts.cumsum(0) - counts
    return UnorderedNodePairs(
        first=(offsets.unsqueeze(1) + local_first.unsqueeze(0))[valid],
        second=(offsets.unsqueeze(1) + local_second.unsqueeze(0))[valid],
        graph=graph_grid.expand_as(valid)[valid],
    )


class PairwiseReciprocalScore(nn.Module):
    """Signed symmetric pair coefficients on a smooth reciprocal ball.

    Unlike a node-amplitude structure-factor factorization, ``b_ij`` is
    produced from a symmetric MLP over the complete unordered pair.  The
    remaining small channel factorization is only over reciprocal radius.
    Pair forces are accumulated with opposite signs, making the output an
    exactly translation-horizontal Cartesian covector field.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        pair_width: int = 32,
        channels: int = 8,
        radial_dim: int = 8,
        cutoff: float = 4.0,
    ) -> None:
        super().__init__()
        if pair_width < 2 or channels < 1:
            raise ValueError("pair reciprocal dimensions must be positive")
        self.channels = int(channels)
        self.radial = GaussianRadialBasis(radial_dim, cutoff)
        self.node_projection = nn.Linear(hidden_dim, pair_width)
        self.pair_channels = nn.Sequential(
            nn.Linear(2 * pair_width, pair_width),
            nn.SiLU(),
            nn.Linear(pair_width, channels),
        )
        self.mode_channels = nn.Sequential(
            nn.Linear(radial_dim, pair_width),
            nn.SiLU(),
            nn.Linear(pair_width, channels),
        )
        # The qualified residual starts at exactly zero without suppressing
        # gradients to the final mode layer on its first optimization step.
        nn.init.zeros_(self.mode_channels[-1].weight)
        nn.init.zeros_(self.mode_channels[-1].bias)

    def forward(
        self,
        nodes: torch.Tensor,
        fractional_coordinates: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if nodes.ndim != 2 or fractional_coordinates.shape != (nodes.shape[0], 3):
            raise ValueError("pair reciprocal node tensors have incompatible shapes")
        if batch.shape != nodes.shape[:1] or batch.dtype != torch.long:
            raise ValueError("pair reciprocal score requires one graph index per node")
        graphs = lattice.shape[0]
        pairs = complete_unordered_node_pairs(batch, graphs)
        ball = projective_reciprocal_ball(lattice, self.radial.cutoff)
        if pairs.first.numel() == 0 or ball.integer_modes.shape[1] == 0:
            return fractional_coordinates.new_zeros(fractional_coordinates.shape)

        projected = self.node_projection(nodes)
        first = projected[pairs.first]
        second = projected[pairs.second]
        pair_channels = self.pair_channels(
            torch.cat((first + second, first * second), dim=-1)
        )
        mode_features = self.radial(ball.norms.reshape(-1)).reshape(
            graphs, ball.norms.shape[1], -1
        )
        mode_channels = self.mode_channels(mode_features)
        phase = 2.0 * math.pi * torch.einsum(
            "pi,pki->pk",
            fractional_coordinates[pairs.first]
            - fractional_coordinates[pairs.second],
            ball.integer_modes[pairs.graph].to(fractional_coordinates),
        )
        coefficient = torch.einsum(
            "pc,pkc->pk", pair_channels, mode_channels[pairs.graph]
        ) / math.sqrt(self.channels)
        coefficient = (
            coefficient
            * phase.sin()
            * ball.mask[pairs.graph].to(coefficient)
        )
        pair_score = math.sqrt(2.0) * torch.einsum(
            "pk,pki->pi",
            coefficient,
            ball.cartesian_covectors[pairs.graph].to(coefficient),
        )
        node_counts = torch.bincount(batch, minlength=graphs).to(pair_score)
        mode_counts = ball.mask.sum(dim=-1).clamp_min(1).to(pair_score)
        normalization = (
            node_counts[pairs.graph] * mode_counts[pairs.graph]
        ).sqrt()
        pair_score = pair_score / normalization.unsqueeze(-1)

        result = fractional_coordinates.new_zeros(fractional_coordinates.shape)
        result.index_add_(0, pairs.first, pair_score.to(result.dtype))
        result.index_add_(0, pairs.second, -pair_score.to(result.dtype))
        return result
