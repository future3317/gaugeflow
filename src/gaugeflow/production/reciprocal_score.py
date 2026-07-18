"""Smooth cell-covariant reciprocal score fields on the periodic quotient."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from gaugeflow.geometry import GaussianRadialBasis


@dataclass(frozen=True)
class ReciprocalBall:
    integer_modes: torch.Tensor
    cartesian_modes: torch.Tensor
    norms: torch.Tensor
    mask: torch.Tensor


def reciprocal_ball(lattice: torch.Tensor, cutoff: float) -> ReciprocalBall:
    """Enumerate the reciprocal ball modulo the exact ``k ~ -k`` pairing.

    Integer modes are padded only across graphs; no node--mode Python loop is
    used.  The physical cutoff and cosine envelope make the set covariant under
    unimodular cell changes and continuous when a mode crosses the boundary.
    """
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError("reciprocal modes require lattice shape [graphs,3,3]")
    if cutoff <= 0.0 or not torch.isfinite(lattice).all():
        raise ValueError("reciprocal cutoff and lattice must be finite and positive")
    graphs = lattice.shape[0]
    # q = 2*pi*k*L^-T and k = q*L^T/(2*pi), hence this row-norm
    # Cauchy--Schwarz bound contains the complete integer reciprocal ball.
    with torch.autocast(device_type=lattice.device.type, enabled=False):
        bounds = torch.ceil(
            cutoff * torch.linalg.vector_norm(lattice, dim=-1) / (2.0 * math.pi)
            + 1.0
        ).to(torch.long)
    maximum = bounds.amax(dim=0).detach().cpu().tolist()
    axes = [
        torch.arange(-int(value), int(value) + 1, device=lattice.device)
        for value in maximum
    ]
    grid = torch.cartesian_prod(*axes)
    within_bounds = (grid.abs().unsqueeze(0) <= bounds.unsqueeze(1)).all(dim=-1)
    with torch.autocast(device_type=lattice.device.type, enabled=False):
        reciprocal_basis = (
            2.0 * math.pi * torch.linalg.inv(lattice).transpose(-1, -2)
        )
        cartesian = torch.einsum(
            "qj,gjk->gqk", grid.to(lattice), reciprocal_basis
        )
        norms = torch.linalg.vector_norm(cartesian, dim=-1)
    positive_representative = (grid[:, 0] > 0) | (
        (grid[:, 0] == 0)
        & (
            (grid[:, 1] > 0)
            | ((grid[:, 1] == 0) & (grid[:, 2] > 0))
        )
    )
    valid = (
        within_bounds
        & positive_representative.unsqueeze(0)
        & (norms < cutoff)
    )
    graph, column = torch.nonzero(valid, as_tuple=True)
    counts = torch.bincount(graph, minlength=graphs)
    maximum_count = int(counts.max().detach().cpu()) if graphs else 0
    integer_modes = grid.new_zeros((graphs, maximum_count, 3))
    cartesian_modes = lattice.new_zeros((graphs, maximum_count, 3))
    padded_norms = lattice.new_zeros((graphs, maximum_count))
    mask = torch.zeros(
        (graphs, maximum_count), dtype=torch.bool, device=lattice.device
    )
    if graph.numel():
        starts = torch.repeat_interleave(
            torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1])), counts
        )
        slot = torch.arange(graph.numel(), device=lattice.device) - starts
        integer_modes[graph, slot] = grid[column]
        cartesian_modes[graph, slot] = cartesian[graph, column]
        padded_norms[graph, slot] = norms[graph, column]
        mask[graph, slot] = True
    return ReciprocalBall(integer_modes, cartesian_modes, padded_norms, mask)


class ReciprocalStructureFactorScore(nn.Module):
    """Low-rank global periodic score with O(N*K*C) structure factors.

    For node channels ``a_ic`` and reciprocal modes ``q_k``, the field uses

        a_ic Im[e^{-i k.f_i} sum_j a_jc e^{i k.f_j}] q_k.

    Every pair contribution is antisymmetric, so common translation is removed
    by construction.  Summing the complete physical reciprocal ball is
    independent of mode enumeration and covariant under GL(3,Z) cell changes.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        channels: int = 8,
        radial_dim: int = 8,
        cutoff: float = 4.0,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("reciprocal score needs at least one channel")
        self.channels = int(channels)
        self.radial = GaussianRadialBasis(radial_dim, cutoff)
        self.node_amplitude = nn.Linear(hidden_dim, channels)
        self.mode_channels = nn.Sequential(
            nn.Linear(radial_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        # Start as an exact zero residual so introducing the spectral basis does
        # not perturb the other trained heads at initialization.
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
            raise ValueError("reciprocal score node tensors have incompatible shapes")
        if batch.shape != nodes.shape[:1] or batch.dtype != torch.long:
            raise ValueError("reciprocal score requires one graph index per node")
        graphs = lattice.shape[0]
        ball = reciprocal_ball(lattice, self.radial.cutoff)
        modes = ball.integer_modes.shape[1]
        if modes == 0:
            return fractional_coordinates.new_zeros(fractional_coordinates.shape)
        phase = 2.0 * math.pi * torch.einsum(
            "ni,nki->nk",
            fractional_coordinates,
            ball.integer_modes[batch].to(fractional_coordinates),
        )
        sine, cosine = phase.sin(), phase.cos()
        amplitude = self.node_amplitude(nodes)
        real = nodes.new_zeros((graphs, modes, self.channels))
        imaginary = nodes.new_zeros((graphs, modes, self.channels))
        real.index_add_(
            0,
            batch,
            (cosine.unsqueeze(-1) * amplitude.unsqueeze(1)).to(real.dtype),
        )
        imaginary.index_add_(
            0,
            batch,
            (sine.unsqueeze(-1) * amplitude.unsqueeze(1)).to(imaginary.dtype),
        )
        counts = torch.bincount(batch, minlength=graphs).to(nodes).sqrt()
        local_imaginary = (
            imaginary[batch] * cosine.unsqueeze(-1)
            - real[batch] * sine.unsqueeze(-1)
        ) / counts[batch, None, None]
        mode_features = self.radial(ball.norms.reshape(-1)).reshape(
            graphs, modes, -1
        )
        mode_channels = self.mode_channels(mode_features)
        channel_response = (
            amplitude.unsqueeze(1) * local_imaginary * mode_channels[batch]
        ).sum(dim=-1) / math.sqrt(self.channels)
        channel_response = channel_response * ball.mask[batch].to(channel_response)
        mode_counts = ball.mask.sum(dim=-1).clamp_min(1).to(nodes).sqrt()
        # The omitted -k member gives the identical sin(phase)*q response.
        # sqrt(2) retains the normalization of the full symmetric ball while
        # halving mode memory and arithmetic exactly.
        return math.sqrt(2.0) * (
            channel_response.unsqueeze(-1) * ball.cartesian_modes[batch]
        ).sum(dim=1) / mode_counts[batch].unsqueeze(-1)
