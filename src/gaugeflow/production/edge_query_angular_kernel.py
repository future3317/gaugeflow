"""Permutation-safe explicit and induced edge-query angular kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .factorized_angular_moments import FactorizedCartesianAngularMoments
from .state_projection import sorted_segment_sum


@dataclass(frozen=True)
class ShellCompleteNeighbors:
    """Dense nearest-K partner indices with the complete boundary shell."""

    edge_index: torch.Tensor
    valid: torch.Tensor
    selected_count: torch.Tensor


@dataclass(frozen=True)
class InducedSlotStatistics:
    """Assignment probabilities and normalized Cartesian slot moments."""

    probability: torch.Tensor
    mass: torch.Tensor
    scalar: torch.Tensor
    vector: torch.Tensor
    stf2: torch.Tensor


def shell_complete_nearest_neighbors(
    distance: torch.Tensor,
    target: torch.Tensor,
    node_count: int,
    *,
    k: int,
    tie_tolerance: float = 1.0e-6,
) -> ShellCompleteNeighbors:
    """Select at least K incoming edges without cutting an equal-distance shell.

    The radius graph is target sorted.  Selection depends only on physical
    distance and includes every edge tied with the Kth distance, so summation
    is invariant to atom/edge enumeration even at a symmetric shell boundary.
    """
    if distance.ndim != 1 or target.shape != distance.shape:
        raise ValueError("shell-complete selection requires one target and distance per edge")
    if target.dtype != torch.long or node_count < 0 or k < 1 or tie_tolerance < 0.0:
        raise ValueError("invalid shell-complete neighbor specification")
    if distance.numel() == 0:
        return ShellCompleteNeighbors(
            target.new_empty((node_count, 0)),
            torch.empty((node_count, 0), dtype=torch.bool, device=target.device),
            target.new_zeros((node_count,)),
        )
    degree = torch.bincount(target, minlength=node_count)
    maximum_degree = int(degree.max().detach().cpu())
    starts = degree.cumsum(0) - degree
    local = torch.arange(target.numel(), device=target.device) - starts[target]
    padded_distance = distance.new_full((node_count, maximum_degree), float("inf"))
    padded_index = target.new_full((node_count, maximum_degree), -1)
    padded_distance[target, local] = distance
    padded_index[target, local] = torch.arange(target.numel(), device=target.device)
    ordered_distance = torch.sort(padded_distance, dim=-1).values
    boundary_column = degree.clamp(min=1, max=k) - 1
    boundary = ordered_distance.gather(1, boundary_column[:, None]).squeeze(1)
    selected = (padded_index >= 0) & (
        padded_distance <= boundary[:, None] + tie_tolerance
    )
    selected_count = selected.sum(dim=-1)
    maximum_selected = int(selected_count.max().detach().cpu())
    packed_index = target.new_full((node_count, maximum_selected), -1)
    rank = selected.cumsum(dim=-1) - 1
    node, column = torch.nonzero(selected, as_tuple=True)
    packed_index[node, rank[node, column]] = padded_index[node, column]
    valid = packed_index >= 0
    return ShellCompleteNeighbors(packed_index, valid, selected_count)


class ShellCompleteTopKTripletKernel(nn.Module):
    """Strong explicit edge-edge angular control with bounded local support."""

    def __init__(self, edge_dim: int, channels: int, *, k: int = 8) -> None:
        super().__init__()
        if edge_dim < 1 or channels < 1 or k < 1:
            raise ValueError("explicit triplet dimensions must be positive")
        self.edge_dim = int(edge_dim)
        self.channels = int(channels)
        self.k = int(k)
        self.partner_value = nn.Linear(edge_dim, channels, bias=False)
        self.partner_key = nn.Linear(edge_dim, channels, bias=False)
        self.query_key = nn.Linear(edge_dim, channels, bias=False)
        self.angular_gate = nn.Sequential(
            nn.Linear(4, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.readout = nn.Sequential(
            nn.Linear(edge_dim + channels, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, 2 * channels),
        )

    @property
    def output_dim(self) -> int:
        return 2 * self.channels

    def forward(
        self,
        edge_state: torch.Tensor,
        edge_target: torch.Tensor,
        edge_direction: torch.Tensor,
        edge_envelope: torch.Tensor,
        neighbors: ShellCompleteNeighbors,
    ) -> torch.Tensor:
        if edge_state.ndim != 2 or edge_state.shape[1] != self.edge_dim:
            raise ValueError("explicit triplet kernel received the wrong edge state")
        if edge_target.shape != edge_state.shape[:1] or edge_target.dtype != torch.long:
            raise ValueError("explicit triplet kernel requires one int64 target per edge")
        if edge_direction.shape != (edge_state.shape[0], 3):
            raise ValueError("explicit triplet kernel requires Cartesian edge directions")
        if edge_envelope.shape != (edge_state.shape[0], 1):
            raise ValueError("explicit triplet kernel requires one edge envelope")
        if edge_state.shape[0] == 0:
            return edge_state.new_empty((0, self.output_dim))
        partner_index = neighbors.edge_index[edge_target]
        partner_valid = neighbors.valid[edge_target]
        partner = partner_index.clamp_min(0)
        cosine = torch.einsum(
            "ei,eki->ek", edge_direction, edge_direction[partner]
        )
        angular_basis = torch.stack(
            (cosine, cosine.square(), cosine.pow(3), cosine.pow(4)), dim=-1
        )
        gate = torch.sigmoid(
            self.query_key(edge_state)[:, None, :]
            + self.partner_key(edge_state)[partner]
            + self.angular_gate(angular_basis)
        )
        message = (
            torch.tanh(self.partner_value(edge_state))[partner]
            * gate
            * edge_envelope[partner]
            * partner_valid[:, :, None].to(edge_state)
        )
        normalization = partner_valid.sum(dim=-1).clamp_min(1).to(edge_state).rsqrt()
        aggregate = message.sum(dim=1) * normalization[:, None]
        return self.readout(torch.cat((edge_state, aggregate), dim=-1))


class InducedEdgeQueryAngularKernel(nn.Module):
    r"""Low-rank induced approximation to a complete edge-edge kernel.

    Each incoming edge is softly assigned to ``R`` latent slots.  Per-slot
    scalar, vector and rank-two STF moments are aggregated once per node.  A
    query edge contracts those covariants with its own direction and combines
    the slots through a state-dependent gate.  No explicit edge pair is built.
    """

    def __init__(
        self,
        edge_dim: int,
        channels: int,
        *,
        slots: int,
        slot_chunk: int = 4,
    ) -> None:
        super().__init__()
        if edge_dim < 1 or channels < 1 or slots < 2 or slot_chunk < 1:
            raise ValueError("induced angular-kernel dimensions are invalid")
        self.edge_dim = int(edge_dim)
        self.channels = int(channels)
        self.slots = int(slots)
        self.slot_chunk = int(slot_chunk)
        self.assignment = nn.Linear(edge_dim, slots, bias=False)
        self.coefficients = nn.Linear(edge_dim, 3 * channels, bias=False)
        self.query_gate = nn.Linear(edge_dim, slots, bias=False)
        self.slot_readout = nn.Sequential(
            nn.Linear(3 * channels, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, 2 * channels),
        )

    @property
    def output_dim(self) -> int:
        return 2 * self.channels

    @staticmethod
    def _quadratic_basis(direction: torch.Tensor) -> torch.Tensor:
        return FactorizedCartesianAngularMoments._quadratic_basis(direction)

    @staticmethod
    def _quadratic_contraction(
        moment: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        # moment is [edges,slots,channels,6].
        x, y, z = direction.unbind(dim=-1)
        return (
            moment[..., 0] * x[:, None, None].square()
            + moment[..., 1] * y[:, None, None].square()
            + moment[..., 2] * z[:, None, None].square()
            + 2.0 * moment[..., 3] * (x * y)[:, None, None]
            + 2.0 * moment[..., 4] * (x * z)[:, None, None]
            + 2.0 * moment[..., 5] * (y * z)[:, None, None]
        )

    def assignment_probabilities(self, edge_state: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.assignment(edge_state), dim=-1)

    def slot_statistics(
        self,
        edge_state: torch.Tensor,
        edge_target: torch.Tensor,
        edge_direction: torch.Tensor,
        edge_envelope: torch.Tensor,
        node_count: int,
    ) -> InducedSlotStatistics:
        """Build all slots once; used by both runtime and causal diagnostics."""
        probability = self.assignment_probabilities(edge_state)
        scalar, first, second = torch.tanh(self.coefficients(edge_state)).split(
            self.channels, dim=-1
        )
        envelope = edge_envelope.squeeze(-1)
        quadratic = self._quadratic_basis(edge_direction)
        slot_mass = sorted_segment_sum(
            probability * envelope[:, None], edge_target, node_count
        )
        normalization = slot_mass.clamp_min(1.0e-6).rsqrt()
        scalar_moment = sorted_segment_sum(
            probability[:, :, None] * scalar[:, None, :] * envelope[:, None, None],
            edge_target,
            node_count,
        ) * normalization[:, :, None]
        first_moment = sorted_segment_sum(
            probability[:, :, None, None]
            * first[:, None, :, None]
            * edge_direction[:, None, None, :]
            * envelope[:, None, None, None],
            edge_target,
            node_count,
        ) * normalization[:, :, None, None]
        second_moment = sorted_segment_sum(
            probability[:, :, None, None]
            * second[:, None, :, None]
            * quadratic[:, None, None, :]
            * envelope[:, None, None, None],
            edge_target,
            node_count,
        ) * normalization[:, :, None, None]
        return InducedSlotStatistics(
            probability=probability,
            mass=slot_mass,
            scalar=scalar_moment,
            vector=first_moment,
            stf2=second_moment,
        )

    def forward(
        self,
        edge_state: torch.Tensor,
        edge_target: torch.Tensor,
        edge_direction: torch.Tensor,
        edge_envelope: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        if edge_state.ndim != 2 or edge_state.shape[1] != self.edge_dim:
            raise ValueError("induced kernel received the wrong edge-state shape")
        if edge_target.shape != edge_state.shape[:1] or edge_target.dtype != torch.long:
            raise ValueError("induced kernel requires one int64 target per edge")
        if edge_direction.shape != (edge_state.shape[0], 3):
            raise ValueError("induced kernel requires Cartesian edge directions")
        if edge_envelope.shape != (edge_state.shape[0], 1):
            raise ValueError("induced kernel requires one edge envelope")
        if node_count < 0:
            raise ValueError("induced kernel requires a nonnegative node count")
        if edge_state.shape[0] == 0:
            return edge_state.new_empty((0, self.output_dim))

        statistics = self.slot_statistics(
            edge_state,
            edge_target,
            edge_direction,
            edge_envelope,
            node_count,
        )

        query_gate = torch.sigmoid(self.query_gate(edge_state))
        output = edge_state.new_zeros((edge_state.shape[0], self.output_dim))
        for start in range(0, self.slots, self.slot_chunk):
            stop = min(start + self.slot_chunk, self.slots)
            selected_scalar = statistics.scalar[edge_target, start:stop]
            selected_first = statistics.vector[edge_target, start:stop]
            linear = torch.einsum(
                "erci,ei->erc", selected_first, edge_direction
            )
            selected_second = statistics.stf2[edge_target, start:stop]
            quadratic_value = self._quadratic_contraction(
                selected_second, edge_direction
            )
            descriptor = torch.cat(
                (selected_scalar, linear, quadratic_value), dim=-1
            )
            slot_value = self.slot_readout(descriptor)
            output = output + (
                slot_value * query_gate[:, start:stop, None]
            ).sum(dim=1)
        return output / math.sqrt(float(self.slots))
