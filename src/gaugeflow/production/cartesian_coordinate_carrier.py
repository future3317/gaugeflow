"""Compact Cartesian moment carriers for the periodic coordinate score."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .state_projection import graph_mean


def _vector_rms_normalize(
    vectors: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    epsilon: float,
) -> torch.Tensor:
    energy = graph_mean(vectors.square().sum(-1) / 3.0, batch, graph_count)
    return vectors * (energy + epsilon).rsqrt()[batch, :, None]


def _stf_rms_normalize(
    tensors: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    epsilon: float,
) -> torch.Tensor:
    energy = graph_mean(
        tensors.square().sum(dim=(-1, -2)) / 5.0, batch, graph_count
    )
    return tensors * (energy + epsilon).rsqrt()[batch, :, None, None]


class CompactCartesianKrylovCarrier(nn.Module):
    """Form bounded polar carriers from first and rank-two edge moments.

    Scalar edge features produce a polar first moment ``m`` and an even
    symmetric-traceless moment ``Q``.  In three dimensions the
    Cayley--Hamilton-closed family ``(m, Qm, Q^2m)`` contains every polynomial
    action of ``Q`` on ``m``.  All learned operations act on scalar channels;
    Cartesian covariance is therefore exact and no frame or harmonic basis is
    constructed.
    """

    def __init__(
        self,
        hidden_dim: int,
        vector_channels: int,
        *,
        moment_channels: int = 16,
        rms_epsilon: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or vector_channels < 1 or moment_channels < 1:
            raise ValueError("Cartesian carrier dimensions must be positive")
        if rms_epsilon <= 0.0:
            raise ValueError("Cartesian carrier RMS epsilon must be positive")
        self.vector_channels = int(vector_channels)
        self.moment_channels = int(moment_channels)
        self.rms_epsilon = float(rms_epsilon)
        self.moment_projection = nn.Linear(
            hidden_dim, 2 * moment_channels, bias=False
        )
        nn.init.orthogonal_(self.moment_projection.weight)

    @property
    def output_channels(self) -> int:
        return self.vector_channels + 3 * self.moment_channels

    def forward(
        self,
        vector_basis: torch.Tensor,
        edge_hidden: torch.Tensor,
        edge_target: torch.Tensor,
        edge_direction: torch.Tensor,
        edge_envelope: torch.Tensor,
        batch: torch.Tensor,
        graph_count: int,
    ) -> torch.Tensor:
        if vector_basis.ndim != 3 or vector_basis.shape[1:] != (
            self.vector_channels,
            3,
        ):
            raise ValueError("vector basis has the wrong carrier shape")
        if edge_hidden.ndim != 2 or edge_hidden.shape[1] != self.moment_projection.in_features:
            raise ValueError("edge hidden features have the wrong carrier shape")
        if edge_target.shape != (edge_hidden.shape[0],) or edge_target.dtype != torch.long:
            raise ValueError("edge targets do not match carrier edges")
        if edge_direction.shape != (edge_hidden.shape[0], 3):
            raise ValueError("edge directions do not match carrier edges")
        if edge_envelope.shape != (edge_hidden.shape[0], 1):
            raise ValueError("edge envelope must have shape [edges,1]")
        if batch.shape != (vector_basis.shape[0],) or batch.dtype != torch.long:
            raise ValueError("carrier batch must provide one graph per node")
        if graph_count < 1:
            raise ValueError("Cartesian carrier needs at least one graph")

        # Geometry reductions stay FP32 under BF16 backbone autocast.  This is
        # one fixed typed path, not a selectable precision or legacy fallback.
        with torch.autocast(device_type=vector_basis.device.type, enabled=False):
            vectors = vector_basis.float()
            hidden = edge_hidden.float()
            directions = edge_direction.float()
            envelope = edge_envelope.float()
            coefficients = torch.tanh(
                F.linear(hidden, self.moment_projection.weight.float())
            )
            first_coefficients, second_coefficients = coefficients.split(
                self.moment_channels, dim=-1
            )
            degree = torch.bincount(
                edge_target, minlength=vector_basis.shape[0]
            ).float()
            degree_scale = degree.clamp_min(1.0).rsqrt()

            first_messages = (
                first_coefficients[:, :, None]
                * envelope[:, :, None]
                * directions[:, None, :]
            )
            first = vectors.new_zeros(
                (vectors.shape[0], self.moment_channels, 3)
            )
            first.index_add_(0, edge_target, first_messages)
            first = first * degree_scale[:, None, None]

            identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
            dyad = torch.einsum("ei,ej->eij", directions, directions)
            dyad = dyad - identity / 3.0
            second_messages = (
                second_coefficients[:, :, None, None]
                * envelope[:, :, None, None]
                * dyad[:, None, :, :]
            )
            second = vectors.new_zeros(
                (vectors.shape[0], self.moment_channels, 3, 3)
            )
            second.index_add_(0, edge_target, second_messages)
            second = second * degree_scale[:, None, None, None]

            vectors = _vector_rms_normalize(
                vectors, batch, graph_count, self.rms_epsilon
            )
            first = _vector_rms_normalize(
                first, batch, graph_count, self.rms_epsilon
            )
            second = _stf_rms_normalize(
                second, batch, graph_count, self.rms_epsilon
            )
            second_first = torch.einsum("ncij,ncj->nci", second, first)
            second_first = _vector_rms_normalize(
                second_first, batch, graph_count, self.rms_epsilon
            )
            second_squared_first = torch.einsum(
                "ncij,ncj->nci", second, second_first
            )
            second_squared_first = _vector_rms_normalize(
                second_squared_first, batch, graph_count, self.rms_epsilon
            )
            carrier = torch.cat(
                (vectors, first, second_first, second_squared_first), dim=1
            )
            return carrier - graph_mean(carrier, batch, graph_count)[batch]
