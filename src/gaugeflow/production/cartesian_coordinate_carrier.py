"""Compact Cartesian moment carriers for the periodic coordinate score."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .state_projection import graph_mean, sorted_segment_sum


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
    symmetric-traceless moment ``Q``. In three dimensions the
    Cayley--Hamilton-closed family ``(m, Qm, Q^2m)`` contains every polynomial
    action of ``Q`` on ``m``. Learned operations act only on scalar channels,
    so Cartesian covariance is exact without frames or harmonic bases.
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
        if (
            edge_hidden.ndim != 2
            or edge_hidden.shape[1] != self.moment_projection.in_features
        ):
            raise ValueError("edge hidden features have the wrong carrier shape")
        if (
            edge_target.shape != (edge_hidden.shape[0],)
            or edge_target.dtype != torch.long
        ):
            raise ValueError("edge targets do not match carrier edges")
        if edge_direction.shape != (edge_hidden.shape[0], 3):
            raise ValueError("edge directions do not match carrier edges")
        if edge_envelope.shape != (edge_hidden.shape[0], 1):
            raise ValueError("edge envelope must have shape [edges,1]")
        if batch.shape != (vector_basis.shape[0],) or batch.dtype != torch.long:
            raise ValueError("carrier batch must provide one graph per node")
        if graph_count < 1:
            raise ValueError("Cartesian carrier needs at least one graph")

        # Geometry reductions stay FP32 under BF16 backbone autocast. This is
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
            first = sorted_segment_sum(
                first_messages, edge_target, vectors.shape[0]
            )
            first = first * degree_scale[:, None, None]

            identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
            dyad = torch.einsum("ei,ej->eij", directions, directions)
            dyad = dyad - identity / 3.0
            second_messages = (
                second_coefficients[:, :, None, None]
                * envelope[:, :, None, None]
                * dyad[:, None, :, :]
            )
            second = sorted_segment_sum(
                second_messages, edge_target, vectors.shape[0]
            )
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


class StateAdaptiveCartesianCarrierMixer(nn.Module):
    """Mix polar Cartesian carriers with invariant node-dependent weights.

    A single global readout can only select one vector from the carrier span
    with the same coefficients at every state and site.  This mixer keeps that
    readout as its base term and adds the bounded low-rank residual

    ``a_i = w_0 + U tanh(V h_i)``.

    ``h_i`` contains scalar node features and ``a_i`` therefore remains an
    invariant coefficient vector.  Mixing polar carriers with these scalars
    preserves Cartesian covariance.  ``U`` uses a small orthogonal
    initialization so both factors receive gradients on the first backward
    pass.  This is the only initialization; there is no compatibility
    dispatch or second coordinate branch.
    """

    def __init__(
        self,
        carrier_channels: int,
        state_dim: int,
        *,
        rank: int = 8,
    ) -> None:
        super().__init__()
        if carrier_channels < 1 or state_dim < 1 or rank < 1:
            raise ValueError("adaptive carrier dimensions must be positive")
        self.carrier_channels = int(carrier_channels)
        self.state_dim = int(state_dim)
        self.base_weight = nn.Parameter(torch.empty(carrier_channels))
        self.state_projection = nn.Linear(state_dim, rank, bias=False)
        self.carrier_projection = nn.Linear(rank, carrier_channels, bias=False)
        nn.init.kaiming_uniform_(self.base_weight.unsqueeze(0), a=5.0**0.5)
        nn.init.orthogonal_(self.carrier_projection.weight, gain=1.0e-2)

    @property
    def rank(self) -> int:
        return self.state_projection.out_features

    def forward(
        self,
        carrier: torch.Tensor,
        scalar_state: torch.Tensor,
    ) -> torch.Tensor:
        if carrier.ndim != 3 or carrier.shape[1:] != (
            self.carrier_channels,
            3,
        ):
            raise ValueError("adaptive mixer received the wrong carrier shape")
        if scalar_state.shape != (carrier.shape[0], self.state_dim):
            raise ValueError("adaptive mixer received the wrong scalar state")
        with torch.autocast(device_type=carrier.device.type, enabled=False):
            latent = torch.tanh(
                F.linear(scalar_state.float(), self.state_projection.weight.float())
            )
            residual = F.linear(
                latent, self.carrier_projection.weight.float()
            )
            coefficients = self.base_weight.float().unsqueeze(0) + residual
            return torch.einsum("nc,ncd->nd", coefficients, carrier.float())
