"""Harmonic relative-frame conditioner for the production diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from e3nn import o3
from torch import nn
from torch_geometric.utils import scatter

from gaugeflow.harmonic import (
    OrbitInvariantConditionEncoder,
    piezo_irrep_blocks,
    rotate_piezo_irreps_on_grid,
)
from gaugeflow.tensor import fixed_lossless_response_probes, piezo_from_irreps, response_field

from .schedules import CosineNoiseSchedule


def _radical_inverse(index: torch.Tensor, base: int) -> torch.Tensor:
    value = torch.zeros_like(index, dtype=torch.float64)
    factor = 1.0 / base
    remaining = index.clone().to(dtype=torch.long)
    while bool((remaining > 0).any()):
        value = value + factor * (remaining % base).to(value)
        remaining = torch.div(remaining, base, rounding_mode="floor")
        factor /= base
    return value


def nested_hopf_so3_grid(
    count: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Nested deterministic Haar-QMC rotations, with identity first.

    Unlike a count-normalized construction, the first ``K`` nodes are exactly
    preserved when the rule is refined.  This makes finite-grid convergence a
    reproducible numerical question rather than a comparison of unrelated
    random grids.
    """
    if count < 1:
        raise ValueError("SO(3) grid count must be positive")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    index = torch.arange(1, count + 1, device=target_device)
    u = _radical_inverse(index, 2).to(device=target_device)
    v = _radical_inverse(index, 3).to(device=target_device)
    w = _radical_inverse(index, 5).to(device=target_device)
    q = torch.stack(
        (
            (1.0 - u).sqrt() * torch.sin(2.0 * math.pi * v),
            (1.0 - u).sqrt() * torch.cos(2.0 * math.pi * v),
            u.sqrt() * torch.sin(2.0 * math.pi * w),
            u.sqrt() * torch.cos(2.0 * math.pi * w),
        ),
        dim=-1,
    )
    x, y, z, scalar = q.unbind(dim=-1)
    rotation = torch.stack(
        (
            1 - 2 * (y.square() + z.square()), 2 * (x * y - z * scalar), 2 * (x * z + y * scalar),
            2 * (x * y + z * scalar), 1 - 2 * (x.square() + z.square()), 2 * (y * z - x * scalar),
            2 * (x * z - y * scalar), 2 * (y * z + x * scalar), 1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(count, 3, 3).to(dtype=dtype)
    rotation[0] = torch.eye(3, dtype=dtype, device=target_device)
    return rotation


@dataclass(frozen=True)
class HarmonicGaugeFlowOutput:
    graph_condition: torch.Tensor
    edge_response: torch.Tensor
    posterior: torch.Tensor
    aligned_irreps: torch.Tensor
    aligned_tensor: torch.Tensor
    gate: torch.Tensor
    entropy: torch.Tensor


def weighted_geometric_harmonic_queries(
    directions: torch.Tensor,
    edge_graph: torch.Tensor,
    edge_query_weights: torch.Tensor,
    graph_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chemistry/metric-weighted O(3)-covariant queries for l=1,2,3.

    A bare mean over a directed edge list would cancel every odd degree because
    each ``n_ij`` is paired with ``-n_ij``.  Condition-free scalar weights from
    the current token/metric state break that artificial cancellation while
    retaining rotation covariance and node-permutation equivariance.
    """
    if edge_query_weights.ndim != 2 or edge_query_weights.shape[0] != directions.shape[0]:
        raise ValueError("edge query weights must have shape [edges,channels]")
    if directions.numel() == 0:
        channels = edge_query_weights.shape[-1]
        return tuple(
            directions.new_zeros((graph_count, channels, degree)) for degree in (3, 5, 7)
        )  # type: ignore[return-value]
    harmonics = o3.spherical_harmonics(
        [1, 2, 3], torch.nn.functional.normalize(directions, dim=-1),
        normalize=True, normalization="component",
    )
    return tuple(
        scatter(
            edge_query_weights.unsqueeze(-1) * harmonics[:, None, start:stop],
            edge_graph,
            dim=0,
            dim_size=graph_count,
            reduce="mean",
        )
        for start, stop in ((0, 3), (3, 8), (8, 15))
    )  # type: ignore[return-value]


def weighted_harmonic_alignment_scores(
    piezo_irreps: torch.Tensor,
    directions: torch.Tensor,
    edge_graph: torch.Tensor,
    edge_query_weights: torch.Tensor,
    rotations: torch.Tensor,
    *,
    coupling_l1: torch.Tensor,
    coupling_l2: torch.Tensor,
    coupling_l3: torch.Tensor,
) -> torch.Tensor:
    """Band-limited continuous score with learned multiplicity couplings."""
    graphs = piezo_irreps.shape[0]
    queries = weighted_geometric_harmonic_queries(
        directions, edge_graph, edge_query_weights, graphs
    )
    rotated = rotate_piezo_irreps_on_grid(piezo_irreps, rotations)
    blocks = piezo_irrep_blocks(rotated.reshape(-1, 18))
    first = blocks[0].reshape(graphs, rotations.shape[0], 2, 3)
    second = blocks[1].reshape(graphs, rotations.shape[0], 1, 5)
    third = blocks[2].reshape(graphs, rotations.shape[0], 1, 7)
    channels = edge_query_weights.shape[-1]
    if coupling_l1.shape != (channels, 2) or coupling_l2.shape != (channels, 1) or coupling_l3.shape != (channels, 1):
        raise ValueError("harmonic coupling matrices do not match query/condition multiplicities")
    return (
        torch.einsum("bfmi,bai,am->bf", first, queries[0], coupling_l1.to(first)) / math.sqrt(3.0)
        + torch.einsum("bfmi,bai,am->bf", second, queries[1], coupling_l2.to(second)) / math.sqrt(5.0)
        + torch.einsum("bfmi,bai,am->bf", third, queries[2], coupling_l3.to(third)) / math.sqrt(7.0)
    )


class HarmonicGaugeFlowConditioner(nn.Module):
    """One graph-level posterior and a lossless aligned response field."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        grid_size: int = 240,
        lambda_max: float = 1.0,
        query_channels: int = 2,
        schedule: CosineNoiseSchedule | None = None,
    ) -> None:
        super().__init__()
        if grid_size < 2 or not 0.0 < lambda_max <= 1.0 or query_channels < 1:
            raise ValueError("harmonic grid and lambda_max are outside their valid range")
        self.lambda_max = float(lambda_max)
        self.query_channels = int(query_channels)
        self.schedule = schedule or CosineNoiseSchedule()
        self.invariant = OrbitInvariantConditionEncoder(hidden_dim)
        # Six 3-vector response probes are lossless for the symmetric final
        # tensor indices. Entropy and top mass describe posterior confidence.
        self.aligned_token = nn.Sequential(
            nn.Linear(20, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.coupling_l1 = nn.Parameter(torch.ones(query_channels, 2))
        self.coupling_l2 = nn.Parameter(torch.ones(query_channels, 1))
        self.coupling_l3 = nn.Parameter(torch.ones(query_channels, 1))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.null_condition = nn.Parameter(torch.zeros(hidden_dim))
        self.present_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.register_buffer("rotations", nested_hopf_so3_grid(grid_size))
        self.register_buffer("probes", fixed_lossless_response_probes())

    def forward(
        self,
        piezo_irreps: torch.Tensor,
        condition_present: torch.Tensor,
        edge_directions: torch.Tensor,
        edge_graph: torch.Tensor,
        edge_query_weights: torch.Tensor,
        time: torch.Tensor,
    ) -> HarmonicGaugeFlowOutput:
        graphs = piezo_irreps.shape[0]
        if piezo_irreps.shape != (graphs, 18):
            raise ValueError("piezo irreps must have shape [graphs,18]")
        if condition_present.shape not in {(graphs,), (graphs, 1)}:
            raise ValueError("condition-present flag must provide one value per graph")
        if time.shape != (graphs,):
            raise ValueError("time must provide one value per graph")
        if edge_directions.ndim != 2 or edge_directions.shape[-1] != 3:
            raise ValueError("edge directions must have shape [edges,3]")
        if edge_graph.shape != edge_directions.shape[:1]:
            raise ValueError("edge graph must provide one index per direction")
        if edge_query_weights.shape != (edge_directions.shape[0], self.query_channels):
            raise ValueError("edge query weights do not match edges and configured channels")
        rotations = self.rotations.to(piezo_irreps)
        score = weighted_harmonic_alignment_scores(
            piezo_irreps,
            edge_directions,
            edge_graph,
            edge_query_weights,
            rotations,
            coupling_l1=self.coupling_l1,
            coupling_l2=self.coupling_l2,
            coupling_l3=self.coupling_l3,
        )
        temperature = self.log_temperature.exp().clamp_min(1e-4).to(score)
        posterior = torch.softmax(score / temperature, dim=-1)
        rotated = rotate_piezo_irreps_on_grid(piezo_irreps, rotations)
        aligned_irreps = (posterior.unsqueeze(-1) * rotated).sum(dim=1)
        aligned_tensor = piezo_from_irreps(aligned_irreps)
        entropy = -(posterior * posterior.clamp_min(torch.finfo(posterior.dtype).tiny).log()).sum(-1)
        confidence = 1.0 - entropy / math.log(float(posterior.shape[-1]))
        snr = self.schedule.snr(time)
        gate = self.lambda_max * (snr / (1.0 + snr)) * confidence
        fixed_response = response_field(
            aligned_tensor.unsqueeze(1), self.probes.to(aligned_tensor).unsqueeze(0)
        ).reshape(graphs, 18)
        aligned_features = torch.cat(
            (fixed_response, entropy.unsqueeze(-1), posterior.max(dim=-1).values.unsqueeze(-1)), dim=-1
        )
        graph_condition = self.invariant(piezo_irreps) + gate.unsqueeze(-1) * self.aligned_token(aligned_features)
        graph_condition = graph_condition + self.present_bias
        present = condition_present.reshape(graphs, 1).to(dtype=torch.bool)
        graph_condition = torch.where(
            present, graph_condition, self.null_condition.unsqueeze(0).expand_as(graph_condition)
        )
        if edge_directions.numel():
            edge_tensor = aligned_tensor[edge_graph]
            edge_response = response_field(edge_tensor, edge_directions)
            edge_response = edge_response * gate[edge_graph].unsqueeze(-1)
            edge_response = torch.where(
                present[edge_graph], edge_response, torch.zeros_like(edge_response)
            )
        else:
            edge_response = edge_directions.new_empty((0, 3))
        return HarmonicGaugeFlowOutput(
            graph_condition=graph_condition,
            edge_response=edge_response,
            posterior=posterior,
            aligned_irreps=aligned_irreps,
            aligned_tensor=aligned_tensor,
            gate=gate,
            entropy=entropy,
        )
