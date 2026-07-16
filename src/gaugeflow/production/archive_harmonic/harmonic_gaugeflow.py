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

from ..schedules import CosineNoiseSchedule
from ..so3_quadrature import nested_hopf_so3_grid


@dataclass(frozen=True)
class HarmonicGaugeFlowOutput:
    graph_condition: torch.Tensor
    edge_response: torch.Tensor
    posterior: torch.Tensor
    aligned_irreps: torch.Tensor
    aligned_tensor: torch.Tensor
    gate: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class GeometryHarmonicQueries:
    """Condition-free graph queries in the rank-three polar SO(3) channels."""

    first: torch.Tensor
    second: torch.Tensor
    third: torch.Tensor


class _GeometryScalarBlock(nn.Module):
    def __init__(self, hidden_dim: int, radial_dim: int, query_channels: int) -> None:
        super().__init__()
        angular_dim = 2 * query_channels
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_dim + 2 * angular_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        radial: torch.Tensor,
        angular: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(nodes)
        if source.numel():
            message = self.message(
                torch.cat(
                    (nodes[source], nodes[target], radial, angular[source], angular[target]), dim=-1
                )
            )
            aggregate.index_add_(0, target, message)
            degree = torch.bincount(target, minlength=nodes.shape[0]).clamp_min(1).to(nodes)
            aggregate = aggregate / degree.unsqueeze(-1)
        return self.norm(nodes + self.update(torch.cat((nodes, aggregate), dim=-1)))


class ConditionFreeGeometryQueryEncoder(nn.Module):
    """Build non-degenerate harmonic queries from current geometry only.

    Scalar message blocks see tokens, distances, and angular invariants.  The
    angular invariants are norms of node-centred l=1/l=2 moments and therefore
    contain pair-angle information without selecting a frame.  Final l=1 and
    l=2 local moments are coupled through an O(3) tensor product to obtain the
    polar l=1,2,3 channels.  No tensor condition is accepted by this module.
    """

    def __init__(
        self,
        hidden_dim: int,
        radial_dim: int,
        *,
        query_channels: int = 2,
        layers: int = 3,
    ) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("geometry query encoder needs at least two blocks")
        self.query_channels = int(query_channels)
        self.blocks = nn.ModuleList(
            [_GeometryScalarBlock(hidden_dim, radial_dim, query_channels) for _ in range(layers)]
        )
        self.moment_weight = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, query_channels),
        )
        self.tensor_product = o3.FullTensorProduct(
            "1o", "2e", filter_ir_out=["1o", "2o", "3o"], irrep_normalization="component"
        )
        if self.tensor_product.irreps_out.dim != 15:
            raise RuntimeError("unexpected l=1 x l=2 tensor-product layout")
        self.product_scale = nn.Parameter(torch.ones(query_channels, 3))

    def _local_moments(
        self,
        nodes: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        directions: torch.Tensor,
        radial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = (
            self.moment_weight(torch.cat((nodes[source], nodes[target], radial), dim=-1))
            if source.numel()
            else nodes.new_empty((0, self.query_channels))
        )
        if not source.numel():
            zeros1 = nodes.new_zeros((nodes.shape[0], self.query_channels, 3))
            zeros2 = nodes.new_zeros((nodes.shape[0], self.query_channels, 5))
            zeros3 = nodes.new_zeros((nodes.shape[0], self.query_channels, 7))
            return zeros1, zeros2, zeros3, weights
        harmonics = o3.spherical_harmonics(
            [1, 2, 3], torch.nn.functional.normalize(directions, dim=-1),
            normalize=True, normalization="component",
        )
        values = []
        for start, stop in ((0, 3), (3, 8), (8, 15)):
            values.append(
                scatter(
                    weights.unsqueeze(-1) * harmonics[:, None, start:stop],
                    target,
                    dim=0,
                    dim_size=nodes.shape[0],
                    reduce="mean",
                )
            )
        return values[0], values[1], values[2], weights

    def forward(
        self,
        initial_nodes: torch.Tensor,
        node_time: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        directions: torch.Tensor,
        radial: torch.Tensor,
        batch: torch.Tensor,
        graph_count: int,
    ) -> GeometryHarmonicQueries:
        nodes = initial_nodes + node_time
        angular = nodes.new_zeros((nodes.shape[0], 2 * self.query_channels))
        for block in self.blocks:
            nodes = block(nodes, source, target, radial, angular)
            local_first, local_second, _, _ = self._local_moments(
                nodes, source, target, directions, radial
            )
            angular = torch.cat(
                (
                    torch.linalg.vector_norm(local_first, dim=-1),
                    torch.linalg.vector_norm(local_second, dim=-1),
                ),
                dim=-1,
            )
        local_first, local_second, local_third, _ = self._local_moments(
            nodes, source, target, directions, radial
        )
        products = []
        for channel in range(self.query_channels):
            products.append(
                self.tensor_product(local_first[:, channel], local_second[:, channel])
            )
        product = torch.stack(products, dim=1)
        product_first, product_second, product_third = (
            product[..., :3], product[..., 3:8], product[..., 8:15]
        )
        scale = self.product_scale.to(product)
        local_first = local_first + scale[:, 0][None, :, None] * product_first
        # l=2 of a polar rank-three tensor has odd parity; the direct Y_2
        # moment is even, so only the 1o x 2e product enters this channel.
        local_second = scale[:, 1][None, :, None] * product_second
        local_third = local_third + scale[:, 2][None, :, None] * product_third
        return GeometryHarmonicQueries(
            first=scatter(local_first, batch, dim=0, dim_size=graph_count, reduce="mean"),
            second=scatter(local_second, batch, dim=0, dim_size=graph_count, reduce="mean"),
            third=scatter(local_third, batch, dim=0, dim_size=graph_count, reduce="mean"),
        )


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


def harmonic_alignment_scores_from_queries(
    piezo_irreps: torch.Tensor,
    queries: GeometryHarmonicQueries,
    rotations: torch.Tensor,
    *,
    coupling_l1: torch.Tensor,
    coupling_l2: torch.Tensor,
    coupling_l3: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the harmonic score from geometry-encoder graph queries."""
    graphs = piezo_irreps.shape[0]
    if queries.first.shape[:2] != (graphs, coupling_l1.shape[0]):
        raise ValueError("l=1 geometry queries do not match graph/channel counts")
    if queries.second.shape != (graphs, coupling_l2.shape[0], 5):
        raise ValueError("l=2 geometry queries do not match graph/channel counts")
    if queries.third.shape != (graphs, coupling_l3.shape[0], 7):
        raise ValueError("l=3 geometry queries do not match graph/channel counts")
    rotated = rotate_piezo_irreps_on_grid(piezo_irreps, rotations)
    blocks = piezo_irrep_blocks(rotated.reshape(-1, 18))
    first = blocks[0].reshape(graphs, rotations.shape[0], 2, 3)
    second = blocks[1].reshape(graphs, rotations.shape[0], 1, 5)
    third = blocks[2].reshape(graphs, rotations.shape[0], 1, 7)
    return (
        torch.einsum("bfmi,bai,am->bf", first, queries.first, coupling_l1.to(first)) / math.sqrt(3.0)
        + torch.einsum("bfmi,bai,am->bf", second, queries.second, coupling_l2.to(second)) / math.sqrt(5.0)
        + torch.einsum("bfmi,bai,am->bf", third, queries.third, coupling_l3.to(third)) / math.sqrt(7.0)
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
        geometry_queries: GeometryHarmonicQueries,
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
        rotations = self.rotations.to(piezo_irreps)
        score = harmonic_alignment_scores_from_queries(
            piezo_irreps,
            geometry_queries,
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
