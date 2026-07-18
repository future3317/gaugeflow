"""Shared equivariant denoiser for the production hybrid diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from gaugeflow.geometry import GaussianRadialBasis, periodic_radius_multigraph

from .cartesian_gauge_atlas import (
    CartesianGaugeAtlasOutput,
    CartesianSTFGeometryQueryEncoder,
    StratifiedCartesianGaugeAtlas,
)
from .lattice_volume_shape import LatticeVolumeShape, project_lattice_state
from .pairwise_reciprocal_score import PairwiseReciprocalScore
from .state_projection import graph_mean, graph_sum


class FourierTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, frequencies: int = 16) -> None:
        super().__init__()
        if frequencies < 1:
            raise ValueError("time embedding needs at least one Fourier frequency")
        values = 2.0 ** torch.arange(frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", values)
        self.network = nn.Sequential(
            nn.Linear(2 * frequencies + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("time embedding requires a graph vector")
        phase = math.pi * time.unsqueeze(-1) * self.frequencies.to(time)
        return self.network(torch.cat((time.unsqueeze(-1), phase.sin(), phase.cos()), dim=-1))


class EquivariantDenoisingBlock(nn.Module):
    """O(3)-typed scalar/vector message block with explicit time and condition FiLM."""

    def __init__(self, hidden_dim: int, vector_dim: int, radial_dim: int) -> None:
        super().__init__()
        scalar_inputs = 2 * hidden_dim + 2 * vector_dim + radial_dim + 2
        self.scalar_message = nn.Sequential(
            nn.Linear(scalar_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.vector_coefficients = nn.Sequential(
            nn.Linear(scalar_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3 * vector_dim)
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.condition_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.state_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.vector_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vector_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        vectors: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        directions: torch.Tensor,
        edge_response: torch.Tensor,
        radial: torch.Tensor,
        edge_envelope: torch.Tensor,
        node_time: torch.Tensor,
        node_condition: torch.Tensor,
        node_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scalar_aggregate = torch.zeros_like(nodes)
        vector_aggregate = torch.zeros_like(vectors)
        if source.numel():
            source_projection = torch.einsum("evc,ec->ev", vectors[source], directions)
            target_projection = torch.einsum("evc,ec->ev", vectors[target], directions)
            response_norm = torch.linalg.vector_norm(edge_response, dim=-1, keepdim=True)
            response_projection = (edge_response * directions).sum(dim=-1, keepdim=True)
            features = torch.cat(
                (
                    nodes[source],
                    nodes[target],
                    source_projection,
                    target_projection,
                    radial,
                    response_norm,
                    response_projection,
                ),
                dim=-1,
            )
            scalar_message = self.scalar_message(features) * edge_envelope
            coefficients = self.vector_coefficients(features).reshape(source.numel(), 3, vectors.shape[1])
            vector_message = (
                coefficients[:, 0, :, None] * directions[:, None, :]
                + coefficients[:, 1, :, None] * edge_response[:, None, :]
                + torch.sigmoid(coefficients[:, 2, :, None]) * vectors[source]
            )
            vector_message = vector_message * edge_envelope.unsqueeze(-1)
            # Keep graph reductions in the FP32 residual-state dtype under
            # BF16 autocast; index_add_ requires an exact dtype match and FP32
            # accumulation is numerically preferable for neighbor sums.
            scalar_aggregate.index_add_(0, target, scalar_message.to(scalar_aggregate.dtype))
            vector_aggregate.index_add_(0, target, vector_message.to(vector_aggregate.dtype))
            degree = torch.bincount(target, minlength=nodes.shape[0]).clamp_min(1).to(nodes)
            scalar_aggregate = scalar_aggregate / degree.unsqueeze(-1)
            vector_aggregate = vector_aggregate / degree[:, None, None]
        update = self.scalar_update(torch.cat((nodes, scalar_aggregate), dim=-1))
        time_scale, time_shift = self.time_film(node_time).chunk(2, dim=-1)
        condition_scale, condition_shift = self.condition_film(node_condition).chunk(2, dim=-1)
        state_scale, state_shift = self.state_film(node_state).chunk(2, dim=-1)
        update = update * (1.0 + time_scale) + time_shift
        update = update * (1.0 + condition_scale) + condition_shift
        update = update * (1.0 + state_scale) + state_shift
        nodes = self.norm(nodes + update)
        vector_scale = torch.sigmoid(
            self.vector_gate(torch.cat((node_time, node_condition, node_state), dim=-1))
        ).unsqueeze(-1)
        vectors = vectors + vector_scale * vector_aggregate
        return nodes, vectors


@dataclass(frozen=True)
class HybridDenoiserOutput:
    clean_element_logits: torch.Tensor
    coordinate_cartesian_scaled_score: torch.Tensor
    coordinate_fractional_scaled_score: torch.Tensor
    clean_volume_latent: torch.Tensor
    clean_shape_latent: torch.Tensor
    gauge_atlas: CartesianGaugeAtlasOutput


class HybridCrystalDenoiser(nn.Module):
    """Paper-defined denoiser with no target-structure inputs or endpoint tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int = 192,
        vector_dim: int = 32,
        layers: int = 4,
        radial_dim: int = 16,
        radial_cutoff: float = 8.0,
        atlas_residual_circle_samples: int = 8,
        reciprocal_pair_width: int = 32,
        reciprocal_channels: int = 8,
        reciprocal_radial_dim: int = 8,
        reciprocal_cutoff: float = 4.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("production denoiser needs at least one message block")
        self.element_embedding = nn.Embedding(119, hidden_dim)
        self.degree_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = FourierTimeEmbedding(hidden_dim)
        # Current generated state only: noisy lattice, graph size and current
        # (possibly masked) composition. No target metadata enters this token.
        self.state_embedding = nn.Sequential(
            nn.Linear(2 * hidden_dim + 9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gauge_atlas = StratifiedCartesianGaugeAtlas(
            hidden_dim, residual_circle_samples=atlas_residual_circle_samples
        )
        self.radial = GaussianRadialBasis(radial_dim, radial_cutoff)
        self.geometry_query_encoder = CartesianSTFGeometryQueryEncoder(
            hidden_dim, radial_dim, query_channels=2, layers=3
        )
        self.blocks = nn.ModuleList(
            [EquivariantDenoisingBlock(hidden_dim, vector_dim, radial_dim) for _ in range(layers)]
        )
        head_inputs = 4 * hidden_dim
        self.element_head = nn.Sequential(nn.Linear(head_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 118))
        self.coordinate_vector_head = nn.Linear(vector_dim, 1, bias=False)
        self.coordinate_control_gate = nn.Linear(3 * hidden_dim, vector_dim)
        self.coordinate_edge_head = nn.Sequential(
            nn.Linear(5 * hidden_dim + radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.coordinate_pair_reciprocal_head = PairwiseReciprocalScore(
            hidden_dim,
            pair_width=reciprocal_pair_width,
            channels=reciprocal_channels,
            radial_dim=reciprocal_radial_dim,
            cutoff=reciprocal_cutoff,
        )
        self.volume_head = nn.Sequential(nn.Linear(head_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.shape_head = nn.Sequential(
            nn.Linear(head_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 5)
        )

    def forward(
        self,
        element_tokens: torch.Tensor,
        frac_coords: torch.Tensor,
        log_volume: torch.Tensor,
        log_shape: torch.Tensor,
        batch: torch.Tensor,
        time: torch.Tensor,
        tensor_condition: torch.Tensor,
        condition_present: torch.Tensor,
        shape_projector: torch.Tensor,
        fractional_to_cartesian: torch.Tensor,
    ) -> HybridDenoiserOutput:
        """Denoise one hybrid state.

        ``shape_projector`` is determined by the sampled space-group blueprint,
        never by a paired target structure.  No target CIF, lattice, space
        group, stabilizer, source ID, or endpoint token is accepted.
        """
        graphs = time.numel()
        if element_tokens.ndim != 1 or element_tokens.dtype != torch.long:
            raise ValueError("element state must be rank-one int64 tokens")
        if element_tokens.numel() and bool(((element_tokens < 0) | (element_tokens > 118)).any()):
            raise ValueError("element state lies outside 118 elements plus MASK")
        if frac_coords.shape != (element_tokens.numel(), 3) or batch.shape != element_tokens.shape:
            raise ValueError("coordinate and batch shapes do not match element tokens")
        if log_volume.shape != (graphs,) or log_shape.shape != (graphs, 6):
            raise ValueError("lattice state must contain graphwise volume and six shape coordinates")
        if tensor_condition.shape != (graphs, 18):
            raise ValueError("tensor condition must have shape [graphs,18]")
        if shape_projector.shape != (graphs, 6, 6):
            raise ValueError("shape projector must have shape [graphs,6,6]")
        if fractional_to_cartesian.shape != (graphs, 3, 3):
            raise ValueError("fractional-to-Cartesian chart must have shape [graphs,3,3]")
        # The diffusion and every reverse step own projection. Re-projecting
        # here is a redundant, non-idempotent FP32 operation whose tiny chart
        # drift is amplified by a large periodic multigraph. Fail closed on a
        # caller that violates the state contract instead of silently fixing it.
        # Lattice exponentials, reciprocal bounds and neighbor selection are
        # geometry kernels, not learned matmuls; retain FP32 under AMP.
        with torch.autocast(device_type=log_volume.device.type, enabled=False):
            projected_shape = project_lattice_state(log_shape, shape_projector)
            if not torch.allclose(log_shape, projected_shape, atol=2e-6, rtol=2e-6):
                raise ValueError(
                    "denoiser lattice shape is outside the blueprint subspace"
                )
            lattice = LatticeVolumeShape(log_volume, log_shape).lattice(
                fractional_to_cartesian
            )
            edges = periodic_radius_multigraph(
                frac_coords, lattice, batch, cutoff=self.radial.cutoff
            )
            radial = self.radial(edges.distance)
            edge_envelope = self.radial.envelope(edges.distance)
        source, target = edges.source, edges.target
        degree = torch.bincount(target, minlength=element_tokens.numel()).to(log_volume)
        node_time = self.time_embedding(time)[batch]
        initial_nodes = self.element_embedding(element_tokens) + self.degree_embedding(
            degree.log1p().unsqueeze(-1)
        )
        counts = torch.bincount(batch, minlength=graphs).to(log_volume)
        composition_mean = graph_mean(initial_nodes, batch, graphs)
        composition_scaled_sum = graph_sum(initial_nodes, batch, graphs) / counts.sqrt().unsqueeze(-1)
        state_features = torch.cat(
            (
                log_volume.unsqueeze(-1),
                log_shape,
                counts.log().unsqueeze(-1),
                counts.reciprocal().unsqueeze(-1),
                composition_mean,
                composition_scaled_sum,
            ),
            dim=-1,
        )
        graph_state = self.state_embedding(state_features)
        node_state = graph_state[batch]
        if bool(condition_present.any()):
            # Tensor-free pretraining uses a learned null token but does not
            # construct an atlas.  Keep its expensive geometry-query encoder
            # outside that path instead of computing an unused tensor.
            edge_graph = batch[source] if source.numel() else batch.new_empty((0,))
            geometry_queries = self.geometry_query_encoder(
                initial_nodes + node_state,
                node_time,
                source,
                target,
                edges.direction,
                radial,
                batch,
                graphs,
            )
            gauge_atlas = self.gauge_atlas(
                tensor_condition, condition_present, edges.direction, edge_graph, geometry_queries, time
            )
        else:
            gauge_atlas = self.gauge_atlas.null_output(
                graph_count=graphs,
                edge_count=edges.direction.shape[0],
                reference=tensor_condition,
            )
        node_condition = gauge_atlas.graph_condition[batch]
        nodes = initial_nodes + node_time + node_condition + node_state
        vectors = nodes.new_zeros((nodes.shape[0], self.coordinate_vector_head.in_features, 3))
        for block in self.blocks:
            nodes, vectors = block(
                nodes,
                vectors,
                source,
                target,
                edges.direction,
                gauge_atlas.edge_response,
                radial,
                edge_envelope,
                node_time,
                node_condition,
                node_state,
            )
        graph_nodes = graph_mean(nodes, batch, graphs)
        graph_time = self.time_embedding(time)
        graph_context = torch.cat(
            (graph_nodes, graph_time, gauge_atlas.graph_condition, graph_state), dim=-1
        )
        node_context = torch.cat((nodes, node_time, node_condition, node_state), dim=-1)
        element_logits = self.element_head(node_context)
        coordinate_control = torch.cat((node_time, node_condition, node_state), dim=-1)
        time_gated_vectors = vectors * torch.sigmoid(
            self.coordinate_control_gate(coordinate_control)
        ).unsqueeze(-1)
        cartesian_score = self.coordinate_vector_head(time_gated_vectors.transpose(-1, -2)).squeeze(-1)
        if source.numel():
            edge_features = torch.cat(
                (
                    nodes[source],
                    nodes[target],
                    node_time[target],
                    node_condition[target],
                    node_state[target],
                    radial,
                ),
                dim=-1,
            )
            edge_score = (
                self.coordinate_edge_head(edge_features)
                * edge_envelope
                * edges.displacement
            )
            edge_aggregate = torch.zeros_like(cartesian_score)
            edge_aggregate.index_add_(
                0, target, edge_score.to(edge_aggregate.dtype)
            )
            cartesian_score = cartesian_score + edge_aggregate / degree.clamp_min(
                1
            ).sqrt().unsqueeze(-1)
        cartesian_score = cartesian_score + self.coordinate_pair_reciprocal_head(
            nodes, frac_coords, lattice, batch
        )
        cartesian_score = cartesian_score - graph_mean(cartesian_score, batch, graphs)[batch]
        # A score is a covector, not a displacement. With r=fL, the chain rule
        # gives grad_f log p = grad_r log p @ L^T. (A Cartesian velocity would
        # instead use L^-1; mixing these two transformations is incorrect.)
        fractional_score = torch.einsum("ni,nij->nj", cartesian_score, lattice[batch].transpose(-1, -2))
        # Fractional zero mean is the translation-horizontal chart used by the
        # coordinate probability path.
        fractional_score = fractional_score - graph_mean(fractional_score, batch, graphs)[batch]
        clean_volume_latent = self.volume_head(graph_context).squeeze(-1)
        clean_shape_latent = self.shape_head(graph_context)
        return HybridDenoiserOutput(
            clean_element_logits=element_logits,
            coordinate_cartesian_scaled_score=cartesian_score,
            coordinate_fractional_scaled_score=fractional_score,
            clean_volume_latent=clean_volume_latent,
            clean_shape_latent=clean_shape_latent,
            gauge_atlas=gauge_atlas,
        )
