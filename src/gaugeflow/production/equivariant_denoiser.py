"""Shared equivariant denoiser for the production hybrid diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from gaugeflow.geometry import GaussianRadialBasis, periodic_radius_multigraph

from .cartesian_coordinate_carrier import (
    CompactCartesianKrylovCarrier,
    StateAdaptiveCartesianCarrierMixer,
)
from .cartesian_gauge_atlas import (
    CartesianGaugeAtlasOutput,
    CartesianSTFGeometryQueryEncoder,
    StratifiedCartesianGaugeAtlas,
)
from .factorized_angular_moments import FactorizedCartesianAngularMoments
from .lattice_volume_shape import LatticeVolumeShape, project_lattice_state
from .state_projection import (
    cartesian_tangent_to_fractional,
    graph_mean,
    graph_sum,
    sorted_segment_sum,
)


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

    def __init__(
        self,
        hidden_dim: int,
        vector_dim: int,
        radial_dim: int,
        edge_dim: int,
        angular_channels: int,
        edge_refresh_rank: int,
    ) -> None:
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
        self.angular_moments = FactorizedCartesianAngularMoments(
            edge_dim, angular_channels
        )
        angular_dim = self.angular_moments.output_dim
        if edge_refresh_rank < 1:
            raise ValueError("edge refresh rank must be positive")
        # Project node-leading context before gathering it onto edges.  This
        # keeps the refresh O(NH+ER) instead of repeating wide H-dimensional
        # transforms for every periodic image.
        self.edge_source_refresh = nn.Linear(
            hidden_dim, edge_refresh_rank, bias=False
        )
        self.edge_target_refresh = nn.Linear(
            hidden_dim, edge_refresh_rank, bias=False
        )
        self.edge_context_refresh = nn.Linear(
            3 * hidden_dim, edge_refresh_rank, bias=False
        )
        self.edge_vector_refresh = nn.Linear(
            2 * vector_dim, edge_refresh_rank, bias=False
        )
        self.edge_update = nn.Sequential(
            nn.Linear(
                edge_dim + angular_dim + radial_dim + 4 * edge_refresh_rank,
                edge_dim,
            ),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.angular_scalar_residual = nn.Sequential(
            nn.Linear(edge_dim + angular_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.angular_vector_residual = nn.Sequential(
            nn.Linear(edge_dim + angular_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * vector_dim, bias=False),
        )
        # Small nonzero output maps avoid the serial one-step gradient delay
        # of exact-zero residual initialization while keeping the initial
        # perturbation controlled.  Every internal angular/edge parameter can
        # therefore learn on the first backward pass.
        nn.init.orthogonal_(self.angular_scalar_residual[-1].weight, gain=1.0e-2)
        nn.init.orthogonal_(self.angular_vector_residual[-1].weight, gain=1.0e-2)
        self.norm = nn.LayerNorm(hidden_dim)

    @property
    def edge_refresh_rank(self) -> int:
        """Low-rank refresh width, derived from the active projection."""
        return self.edge_source_refresh.out_features

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
        edge_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scalar_aggregate = torch.zeros_like(nodes)
        vector_aggregate = torch.zeros_like(vectors)
        if source.numel():
            source_projection = torch.einsum(
                "evc,ec->ev", vectors[source], directions
            )
            target_projection = torch.einsum(
                "evc,ec->ev", vectors[target], directions
            )
            node_refresh = self.edge_source_refresh(nodes)[source]
            target_refresh = self.edge_target_refresh(nodes)[target]
            graph_refresh = self.edge_context_refresh(
                torch.cat((node_time, node_condition, node_state), dim=-1)
            )[target]
            vector_refresh = self.edge_vector_refresh(
                torch.cat((source_projection, target_projection), dim=-1)
            )
            angular = self.angular_moments(
                edge_state,
                target,
                directions,
                edge_envelope,
                nodes.shape[0],
            )
            refresh_context = torch.cat(
                (
                    edge_state,
                    angular,
                    radial,
                    node_refresh,
                    target_refresh,
                    graph_refresh,
                    vector_refresh,
                ),
                dim=-1,
            )
            edge_state = self.edge_norm(
                edge_state + self.edge_update(refresh_context)
            )
            # Keep the unnormalized angular contractions in the residual
            # context.  Layer-normalizing the persistent state stabilizes
            # depth, while the raw contractions retain local coordination
            # amplitude instead of silently quotienting it out.
            edge_context = torch.cat((edge_state, angular), dim=-1)
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
            scalar_message = (
                self.scalar_message(features)
                + self.angular_scalar_residual(edge_context)
            ) * edge_envelope
            coefficients = (
                self.vector_coefficients(features)
                + self.angular_vector_residual(edge_context)
            ).reshape(source.numel(), 3, vectors.shape[1])
            vector_message = (
                coefficients[:, 0, :, None] * directions[:, None, :]
                + coefficients[:, 1, :, None] * edge_response[:, None, :]
                + torch.sigmoid(coefficients[:, 2, :, None]) * vectors[source]
            )
            vector_message = vector_message * edge_envelope.unsqueeze(-1)
            # Keep graph reductions in the FP32 residual-state dtype under
            # BF16 autocast; index_add_ requires an exact dtype match and FP32
            # accumulation is numerically preferable for neighbor sums.
            scalar_aggregate = sorted_segment_sum(
                scalar_message.to(scalar_aggregate.dtype), target, nodes.shape[0]
            )
            vector_aggregate = sorted_segment_sum(
                vector_message.to(vector_aggregate.dtype), target, nodes.shape[0]
            )
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
        return nodes, vectors, edge_state


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

    coordinate_chart = "volume_normalized_cartesian_tangent_v1"

    def __init__(
        self,
        *,
        hidden_dim: int = 192,
        vector_dim: int = 32,
        layers: int = 4,
        radial_dim: int = 16,
        radial_cutoff: float = 8.0,
        atlas_residual_circle_samples: int = 8,
        edge_dim: int = 64,
        angular_channels: int = 8,
        edge_refresh_rank: int = 16,
        independent_modality_times: bool = False,
        modality_time_conditioning: str | None = None,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("production denoiser needs at least one message block")
        if edge_dim < 1 or angular_channels < 1 or edge_refresh_rank < 1:
            raise ValueError("edge, angular and refresh dimensions must be positive")
        self.edge_dim = int(edge_dim)
        if modality_time_conditioning is None:
            modality_time_conditioning = (
                "separate" if independent_modality_times else "coordinate"
            )
        if modality_time_conditioning not in {
            "coordinate",
            "matched_single",
            "side_mean",
            "separate",
        }:
            raise ValueError("unknown modality-time conditioning mode")
        if independent_modality_times and modality_time_conditioning != "separate":
            raise ValueError(
                "independent_modality_times is only the archived name for separate clocks"
            )
        self.modality_time_conditioning = modality_time_conditioning
        self.uses_side_modality_times = modality_time_conditioning in {
            "side_mean",
            "separate",
        }
        self.element_embedding = nn.Embedding(119, hidden_dim)
        self.degree_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = FourierTimeEmbedding(hidden_dim)
        self.element_time_embedding: FourierTimeEmbedding | None
        self.lattice_time_embedding: FourierTimeEmbedding | None
        self.modality_time_fusion: nn.Linear | None
        if self.modality_time_conditioning != "coordinate":
            self.element_time_embedding = FourierTimeEmbedding(hidden_dim)
            self.lattice_time_embedding = FourierTimeEmbedding(hidden_dim)
            self.modality_time_fusion = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)
            nn.init.orthogonal_(self.modality_time_fusion.weight)
        else:
            self.element_time_embedding = None
            self.lattice_time_embedding = None
            self.modality_time_fusion = None
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
            [
                EquivariantDenoisingBlock(
                    hidden_dim,
                    vector_dim,
                    radial_dim,
                    edge_dim,
                    angular_channels,
                    edge_refresh_rank,
                )
                for _ in range(layers)
            ]
        )
        head_inputs = 4 * hidden_dim
        self.element_head = nn.Sequential(nn.Linear(head_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 118))
        self.coordinate_control_gate = nn.Linear(3 * hidden_dim, vector_dim)
        self.coordinate_edge_encoder = nn.Sequential(
            nn.Linear(5 * hidden_dim + radial_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_state_initializer = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_dim + 1, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )
        self.coordinate_edge_residual = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        nn.init.orthogonal_(self.coordinate_edge_residual[-1].weight, gain=1.0e-2)
        self.coordinate_carrier = CompactCartesianKrylovCarrier(
            hidden_dim, vector_dim, moment_channels=16, rms_epsilon=1.0e-4
        )
        self.coordinate_carrier_mixer = StateAdaptiveCartesianCarrierMixer(
            self.coordinate_carrier.output_channels,
            hidden_dim,
            rank=8,
        )
        self.volume_head = nn.Sequential(nn.Linear(head_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.shape_head = nn.Sequential(
            nn.Linear(head_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 5)
        )

    @property
    def angular_channels(self) -> int:
        return self.blocks[0].angular_moments.channels

    @property
    def edge_refresh_rank(self) -> int:
        return self.blocks[0].edge_refresh_rank

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
        *,
        element_time: torch.Tensor | None = None,
        lattice_time: torch.Tensor | None = None,
    ) -> HybridDenoiserOutput:
        """Denoise one hybrid state.

        ``shape_projector`` is determined by the sampled space-group blueprint,
        never by a paired target structure.  No target CIF, lattice, space
        group, stabilizer, source ID, or endpoint token is accepted.
        """
        graphs = time.numel()
        if time.ndim != 1:
            raise ValueError("coordinate time must be a graph vector")
        if self.uses_side_modality_times:
            if element_time is None or lattice_time is None:
                raise ValueError(
                    "side-time conditioning requires explicit element and lattice times"
                )
            if element_time.shape != time.shape or lattice_time.shape != time.shape:
                raise ValueError("all modality times must match the graph vector")
            assert self.element_time_embedding is not None
            assert self.lattice_time_embedding is not None
            assert self.modality_time_fusion is not None
            coordinate_clock = self.time_embedding(time)
            if self.modality_time_conditioning == "separate":
                side_clocks = (
                    self.element_time_embedding(element_time),
                    self.lattice_time_embedding(lattice_time),
                )
            else:
                # C1 exposes exactly one side-uncertainty scalar.  The third
                # clock MLP and the final H columns of the fusion map are
                # parameter-matching dummies and do not enter the forward map.
                side_mean = 0.5 * (element_time + lattice_time)
                side_clocks = (
                    self.element_time_embedding(side_mean),
                    torch.zeros_like(coordinate_clock),
                )
            graph_time = self.modality_time_fusion(
                torch.cat((coordinate_clock, *side_clocks), dim=-1)
            )
        else:
            if (element_time is None) != (lattice_time is None):
                raise ValueError("element and lattice times must be supplied together")
            if element_time is not None and (
                not torch.equal(element_time, time) or not torch.equal(lattice_time, time)
            ):
                raise ValueError(
                    "shared-time denoising cannot silently consume different modality times"
                )
            # ``matched_single`` owns the same clock bank as the two controls,
            # but the dummy bank is deliberately absent from this graph.  It
            # therefore sees only t_F while matching total parameter count.
            graph_time = self.time_embedding(time)
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
        node_time = graph_time[batch]
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
        if source.numel():
            self_image = (
                (source == target)
                & edges.image_shift.ne(0).any(dim=-1)
            ).to(nodes).unsqueeze(-1)
            edge_state = self.edge_state_initializer(
                torch.cat((nodes[source], nodes[target], radial, self_image), dim=-1)
            )
        else:
            edge_state = nodes.new_empty((0, self.edge_dim))
        vectors = nodes.new_zeros(
            (nodes.shape[0], self.coordinate_carrier.vector_channels, 3)
        )
        # BF16 has only seven mantissa bits and turns sub-microangstrom changes
        # in periodic directions into percent-level coordinate-field jumps.
        # Geometry-dependent message propagation is therefore one fixed FP32
        # typed path. Scalar terminal heads remain AMP eligible; this is not a
        # runtime precision fallback.
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            for block in self.blocks:
                nodes, vectors, edge_state = block(
                    nodes.float(),
                    vectors.float(),
                    source,
                    target,
                    edges.direction.float(),
                    gauge_atlas.edge_response.float(),
                    radial.float(),
                    edge_envelope.float(),
                    node_time.float(),
                    node_condition.float(),
                    node_state.float(),
                    edge_state.float(),
                )
        graph_nodes = graph_mean(nodes, batch, graphs)
        graph_context = torch.cat(
            (graph_nodes, graph_time, gauge_atlas.graph_condition, graph_state), dim=-1
        )
        node_context = torch.cat((nodes, node_time, node_condition, node_state), dim=-1)
        element_logits = self.element_head(node_context)
        coordinate_control = torch.cat((node_time, node_condition, node_state), dim=-1)
        time_gated_vectors = vectors * torch.sigmoid(
            self.coordinate_control_gate(coordinate_control)
        ).unsqueeze(-1)
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
            with torch.autocast(device_type=edge_features.device.type, enabled=False):
                edge_hidden = self.coordinate_edge_encoder(edge_features.float())
                edge_hidden = edge_hidden + self.coordinate_edge_residual(
                    edge_state.float()
                )
        else:
            edge_hidden = nodes.new_empty(
                (0, self.coordinate_carrier.moment_projection.in_features)
            )
        carrier = self.coordinate_carrier(
            time_gated_vectors,
            edge_hidden,
            target,
            edges.direction,
            edge_envelope,
            batch,
            graphs,
        )
        with torch.autocast(device_type=carrier.device.type, enabled=False):
            normalized_cartesian_score = self.coordinate_carrier_mixer(carrier, nodes)
            normalized_cartesian_score = normalized_cartesian_score - graph_mean(
                normalized_cartesian_score, batch, graphs
            )[batch]
            # The fractional torus path is unchanged.  Only the learned output
            # chart is made dimensionless with the O(3)- and GL(3,Z)-invariant
            # cell scale V^(1/3).  The physical Cartesian tangent is restored
            # before the exact fractional pullback consumed by the sampler.
            cell_scale = torch.exp(log_volume.float() / 3.0)
            cartesian_score = normalized_cartesian_score * cell_scale[
                batch, None
            ]
            # The reverse sampler consumes a tangent drift because it adds the
            # network output to fractional coordinates. For r=fL, a Cartesian
            # tangent vector obeys v_r=v_f L and hence v_f=v_r L^-1. The prior
            # L^T covector pullback was an index-type error: it produced a
            # covector and then silently used it as a vector. Solve the
            # transposed row-vector system without forming an inverse, and keep
            # this physical chart change in FP32 under BF16 execution.
            fractional_score = cartesian_tangent_to_fractional(
                cartesian_score, lattice, batch
            )
            # Fractional zero mean is the translation-horizontal tangent chart
            # used by the coordinate probability path.
            fractional_score = fractional_score - graph_mean(
                fractional_score, batch, graphs
            )[batch]
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
