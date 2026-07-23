"""Shared equivariant denoiser for the production hybrid diffusion."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import torch
from torch import nn

from gaugeflow.geometry import GaussianRadialBasis, periodic_radius_multigraph
from gaugeflow.tensor import piezo_from_irreps
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT

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
        self.angular_moments = FactorizedCartesianAngularMoments(edge_dim, angular_channels)
        angular_dim = self.angular_moments.output_dim
        if edge_refresh_rank < 1:
            raise ValueError("edge refresh rank must be positive")
        # Project node-leading context before gathering it onto edges.  This
        # keeps the refresh O(NH+ER) instead of repeating wide H-dimensional
        # transforms for every periodic image.
        self.edge_source_refresh = nn.Linear(hidden_dim, edge_refresh_rank, bias=False)
        self.edge_target_refresh = nn.Linear(hidden_dim, edge_refresh_rank, bias=False)
        self.edge_context_refresh = nn.Linear(3 * hidden_dim, edge_refresh_rank, bias=False)
        self.edge_vector_refresh = nn.Linear(2 * vector_dim, edge_refresh_rank, bias=False)
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
            source_projection = torch.einsum("evc,ec->ev", vectors[source], directions)
            target_projection = torch.einsum("evc,ec->ev", vectors[target], directions)
            node_refresh = self.edge_source_refresh(nodes)[source]
            target_refresh = self.edge_target_refresh(nodes)[target]
            graph_refresh = self.edge_context_refresh(torch.cat((node_time, node_condition, node_state), dim=-1))[
                target
            ]
            vector_refresh = self.edge_vector_refresh(torch.cat((source_projection, target_projection), dim=-1))
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
            edge_state = self.edge_norm(edge_state + self.edge_update(refresh_context))
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
                self.scalar_message(features) + self.angular_scalar_residual(edge_context)
            ) * edge_envelope
            coefficients = (self.vector_coefficients(features) + self.angular_vector_residual(edge_context)).reshape(
                source.numel(), 3, vectors.shape[1]
            )
            vector_message = (
                coefficients[:, 0, :, None] * directions[:, None, :]
                + coefficients[:, 1, :, None] * edge_response[:, None, :]
                + torch.sigmoid(coefficients[:, 2, :, None]) * vectors[source]
            )
            vector_message = vector_message * edge_envelope.unsqueeze(-1)
            # Keep graph reductions in the FP32 residual-state dtype under
            # BF16 autocast; index_add_ requires an exact dtype match and FP32
            # accumulation is numerically preferable for neighbor sums.
            scalar_aggregate = sorted_segment_sum(scalar_message.to(scalar_aggregate.dtype), target, nodes.shape[0])
            vector_aggregate = sorted_segment_sum(vector_message.to(vector_aggregate.dtype), target, nodes.shape[0])
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
    clean_composition_logits: torch.Tensor
    coordinate_cartesian_scaled_score: torch.Tensor
    coordinate_fractional_scaled_score: torch.Tensor
    clean_volume_latent: torch.Tensor
    clean_shape_latent: torch.Tensor
    gauge_atlas: CartesianGaugeAtlasOutput


@dataclass(frozen=True)
class LatticeDenoiserOutput:
    """Coordinate-free lattice endpoint prediction."""

    clean_volume_latent: torch.Tensor
    clean_shape_latent: torch.Tensor


@dataclass(frozen=True)
class HybridBackboneFeatures:
    """Cartesian node features shared by generation and physical transfer."""

    node_scalar: torch.Tensor
    node_vectors: torch.Tensor


@dataclass(frozen=True)
class _EncodedHybridState:
    nodes: torch.Tensor
    vectors: torch.Tensor
    edge_state: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    direction: torch.Tensor
    radial: torch.Tensor
    edge_envelope: torch.Tensor
    graph_time: torch.Tensor
    graph_state: torch.Tensor
    lattice_context: torch.Tensor
    node_time: torch.Tensor
    node_condition: torch.Tensor
    node_state: torch.Tensor
    lattice: torch.Tensor
    gauge_atlas: CartesianGaugeAtlasOutput


class CenteredResidualAdapter(nn.Module):
    """Exact-function-preserving residual with immediate internal gradients."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.active = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reference = copy.deepcopy(self.active)
        self.reference.requires_grad_(False)
        self.reference.eval()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reference = self.reference(value)
        return self.active(value) - reference


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
        self.hidden_dim = int(hidden_dim)
        self.tensor_residual_adapter: CenteredResidualAdapter | None = None
        if modality_time_conditioning is None:
            modality_time_conditioning = "separate" if independent_modality_times else "coordinate"
        if modality_time_conditioning not in {
            "coordinate",
            "matched_single",
            "side_mean",
            "separate",
        }:
            raise ValueError("unknown modality-time conditioning mode")
        if independent_modality_times and modality_time_conditioning != "separate":
            raise ValueError("independent_modality_times is only the archived name for separate clocks")
        self.modality_time_conditioning = modality_time_conditioning
        self.uses_side_modality_times = modality_time_conditioning in {
            "side_mean",
            "separate",
        }
        self.element_embedding = nn.Embedding(119, hidden_dim)
        self.degree_embedding = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
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
        graph_head_inputs = 4 * hidden_dim
        node_head_inputs = 5 * hidden_dim
        self.composition_head = nn.Sequential(
            nn.Linear(graph_head_inputs + CHEMICAL_ELEMENT_COUNT + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, CHEMICAL_ELEMENT_COUNT),
        )
        nn.init.orthogonal_(self.composition_head[-1].weight, gain=1.0e-2)
        nn.init.zeros_(self.composition_head[-1].bias)
        self.element_head = nn.Sequential(
            nn.Linear(node_head_inputs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 118),
        )
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
        self.volume_head = nn.Sequential(nn.Linear(graph_head_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.shape_head = nn.Sequential(nn.Linear(graph_head_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 5))

    def attach_tensor_residual_adapter(self) -> None:
        """Attach the optional E-stage adapter after loading a base checkpoint."""

        if self.tensor_residual_adapter is None:
            # The base denoiser is normally restored on CUDA before E2
            # attaches its optional module.  Constructing on CPU and relying
            # on a later implicit move would make the first forward fail (and
            # would silently break non-FP32 evaluation), so inherit both
            # device and parameter dtype from the existing backbone.
            reference = self.element_embedding.weight
            self.tensor_residual_adapter = CenteredResidualAdapter(self.hidden_dim).to(
                device=reference.device,
                dtype=reference.dtype,
            )

    @property
    def angular_channels(self) -> int:
        return self.blocks[0].angular_moments.channels

    @property
    def edge_refresh_rank(self) -> int:
        return self.blocks[0].edge_refresh_rank

    def _embed_modality_times(
        self,
        time: torch.Tensor,
        element_time: torch.Tensor | None,
        lattice_time: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embed the explicit clocks while enforcing their declared contract."""

        if time.ndim != 1:
            raise ValueError("coordinate time must be a graph vector")
        if self.uses_side_modality_times:
            if element_time is None or lattice_time is None:
                raise ValueError("side-time conditioning requires explicit element and lattice times")
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
                side_mean = 0.5 * (element_time + lattice_time)
                side_clocks = (
                    self.element_time_embedding(side_mean),
                    torch.zeros_like(coordinate_clock),
                )
            return self.modality_time_fusion(torch.cat((coordinate_clock, *side_clocks), dim=-1))
        if (element_time is None) != (lattice_time is None):
            raise ValueError("element and lattice times must be supplied together")
        if element_time is not None and (not torch.equal(element_time, time) or not torch.equal(lattice_time, time)):
            raise ValueError("shared-time denoising cannot silently consume different modality times")
        return self.time_embedding(time)

    def _composition_lattice_context(
        self,
        element_tokens: torch.Tensor,
        log_volume: torch.Tensor,
        log_shape: torch.Tensor,
        batch: torch.Tensor,
        graph_time: torch.Tensor,
        composition_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Summarize current state and, when supplied, sampled composition."""

        graphs = graph_time.shape[0]
        element_nodes = self.element_embedding(element_tokens)
        counts = torch.bincount(batch, minlength=graphs).to(log_volume)
        composition_mean = graph_mean(element_nodes, batch, graphs)
        composition_scaled_sum = graph_sum(element_nodes, batch, graphs) / counts.sqrt().unsqueeze(-1)
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
        if composition_counts is not None:
            if (
                composition_counts.shape != (graphs, CHEMICAL_ELEMENT_COUNT)
                or composition_counts.dtype != torch.long
                or composition_counts.device != element_tokens.device
                or bool((composition_counts < 0).any())
            ):
                raise ValueError("composition condition must be nonnegative [graphs,elements] int64")
            if not torch.equal(composition_counts.sum(dim=1), counts.long()):
                raise ValueError("composition condition does not close on packed node counts")
            composition_probability = composition_counts.to(element_nodes)
            composition_probability = composition_probability / counts.unsqueeze(-1)
            composition_token = composition_probability @ self.element_embedding.weight[:CHEMICAL_ELEMENT_COUNT]
            graph_state = graph_state + composition_token
        lattice_context = torch.cat(
            (composition_mean, composition_scaled_sum, graph_time, graph_state),
            dim=-1,
        )
        return element_nodes, graph_state, lattice_context

    def _tensor_condition_embedding(self, tensor_condition: torch.Tensor) -> torch.Tensor:
        """Return the geometry-free invariant token shared by side-state paths.

        The lattice-only sampler has no edge geometry and therefore cannot use
        the atlas' aligned-frame token.  A residual adapter must nevertheless
        receive the same condition coordinates in the lattice and hybrid
        paths; otherwise it is trained on one input distribution and sampled
        on another.  The invariant tensor token is the common, representative-
        independent part of the condition and is valid for scalar lattice
        readouts as well as the graph field.
        """

        invariant, _ = self.gauge_atlas.invariant(
            piezo_from_irreps(tensor_condition)
        )
        return invariant + self.gauge_atlas.present_bias

    def _centered_tensor_residual(self, tensor_condition: torch.Tensor) -> torch.Tensor:
        """Map the shared condition token through an exact zero residual."""

        condition_token = self._tensor_condition_embedding(tensor_condition)
        null_token = self.gauge_atlas.null_condition.unsqueeze(0).expand_as(condition_token)
        if self.tensor_residual_adapter is None:
            return condition_token - null_token
        return self.tensor_residual_adapter(condition_token - null_token)

    def forward_lattice(
        self,
        element_tokens: torch.Tensor,
        log_volume: torch.Tensor,
        log_shape: torch.Tensor,
        batch: torch.Tensor,
        lattice_time: torch.Tensor,
        shape_projector: torch.Tensor,
        *,
        composition_counts: torch.Tensor | None = None,
        tensor_condition: torch.Tensor | None = None,
        condition_present: torch.Tensor | None = None,
    ) -> LatticeDenoiserOutput:
        """Denoise ``L_t`` without accepting coordinates or building edges."""

        graphs = lattice_time.numel()
        if self.modality_time_conditioning != "separate":
            raise ValueError("lattice-only denoising requires the unified separate-clock backbone")
        if element_tokens.ndim != 1 or element_tokens.dtype != torch.long:
            raise ValueError("element state must be rank-one int64 tokens")
        if batch.shape != element_tokens.shape or batch.dtype != torch.long:
            raise ValueError("batch must provide one graph index per element token")
        if element_tokens.numel() and bool(((element_tokens < 0) | (element_tokens > 118)).any()):
            raise ValueError("element state lies outside 118 elements plus MASK")
        if log_volume.shape != (graphs,) or log_shape.shape != (graphs, 6):
            raise ValueError("lattice state must contain graphwise volume and six shape coordinates")
        if shape_projector.shape != (graphs, 6, 6):
            raise ValueError("shape projector must have shape [graphs,6,6]")
        if batch.numel() == 0 or int(batch.min()) != 0 or int(batch.max()) + 1 != graphs:
            raise ValueError("lattice batch must contain every graph index")
        with torch.autocast(device_type=log_volume.device.type, enabled=False):
            projected_shape = project_lattice_state(log_shape, shape_projector)
            if not torch.allclose(log_shape, projected_shape, atol=2e-6, rtol=2e-6):
                raise ValueError("denoiser lattice shape is outside the blueprint subspace")
        clean_time = torch.zeros_like(lattice_time)
        graph_time = self._embed_modality_times(clean_time, clean_time, lattice_time)
        element_nodes, graph_state, _ = self._composition_lattice_context(
            element_tokens,
            log_volume,
            log_shape,
            batch,
            graph_time,
            composition_counts,
        )
        if tensor_condition is not None or condition_present is not None:
            if tensor_condition is None or condition_present is None:
                raise ValueError("lattice tensor condition requires both values")
            if tensor_condition.shape != (graphs, 18):
                raise ValueError("lattice tensor condition must have shape [graphs,18]")
            if condition_present.shape not in {(graphs,), (graphs, 1)} or condition_present.dtype != torch.bool:
                raise ValueError("lattice condition-present flag must provide one boolean per graph")
            if tensor_condition.device != log_volume.device or not bool(torch.isfinite(tensor_condition).all()):
                raise ValueError("lattice tensor condition must be finite and on the lattice device")
            # The lattice-only path has no edge geometry, so it uses the same
            # proper-SO invariant graph token as the full hybrid path with a
            # zero available-edge gate.  This keeps the head dimension and
            # checkpoint schema unchanged while making the condition explicit.
            present = condition_present.reshape(graphs, 1)
            if self.tensor_residual_adapter is not None:
                # Use the same geometry-free condition coordinates as the
                # hybrid path.  This path is explicitly exercised by the
                # generated-lattice sampler; it must not be an unseen input
                # distribution for an adapter trained only on hybrid states.
                condition_token = self._centered_tensor_residual(tensor_condition)
                condition_token = condition_token * present.to(condition_token)
            else:
                condition_token, _ = self.gauge_atlas.invariant(
                    piezo_from_irreps(tensor_condition)
                )
                condition_token = torch.where(
                    present,
                    condition_token + self.gauge_atlas.present_bias,
                    self.gauge_atlas.null_condition.unsqueeze(0).expand_as(condition_token),
                )
            graph_state = graph_state + condition_token
        counts = torch.bincount(batch, minlength=graphs).to(log_volume)
        composition_mean = graph_mean(element_nodes, batch, graphs)
        composition_scaled_sum = graph_sum(element_nodes, batch, graphs) / counts.sqrt().unsqueeze(-1)
        lattice_context = torch.cat(
            (composition_mean, composition_scaled_sum, graph_time, graph_state),
            dim=-1,
        )
        return LatticeDenoiserOutput(
            clean_volume_latent=self.volume_head(lattice_context).squeeze(-1),
            clean_shape_latent=self.shape_head(lattice_context),
        )

    def _encode_hybrid_state(
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
        element_time: torch.Tensor | None,
        lattice_time: torch.Tensor | None,
        composition_counts: torch.Tensor | None,
        geometry_lattice: torch.Tensor | None = None,
    ) -> _EncodedHybridState:
        """Run the sole geometry/message path shared by every terminal head."""

        graphs = time.numel()
        graph_time = self._embed_modality_times(time, element_time, lattice_time)
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
        if condition_present.shape != (graphs, 1) or condition_present.dtype != torch.bool:
            raise ValueError("condition presence must contain one boolean per graph")
        if shape_projector.shape != (graphs, 6, 6):
            raise ValueError("shape projector must have shape [graphs,6,6]")
        if fractional_to_cartesian.shape != (graphs, 3, 3):
            raise ValueError("fractional-to-Cartesian chart must have shape [graphs,3,3]")
        if batch.numel() == 0 or int(batch.min()) != 0 or int(batch.max()) + 1 != graphs:
            raise ValueError("hybrid batch must contain every graph index")
        # Geometry kernels and state charts stay FP32 under learned-path AMP.
        with torch.autocast(device_type=log_volume.device.type, enabled=False):
            projected_shape = project_lattice_state(log_shape, shape_projector)
            if not torch.allclose(log_shape, projected_shape, atol=2e-6, rtol=2e-6):
                raise ValueError("denoiser lattice shape is outside the blueprint subspace")
            if geometry_lattice is None:
                lattice = LatticeVolumeShape(log_volume, log_shape).lattice(
                    fractional_to_cartesian
                )
            else:
                if geometry_lattice.shape != (graphs, 3, 3):
                    raise ValueError("explicit geometry lattice must have shape [graphs,3,3]")
                # The only explicit-geometry caller is forward_physical_features,
                # which derives log_volume/log_shape from this same clean lattice.
                # Re-decoding an ill-conditioned FP32 SPD chart is neither an
                # independent consistency check nor a stable way to build edges.
                lattice = geometry_lattice.float()
            edges = periodic_radius_multigraph(frac_coords, lattice, batch, cutoff=self.radial.cutoff)
            radial = self.radial(edges.distance)
            edge_envelope = self.radial.envelope(edges.distance)
        source, target = edges.source, edges.target
        degree = torch.bincount(target, minlength=element_tokens.numel()).to(log_volume)
        node_time = graph_time[batch]
        element_nodes, graph_state, lattice_context = self._composition_lattice_context(
            element_tokens,
            log_volume,
            log_shape,
            batch,
            graph_time,
            composition_counts,
        )
        initial_nodes = element_nodes + self.degree_embedding(degree.log1p().unsqueeze(-1))
        node_state = graph_state[batch]
        if bool(condition_present.any()):
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
                tensor_condition,
                condition_present,
                edges.direction,
                edge_graph,
                geometry_queries,
                time,
            )
        else:
            gauge_atlas = self.gauge_atlas.null_output(
                graph_count=graphs,
                edge_count=edges.direction.shape[0],
                reference=tensor_condition,
            )
        node_condition = gauge_atlas.graph_condition[batch]
        if self.tensor_residual_adapter is not None and bool(condition_present.any()):
            present = condition_present.to(gauge_atlas.graph_condition)
            null_condition = self.gauge_atlas.null_condition.unsqueeze(0).expand_as(
                gauge_atlas.graph_condition
            )
            # Use only a centered residual over the frozen Stage-C null
            # field.  This removes the untrained raw present-atlas path from
            # the initialization while retaining an immediate adapter
            # gradient.  The same residual controls edge response, so no
            # tensor-dependent equivariant vector can leak into the backbone
            # before the adapter has learned it.
            residual = self._centered_tensor_residual(tensor_condition)
            graph_condition = null_condition + present * residual
            graph_state = graph_state + present * residual
            edge_gate = torch.tanh(residual.mean(dim=-1, keepdim=True))
            edge_graph = batch[source] if source.numel() else batch.new_empty((0,))
            edge_response = gauge_atlas.edge_response * edge_gate[edge_graph]
            edge_response = torch.where(
                condition_present.reshape(-1)[edge_graph].unsqueeze(-1),
                edge_response,
                torch.zeros_like(edge_response),
            )
            gauge_atlas = replace(gauge_atlas, graph_condition=graph_condition)
            gauge_atlas = replace(gauge_atlas, edge_response=edge_response)
            node_condition = graph_condition[batch]
        nodes = initial_nodes + node_time + node_condition + node_state
        # The lattice readout is part of the shared product-space field.  It
        # must see the effective (possibly centered-adapter) graph condition;
        # the pre-atlas ``lattice_context`` captured the Stage-C state before
        # this branch and would otherwise make tensor conditioning silently
        # disappear from hybrid volume/shape heads.
        lattice_context = torch.cat(
            (lattice_context[..., : 3 * self.hidden_dim], graph_state), dim=-1
        )
        if source.numel():
            self_image = ((source == target) & edges.image_shift.ne(0).any(dim=-1)).to(nodes).unsqueeze(-1)
            edge_state = self.edge_state_initializer(
                torch.cat((nodes[source], nodes[target], radial, self_image), dim=-1)
            )
        else:
            edge_state = nodes.new_empty((0, self.edge_dim))
        vectors = nodes.new_zeros((nodes.shape[0], self.coordinate_carrier.vector_channels, 3))
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
        return _EncodedHybridState(
            nodes=nodes,
            vectors=vectors,
            edge_state=edge_state,
            source=source,
            target=target,
            direction=edges.direction,
            radial=radial,
            edge_envelope=edge_envelope,
            graph_time=graph_time,
            graph_state=graph_state,
            lattice_context=lattice_context,
            node_time=node_time,
            node_condition=node_condition,
            node_state=node_state,
            lattice=lattice,
            gauge_atlas=gauge_atlas,
        )

    def forward_physical_features(
        self,
        element_tokens: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
    ) -> HybridBackboneFeatures:
        """Encode clean periodic structures without evaluating generation heads."""

        if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
            raise ValueError("physical lattice must have shape [graphs,3,3]")
        graphs = lattice.shape[0]
        if element_tokens.ndim != 1 or element_tokens.dtype != torch.long:
            raise ValueError("physical elements must be rank-one int64 tokens")
        if element_tokens.numel() == 0 or bool(
            ((element_tokens < 0) | (element_tokens >= CHEMICAL_ELEMENT_COUNT)).any()
        ):
            raise ValueError("physical structures require nonempty chemical element tokens")
        if batch.shape != element_tokens.shape or batch.dtype != torch.long:
            raise ValueError("physical batch must index every node")
        if frac_coords.shape != (element_tokens.numel(), 3):
            raise ValueError("physical fractional coordinates must have shape [nodes,3]")
        if (
            element_tokens.device != lattice.device
            or frac_coords.device != lattice.device
            or batch.device != lattice.device
        ):
            raise ValueError("physical structure tensors must share one device")
        identity_chart = torch.eye(3, dtype=lattice.dtype, device=lattice.device).expand(graphs, 3, 3)
        with torch.autocast(device_type=lattice.device.type, enabled=False):
            lattice_state = LatticeVolumeShape.from_lattice(
                lattice.float(),
                identity_chart.float(),
            )
        shape_projector = torch.eye(6, dtype=lattice.dtype, device=lattice.device).expand(graphs, 6, 6)
        clean_time = lattice_state.log_volume.new_zeros(graphs)
        tensor_condition = lattice_state.log_volume.new_zeros(graphs, 18)
        condition_present = torch.zeros((graphs, 1), dtype=torch.bool, device=lattice.device)
        flat = batch * CHEMICAL_ELEMENT_COUNT + element_tokens
        composition_counts = torch.bincount(
            flat,
            minlength=graphs * CHEMICAL_ELEMENT_COUNT,
        ).reshape(graphs, CHEMICAL_ELEMENT_COUNT)
        encoded = self._encode_hybrid_state(
            element_tokens,
            frac_coords,
            lattice_state.log_volume,
            lattice_state.log_shape,
            batch,
            clean_time,
            tensor_condition,
            condition_present,
            shape_projector,
            identity_chart,
            element_time=clean_time,
            lattice_time=clean_time,
            composition_counts=composition_counts,
            geometry_lattice=lattice,
        )
        return HybridBackboneFeatures(
            node_scalar=encoded.nodes,
            node_vectors=encoded.vectors,
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
        *,
        element_time: torch.Tensor | None = None,
        lattice_time: torch.Tensor | None = None,
        composition_counts: torch.Tensor | None = None,
    ) -> HybridDenoiserOutput:
        """Denoise one hybrid state.

        ``shape_projector`` is determined by the sampled space-group blueprint,
        never by a paired target structure.  No target CIF, lattice, space
        group, stabilizer, source ID, or endpoint token is accepted.
        """
        graphs = time.numel()
        encoded = self._encode_hybrid_state(
            element_tokens,
            frac_coords,
            log_volume,
            log_shape,
            batch,
            time,
            tensor_condition,
            condition_present,
            shape_projector,
            fractional_to_cartesian,
            element_time=element_time,
            lattice_time=lattice_time,
            composition_counts=composition_counts,
        )
        nodes = encoded.nodes
        vectors = encoded.vectors
        edge_state = encoded.edge_state
        source = encoded.source
        target = encoded.target
        radial = encoded.radial
        edge_envelope = encoded.edge_envelope
        graph_time = encoded.graph_time
        graph_state = encoded.graph_state
        lattice_context = encoded.lattice_context
        node_time = encoded.node_time
        node_condition = encoded.node_condition
        node_state = encoded.node_state
        lattice = encoded.lattice
        gauge_atlas = encoded.gauge_atlas
        graph_nodes = graph_mean(nodes, batch, graphs)
        graph_context = torch.cat((graph_nodes, graph_time, gauge_atlas.graph_condition, graph_state), dim=-1)
        if composition_counts is None:
            chemical = element_tokens < CHEMICAL_ELEMENT_COUNT
            flat_chemical = batch[chemical] * CHEMICAL_ELEMENT_COUNT + element_tokens[chemical]
            current_counts = torch.bincount(
                flat_chemical,
                minlength=graphs * CHEMICAL_ELEMENT_COUNT,
            ).reshape(graphs, CHEMICAL_ELEMENT_COUNT)
            observed_total = current_counts.sum(dim=-1, keepdim=True)
            uniform_composition = current_counts.new_full(
                current_counts.shape,
                1.0 / CHEMICAL_ELEMENT_COUNT,
                dtype=nodes.dtype,
            )
            current_composition = current_counts.to(nodes) / observed_total.clamp_min(1).to(nodes)
            current_composition = torch.where(
                observed_total > 0,
                current_composition,
                uniform_composition,
            )
            active_element_time = element_time if element_time is not None else time
            categorical_survival = torch.cos(0.5 * math.pi * active_element_time).square()
            base_composition = (
                categorical_survival.unsqueeze(-1) * current_composition
                + (1.0 - categorical_survival.unsqueeze(-1)) * uniform_composition
            )
            composition_residual = self.composition_head(
                torch.cat(
                    (
                        graph_context,
                        current_composition,
                        categorical_survival.unsqueeze(-1),
                    ),
                    dim=-1,
                )
            )
            composition_logits = (
                base_composition.clamp_min(1.0e-8).log()
                + (1.0 - categorical_survival.unsqueeze(-1)) * composition_residual
            )
            composition_probability = torch.softmax(composition_logits.float(), dim=-1)
        else:
            composition_probability = composition_counts.to(nodes)
            composition_probability = composition_probability / composition_probability.sum(dim=-1, keepdim=True)
            composition_logits = composition_probability.clamp_min(1.0e-8).log()
        # A sampled composition is a graph-level observed state, not a target
        # endpoint.  In the product-space path it is injected before every
        # message block and again here for the categorical readout.
        composition_token = composition_probability @ self.element_embedding.weight[:118].float()
        node_context = torch.cat(
            (
                nodes,
                node_time,
                node_condition,
                node_state,
                composition_token[batch],
            ),
            dim=-1,
        )
        element_logits = self.element_head(node_context)
        coordinate_control = torch.cat((node_time, node_condition, node_state), dim=-1)
        time_gated_vectors = vectors * torch.sigmoid(self.coordinate_control_gate(coordinate_control)).unsqueeze(-1)
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
                edge_hidden = edge_hidden + self.coordinate_edge_residual(edge_state.float())
        else:
            edge_hidden = nodes.new_empty((0, self.coordinate_carrier.moment_projection.in_features))
        carrier = self.coordinate_carrier(
            time_gated_vectors,
            edge_hidden,
            target,
            encoded.direction,
            edge_envelope,
            batch,
            graphs,
        )
        with torch.autocast(device_type=carrier.device.type, enabled=False):
            normalized_cartesian_score = self.coordinate_carrier_mixer(carrier, nodes)
            normalized_cartesian_score = (
                normalized_cartesian_score - graph_mean(normalized_cartesian_score, batch, graphs)[batch]
            )
            # The fractional torus path is unchanged.  Only the learned output
            # chart is made dimensionless with the O(3)- and GL(3,Z)-invariant
            # cell scale V^(1/3).  The physical Cartesian tangent is restored
            # before the exact fractional pullback consumed by the sampler.
            cell_scale = torch.exp(log_volume.float() / 3.0)
            cartesian_score = normalized_cartesian_score * cell_scale[batch, None]
            # The reverse sampler consumes a tangent drift because it adds the
            # network output to fractional coordinates. For r=fL, a Cartesian
            # tangent vector obeys v_r=v_f L and hence v_f=v_r L^-1. The prior
            # L^T covector pullback was an index-type error: it produced a
            # covector and then silently used it as a vector. Solve the
            # transposed row-vector system without forming an inverse, and keep
            # this physical chart change in FP32 under BF16 execution.
            fractional_score = cartesian_tangent_to_fractional(cartesian_score, lattice, batch)
            # Fractional zero mean is the translation-horizontal tangent chart
            # used by the coordinate probability path.
            fractional_score = fractional_score - graph_mean(fractional_score, batch, graphs)[batch]
        clean_volume_latent = self.volume_head(lattice_context).squeeze(-1)
        clean_shape_latent = self.shape_head(lattice_context)
        return HybridDenoiserOutput(
            clean_element_logits=element_logits,
            clean_composition_logits=composition_logits,
            coordinate_cartesian_scaled_score=cartesian_score,
            coordinate_fractional_scaled_score=fractional_score,
            clean_volume_latent=clean_volume_latent,
            clean_shape_latent=clean_shape_latent,
            gauge_atlas=gauge_atlas,
        )
