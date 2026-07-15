"""Stabilizer-aware response-field vector field for standalone GaugeFlow."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.profiler import record_function

from .manifold import log_vector_to_lattice, torus_logmap
from .stabilizer import (
    batched_soft_crystal_stabilizer_actions,
    proper_unimodular_candidates,
)
from .tensor import (
    fixed_lossless_response_probes,
    fixed_so3_frames,
    isotypic_slices,
    piezo_change_of_basis,
    piezo_from_irreps,
    piezo_to_irreps,
    rotate_rank3,
)
from .direct_irrep import CompleteDirectIrrepCoupling
from .geometry import GaussianRadialBasis, periodic_closest_image_edges
from .harmonic import HarmonicDoubleCosetConditionEncoder, OrbitInvariantConditionEncoder
from torch_geometric.utils import scatter
from .uncertainty import VelocityUncertainty, bounded_log_std


def scatter_mean(value: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    return scatter(value, index, dim=0, dim_size=dim_size, reduce="mean")


def safe_norm(
    value: torch.Tensor, dim: int | tuple[int, ...] | None = None, *, keepdim: bool = False
) -> torch.Tensor:
    """Smooth Euclidean norm with a defined zero gradient at a physical zero.

    Adding ``eps^2`` inside the square root avoids the flat, artificial
    near-zero region introduced by ``sqrt(clamp(sum(x^2), eps))`` while still
    making the exact zero tensor finite in forward and backward passes.
    """
    eps = torch.finfo(value.dtype).eps
    return (value.square().sum(dim=dim, keepdim=keepdim) + eps * eps).sqrt()


def direct_irrep_cartesian_products(
    tensors: torch.Tensor, directions: torch.Tensor, edge_graph: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two exact Cartesian tensor-product paths from ``e`` and an edge vector.

    ``e:(n outer n)`` and ``n_i e_ijk n_k`` are Cartesian forms of coupling the
    ``2x1o + 1x2o + 1x3o`` condition token to local vector/tensor geometry.
    They are algebraically equivalent to Clebsch--Gordan contractions, but use
    only ``einsum`` over Cartesian tensors: no spherical-harmonic evaluation is
    needed.
    """
    if directions.numel() == 0:
        empty = directions.new_empty((0, 3))
        return empty, empty
    edge_tensors = tensors[edge_graph]
    response = torch.einsum("eijk,ej,ek->ei", edge_tensors, directions, directions)
    crossed = torch.einsum("eijk,ei,ek->ej", edge_tensors, directions, directions)
    return response, crossed


class OrbitResponseFieldEncoder(nn.Module):
    """Discrete orbit-alignment constitutive-query encoder.

    The legacy finite-frame modes use seeded Haar-Monte-Carlo nodes, not an
    exact finite quadrature; representative invariance is approximate in node
    count.  ``harmonic_alignment_v1`` is a separate, deterministic-grid
    implementation with an early invariant channel.  The encoder consumes
    only the tensor condition and the evolving flow state.  In particular, it
    never accepts a target-CIF stabilizer during training because that object is
    unavailable in tensor-only sampling.
    """

    def __init__(self, hidden_dim: int, orbit_frames: int = 24, mode: str = "orbit_alignment"):
        super().__init__()
        if mode == "double_coset":
            # Kept only to load older smoke checkpoints. The old mode used a
            # target-CIF stabilizer during training, which tensor-only sampling
            # cannot provide.
            mode = "orbit_alignment"
        if mode not in {
            "raw_tensor", "direct_irrep", "direct_irrep_complete_v1",
            "invariant_only_v1", "stabilizer_pooling", "orbit_alignment", "harmonic_alignment_v1",
        }:
            raise ValueError(
                "unknown conditioning mode"
            )
        self.mode = mode
        self.register_buffer("rotations", fixed_so3_frames(orbit_frames))
        self.register_buffer("fixed_probes", fixed_lossless_response_probes())
        self.register_buffer("piezo_basis", piezo_change_of_basis(), persistent=False)
        catalogue = (
            proper_unimodular_candidates()
            if mode == "orbit_alignment"
            else torch.empty((0, 3, 3), dtype=torch.float32)
        )
        self.register_buffer("automorphism_candidates", catalogue, persistent=False)
        self.complete_direct = CompleteDirectIrrepCoupling() if mode == "direct_irrep_complete_v1" else None
        self.harmonic = (
            HarmonicDoubleCosetConditionEncoder(hidden_dim, grid_size=orbit_frames)
            if mode == "harmonic_alignment_v1" else None
        )
        self.invariant_only = (
            OrbitInvariantConditionEncoder(hidden_dim) if mode == "invariant_only_v1" else None
        )
        # Six scalar diagnostics plus six fixed three-vector constitutive
        # probes. These span Sym?(R?), so information is not limited to the
        # accidental local bond directions of a noisy state.
        self.candidate = nn.Sequential(nn.Linear(24, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.complete_candidate = nn.Sequential(
            nn.Linear(24, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Deliberately non-equivariant lab-frame control.  It is kept separate
        # from the Cartesian direct-irrep baseline so Gate A can distinguish a
        # raw component shortcut from an actual tensor--geometry interaction.
        self.raw_tensor = nn.Sequential(nn.Linear(18, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.null_condition = nn.Parameter(torch.zeros(hidden_dim))
        self.present_bias = nn.Parameter(torch.zeros(hidden_dim))

    def precompute_condition_orbit(self, piezo_irreps: torch.Tensor) -> torch.Tensor:
        """Build the fixed Cartesian tensor orbit once for a static condition."""
        tensors = piezo_from_irreps(piezo_irreps, self.piezo_basis)
        return rotate_rank3(
            tensors.unsqueeze(1), self.rotations.to(piezo_irreps).unsqueeze(0)
        )

    def forward(
        self,
        piezo_irreps: torch.Tensor,
        present: torch.Tensor,
        graph_query: torch.Tensor,
        edge_directions: torch.Tensor,
        edge_graph: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        batch: torch.Tensor,
        type_state: torch.Tensor,
        framed_tensors: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        graphs = piezo_irreps.shape[0]
        if self.mode == "invariant_only_v1":
            if self.invariant_only is None:
                raise RuntimeError("invariant-only encoder was not constructed")
            if framed_tensors is not None:
                raise ValueError("invariant-only conditioning does not accept a tensor-orbit cache")
            values = self.invariant_only(piezo_irreps).unsqueeze(1)
            weights = torch.ones((graphs, 1), dtype=values.dtype, device=values.device)
            aligned = values[:, 0] + self.present_bias
            mask = present.to(dtype=torch.bool)
            graph_condition = torch.where(
                mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned)
            )
            empty_field = edge_directions.new_zeros((edge_directions.shape[0], 3))
            outputs = (graph_condition, empty_field, empty_field, weights)
            if not return_diagnostics:
                return outputs
            return (*outputs, {
                "raw_condition_embedding": values[:, 0],
                "frame_candidate_embeddings": values,
                "frame_weights": weights,
                "uniform_pooled_embedding": values[:, 0],
                "aligned_embedding": aligned,
                "pool_then_embed": values[:, 0],
                "stabilizer_posterior": None,
            })
        if self.mode == "harmonic_alignment_v1":
            if self.harmonic is None:
                raise RuntimeError("harmonic encoder was not constructed")
            if framed_tensors is not None:
                raise ValueError("harmonic alignment does not accept legacy finite-frame caches")
            if time is None:
                raise ValueError("harmonic alignment requires path time")
            harmonic_outputs = self.harmonic(
                piezo_irreps, present, edge_directions, edge_graph, time,
                return_diagnostics=return_diagnostics,
            )
            if not return_diagnostics:
                return harmonic_outputs  # type: ignore[return-value]
            graph_condition, field, auxiliary, weights, diagnostics = harmonic_outputs
            return graph_condition, field, auxiliary, weights, {
                "raw_condition_embedding": diagnostics["invariant_embedding"],
                "frame_candidate_embeddings": diagnostics["aligned_irreps"].unsqueeze(1),
                "frame_weights": weights,
                "uniform_pooled_embedding": diagnostics["invariant_embedding"],
                "aligned_embedding": diagnostics["aligned_embedding"],
                "pool_then_embed": diagnostics["aligned_embedding"],
                "stabilizer_posterior": None,
                "harmonic": diagnostics,
            }
        if self.mode == "raw_tensor":
            values = self.raw_tensor(piezo_irreps).unsqueeze(1)
            weights = torch.ones((graphs, 1), dtype=values.dtype, device=values.device)
            aligned = values[:, 0] + self.present_bias
            mask = present.to(dtype=torch.bool)
            graph_condition = torch.where(mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned))
            empty_field = edge_directions.new_zeros((edge_directions.shape[0], 3))
            outputs = (graph_condition, empty_field, empty_field, weights)
            if not return_diagnostics:
                return outputs
            return (*outputs, {
                "raw_condition_embedding": values[:, 0],
                "frame_candidate_embeddings": values,
                "frame_weights": weights,
                "uniform_pooled_embedding": values[:, 0],
                "aligned_embedding": aligned,
                "pool_then_embed": values[:, 0],
                "stabilizer_posterior": None,
            })

        if self.mode in {"direct_irrep", "direct_irrep_complete_v1"}:
            tensors = piezo_from_irreps(piezo_irreps, self.piezo_basis)
            if self.mode == "direct_irrep_complete_v1":
                if self.complete_direct is None:
                    raise RuntimeError("complete direct-irrep module was not constructed")
                edge_tensors = tensors[edge_graph]
                fields = self.complete_direct(edge_tensors, edge_directions)
                path_norms = scatter_mean(safe_norm(fields, dim=-1), edge_graph, graphs)
                pairwise = torch.einsum("efi,egi->efg", fields, fields)
                upper = torch.triu_indices(6, 6, offset=1, device=fields.device)
                path_pairs = scatter_mean(pairwise[:, upper[0], upper[1]], edge_graph, graphs)
                component_norms = torch.stack(
                    [safe_norm(piezo_irreps[:, block], dim=-1) for block in isotypic_slices()], dim=-1
                )
                features = torch.cat((component_norms, path_norms, path_pairs), dim=-1)
                values = self.complete_candidate(features).unsqueeze(1)
                weights = torch.ones((graphs, 1), dtype=values.dtype, device=values.device)
                aligned = values[:, 0] + self.present_bias
                mask = present.to(dtype=torch.bool)
                graph_condition = torch.where(mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned))
                selected_field = torch.where(
                    mask[edge_graph].view(-1, 1, 1), fields, torch.zeros_like(fields)
                ) if fields.numel() else fields
                auxiliary = edge_directions.new_zeros((edge_directions.shape[0], 3))
                outputs = (graph_condition, selected_field, auxiliary, weights)
                if not return_diagnostics:
                    return outputs
                return (*outputs, {
                    "raw_condition_embedding": values[:, 0],
                    "frame_candidate_embeddings": values,
                    "frame_weights": weights,
                    "uniform_pooled_embedding": values[:, 0],
                    "aligned_embedding": aligned,
                    "pool_then_embed": values[:, 0],
                    "stabilizer_posterior": None,
                    "complete_direct_path_norms": path_norms,
                })
            primary, secondary = direct_irrep_cartesian_products(tensors, edge_directions, edge_graph)
            field_norm = scatter_mean(safe_norm(primary, dim=-1), edge_graph, graphs)
            auxiliary_norm = scatter_mean(safe_norm(secondary, dim=-1), edge_graph, graphs)
            cross_invariant = scatter_mean(
                (primary * secondary).sum(dim=-1), edge_graph, graphs
            )
            component_norms = torch.stack(
                [safe_norm(piezo_irreps[:, block], dim=-1) for block in isotypic_slices()],
                dim=-1,
            )
            features = torch.cat(
                (
                    component_norms,
                    field_norm.unsqueeze(-1),
                    auxiliary_norm.unsqueeze(-1),
                    cross_invariant.unsqueeze(-1),
                    tensors.new_zeros((graphs, 18)),
                ),
                dim=-1,
            )
            values = self.candidate(features).unsqueeze(1)
            weights = torch.ones((graphs, 1), dtype=values.dtype, device=values.device)
            aligned = values[:, 0] + self.present_bias
            mask = present.to(dtype=torch.bool)
            graph_condition = torch.where(mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned))
            selected_field = primary
            edge_auxiliary = secondary
            if selected_field.numel():
                selected_field = torch.where(mask[edge_graph], selected_field, torch.zeros_like(selected_field))
                edge_auxiliary = torch.where(mask[edge_graph], edge_auxiliary, torch.zeros_like(edge_auxiliary))
            outputs = (graph_condition, selected_field, edge_auxiliary, weights)
            if not return_diagnostics:
                return outputs
            return (*outputs, {
                "raw_condition_embedding": values[:, 0],
                "frame_candidate_embeddings": values,
                "frame_weights": weights,
                "uniform_pooled_embedding": values[:, 0],
                "aligned_embedding": aligned,
                "pool_then_embed": values[:, 0],
                "stabilizer_posterior": None,
            })

        tensors = piezo_from_irreps(piezo_irreps, self.piezo_basis)
        frames = self.rotations.shape[0]
        with record_function("model.tensor_orbit_rotation"):
            if framed_tensors is None:
                framed = self.precompute_condition_orbit(piezo_irreps)
            else:
                expected = (graphs, frames, 3, 3, 3)
                if framed_tensors.shape != expected:
                    raise ValueError(
                        f"Expected cached condition orbit {expected}, got {tuple(framed_tensors.shape)}"
                    )
                framed = framed_tensors.to(piezo_irreps)
            automorphism_weights: torch.Tensor | None = None
            if self.mode == "stabilizer_pooling":
                transformed = framed
            else:
                automorphisms, automorphism_weights = batched_soft_crystal_stabilizer_actions(
                    frac_coords,
                    lattices,
                    type_state,
                    batch,
                    candidates=self.automorphism_candidates,
                )
            # A posterior over latent proper automorphisms, derived only from
            # the evolving state. It is not an assertion that noisy x_t itself
            # has this exact point group. Right tensor-stabilizer actions need
            # no separate pooling: rho(R h)e == rho(R)e whenever h stabilizes e.
                transformed_by_action = rotate_rank3(
                    framed.unsqueeze(2), automorphisms.unsqueeze(1)
                )
                transformed = (
                    automorphism_weights[:, None, :, None, None, None]
                    * transformed_by_action
                ).sum(dim=2)
            transformed_irreps = piezo_to_irreps(transformed, self.piezo_basis)
        with record_function("model.response_queries"):
            fixed_fields = torch.einsum(
                "bfijk,mj,mk->bfmi",
                transformed,
                self.fixed_probes.to(transformed),
                self.fixed_probes.to(transformed),
            ).reshape(graphs, frames, -1)
            if edge_directions.numel():
                edge_fields = torch.einsum(
                    "efijk,ej,ek->efi",
                    transformed[edge_graph],
                    edge_directions,
                    edge_directions,
                )
                graph_field = scatter_mean(edge_fields, edge_graph, graphs)
            else:
                edge_fields = transformed.new_empty((0, frames, 3))
                graph_field = transformed.new_zeros((graphs, frames, 3))
        isotypic_norms = torch.stack(
            [safe_norm(transformed_irreps[..., block], dim=-1) for block in isotypic_slices()],
            dim=-1,
        )
        features = torch.cat(
            (
                isotypic_norms,
                safe_norm(graph_field, dim=-1, keepdim=True),
                isotypic_norms.new_zeros((graphs, frames, 2)),
                fixed_fields,
            ),
            dim=-1,
        )
        values = self.candidate(features)
        # This control exposes the same finite orbit features but removes the
        # coherent, state-conditioned relative-frame posterior.  It is an
        # orbit/stabilizer pooling baseline, not an alignment method.
        if self.mode == "stabilizer_pooling":
            weights = torch.full(
                (graphs, frames), 1.0 / frames, dtype=values.dtype, device=values.device
            )
        else:
            scores = (
                self.key(values) * self.query(graph_query).unsqueeze(1)
            ).sum(dim=-1) / math.sqrt(values.shape[-1])
            weights = torch.softmax(scores, dim=-1)
        uniform_pooled_embedding = values.mean(dim=1)
        aligned = (weights.unsqueeze(-1) * values).sum(dim=1) + self.present_bias
        # This is diagnostic-only: the production path remains ``sum_k q_k
        # phi(tensor_k)``.  Returning ``phi(sum_k q_k tensor_k)`` alongside it
        # lets Gate A.1 locate collapse caused by pooling before a nonlinear
        # embedding without changing the checkpoint's conditioning path.
        if return_diagnostics:
            pooled_tensor = (weights[..., None, None, None] * transformed).sum(dim=1)
            pooled_irreps = piezo_to_irreps(pooled_tensor, self.piezo_basis)
            pooled_fixed_fields = torch.einsum(
                "bijk,mj,mk->bmi",
                pooled_tensor,
                self.fixed_probes.to(pooled_tensor),
                self.fixed_probes.to(pooled_tensor),
            ).reshape(graphs, -1)
            pooled_graph_field = (weights.unsqueeze(-1) * graph_field).sum(dim=1)
            pooled_isotypic_norms = torch.stack(
                [safe_norm(pooled_irreps[..., block], dim=-1) for block in isotypic_slices()],
                dim=-1,
            )
            pooled_features = torch.cat(
                (
                    pooled_isotypic_norms,
                    safe_norm(pooled_graph_field, dim=-1, keepdim=True),
                    pooled_isotypic_norms.new_zeros((graphs, 2)),
                    pooled_fixed_fields,
                ),
                dim=-1,
            )
            pool_then_embed = self.candidate(pooled_features)
        mask = present.to(dtype=torch.bool)
        graph_condition = torch.where(mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned))
        if edge_fields.numel():
            selected_field = (weights[edge_graph].unsqueeze(-1) * edge_fields).sum(dim=1)
            selected_field = torch.where(mask[edge_graph], selected_field, torch.zeros_like(selected_field))
        else:
            selected_field = edge_directions.new_empty((0, 3))
        auxiliary = torch.zeros_like(selected_field)
        outputs = (graph_condition, selected_field, auxiliary, weights)
        if not return_diagnostics:
            return outputs
        return (*outputs, {
            "raw_condition_embedding": values[:, 0],
            "frame_candidate_embeddings": values,
            "frame_weights": weights,
            "uniform_pooled_embedding": uniform_pooled_embedding,
            "aligned_embedding": aligned,
            "pool_then_embed": pool_then_embed,
            "stabilizer_posterior": automorphism_weights,
        })


class ResponseMessageLayer(nn.Module):
    """SO(3)-equivariant scalar/vector message update.

    Scalar networks see only norms and dot products.  Every Cartesian vector
    is formed as a scalar-weighted sum of an edge direction, a response vector,
    and an incoming vector feature, so it transforms covariantly under SO(3).
    """

    def __init__(
        self, hidden_dim: int, vector_dim: int, radial_basis_dim: int, response_channels: int = 1,
    ):
        super().__init__()
        if response_channels < 1 or radial_basis_dim < 2:
            raise ValueError("response_channels must be positive and radial_basis_dim at least two")
        self.vector_dim = vector_dim
        self.response_channels = response_channels
        self.radial_basis_dim = radial_basis_dim
        # Preserve the historical K=1 parameter shapes and forward equations
        # exactly.  K=6 is used only by the new complete-CG baseline.
        field_count = response_channels + 1  # primary fields plus auxiliary
        invariant_features = 1 + 2 * field_count + field_count * (field_count - 1) // 2
        scalar_features = hidden_dim * 3 + (6 if response_channels == 1 else invariant_features) + radial_basis_dim
        self.scalar_message = nn.Sequential(
            nn.Linear(scalar_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.vector_gates = nn.Sequential(
            nn.Linear(scalar_features, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, vector_dim * (4 if response_channels == 1 else response_channels + 3))
        )
        self.vector_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vector_dim), nn.Sigmoid()
        )

    def forward(
        self, nodes: torch.Tensor, vectors: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
        directions: torch.Tensor, response: torch.Tensor, auxiliary_response: torch.Tensor,
        node_condition: torch.Tensor, radial_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.numel() == 0:
            return nodes, vectors
        if radial_basis.shape != (source.numel(), self.radial_basis_dim):
            raise ValueError("metric layer received an incompatible radial basis")
        if self.response_channels != 1:
            if response.ndim != 3 or response.shape[1:] != (self.response_channels, 3):
                raise ValueError("multi-response layer received an incompatible response field")
            if auxiliary_response.shape != (response.shape[0], 3):
                raise ValueError("multi-response layer requires one auxiliary polar vector per edge")
            fields = torch.cat((response, auxiliary_response.unsqueeze(1)), dim=1)
            direction_norm = safe_norm(directions, dim=-1, keepdim=True)
            direction_dots = (fields * directions.unsqueeze(1)).sum(dim=-1)
            field_norms = safe_norm(fields, dim=-1)
            pairwise = torch.einsum("efi,egi->efg", fields, fields)
            upper = torch.triu_indices(fields.shape[1], fields.shape[1], offset=1, device=fields.device)
            pair_features = pairwise[:, upper[0], upper[1]]
            invariants = torch.cat((direction_norm, direction_dots, field_norms, pair_features), dim=-1)
            edge_scalars = torch.cat(
                (nodes[source], nodes[target], node_condition[source], invariants, radial_basis), dim=-1
            )
            messages = self.scalar_message(edge_scalars)
            aggregate = nodes.new_zeros(nodes.shape)
            aggregate.index_add_(0, target, messages)
            updated_nodes = nodes + self.update(torch.cat((nodes, aggregate), dim=-1))
            gates = self.vector_gates(edge_scalars).reshape(
                -1, self.response_channels + 3, self.vector_dim, 1
            )
            vector_messages = gates[:, 0] * directions.unsqueeze(1) + gates[:, -1] * vectors[source]
            for field_index in range(fields.shape[1]):
                vector_messages = vector_messages + gates[:, field_index + 1] * fields[:, field_index].unsqueeze(1)
            vector_aggregate = vectors.new_zeros(vectors.shape)
            vector_aggregate.index_add_(0, target, vector_messages)
            return updated_nodes, vectors + self.vector_update(updated_nodes).unsqueeze(-1) * vector_aggregate
        invariants = torch.stack(
            (
                (directions * response).sum(dim=-1),
                safe_norm(response, dim=-1),
                safe_norm(directions, dim=-1),
                (directions * auxiliary_response).sum(dim=-1),
                safe_norm(auxiliary_response, dim=-1),
                (response * auxiliary_response).sum(dim=-1),
            ),
            dim=-1,
        )
        edge_scalars = torch.cat(
            (nodes[source], nodes[target], node_condition[source], invariants, radial_basis), dim=-1
        )
        messages = self.scalar_message(edge_scalars)
        aggregate = nodes.new_zeros(nodes.shape)
        aggregate.index_add_(0, target, messages)
        updated_nodes = nodes + self.update(torch.cat((nodes, aggregate), dim=-1))

        gates = self.vector_gates(edge_scalars).reshape(-1, 4, self.vector_dim, 1)
        vector_messages = (
            gates[:, 0] * directions.unsqueeze(1)
            + gates[:, 1] * response.unsqueeze(1)
            + gates[:, 2] * auxiliary_response.unsqueeze(1)
            + gates[:, 3] * vectors[source]
        )
        vector_aggregate = vectors.new_zeros(vectors.shape)
        vector_aggregate.index_add_(0, target, vector_messages)
        return updated_nodes, vectors + self.vector_update(updated_nodes).unsqueeze(-1) * vector_aggregate


class ConditionedResidualBlock(nn.Module):
    """A per-message-block FiLM-gated conditional tangent residual.

    This module is intentionally separate from the base field.  Its input may
    read the tensor token and tensor response vectors, whereas the base path
    receives explicit zeros for both objects.  The returned values are a
    residual state, not a replacement for the condition-free message field.
    """

    def __init__(
        self, hidden_dim: int, vector_dim: int, radial_basis_dim: int, response_channels: int = 1,
    ):
        super().__init__()
        self.message = ResponseMessageLayer(
            hidden_dim, vector_dim, response_channels=response_channels, radial_basis_dim=radial_basis_dim
        )
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.residual_gate = nn.Linear(hidden_dim, hidden_dim)
        self.vector_gate = nn.Linear(hidden_dim, vector_dim)

    def forward(
        self,
        base_nodes: torch.Tensor,
        delta_nodes: torch.Tensor,
        delta_vectors: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        directions: torch.Tensor,
        response: torch.Tensor,
        auxiliary_response: torch.Tensor,
        node_condition: torch.Tensor,
        radial_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.film(node_condition).chunk(2, dim=-1)
        context = (base_nodes + delta_nodes) * (1.0 + scale) + shift
        vector_scale = torch.sigmoid(self.vector_gate(node_condition)).unsqueeze(-1)
        vector_context = delta_vectors * vector_scale
        updated_nodes, updated_vectors = self.message(
            context, vector_context, source, target, directions, response,
            auxiliary_response, node_condition, radial_basis,
        )
        node_update = updated_nodes - context
        vector_update = updated_vectors - vector_context
        return (
            delta_nodes + torch.sigmoid(self.residual_gate(node_condition)) * node_update,
            delta_vectors + vector_scale * vector_update,
        )


class EndpointIdConditionEncoder(nn.Module):
    """Two-class endpoint-ID control used only by the A4 substrate audit.

    This deliberately has no tensor interpretation, no tensor orbit and no
    response field.  It exposes the exact same graph-token interface to the
    existing message-passing backbone so a failed endpoint-ID experiment is a
    generator-substrate failure, rather than a tensor-conditioning failure.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.null_condition = nn.Parameter(torch.zeros(hidden_dim))
        self.present_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        endpoint_id: torch.Tensor,
        present: torch.Tensor,
        graph_query: torch.Tensor,
        edge_directions: torch.Tensor,
        edge_graph: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        batch: torch.Tensor,
        type_state: torch.Tensor,
        framed_tensors: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        del graph_query, edge_graph, frac_coords, lattices, batch, type_state, time
        if framed_tensors is not None:
            raise ValueError("endpoint-ID control has no tensor orbit cache")
        if endpoint_id.ndim != 2 or endpoint_id.shape[-1] != 2:
            raise ValueError(
                f"endpoint-ID control requires [num_graphs, 2] one-hot IDs, got {tuple(endpoint_id.shape)}"
            )
        if not torch.allclose(endpoint_id.sum(dim=-1), torch.ones_like(endpoint_id[:, 0])):
            raise ValueError("endpoint-ID control requires normalized one-hot IDs")
        values = self.embedding(endpoint_id).unsqueeze(1)
        aligned = values[:, 0] + self.present_bias
        mask = present.to(dtype=torch.bool)
        graph_condition = torch.where(
            mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned)
        )
        empty_field = edge_directions.new_zeros((edge_directions.shape[0], 3))
        weights = torch.ones((endpoint_id.shape[0], 1), device=endpoint_id.device, dtype=endpoint_id.dtype)
        outputs = (graph_condition, empty_field, empty_field, weights)
        if not return_diagnostics:
            return outputs
        return (*outputs, {
            "raw_condition_embedding": values[:, 0],
            "frame_candidate_embeddings": values,
            "frame_weights": weights,
            "uniform_pooled_embedding": values[:, 0],
            "aligned_embedding": aligned,
            "pool_then_embed": values[:, 0],
            "stabilizer_posterior": None,
        })


class GaugeFlowVectorField(nn.Module):
    """Standalone vector field over atom logits, torus coordinates, and SPD lattice logs."""

    def __init__(
        self,
        hidden_dim: int = 256,
        layers: int = 4,
        orbit_frames: int = 24,
        atom_types: int = 119,
        vector_dim: int | None = None,
        conditioning_mode: str = "orbit_alignment",
        conditional_control: str = "original_injection",
        residual_g_min: float = 0.25,
        coordinate_rbf_dim: int = 16,
        coordinate_rbf_cutoff: float = 8.0,
        composition_max_atoms: int | None = None,
        composition_atom_types: int | None = None,
    ):
        super().__init__()
        if conditional_control not in {"original_injection", "residual_field"}:
            raise ValueError("conditional_control must be 'original_injection' or 'residual_field'")
        if not 0.0 <= residual_g_min <= 1.0:
            raise ValueError("residual_g_min must lie in [0, 1]")
        if coordinate_rbf_dim < 2 or coordinate_rbf_cutoff <= 0:
            raise ValueError("coordinate RBF requires at least two features and a positive cutoff")
        if composition_max_atoms is not None and composition_max_atoms < 1:
            raise ValueError("composition_max_atoms must be positive when set")
        if composition_atom_types is not None and not 1 <= composition_atom_types <= atom_types:
            raise ValueError("composition_atom_types must lie in [1, atom_types]")
        self.atom_types = atom_types
        self.conditional_control = conditional_control
        self.residual_g_min = residual_g_min
        self.coordinate_rbf_dim = coordinate_rbf_dim
        self.composition_max_atoms = composition_max_atoms
        self.composition_atom_types = composition_atom_types or atom_types
        self.type_input = nn.Linear(atom_types, hidden_dim)
        self.time_input = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conditioning_mode = conditioning_mode
        if conditioning_mode == "unconditional":
            # P5-D0 is deliberately a no-condition substrate qualification.
            # Do not construct a tensor, harmonic, null-token, or endpoint-ID
            # encoder here: its forward interface receives no condition input.
            self.response = None
        else:
            self.response = (
                EndpointIdConditionEncoder(hidden_dim)
                if conditioning_mode == "endpoint_id"
                else OrbitResponseFieldEncoder(hidden_dim, orbit_frames, conditioning_mode)
            )
        axis = torch.arange(-2, 3, dtype=torch.float32)
        self.register_buffer(
            "periodic_shifts", torch.cartesian_prod(axis, axis, axis), persistent=False
        )
        self.vector_dim = vector_dim or max(16, hidden_dim // 8)
        self.response_channels = 6 if conditioning_mode == "direct_irrep_complete_v1" else 1
        self.coordinate_rbf = GaussianRadialBasis(coordinate_rbf_dim, coordinate_rbf_cutoff)
        self.layers = nn.ModuleList([
            ResponseMessageLayer(
                hidden_dim, self.vector_dim, response_channels=self.response_channels,
                radial_basis_dim=self.coordinate_rbf_dim,
            )
            for _ in range(layers)
        ])
        # Construct these adapters in both A2 variants so hidden width, depth,
        # and total parameter capacity remain fixed across the pre-registered
        # mechanism comparison.  They are deliberately inactive in the
        # original-injection control.
        self.conditional_layers = nn.ModuleList(
            [
                ConditionedResidualBlock(
                    hidden_dim, self.vector_dim, response_channels=self.response_channels,
                    radial_basis_dim=self.coordinate_rbf_dim,
                )
                for _ in range(layers)
            ]
        )
        self.type_out = nn.Linear(hidden_dim, atom_types)
        self.coord_out = nn.Linear(self.vector_dim, 1, bias=False)
        self.lattice_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 6))
        self.delta_type_out = nn.Linear(hidden_dim, atom_types)
        self.delta_coord_out = nn.Linear(self.vector_dim, 1, bias=False)
        self.delta_lattice_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 6))
        self.composition_count_out = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, self.composition_atom_types * (composition_max_atoms + 1)),
            )
            if composition_max_atoms is not None else None
        )
        # Scalar heads: coordinates are explicitly interpreted in Cartesian
        # tangent space, where one scale gives the equivariant covariance sigma^2 I.
        self.type_log_std_out = nn.Linear(hidden_dim, 1)
        self.coord_log_std_out = nn.Linear(hidden_dim, 1)
        self.lattice_log_std_out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        type_state: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice_log: torch.Tensor,
        batch: torch.Tensor,
        time: torch.Tensor,
        piezo_irreps: torch.Tensor | None = None,
        condition_present: torch.Tensor | None = None,
        condition_orbit: torch.Tensor | None = None,
        return_uncertainty: bool = False,
        return_condition_diagnostics: bool = False,
        return_velocity_components: bool = False,
        return_composition_counts: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        lattices = log_vector_to_lattice(lattice_log)
        edge_geometry = periodic_closest_image_edges(
            frac_coords, lattices, batch, shifts=self.periodic_shifts
        )
        source, target = edge_geometry.source, edge_geometry.target
        directions = edge_geometry.direction
        radial_basis = self.coordinate_rbf(edge_geometry.distance)
        edge_graph = batch[source] if source.numel() else batch.new_empty((0,))
        nodes = self.type_input(type_state)
        # Flow matching needs the scalar path time in the vector-field state
        # itself.  Supplying it only through ``graph_query`` is insufficient:
        # endpoint-ID, raw-tensor, and direct-irrep condition encoders do not
        # consume that query, which previously made original-injection fields
        # time-blind.  This broadcast is condition-independent and is applied
        # before every message-passing block in all conditioning modes.
        node_time_embedding = self.time_input(time.unsqueeze(-1))[batch]
        vectors = nodes.new_zeros((nodes.shape[0], self.vector_dim, 3))
        graph_seed = scatter_mean(nodes + node_time_embedding, batch, lattices.shape[0])
        if self.conditioning_mode == "unconditional":
            if piezo_irreps is not None or condition_present is not None or condition_orbit is not None:
                raise ValueError("unconditional mode must not receive tensor, mask, or condition-orbit inputs")
            graph_condition = nodes.new_zeros((lattices.shape[0], nodes.shape[-1]))
            edge_response = directions.new_zeros((directions.shape[0], 3))
            edge_auxiliary = directions.new_zeros((directions.shape[0], 3))
            alignment = torch.ones((lattices.shape[0], 1), dtype=nodes.dtype, device=nodes.device)
            condition_diagnostics = None
        else:
            if piezo_irreps is None or condition_present is None:
                raise ValueError("conditional modes require piezo_irreps and condition_present")
            if self.response is None:
                raise RuntimeError("conditional response encoder was not constructed")
            response_outputs = self.response(
                piezo_irreps, condition_present, graph_seed, directions, edge_graph,
                frac_coords, lattices, batch, type_state, condition_orbit,
                return_diagnostics=return_condition_diagnostics, time=time,
            )
            graph_condition, edge_response, edge_auxiliary, alignment = response_outputs[:4]
            condition_diagnostics = response_outputs[4] if return_condition_diagnostics else None

        if self.conditional_control == "original_injection":
            nodes = nodes + node_time_embedding + graph_condition[batch]
            for layer in self.layers:
                nodes, vectors = layer(
                    nodes, vectors, source, target, directions, edge_response,
                    edge_auxiliary, graph_condition[batch], radial_basis,
                )
            graph_nodes = scatter_mean(nodes, batch, lattices.shape[0])
            composition_graph_nodes = graph_nodes
            cartesian_velocity = self.coord_out(vectors.transpose(-1, -2)).squeeze(-1)
            lattice_nodes = lattices[batch]
            fractional_velocity = torch.einsum("ni,nij->nj", cartesian_velocity, torch.linalg.inv(lattice_nodes))
            type_velocity = self.type_out(nodes)
            lattice_velocity = self.lattice_out(graph_nodes)
            components = {
                "mode": self.conditional_control,
                "gate": torch.ones_like(time),
                "type_base": type_velocity,
                "type_conditional_residual": torch.zeros_like(type_velocity),
                "coordinate_base": fractional_velocity,
                "coordinate_conditional_residual": torch.zeros_like(fractional_velocity),
                "lattice_base": lattice_velocity,
                "lattice_conditional_residual": torch.zeros_like(lattice_velocity),
            }
            uncertainty_nodes, uncertainty_graph_nodes = nodes, graph_nodes
        else:
            # v_base is condition-free: no tensor token, response vector, or
            # tensor-derived edge feature reaches this path.
            base_nodes = nodes + node_time_embedding
            base_vectors = vectors
            delta_nodes = nodes.new_zeros(nodes.shape)
            delta_vectors = vectors.new_zeros(vectors.shape)
            zero_condition = torch.zeros_like(graph_condition[batch])
            zero_response = torch.zeros_like(edge_response)
            zero_auxiliary = torch.zeros_like(edge_auxiliary)
            for base_layer, conditional_layer in zip(self.layers, self.conditional_layers):
                base_nodes, base_vectors = base_layer(
                    base_nodes, base_vectors, source, target, directions, zero_response,
                    zero_auxiliary, zero_condition, radial_basis,
                )
                delta_nodes, delta_vectors = conditional_layer(
                    base_nodes, delta_nodes, delta_vectors, source, target, directions,
                    edge_response, edge_auxiliary, graph_condition[batch], radial_basis,
                )
            base_graph_nodes = scatter_mean(base_nodes, batch, lattices.shape[0])
            delta_graph_nodes = scatter_mean(delta_nodes, batch, lattices.shape[0])
            composition_graph_nodes = base_graph_nodes + delta_graph_nodes
            base_type = self.type_out(base_nodes)
            base_cartesian = self.coord_out(base_vectors.transpose(-1, -2)).squeeze(-1)
            delta_cartesian = self.delta_coord_out(delta_vectors.transpose(-1, -2)).squeeze(-1)
            lattice_nodes = lattices[batch]
            base_coordinate = torch.einsum("ni,nij->nj", base_cartesian, torch.linalg.inv(lattice_nodes))
            delta_coordinate = torch.einsum("ni,nij->nj", delta_cartesian, torch.linalg.inv(lattice_nodes))
            base_lattice = self.lattice_out(base_graph_nodes)
            delta_type = self.delta_type_out(delta_nodes)
            delta_lattice = self.delta_lattice_out(delta_graph_nodes)
            gate = self.residual_g_min + (1.0 - self.residual_g_min) * 4.0 * time * (1.0 - time)
            node_gate = gate[batch].unsqueeze(-1)
            type_velocity = base_type + node_gate * delta_type
            fractional_velocity = base_coordinate + node_gate * delta_coordinate
            lattice_velocity = base_lattice + gate.unsqueeze(-1) * delta_lattice
            components = {
                "mode": self.conditional_control,
                "gate": gate,
                "type_base": base_type,
                "type_conditional_residual": delta_type,
                "coordinate_base": base_coordinate,
                "coordinate_conditional_residual": delta_coordinate,
                "lattice_base": base_lattice,
                "lattice_conditional_residual": delta_lattice,
            }
            uncertainty_nodes = base_nodes + delta_nodes
            uncertainty_graph_nodes = base_graph_nodes + delta_graph_nodes
        outputs = (type_velocity, fractional_velocity, lattice_velocity, alignment)
        if return_uncertainty:
            uncertainty = VelocityUncertainty(
                type_log_std=bounded_log_std(self.type_log_std_out(uncertainty_nodes)),
                coord_log_std=bounded_log_std(self.coord_log_std_out(uncertainty_nodes)),
                lattice_log_std=bounded_log_std(self.lattice_log_std_out(uncertainty_graph_nodes)),
            )
            outputs = (*outputs, uncertainty)
        if return_condition_diagnostics:
            outputs = (*outputs, condition_diagnostics)
        if return_velocity_components:
            outputs = (*outputs, components)
        if return_composition_counts:
            if self.composition_count_out is None or self.composition_max_atoms is None:
                raise ValueError("composition-count outputs require composition_max_atoms at model construction")
            composition_logits = self.composition_count_out(composition_graph_nodes).reshape(
                -1, self.composition_atom_types, self.composition_max_atoms + 1
            )
            outputs = (*outputs, composition_logits)
        return outputs
