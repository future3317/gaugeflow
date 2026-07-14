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
from torch_geometric.utils import scatter
from .uncertainty import VelocityUncertainty, bounded_log_std


def scatter_mean(value: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    return scatter(value, index, dim=0, dim_size=dim_size, reduce="mean")


def safe_norm(
    value: torch.Tensor, dim: int | tuple[int, ...] | None = None, *, keepdim: bool = False
) -> torch.Tensor:
    """Euclidean norm with a defined zero gradient at the physical zero tensor."""
    return value.square().sum(dim=dim, keepdim=keepdim).clamp_min(1e-12).sqrt()


def periodic_complete_edges(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    shifts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense edges queried by nearest periodic Cartesian displacements.

    The query directions are local physical bonds, not the three columns of a
    chosen unit cell.  We solve the finite-shell closest-image problem directly
    in Cartesian space; the shell is intentionally explicit so callers can
    stress-test it under unimodular cell changes.
    """
    with record_function("model.periodic_neighbor_graph"):
        atom_count = frac_coords.shape[0]
        same_graph = batch[:, None] == batch[None, :]
        keep = same_graph & ~torch.eye(atom_count, dtype=torch.bool, device=batch.device)
        source, target = torch.nonzero(keep, as_tuple=True)
        if source.numel() == 0:
            empty = batch.new_empty((0,))
            return empty, empty, frac_coords.new_empty((0, 3))
        delta = frac_coords[target] - frac_coords[source]
        if shifts is None:
            axis = torch.arange(-2, 3, device=delta.device, dtype=delta.dtype)
            shifts = torch.cartesian_prod(axis, axis, axis)
        else:
            shifts = shifts.to(delta)
        images = delta.unsqueeze(1) + shifts.unsqueeze(0)
        edge_lattice = lattice[batch[source]]
        cart_images = torch.einsum("esi,eij->esj", images, edge_lattice)
        closest = cart_images.square().sum(dim=-1).argmin(dim=-1)
        cart = cart_images[
            torch.arange(cart_images.shape[0], device=cart_images.device), closest
        ]
        directions = cart / torch.linalg.vector_norm(cart, dim=-1, keepdim=True).clamp_min(1e-8)
        return source, target, directions


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

    SO(3) integration is a finite quadrature, hence representative invariance
    is approximate in the number of nodes.  The encoder intentionally consumes
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
        if mode not in {"raw_tensor", "direct_irrep", "stabilizer_pooling", "orbit_alignment"}:
            raise ValueError(
                "mode must be 'raw_tensor', 'direct_irrep', 'stabilizer_pooling', or 'orbit_alignment'"
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
        # Six scalar diagnostics plus six fixed three-vector constitutive
        # probes. These span Sym?(R?), so information is not limited to the
        # accidental local bond directions of a noisy state.
        self.candidate = nn.Sequential(nn.Linear(24, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
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
    ) -> tuple[torch.Tensor, ...]:
        graphs = piezo_irreps.shape[0]
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

        if self.mode == "direct_irrep":
            tensors = piezo_from_irreps(piezo_irreps, self.piezo_basis)
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

    def __init__(self, hidden_dim: int, vector_dim: int):
        super().__init__()
        self.vector_dim = vector_dim
        scalar_features = hidden_dim * 3 + 6
        self.scalar_message = nn.Sequential(
            nn.Linear(scalar_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.vector_gates = nn.Sequential(
            nn.Linear(scalar_features, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vector_dim * 4)
        )
        self.vector_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vector_dim), nn.Sigmoid()
        )

    def forward(
        self, nodes: torch.Tensor, vectors: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
        directions: torch.Tensor, response: torch.Tensor, auxiliary_response: torch.Tensor,
        node_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.numel() == 0:
            return nodes, vectors
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
        edge_scalars = torch.cat((nodes[source], nodes[target], node_condition[source], invariants), dim=-1)
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

    def __init__(self, hidden_dim: int, vector_dim: int):
        super().__init__()
        self.message = ResponseMessageLayer(hidden_dim, vector_dim)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.film(node_condition).chunk(2, dim=-1)
        context = (base_nodes + delta_nodes) * (1.0 + scale) + shift
        vector_scale = torch.sigmoid(self.vector_gate(node_condition)).unsqueeze(-1)
        vector_context = delta_vectors * vector_scale
        updated_nodes, updated_vectors = self.message(
            context, vector_context, source, target, directions, response,
            auxiliary_response, node_condition,
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
    ) -> tuple[torch.Tensor, ...]:
        del graph_query, edge_graph, frac_coords, lattices, batch, type_state
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
        composition_max_atoms: int | None = None,
        composition_atom_types: int | None = None,
    ):
        super().__init__()
        if conditional_control not in {"original_injection", "residual_field"}:
            raise ValueError("conditional_control must be 'original_injection' or 'residual_field'")
        if not 0.0 <= residual_g_min <= 1.0:
            raise ValueError("residual_g_min must lie in [0, 1]")
        if composition_max_atoms is not None and composition_max_atoms < 1:
            raise ValueError("composition_max_atoms must be positive when set")
        if composition_atom_types is not None and not 1 <= composition_atom_types <= atom_types:
            raise ValueError("composition_atom_types must lie in [1, atom_types]")
        self.atom_types = atom_types
        self.conditional_control = conditional_control
        self.residual_g_min = residual_g_min
        self.composition_max_atoms = composition_max_atoms
        self.composition_atom_types = composition_atom_types or atom_types
        self.type_input = nn.Linear(atom_types, hidden_dim)
        self.time_input = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conditioning_mode = conditioning_mode
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
        self.layers = nn.ModuleList([ResponseMessageLayer(hidden_dim, self.vector_dim) for _ in range(layers)])
        # Construct these adapters in both A2 variants so hidden width, depth,
        # and total parameter capacity remain fixed across the pre-registered
        # mechanism comparison.  They are deliberately inactive in the
        # original-injection control.
        self.conditional_layers = nn.ModuleList(
            [ConditionedResidualBlock(hidden_dim, self.vector_dim) for _ in range(layers)]
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
        piezo_irreps: torch.Tensor,
        condition_present: torch.Tensor,
        condition_orbit: torch.Tensor | None = None,
        return_uncertainty: bool = False,
        return_condition_diagnostics: bool = False,
        return_velocity_components: bool = False,
        return_composition_counts: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        lattices = log_vector_to_lattice(lattice_log)
        source, target, directions = periodic_complete_edges(
            frac_coords, lattices, batch, self.periodic_shifts
        )
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
        response_outputs = self.response(
            piezo_irreps, condition_present, graph_seed, directions, edge_graph,
            frac_coords, lattices, batch, type_state, condition_orbit,
            return_diagnostics=return_condition_diagnostics,
        )
        graph_condition, edge_response, edge_auxiliary, alignment = response_outputs[:4]
        condition_diagnostics = response_outputs[4] if return_condition_diagnostics else None

        if self.conditional_control == "original_injection":
            nodes = nodes + node_time_embedding + graph_condition[batch]
            for layer in self.layers:
                nodes, vectors = layer(
                    nodes, vectors, source, target, directions, edge_response,
                    edge_auxiliary, graph_condition[batch],
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
                    zero_auxiliary, zero_condition,
                )
                delta_nodes, delta_vectors = conditional_layer(
                    base_nodes, delta_nodes, delta_vectors, source, target, directions,
                    edge_response, edge_auxiliary, graph_condition[batch],
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
