"""Stabilizer-aware response-field vector field for standalone GaugeFlow."""

from __future__ import annotations

import math

import torch
from torch import nn

from .manifold import log_vector_to_lattice, torus_logmap
from .stabilizer import soft_crystal_stabilizer_actions
from .tensor import (
    fixed_lossless_response_probes,
    fixed_so3_frames,
    isotypic_slices,
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
    frac_coords: torch.Tensor, lattice: torch.Tensor, batch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense edges queried by nearest periodic Cartesian displacements.

    The query directions are local physical bonds, not the three columns of a
    chosen unit cell.  We solve the finite-shell closest-image problem directly
    in Cartesian space; the shell is intentionally explicit so callers can
    stress-test it under unimodular cell changes.
    """
    source, target, directions = [], [], []
    for graph in range(lattice.shape[0]):
        nodes = torch.nonzero(batch == graph, as_tuple=False).flatten()
        if nodes.numel() < 2:
            continue
        ii, jj = torch.meshgrid(nodes, nodes, indexing="ij")
        keep = ii != jj
        src, dst = ii[keep], jj[keep]
        delta = frac_coords[dst] - frac_coords[src]
        shifts = torch.cartesian_prod(
            *[torch.arange(-2, 3, device=delta.device, dtype=delta.dtype) for _ in range(3)]
        )
        images = delta.unsqueeze(1) + shifts.unsqueeze(0)
        cart_images = torch.einsum("esi,ij->esj", images, lattice[graph])
        closest = cart_images.square().sum(dim=-1).argmin(dim=-1)
        cart = cart_images[torch.arange(cart_images.shape[0], device=cart_images.device), closest]
        source.append(src)
        target.append(dst)
        directions.append(cart / torch.linalg.vector_norm(cart, dim=-1, keepdim=True).clamp_min(1e-8))
    if not source:
        empty = batch.new_empty((0,))
        return empty, empty, frac_coords.new_empty((0, 3))
    return torch.cat(source), torch.cat(target), torch.cat(directions)


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
        # Six scalar diagnostics plus six fixed three-vector constitutive
        # probes. These span Sym²(R³), so information is not limited to the
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        graphs = piezo_irreps.shape[0]
        if self.mode == "raw_tensor":
            values = self.raw_tensor(piezo_irreps).unsqueeze(1)
            weights = torch.ones((graphs, 1), dtype=values.dtype, device=values.device)
            aligned = values[:, 0] + self.present_bias
            mask = present.to(dtype=torch.bool)
            graph_condition = torch.where(mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned))
            empty_field = edge_directions.new_zeros((edge_directions.shape[0], 3))
            return graph_condition, empty_field, empty_field, weights

        if self.mode == "direct_irrep":
            tensors = piezo_from_irreps(piezo_irreps)
            edge_fields = edge_directions.new_zeros((edge_directions.shape[0], 1, 3))
            edge_auxiliary = edge_directions.new_zeros((edge_directions.shape[0], 3))
            primary, secondary = direct_irrep_cartesian_products(tensors, edge_directions, edge_graph)
            if primary.numel():
                edge_fields[:, 0] = primary
                edge_auxiliary = secondary
            graph_values = []
            for graph in range(graphs):
                edge_ids = torch.nonzero(edge_graph == graph, as_tuple=False).flatten()
                if edge_ids.numel():
                    field_norm = safe_norm(primary[edge_ids], dim=-1).mean().reshape(1)
                    auxiliary_norm = safe_norm(secondary[edge_ids], dim=-1).mean().reshape(1)
                    cross_invariant = (primary[edge_ids] * secondary[edge_ids]).sum(dim=-1).mean().reshape(1)
                else:
                    field_norm = tensors.new_zeros((1,))
                    auxiliary_norm = tensors.new_zeros((1,))
                    cross_invariant = tensors.new_zeros((1,))
                component_norms = torch.stack(
                    [safe_norm(piezo_irreps[graph, block]) for block in isotypic_slices()]
                )
                graph_values.append(
                    self.candidate(
                        torch.cat((
                            component_norms, field_norm, auxiliary_norm, cross_invariant,
                            tensors.new_zeros((18,)),
                        )).unsqueeze(0)
                    ).squeeze(0)
                )
            values = torch.stack(graph_values).unsqueeze(1)
            weights = torch.ones((graphs, 1), dtype=values.dtype, device=values.device)
            aligned = values[:, 0] + self.present_bias
            mask = present.to(dtype=torch.bool)
            graph_condition = torch.where(mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned))
            selected_field = edge_fields[:, 0] if edge_fields.numel() else edge_directions.new_empty((0, 3))
            if selected_field.numel():
                selected_field = torch.where(mask[edge_graph], selected_field, torch.zeros_like(selected_field))
                edge_auxiliary = torch.where(mask[edge_graph], edge_auxiliary, torch.zeros_like(edge_auxiliary))
            return graph_condition, selected_field, edge_auxiliary, weights

        tensors = piezo_from_irreps(piezo_irreps)
        frames = self.rotations.shape[0]
        edge_fields = edge_directions.new_zeros((edge_directions.shape[0], frames, 3))
        graph_values = []
        for graph in range(graphs):
            nodes = torch.nonzero(batch == graph, as_tuple=False).flatten()
            automorphisms, automorphism_weights = soft_crystal_stabilizer_actions(
                frac_coords[nodes], lattices[graph], type_state[nodes]
            )
            # A posterior over latent proper automorphisms, derived only from
            # the evolving state. It is not an assertion that noisy x_t itself
            # has this exact point group. Right tensor-stabilizer actions need
            # no separate pooling: rho(R h)e == rho(R)e whenever h stabilizes e.
            framed = rotate_rank3(tensors[graph], self.rotations.to(piezo_irreps))
            transformed = rotate_rank3(
                framed.unsqueeze(1), automorphisms.to(piezo_irreps).unsqueeze(0)
            )
            transformed = (automorphism_weights.to(piezo_irreps).view(1, -1, 1, 1, 1) * transformed).sum(dim=1)
            transformed_irreps = piezo_to_irreps(transformed)
            fixed_fields = torch.einsum(
                "fijk,mj,mk->fmi", transformed,
                self.fixed_probes.to(transformed), self.fixed_probes.to(transformed),
            ).reshape(frames, -1)
            edge_ids = torch.nonzero(edge_graph == graph, as_tuple=False).flatten()
            if edge_ids.numel():
                fields = torch.einsum(
                    "fijk,ej,ek->efi", transformed, edge_directions[edge_ids], edge_directions[edge_ids]
                )
                graph_field = fields.mean(dim=0)
                edge_fields[edge_ids] = fields
            else:
                graph_field = transformed.new_zeros((frames, 3))
            isotypic_norms = torch.stack(
                [safe_norm(transformed_irreps[..., block], dim=-1) for block in isotypic_slices()],
                dim=-1,
            )
            features = torch.cat(
                (
                    isotypic_norms,
                    safe_norm(graph_field, dim=-1, keepdim=True),
                    isotypic_norms.new_zeros((frames, 2)),
                    fixed_fields,
                ),
                dim=-1,
            )
            graph_values.append(self.candidate(features))
        values = torch.stack(graph_values)
        scores = (self.key(values) * self.query(graph_query).unsqueeze(1)).sum(dim=-1) / math.sqrt(values.shape[-1])
        # This control exposes the same finite orbit features but removes the
        # coherent, state-conditioned relative-frame posterior.  It is an
        # orbit/stabilizer pooling baseline, not an alignment method.
        if self.mode == "stabilizer_pooling":
            weights = torch.full_like(scores, 1.0 / scores.shape[-1])
        else:
            weights = torch.softmax(scores, dim=-1)
        aligned = (weights.unsqueeze(-1) * values).sum(dim=1) + self.present_bias
        mask = present.to(dtype=torch.bool)
        graph_condition = torch.where(mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned))
        if edge_fields.numel():
            selected_field = (weights[edge_graph].unsqueeze(-1) * edge_fields).sum(dim=1)
            selected_field = torch.where(mask[edge_graph], selected_field, torch.zeros_like(selected_field))
        else:
            selected_field = edge_directions.new_empty((0, 3))
        auxiliary = torch.zeros_like(selected_field)
        return graph_condition, selected_field, auxiliary, weights


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
    ):
        super().__init__()
        self.atom_types = atom_types
        self.type_input = nn.Linear(atom_types, hidden_dim)
        self.time_input = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.response = OrbitResponseFieldEncoder(hidden_dim, orbit_frames, conditioning_mode)
        self.vector_dim = vector_dim or max(16, hidden_dim // 8)
        self.layers = nn.ModuleList([ResponseMessageLayer(hidden_dim, self.vector_dim) for _ in range(layers)])
        self.type_out = nn.Linear(hidden_dim, atom_types)
        self.coord_out = nn.Linear(self.vector_dim, 1, bias=False)
        self.lattice_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 6))
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
        return_uncertainty: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        lattices = log_vector_to_lattice(lattice_log)
        source, target, directions = periodic_complete_edges(frac_coords, lattices, batch)
        edge_graph = batch[source] if source.numel() else batch.new_empty((0,))
        nodes = self.type_input(type_state)
        vectors = nodes.new_zeros((nodes.shape[0], self.vector_dim, 3))
        graph_seed = scatter_mean(nodes, batch, lattices.shape[0]) + self.time_input(time.unsqueeze(-1))
        graph_condition, edge_response, edge_auxiliary, alignment = self.response(
            piezo_irreps, condition_present, graph_seed, directions, edge_graph,
            frac_coords, lattices, batch, type_state,
        )
        nodes = nodes + graph_condition[batch]
        for layer in self.layers:
            nodes, vectors = layer(
                nodes, vectors, source, target, directions, edge_response,
                edge_auxiliary, graph_condition[batch],
            )
        graph_nodes = scatter_mean(nodes, batch, lattices.shape[0])
        cartesian_velocity = self.coord_out(vectors.transpose(-1, -2)).squeeze(-1)
        lattice_nodes = lattices[batch]
        fractional_velocity = torch.einsum("ni,nij->nj", cartesian_velocity, torch.linalg.inv(lattice_nodes))
        outputs = (self.type_out(nodes), fractional_velocity, self.lattice_out(graph_nodes), alignment)
        if not return_uncertainty:
            return outputs
        uncertainty = VelocityUncertainty(
            type_log_std=bounded_log_std(self.type_log_std_out(nodes)),
            coord_log_std=bounded_log_std(self.coord_log_std_out(nodes)),
            lattice_log_std=bounded_log_std(self.lattice_log_std_out(graph_nodes)),
        )
        return (*outputs, uncertainty)
