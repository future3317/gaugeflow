"""Stabilizer-aware response-field vector field for standalone GaugeFlow."""

from __future__ import annotations

import math

import torch
from torch import nn

from .manifold import log_vector_to_lattice, torus_logmap
from .tensor import (
    fixed_so3_frames,
    isotypic_slices,
    orbit_irreps,
    piezo_from_irreps,
    piezo_to_irreps,
    rotate_rank3,
)
from torch_geometric.utils import scatter


def scatter_mean(value: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    return scatter(value, index, dim=0, dim_size=dim_size, reduce="mean")


def periodic_complete_edges(
    frac_coords: torch.Tensor, lattice: torch.Tensor, batch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense intra-crystal edges and normalized Cartesian directions."""
    source, target, directions = [], [], []
    for graph in range(lattice.shape[0]):
        nodes = torch.nonzero(batch == graph, as_tuple=False).flatten()
        if nodes.numel() < 2:
            continue
        ii, jj = torch.meshgrid(nodes, nodes, indexing="ij")
        keep = ii != jj
        src, dst = ii[keep], jj[keep]
        delta = torus_logmap(frac_coords[src], frac_coords[dst])
        cart = delta @ lattice[graph]
        source.append(src)
        target.append(dst)
        directions.append(cart / torch.linalg.vector_norm(cart, dim=-1, keepdim=True).clamp_min(1e-8))
    if not source:
        empty = batch.new_empty((0,))
        return empty, empty, frac_coords.new_empty((0, 3))
    return torch.cat(source), torch.cat(target), torch.cat(directions)


class OrbitResponseFieldEncoder(nn.Module):
    """Orbit-set encoder with one coherent graph-level latent alignment."""

    def __init__(self, hidden_dim: int, orbit_frames: int = 24):
        super().__init__()
        self.register_buffer("rotations", fixed_so3_frames(orbit_frames))
        # Three isotypic norms plus a response-field norm are SO(3) scalars.
        # Raw irreps are deliberately not fed to a scalar MLP here.
        self.candidate = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
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
        stabilizer_rotations: torch.Tensor | None = None,
        stabilizer_count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graphs = piezo_irreps.shape[0]
        candidates = orbit_irreps(piezo_irreps, self.rotations)
        tensors = piezo_from_irreps(candidates)
        frames = candidates.shape[1]
        edge_fields = edge_directions.new_zeros((edge_directions.shape[0], frames, 3))
        if stabilizer_rotations is None or stabilizer_count is None:
            stabilizer_rotations = torch.eye(3, dtype=piezo_irreps.dtype, device=piezo_irreps.device).unsqueeze(0).expand(graphs, -1, -1)
            stabilizer_count = torch.ones(graphs, dtype=torch.long, device=piezo_irreps.device)
        else:
            stabilizer_rotations = stabilizer_rotations.to(piezo_irreps)
            stabilizer_count = stabilizer_count.reshape(-1).to(device=piezo_irreps.device, dtype=torch.long)
        if stabilizer_count.shape[0] != graphs or int(stabilizer_count.sum()) != stabilizer_rotations.shape[0]:
            raise ValueError("Stabilizer rotations must be concatenated graph-by-graph")

        # Pool candidate encodings over the *proper* crystal stabilizer.  This
        # quotients residual high-symmetry alignments without treating a mirror
        # as an SO(3) gauge transformation.
        graph_values = []
        offset = 0
        for graph, count in enumerate(stabilizer_count.tolist()):
            rotations = stabilizer_rotations[offset:offset + count]
            offset += count
            transformed = rotate_rank3(
                tensors[graph].unsqueeze(1), rotations.unsqueeze(0)
            )  # [frames, stabilizer, 3, 3, 3]
            # Transform once in Cartesian form, then return to irreps; this
            # retains all 18 degrees of freedom before nonlinear pooling.
            transformed_irreps = piezo_to_irreps(transformed)
            edge_ids = torch.nonzero(edge_graph == graph, as_tuple=False).flatten()
            if edge_ids.numel():
                fields = torch.einsum(
                    "fsijk,ej,ek->efsi", transformed, edge_directions[edge_ids], edge_directions[edge_ids]
                )
                graph_field = fields.mean(dim=0)
                edge_fields[edge_ids] = fields.mean(dim=2)
            else:
                graph_field = transformed.new_zeros((frames, count, 3))
            isotypic_norms = torch.stack(
                [transformed_irreps[..., block].square().sum(dim=-1).sqrt() for block in isotypic_slices()],
                dim=-1,
            )
            features = torch.cat((isotypic_norms, graph_field.square().sum(dim=-1, keepdim=True).sqrt()), dim=-1)
            graph_values.append(self.candidate(features).mean(dim=1))
        values = torch.stack(graph_values)
        scores = (self.key(values) * self.query(graph_query).unsqueeze(1)).sum(dim=-1) / math.sqrt(values.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        aligned = (weights.unsqueeze(-1) * values).sum(dim=1) + self.present_bias
        mask = present.to(dtype=torch.bool)
        graph_condition = torch.where(mask, aligned, self.null_condition.unsqueeze(0).expand_as(aligned))
        if edge_fields.numel():
            selected_field = (weights[edge_graph].unsqueeze(-1) * edge_fields).sum(dim=1)
            selected_field = torch.where(mask[edge_graph], selected_field, torch.zeros_like(selected_field))
        else:
            selected_field = edge_directions.new_empty((0, 3))
        return graph_condition, selected_field, weights


class ResponseMessageLayer(nn.Module):
    """SO(3)-equivariant scalar/vector message update.

    Scalar networks see only norms and dot products.  Every Cartesian vector
    is formed as a scalar-weighted sum of an edge direction, a response vector,
    and an incoming vector feature, so it transforms covariantly under SO(3).
    """

    def __init__(self, hidden_dim: int, vector_dim: int):
        super().__init__()
        self.vector_dim = vector_dim
        scalar_features = hidden_dim * 3 + 3
        self.scalar_message = nn.Sequential(
            nn.Linear(scalar_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.vector_gates = nn.Sequential(
            nn.Linear(scalar_features, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vector_dim * 3)
        )
        self.vector_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vector_dim), nn.Sigmoid()
        )

    def forward(
        self, nodes: torch.Tensor, vectors: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
        directions: torch.Tensor, response: torch.Tensor, node_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.numel() == 0:
            return nodes, vectors
        invariants = torch.stack(
            (
                (directions * response).sum(dim=-1),
                response.square().sum(dim=-1).sqrt(),
                directions.square().sum(dim=-1).sqrt(),
            ),
            dim=-1,
        )
        edge_scalars = torch.cat((nodes[source], nodes[target], node_condition[source], invariants), dim=-1)
        messages = self.scalar_message(edge_scalars)
        aggregate = nodes.new_zeros(nodes.shape)
        aggregate.index_add_(0, target, messages)
        updated_nodes = nodes + self.update(torch.cat((nodes, aggregate), dim=-1))

        gates = self.vector_gates(edge_scalars).reshape(-1, 3, self.vector_dim, 1)
        vector_messages = (
            gates[:, 0] * directions.unsqueeze(1)
            + gates[:, 1] * response.unsqueeze(1)
            + gates[:, 2] * vectors[source]
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
    ):
        super().__init__()
        self.atom_types = atom_types
        self.type_input = nn.Linear(atom_types, hidden_dim)
        self.time_input = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.response = OrbitResponseFieldEncoder(hidden_dim, orbit_frames)
        self.vector_dim = vector_dim or max(16, hidden_dim // 8)
        self.layers = nn.ModuleList([ResponseMessageLayer(hidden_dim, self.vector_dim) for _ in range(layers)])
        self.type_out = nn.Linear(hidden_dim, atom_types)
        self.coord_out = nn.Linear(self.vector_dim, 1, bias=False)
        self.lattice_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 6))

    def forward(
        self,
        type_state: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice_log: torch.Tensor,
        batch: torch.Tensor,
        time: torch.Tensor,
        piezo_irreps: torch.Tensor,
        condition_present: torch.Tensor,
        stabilizer_rotations: torch.Tensor | None = None,
        stabilizer_count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lattices = log_vector_to_lattice(lattice_log)
        source, target, directions = periodic_complete_edges(frac_coords, lattices, batch)
        edge_graph = batch[source] if source.numel() else batch.new_empty((0,))
        nodes = self.type_input(type_state)
        vectors = nodes.new_zeros((nodes.shape[0], self.vector_dim, 3))
        graph_seed = scatter_mean(nodes, batch, lattices.shape[0]) + self.time_input(time.unsqueeze(-1))
        graph_condition, edge_response, alignment = self.response(
            piezo_irreps, condition_present, graph_seed, directions, edge_graph,
            stabilizer_rotations, stabilizer_count,
        )
        nodes = nodes + graph_condition[batch]
        for layer in self.layers:
            nodes, vectors = layer(nodes, vectors, source, target, directions, edge_response, graph_condition[batch])
        graph_nodes = scatter_mean(nodes, batch, lattices.shape[0])
        cartesian_velocity = self.coord_out(vectors.transpose(-1, -2)).squeeze(-1)
        lattice_nodes = lattices[batch]
        fractional_velocity = torch.einsum("ni,nij->nj", cartesian_velocity, torch.linalg.inv(lattice_nodes))
        return self.type_out(nodes), fractional_velocity, self.lattice_out(graph_nodes), alignment
