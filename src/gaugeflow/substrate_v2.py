"""Geometry-aware categorical site scorer for the post-A11 substrate protocol.

This module is intentionally isolated from ``GaugeFlowVectorField`` so it
cannot mutate or reinterpret frozen A5--A11 results.  It is a fixed-geometry,
discrete-decoration scorer: it receives masked site tokens, physical periodic
geometry and a graph-level endpoint ID, then emits dense chemical-token scores.
It never accepts a target CIF row order, a species mapping, a target
composition, a stabilizer, or a tensor condition.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter

from .geometry import GaussianRadialBasis, periodic_closest_image_edges
from .vocabulary import MASK_TOKEN, TYPE_STATE_DIM, validate_type_tokens


def _scatter_mean(value: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    return scatter(value, index, dim=0, dim_size=dim_size, reduce="mean")


class _GeometryMessageBlock(nn.Module):
    """One scalar/vector message block with metric and vector invariants."""

    def __init__(
        self,
        hidden_dim: int,
        vector_channels: int,
        rbf_dim: int,
        *,
        use_rbf: bool,
        use_vector_invariants: bool,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vector_channels = vector_channels
        self.use_rbf = use_rbf
        self.use_vector_invariants = use_vector_invariants
        invariant_dim = 3 * vector_channels if use_vector_invariants else 0
        scalar_input = 2 * hidden_dim + (rbf_dim if use_rbf else 0) + invariant_dim
        self.edge_scalar = nn.Sequential(
            nn.Linear(scalar_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_vector = nn.Sequential(
            nn.Linear(scalar_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vector_channels)
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(2 * hidden_dim + vector_channels, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.vector_self_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vector_channels)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        direction: torch.Tensor,
        rbf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vector_norm2 = vector.square().sum(dim=-1)
        vector_dot = (vector[source] * vector[target]).sum(dim=-1)
        parts = [scalar[source], scalar[target]]
        if self.use_rbf:
            parts.append(rbf)
        if self.use_vector_invariants:
            parts.extend((vector_norm2[source], vector_norm2[target], vector_dot))
        edge_features = torch.cat(parts, dim=-1)
        scalar_message = self.edge_scalar(edge_features)
        vector_message = self.edge_vector(edge_features).unsqueeze(-1) * direction.unsqueeze(1)
        aggregated_scalar = _scatter_mean(scalar_message, target, dim_size=scalar.shape[0])
        aggregated_vector = _scatter_mean(vector_message, target, dim_size=scalar.shape[0])
        self_vector = vector * self.vector_self_gate(scalar).unsqueeze(-1)
        next_vector = vector + aggregated_vector + self_vector
        next_invariant = next_vector.square().sum(dim=-1)
        update = self.scalar_update(torch.cat((scalar, aggregated_scalar, next_invariant), dim=-1))
        return self.norm(scalar + update), next_vector


class GeometryAwareSiteScorer(nn.Module):
    """Permutation-equivariant dense 118-element scorer for fixed PBC geometry.

    The equivariant vector channels transform with the physical edge directions;
    every scalar update receives distance RBF features (when enabled) and
    scalar invariants of those vector channels.  Thus the chemical score head
    can depend on periodic metric geometry without using atom row indices.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        layers: int = 4,
        vector_channels: int = 16,
        rbf_dim: int = 16,
        cutoff: float = 8.0,
        endpoint_classes: int = 2,
        use_rbf: bool = True,
        use_vector_invariants: bool = True,
    ):
        super().__init__()
        if hidden_dim < 1 or layers < 1 or vector_channels < 1 or endpoint_classes < 1:
            raise ValueError("hidden_dim, layers, vector_channels and endpoint_classes must be positive")
        self.hidden_dim = hidden_dim
        self.endpoint_classes = endpoint_classes
        self.type_embedding = nn.Embedding(TYPE_STATE_DIM, hidden_dim)
        self.endpoint_embedding = nn.Embedding(endpoint_classes, hidden_dim)
        self.rbf = GaussianRadialBasis(rbf_dim, cutoff)
        self.blocks = nn.ModuleList(
            _GeometryMessageBlock(
                hidden_dim,
                vector_channels,
                rbf_dim,
                use_rbf=use_rbf,
                use_vector_invariants=use_vector_invariants,
            )
            for _ in range(layers)
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + vector_channels, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, MASK_TOKEN)
        )

    def forward(
        self,
        type_tokens: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
        endpoint_id: torch.Tensor,
    ) -> torch.Tensor:
        """Return score matrix ``[nodes, 118]`` for dense chemical tokens."""
        tokens = validate_type_tokens(type_tokens, allow_mask=True).to(device=frac_coords.device)
        if tokens.shape != (frac_coords.shape[0],):
            raise ValueError("type_tokens must contain exactly one token per node")
        if endpoint_id.ndim != 1 or endpoint_id.shape != (lattice.shape[0],):
            raise ValueError("endpoint_id must contain one endpoint class per graph")
        endpoint_id = endpoint_id.to(device=frac_coords.device, dtype=torch.long)
        if endpoint_id.numel() and ((endpoint_id < 0) | (endpoint_id >= self.endpoint_classes)).any():
            raise ValueError("endpoint_id lies outside endpoint_classes")
        edges = periodic_closest_image_edges(frac_coords, lattice, batch)
        if edges.source.numel() == 0:
            raise ValueError("the site scorer requires at least two sites per graph")
        scalar = self.type_embedding(tokens) + self.endpoint_embedding(endpoint_id[batch])
        vector = scalar.new_zeros((scalar.shape[0], self.blocks[0].vector_channels, 3))
        rbf = self.rbf(edges.distance)
        for block in self.blocks:
            scalar, vector = block(scalar, vector, edges.source, edges.target, edges.direction, rbf)
        invariants = vector.square().sum(dim=-1)
        return self.score_head(torch.cat((scalar, invariants), dim=-1))
