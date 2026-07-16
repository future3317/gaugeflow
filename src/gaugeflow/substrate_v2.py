"""Geometry-aware categorical site scorer for the post-A11 substrate protocol.

This module is intentionally isolated from ``GaugeFlowVectorField`` so it
cannot mutate or reinterpret frozen A5--A11 results.  It is a fixed-geometry,
discrete-decoration scorer: it receives masked site tokens, physical periodic
geometry and (for the legacy two-endpoint control only) an optional graph-level
endpoint ID, then emits dense chemical-token scores.
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
    """Permutation-stable tiny-graph mean aggregation.

    Assignment likelihood can become extremely sharp.  With a float32
    scatter reduction, merely relabeling nodes changes the order of the three
    incoming messages and the resulting round-off is amplified into a large
    categorical log-probability difference.  Accumulating this small physical
    neighbourhood in float64, then returning to the model dtype, prevents an
    arbitrary input row order from becoming a chemical signal.  Gradients
    remain connected to the original parameter dtype.
    """
    if not value.dtype.is_floating_point:
        raise ValueError("message values must be floating point")
    accumulated = scatter(value.to(torch.float64), index, dim=0, dim_size=dim_size, reduce="sum")
    counts = torch.bincount(index, minlength=dim_size).to(device=value.device, dtype=torch.float64)
    shape = (dim_size,) + (1,) * (value.ndim - 1)
    return (accumulated / counts.reshape(shape)).to(value.dtype)


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
        scalar_update_input = 2 * hidden_dim + (vector_channels if use_vector_invariants else 0)
        self.scalar_update = nn.Sequential(
            nn.Linear(scalar_update_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
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
        scalar_parts = [scalar, aggregated_scalar]
        if self.use_vector_invariants:
            scalar_parts.append(next_invariant)
        update = self.scalar_update(torch.cat(scalar_parts, dim=-1))
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
        endpoint_classes: int | None = 2,
        score_bound: float = 20.0,
        use_rbf: bool = True,
        use_vector_invariants: bool = True,
    ):
        super().__init__()
        if (
            hidden_dim < 1 or layers < 1 or vector_channels < 1 or score_bound <= 0
            or (endpoint_classes is not None and endpoint_classes < 1)
        ):
            raise ValueError("hidden_dim, layers, vector_channels, optional endpoint_classes and score_bound must be positive")
        self.endpoint_classes = endpoint_classes
        self.score_bound = float(score_bound)
        self.use_vector_invariants = use_vector_invariants
        self.type_embedding = nn.Embedding(TYPE_STATE_DIM, hidden_dim)
        self.endpoint_embedding = (
            nn.Embedding(endpoint_classes, hidden_dim) if endpoint_classes is not None else None
        )
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
        score_input = hidden_dim + (vector_channels if use_vector_invariants else 0)
        self.score_head = nn.Sequential(nn.Linear(score_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, MASK_TOKEN))

    def node_features(
        self,
        type_tokens: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
        endpoint_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return permutation-equivariant site features before chemical scores."""
        tokens = validate_type_tokens(type_tokens, allow_mask=True).to(device=frac_coords.device)
        if tokens.shape != (frac_coords.shape[0],):
            raise ValueError("type_tokens must contain exactly one token per node")
        if self.endpoint_embedding is None:
            if endpoint_id is not None:
                raise ValueError("this geometry-only scorer was constructed without endpoint IDs")
        else:
            if endpoint_id is None or endpoint_id.ndim != 1 or endpoint_id.shape != (lattice.shape[0],):
                raise ValueError("endpoint_id must contain one endpoint class per graph")
            endpoint_id = endpoint_id.to(device=frac_coords.device, dtype=torch.long)
            if endpoint_id.numel() and ((endpoint_id < 0) | (endpoint_id >= self.endpoint_classes)).any():
                raise ValueError("endpoint_id lies outside endpoint_classes")
        edges = periodic_closest_image_edges(frac_coords, lattice, batch)
        if edges.source.numel() == 0:
            raise ValueError("the site scorer requires at least two sites per graph")
        scalar = self.type_embedding(tokens)
        if self.endpoint_embedding is not None:
            scalar = scalar + self.endpoint_embedding(endpoint_id[batch])
        vector = scalar.new_zeros((scalar.shape[0], self.blocks[0].vector_channels, 3))
        rbf = self.rbf(edges.distance)
        for block in self.blocks:
            scalar, vector = block(scalar, vector, edges.source, edges.target, edges.direction, rbf)
        if self.use_vector_invariants:
            invariants = vector.square().sum(dim=-1)
            scalar = torch.cat((scalar, invariants), dim=-1)
        return scalar

    def scores_from_node_features(self, node_features: torch.Tensor) -> torch.Tensor:
        """Map site features to bounded dense chemical assignment scores."""
        if node_features.ndim != 2 or node_features.shape[-1] != self.score_head[0].in_features:
            raise ValueError("node_features have an incompatible shape")
        # The exact finite assignment law is insensitive to score offsets but
        # becomes numerically ill-conditioned when unconstrained logits grow
        # without bound after NLL saturation.  A fixed, declared tanh bound
        # retains a differentiable categorical energy while preventing an
        # arbitrary FP32 reduction order from turning into a chemical label.
        raw_scores = self.score_head(node_features)
        return self.score_bound * torch.tanh(raw_scores / self.score_bound)

    def forward(
        self,
        type_tokens: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
        endpoint_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return bounded score matrix ``[nodes, 118]`` for dense elements."""
        return self.scores_from_node_features(
            self.node_features(type_tokens, frac_coords, lattice, batch, endpoint_id)
        )
