"""Explicit O(3)-typed message primitives for parity-sensitive GaugeFlow work.

The legacy scalar/polar-vector layer is SO(3)-equivariant, but it cannot retain
the distinction between a pseudoscalar (0o) and an ordinary scalar (0e), or
between axial (1e) and polar (1o) vectors.  This module keeps those four types
separate.  It is a small Cartesian implementation of the relevant O(3) tensor
products, intended for the next versioned backbone rather than historical
checkpoints.
"""

from __future__ import annotations

import torch
from torch import nn


def transform_axial(vectors: torch.Tensor, orthogonal: torch.Tensor) -> torch.Tensor:
    """Transform axial vectors: ``a -> det(P) P a`` for an O(3) matrix ``P``."""
    if orthogonal.shape != (3, 3):
        raise ValueError("orthogonal transform must have shape [3, 3]")
    determinant = torch.linalg.det(orthogonal)
    if not torch.allclose(determinant.abs(), determinant.new_tensor(1.0), atol=2e-5, rtol=2e-5):
        raise ValueError("axial-vector transform requires an orthogonal matrix")
    return determinant * (vectors @ orthogonal.T)


def parity_edge_features(polar_fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 0e, 0o, and 1e features from polar edge vectors.

    ``polar_fields`` has shape ``[edges, fields, 3]`` and normally includes a
    bond direction plus complete-CG response vectors.  Pairwise dot products
    are 0e, pairwise crosses are axial 1e, and scalar triple products are 0o.
    All features are retained separately so a later neural layer cannot
    accidentally mix reflection-even and reflection-odd quantities.
    """
    if polar_fields.ndim != 3 or polar_fields.shape[-1] != 3:
        raise ValueError("polar_fields must have shape [edges, fields, 3]")
    fields = polar_fields.shape[1]
    if fields < 2:
        raise ValueError("at least two polar fields are needed for parity features")
    pair = torch.triu_indices(fields, fields, offset=1, device=polar_fields.device)
    even = (polar_fields[:, pair[0]] * polar_fields[:, pair[1]]).sum(dim=-1)
    axial = torch.cross(polar_fields[:, pair[0]], polar_fields[:, pair[1]], dim=-1)
    if fields < 3:
        odd = polar_fields.new_empty((polar_fields.shape[0], 0))
    else:
        triples = torch.combinations(torch.arange(fields, device=polar_fields.device), r=3)
        odd = (
            torch.cross(polar_fields[:, triples[:, 0]], polar_fields[:, triples[:, 1]], dim=-1)
            * polar_fields[:, triples[:, 2]]
        ).sum(dim=-1)
    return even, odd, axial


class ParityAwareResponseBlock(nn.Module):
    """One O(3)-equivariant message update with 0e/0o/1o/1e state channels.

    The block uses only parity-valid Cartesian products:

    * 0e x 1o -> 1o and 0e x 1e -> 1e;
    * 0o x 1e -> 1o and 0o x 1o -> 1e;
    * 0o x 0o -> 0e.

    It intentionally does not collapse the 0o or 1e state into ordinary
    scalars/vectors.  This provides the missing expressive ingredient for a
    later mirrored-representative/chiral gate.
    """

    def __init__(self, scalar_dim: int, vector_dim: int, polar_edge_fields: int) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1 or vector_dim > scalar_dim or polar_edge_fields < 2:
            raise ValueError("invalid parity-aware block dimensions")
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.polar_edge_fields = polar_edge_fields
        pair_count = polar_edge_fields * (polar_edge_fields - 1) // 2
        triple_count = polar_edge_fields * (polar_edge_fields - 1) * (polar_edge_fields - 2) // 6
        self.even_message = nn.Sequential(
            nn.Linear(3 * scalar_dim + pair_count + triple_count, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, scalar_dim)
        )
        # A generic biased MLP on 0o inputs would violate reflection symmetry.
        # These gates depend only on 0e inputs and multiply explicitly 0o
        # bases, so the result remains exactly odd under a reflection.
        self.odd_gates = nn.Sequential(
            nn.Linear(2 * scalar_dim + pair_count + triple_count, scalar_dim), nn.SiLU(),
            nn.Linear(scalar_dim, 3 * scalar_dim),
        )
        self.odd_edge = nn.Linear(triple_count, scalar_dim, bias=False)
        self.odd_update_gate = nn.Sequential(
            nn.Linear(2 * scalar_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, scalar_dim), nn.Sigmoid()
        )
        self.polar_gates = nn.Sequential(
            nn.Linear(2 * scalar_dim + pair_count + triple_count, scalar_dim), nn.SiLU(),
            nn.Linear(scalar_dim, vector_dim * (polar_edge_fields + 1)),
        )
        self.axial_gates = nn.Sequential(
            nn.Linear(2 * scalar_dim + pair_count + triple_count, scalar_dim), nn.SiLU(),
            nn.Linear(scalar_dim, vector_dim * (pair_count + 1)),
        )
        self.scalar_update = nn.Sequential(nn.Linear(scalar_dim * 2, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, scalar_dim))

    def forward(
        self,
        scalar_even: torch.Tensor,
        scalar_odd: torch.Tensor,
        polar: torch.Tensor,
        axial: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_polar_fields: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_count = scalar_even.shape[0]
        if scalar_even.shape != scalar_odd.shape or scalar_even.shape[1] != self.scalar_dim:
            raise ValueError("even and odd scalar states must have matching [nodes, scalar_dim] shape")
        if polar.shape != axial.shape or polar.shape != (node_count, self.vector_dim, 3):
            raise ValueError("polar and axial states must have shape [nodes, vector_dim, 3]")
        if edge_polar_fields.shape != (source.numel(), self.polar_edge_fields, 3) or target.shape != source.shape:
            raise ValueError("edge fields and incidence tensors have incompatible shapes")
        if source.numel() == 0:
            return scalar_even, scalar_odd, polar, axial
        edge_even, edge_odd, edge_axial = parity_edge_features(edge_polar_fields)
        # 0o x 0o is even; multiplying source/target odd channels is therefore
        # legal input to the even scalar update.
        even_input = torch.cat(
            (scalar_even[source], scalar_even[target], scalar_odd[source] * scalar_odd[target], edge_even, edge_odd.square()),
            dim=-1,
        )
        even_messages = self.even_message(even_input)
        odd_gate_input = torch.cat((scalar_even[source], scalar_even[target], edge_even, edge_odd.square()), dim=-1)
        odd_gates = self.odd_gates(odd_gate_input).reshape(-1, 3, self.scalar_dim)
        odd_basis = torch.stack(
            (scalar_odd[source], scalar_odd[target], scalar_even[source] * scalar_odd[target] + self.odd_edge(edge_odd)),
            dim=1,
        )
        odd_messages = (odd_gates * odd_basis).sum(dim=1)
        even_aggregate = scalar_even.new_zeros(scalar_even.shape)
        odd_aggregate = scalar_odd.new_zeros(scalar_odd.shape)
        even_aggregate.index_add_(0, target, even_messages)
        odd_aggregate.index_add_(0, target, odd_messages)
        next_even = scalar_even + self.scalar_update(torch.cat((scalar_even, even_aggregate), dim=-1))
        next_odd = scalar_odd + self.odd_update_gate(torch.cat((scalar_even, even_aggregate), dim=-1)) * odd_aggregate

        gate_input = torch.cat((scalar_even[source], scalar_even[target], edge_even, edge_odd.square()), dim=-1)
        polar_gates = self.polar_gates(gate_input).reshape(-1, self.polar_edge_fields + 1, self.vector_dim, 1)
        axial_gates = self.axial_gates(gate_input).reshape(-1, edge_axial.shape[1] + 1, self.vector_dim, 1)
        polar_message = polar_gates[:, -1] * polar[source]
        for index in range(self.polar_edge_fields):
            polar_message = polar_message + polar_gates[:, index] * edge_polar_fields[:, index].unsqueeze(1)
        # 0o times a 1e feature is polar.  This is the parity-sensitive path
        # absent from the legacy scalar/polar implementation.
        polar_message = polar_message + scalar_odd[source, : self.vector_dim].unsqueeze(-1) * edge_axial[:, 0].unsqueeze(1)
        axial_message = axial_gates[:, -1] * axial[source]
        for index in range(edge_axial.shape[1]):
            axial_message = axial_message + axial_gates[:, index] * edge_axial[:, index].unsqueeze(1)
        # 0o times polar is axial.
        axial_message = axial_message + scalar_odd[source, : self.vector_dim].unsqueeze(-1) * edge_polar_fields[:, 0].unsqueeze(1)
        polar_aggregate = polar.new_zeros(polar.shape)
        axial_aggregate = axial.new_zeros(axial.shape)
        polar_aggregate.index_add_(0, target, polar_message)
        axial_aggregate.index_add_(0, target, axial_message)
        return next_even, next_odd, polar + polar_aggregate, axial + axial_aggregate
