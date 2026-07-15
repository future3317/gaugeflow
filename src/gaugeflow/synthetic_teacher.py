"""Non-cancelling synthetic rank-three tensor teachers for exact control gates.

The synthetic control intentionally tests tensor-orbit plumbing without an
external tensor oracle.  A symmetric sum of ``rhat outer rhat outer rhat`` on
both directed versions of every bond is identically zero.  This module instead
uses an antisymmetric directed species weight, so reversing an edge flips both
the weight and the odd rank-three dyad, making paired contributions add.
"""

from __future__ import annotations

import torch

from .geometry import periodic_closest_image_edges


def directed_species_rank3_teacher(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    species_scalar: torch.Tensor,
    *,
    distance_decay: float = 1.0,
) -> torch.Tensor:
    """Construct a translation/PBC/SO(3)-equivariant rank-three condition.

    For each directed closest-image edge ``i -> j`` the contribution is

    ``(a_j - a_i) exp(-r_ij / decay) n_ij outer n_ij outer n_ij``.

    The tensor is symmetric in its final two indices (indeed all three) and is
    non-cancelling for heterogeneous species.  It is a synthetic control, not
    a claimed constitutive model of piezoelectricity.
    """
    if species_scalar.ndim != 1 or species_scalar.shape != batch.shape:
        raise ValueError("species_scalar must have one finite scalar per node")
    if not torch.isfinite(species_scalar).all() or distance_decay <= 0:
        raise ValueError("species_scalar must be finite and distance_decay positive")
    edges = periodic_closest_image_edges(frac_coords, lattice, batch)
    edge_weight = (species_scalar[edges.target] - species_scalar[edges.source]) * torch.exp(
        -edges.distance / distance_decay
    )
    outer = torch.einsum("ei,ej,ek->eijk", edges.direction, edges.direction, edges.direction)
    contributions = edge_weight[:, None, None, None] * outer
    result = contributions.new_zeros((lattice.shape[0], 3, 3, 3))
    result.index_add_(0, batch[edges.source], contributions)
    return result


def symmetric_weighted_directed_rank3_sum(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    species_scalar: torch.Tensor,
) -> torch.Tensor:
    """Diagnostic anti-example: symmetric edge weights cancel for paired edges."""
    edges = periodic_closest_image_edges(frac_coords, lattice, batch)
    weights = species_scalar[edges.target] + species_scalar[edges.source]
    outer = torch.einsum("ei,ej,ek->eijk", edges.direction, edges.direction, edges.direction)
    result = outer.new_zeros((lattice.shape[0], 3, 3, 3))
    result.index_add_(0, batch[edges.source], weights[:, None, None, None] * outer)
    return result
