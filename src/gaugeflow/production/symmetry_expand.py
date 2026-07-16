"""Exact space-group expansion of an asymmetric unit."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.geometry import closest_image_displacement
from gaugeflow.manifold import wrap01


@dataclass(frozen=True)
class ExpandedCrystal:
    frac_coords: torch.Tensor
    species: torch.Tensor
    asymmetric_site: torch.Tensor
    operation_index: torch.Tensor


def expand_asymmetric_unit(
    frac_coords: torch.Tensor,
    species: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    lattice: torch.Tensor,
    *,
    merge_tolerance_angstrom: float = 1e-5,
) -> ExpandedCrystal:
    """Expand fractional row coordinates and merge exact Wyckoff duplicates.

    Rotations and translations act as ``f -> f @ R.T + t``. Duplicate tests
    use the exact triclinic closest-vector solver in Cartesian distance.
    Different species at the same periodic position are rejected rather than
    silently merged into an invalid ordered crystal.
    """
    if frac_coords.ndim != 2 or frac_coords.shape[-1] != 3:
        raise ValueError("asymmetric coordinates must have shape [sites,3]")
    if species.shape != frac_coords.shape[:1] or species.dtype != torch.long:
        raise ValueError("species must provide one int64 token per asymmetric site")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("fractional rotations must have shape [operations,3,3]")
    if translations.shape != (rotations.shape[0], 3):
        raise ValueError("translations must have shape [operations,3]")
    if lattice.shape != (3, 3):
        raise ValueError("lattice must have shape [3,3]")
    if merge_tolerance_angstrom <= 0:
        raise ValueError("merge tolerance must be positive")
    transformed = wrap01(
        torch.einsum("si,oij->osj", frac_coords, rotations.to(frac_coords).transpose(-1, -2))
        + translations.to(frac_coords).unsqueeze(1)
    )
    kept_coords: list[torch.Tensor] = []
    kept_species: list[torch.Tensor] = []
    kept_sites: list[int] = []
    kept_operations: list[int] = []
    for operation in range(rotations.shape[0]):
        for site in range(frac_coords.shape[0]):
            coordinate = transformed[operation, site]
            duplicate = None
            for index, seen in enumerate(kept_coords):
                displacement, _ = closest_image_displacement(coordinate - seen, lattice)
                if float(torch.linalg.vector_norm(displacement)) <= merge_tolerance_angstrom:
                    duplicate = index
                    break
            if duplicate is not None:
                if int(kept_species[duplicate]) != int(species[site]):
                    raise ValueError("space-group expansion placed different species on one site")
                continue
            kept_coords.append(coordinate)
            kept_species.append(species[site])
            kept_sites.append(site)
            kept_operations.append(operation)
    return ExpandedCrystal(
        frac_coords=torch.stack(kept_coords),
        species=torch.stack(kept_species),
        asymmetric_site=torch.tensor(kept_sites, dtype=torch.long, device=frac_coords.device),
        operation_index=torch.tensor(kept_operations, dtype=torch.long, device=frac_coords.device),
    )

