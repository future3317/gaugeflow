"""Post-reconstruction symmetry and tensor-compatibility qualification.

The reachable-child router supplies a path prior.  It does not prove that a
numerically reconstructed or relaxed child realizes the declared subgroup:
zero amplitudes and accidental cancellations can restore symmetry, while a
noisy residual can lower it.  This module performs that independent terminal
check in the actual Cartesian frame of each generated structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
import torch

from .so3_quadrature import nested_hopf_so3_grid
from .space_group_router import orbit_compatibility_residual


@dataclass(frozen=True)
class DetectedPointGroup:
    space_group_number: int
    international_symbol: str
    cartesian_operations: torch.Tensor


@dataclass(frozen=True)
class TerminalSymmetryAudit:
    declared_space_group: int
    raw_space_group: int
    relaxed_space_group: int | None
    declared_operation_count: int
    raw_operation_count: int
    relaxed_operation_count: int | None
    declared_to_raw_relation: str
    raw_to_relaxed_relation: str | None
    raw_compatibility_residual: float
    relaxed_compatibility_residual: float | None
    compatibility_retained_after_relaxation: bool | None


def _unique_operations(operations: torch.Tensor, *, tolerance: float) -> torch.Tensor:
    if operations.ndim != 3 or operations.shape[-2:] != (3, 3):
        raise ValueError("Cartesian operations must have shape [operations,3,3]")
    if not bool(torch.isfinite(operations).all()):
        raise ValueError("Cartesian operations contain non-finite values")
    unique: list[torch.Tensor] = []
    for operation in operations:
        if not any(torch.allclose(operation, seen, atol=tolerance, rtol=0.0) for seen in unique):
            unique.append(operation)
    if not unique:
        raise ValueError("a point group must contain at least the identity")
    return torch.stack(unique)


def _operation_set_is_subset(
    subset: torch.Tensor,
    superset: torch.Tensor,
    *,
    tolerance: float,
) -> bool:
    return all(
        any(torch.allclose(operation, candidate, atol=tolerance, rtol=0.0) for candidate in superset)
        for operation in subset
    )


def classify_group_relation(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    tolerance: float = 2e-5,
) -> str:
    """Classify two operation sets expressed in one Cartesian frame."""
    left = _unique_operations(before, tolerance=tolerance)
    right = _unique_operations(after, tolerance=tolerance)
    left_in_right = _operation_set_is_subset(left, right, tolerance=tolerance)
    right_in_left = _operation_set_is_subset(right, left, tolerance=tolerance)
    if left_in_right and right_in_left:
        return "equal"
    if left_in_right:
        return "symmetry_restoration"
    if right_in_left:
        return "symmetry_lowering"
    return "changed_non_nested"


def detect_cartesian_point_group(
    species: torch.Tensor,
    fractional_coordinates: torch.Tensor,
    lattice: torch.Tensor,
    *,
    symprec: float = 1e-3,
    angle_tolerance: float = -1.0,
    operation_tolerance: float = 2e-5,
) -> DetectedPointGroup:
    """Detect the realized full-O(3) point group with spglib.

    GaugeFlow stores row lattice vectors and row fractional coordinates.  A
    spglib fractional column operation ``f' = R f + t`` becomes the Cartesian
    column operation ``Q = L^T R L^{-T}``.
    """
    if species.ndim != 1 or species.dtype != torch.long:
        raise ValueError("species must be an int64 vector")
    if fractional_coordinates.shape != (species.numel(), 3):
        raise ValueError("fractional coordinates must have shape [nodes,3]")
    if lattice.shape != (3, 3) or not bool(torch.isfinite(lattice).all()):
        raise ValueError("lattice must be a finite 3x3 row-basis matrix")
    if abs(float(torch.linalg.det(lattice))) <= 1e-10:
        raise ValueError("lattice is singular")
    cell = (
        lattice.detach().cpu().to(torch.float64).numpy(),
        fractional_coordinates.detach().cpu().to(torch.float64).remainder(1.0).numpy(),
        species.detach().cpu().numpy(),
    )
    dataset = spglib.get_symmetry_dataset(
        cell,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    symmetry = spglib.get_symmetry(
        cell,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if dataset is None or symmetry is None:
        raise ValueError("spglib could not identify the generated child symmetry")
    fractional_rotations = torch.from_numpy(np.asarray(symmetry["rotations"])).to(torch.float64)
    row_lattice = lattice.detach().cpu().to(torch.float64)
    cartesian = (
        row_lattice.transpose(0, 1).unsqueeze(0)
        @ fractional_rotations
        @ torch.linalg.inv(row_lattice).transpose(0, 1).unsqueeze(0)
    )
    cartesian = _unique_operations(cartesian, tolerance=operation_tolerance)
    orthogonality = cartesian.transpose(-1, -2) @ cartesian
    identity = torch.eye(3, dtype=cartesian.dtype).expand_as(orthogonality)
    if not torch.allclose(orthogonality, identity, atol=5e-5, rtol=5e-5):
        raise ValueError("detected symmetry operations are not Cartesian O(3) matrices")
    return DetectedPointGroup(
        space_group_number=int(dataset.number),
        international_symbol=str(dataset.international),
        cartesian_operations=cartesian,
    )


def audit_terminal_symmetry(
    *,
    piezo_irreps: torch.Tensor,
    declared_space_group: int,
    declared_cartesian_operations: torch.Tensor,
    raw_species: torch.Tensor,
    raw_fractional_coordinates: torch.Tensor,
    raw_lattice: torch.Tensor,
    relaxed_species: torch.Tensor | None = None,
    relaxed_fractional_coordinates: torch.Tensor | None = None,
    relaxed_lattice: torch.Tensor | None = None,
    symprec: float = 1e-3,
    compatibility_tolerance: float = 1e-3,
    rotation_count: int = 240,
) -> TerminalSymmetryAudit:
    """Audit declared, reconstructed, and optionally relaxed terminal groups."""
    if piezo_irreps.shape != (18,):
        raise ValueError("one terminal audit requires one 18-dimensional piezo condition")
    relaxed_values = (
        relaxed_species,
        relaxed_fractional_coordinates,
        relaxed_lattice,
    )
    if any(value is None for value in relaxed_values) and not all(
        value is None for value in relaxed_values
    ):
        raise ValueError("relaxed species, coordinates, and lattice must be supplied together")
    raw = detect_cartesian_point_group(
        raw_species,
        raw_fractional_coordinates,
        raw_lattice,
        symprec=symprec,
    )
    declared = _unique_operations(declared_cartesian_operations.to(torch.float64), tolerance=2e-5)
    rotations = nested_hopf_so3_grid(rotation_count, dtype=torch.float64)
    condition = piezo_irreps.to(torch.float64).unsqueeze(0)
    raw_residual = float(
        orbit_compatibility_residual(condition, raw.cartesian_operations, rotations).item()
    )

    relaxed: DetectedPointGroup | None = None
    relaxed_residual: float | None = None
    relaxed_relation: str | None = None
    retained: bool | None = None
    if all(value is not None for value in relaxed_values):
        assert relaxed_species is not None
        assert relaxed_fractional_coordinates is not None
        assert relaxed_lattice is not None
        relaxed = detect_cartesian_point_group(
            relaxed_species,
            relaxed_fractional_coordinates,
            relaxed_lattice,
            symprec=symprec,
        )
        relaxed_residual = float(
            orbit_compatibility_residual(
                condition,
                relaxed.cartesian_operations,
                rotations,
            ).item()
        )
        relaxed_relation = classify_group_relation(
            raw.cartesian_operations,
            relaxed.cartesian_operations,
        )
        retained = relaxed_residual <= compatibility_tolerance

    return TerminalSymmetryAudit(
        declared_space_group=declared_space_group,
        raw_space_group=raw.space_group_number,
        relaxed_space_group=None if relaxed is None else relaxed.space_group_number,
        declared_operation_count=declared.shape[0],
        raw_operation_count=raw.cartesian_operations.shape[0],
        relaxed_operation_count=None if relaxed is None else relaxed.cartesian_operations.shape[0],
        declared_to_raw_relation=classify_group_relation(
            declared,
            raw.cartesian_operations,
        ),
        raw_to_relaxed_relation=relaxed_relation,
        raw_compatibility_residual=raw_residual,
        relaxed_compatibility_residual=relaxed_residual,
        compatibility_retained_after_relaxation=retained,
    )
