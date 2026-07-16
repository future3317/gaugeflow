"""Explicit tensor-source conversion and projection invariants.

The cache consumer alone cannot establish a JARVIS tensor's source convention.
These helpers are used by the v2 raw-data builder and deliberately require the
upstream Voigt order and engineering-shear declaration instead of guessing.
"""

from __future__ import annotations

import torch

from .tensor import piezo_voigt_to_cartesian, rotate_rank3

CANONICAL_ENGINEERING_VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")


def canonicalize_engineering_piezo_voigt(
    value: torch.Tensor,
    source_order: tuple[str, ...] | list[str],
    *,
    engineering_shear: bool,
) -> torch.Tensor:
    """Convert an explicit 3x6 engineering-Voigt source order to canonical order."""
    source = tuple(str(entry) for entry in source_order)
    if value.shape != (3, 6) or not value.dtype.is_floating_point or not torch.isfinite(value).all():
        raise ValueError("piezo Voigt value must be a finite floating [3, 6] tensor")
    if len(source) != 6 or set(source) != set(CANONICAL_ENGINEERING_VOIGT_ORDER):
        raise ValueError("source Voigt order must be a permutation of xx, yy, zz, yz, xz, xy")
    if not engineering_shear:
        raise ValueError("tensor source must explicitly use engineering shear; tensor-shear conversion is not inferred")
    column = [source.index(name) for name in CANONICAL_ENGINEERING_VOIGT_ORDER]
    return value[:, column]


def _validate_finite_orthogonal_group(rotations: torch.Tensor) -> None:
    """Check that a finite numerical O(3) operation list is a group.

    Reynolds averaging is invariant only for a group.  Checking closure here
    prevents a source builder from averaging an arbitrary subset of crystal
    operations and then mislabelling the result as a point-group projection.
    """
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3) or rotations.shape[0] < 1:
        raise ValueError("rotations must have shape [count, 3, 3]")
    if not torch.isfinite(rotations).all():
        raise ValueError("rotations must be finite")
    identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device).expand_as(rotations)
    if not torch.allclose(rotations @ rotations.transpose(-1, -2), identity, atol=1e-4, rtol=1e-4):
        raise ValueError("Reynolds group contains a non-orthogonal operation")
    determinant = torch.linalg.det(rotations)
    if not torch.allclose(determinant.abs(), torch.ones_like(determinant), atol=1e-4, rtol=1e-4):
        raise ValueError("Reynolds group contains an operation outside O(3)")
    identity_error = (rotations - identity).abs().amax(dim=(-1, -2))
    if not bool((identity_error <= 1e-4).any()):
        raise ValueError("Reynolds group does not contain the identity")
    products = torch.einsum("aij,bjk->abik", rotations, rotations)
    closure_error = (products[:, :, None] - rotations[None, None]).abs().amax(dim=(-1, -2)).amin(dim=-1)
    if float(closure_error.max()) > 5e-4:
        raise ValueError("Reynolds operations are not closed under composition")


def reynolds_project_crystal_rank3(
    source_voigt: torch.Tensor,
    point_group_operations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enforce O(3) crystal point-group compatibility for a polar rank-three tensor.

    ``point_group_operations`` must be the full crystallographic point group,
    including determinant-negative operations when present.  This is *not* an
    SO(3) orbit quotient: a mirror remains physically observable for a polar
    rank-three condition, while an inversion correctly projects every such
    tensor to zero.
    """
    _validate_finite_orthogonal_group(point_group_operations)
    source = piezo_voigt_to_cartesian(source_voigt)
    operations = point_group_operations.to(source)
    projected = rotate_rank3(source.unsqueeze(0), operations).mean(dim=0)
    residual = (rotate_rank3(projected.unsqueeze(0), operations) - projected).abs().max()
    return projected, residual


def reynolds_project_proper_rank3(
    source_voigt: torch.Tensor,
    rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy SO(3)-only Reynolds helper, retained for frozen v1 artifacts.

    New raw tensor builds must use :func:`reynolds_project_crystal_rank3` with
    the full O(3) crystal point group.  This compatibility wrapper explicitly
    rejects improper operations rather than silently dropping them.
    """
    determinant = torch.linalg.det(rotations)
    if not torch.allclose(
        determinant, torch.ones_like(determinant), atol=1e-4, rtol=1e-4
    ):
        raise ValueError("Legacy proper Reynolds projection accepts SO(3) operations only")
    return reynolds_project_crystal_rank3(source_voigt, rotations)
