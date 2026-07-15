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


def reynolds_project_proper_rank3(
    source_voigt: torch.Tensor,
    rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average a piezo tensor over verified proper crystallographic rotations."""
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3) or rotations.shape[0] < 1:
        raise ValueError("rotations must have shape [count, 3, 3]")
    if not torch.isfinite(rotations).all():
        raise ValueError("rotations must be finite")
    identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device).expand_as(rotations)
    if not torch.allclose(rotations @ rotations.transpose(-1, -2), identity, atol=1e-4, rtol=1e-4):
        raise ValueError("Reynolds group contains a non-orthogonal operation")
    if not torch.allclose(torch.linalg.det(rotations), torch.ones(rotations.shape[0], device=rotations.device, dtype=rotations.dtype), atol=1e-4, rtol=1e-4):
        raise ValueError("Reynolds projection may use proper SO(3) rotations only")
    source = piezo_voigt_to_cartesian(source_voigt)
    projected = rotate_rank3(source.unsqueeze(0), rotations.to(source)).mean(dim=0)
    residual = (rotate_rank3(projected.unsqueeze(0), rotations.to(projected)) - projected).abs().max()
    return projected, residual
