"""Inference-consistent conditioning utilities."""

from __future__ import annotations

import torch

from .tensor import piezo_from_irreps, piezo_to_irreps, rotate_rank3


def apply_condition_dropout(
    condition_present: torch.Tensor,
    probability: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop complete conditions for classifier-free training.

    A physical zero tensor remains a present condition.  Missingness is carried
    solely by the boolean mask, so CFG never conflates ``e == 0`` with the
    learned null condition.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("condition dropout probability must lie in [0, 1]")
    present = condition_present.to(dtype=torch.bool)
    if probability == 0.0:
        return present
    random = torch.rand(
        present.shape,
        dtype=torch.float32,
        device=present.device,
        generator=generator,
    )
    return present & (random >= probability)


def randomize_tensor_orbit_representative(
    piezo_irreps: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one proper SO(3) representative for every tensor orbit.

    This is used only by the direct-interaction baseline.  It prevents that
    control from exploiting a fixed laboratory representative while preserving
    the complete physical tensor orbit.  The operation is Cartesian and does
    not evaluate spherical harmonics.
    """
    if piezo_irreps.ndim != 2:
        raise ValueError("Expected [graphs, irreps] tensor conditions")
    matrices = torch.randn(
        (piezo_irreps.shape[0], 3, 3),
        dtype=piezo_irreps.dtype,
        device=piezo_irreps.device,
        generator=generator,
    )
    rotations, upper = torch.linalg.qr(matrices)
    signs = torch.where(
        torch.diagonal(upper, dim1=-2, dim2=-1) < 0,
        -torch.ones((), dtype=piezo_irreps.dtype, device=piezo_irreps.device),
        torch.ones((), dtype=piezo_irreps.dtype, device=piezo_irreps.device),
    )
    rotations = rotations * signs.unsqueeze(-2)
    improper = torch.linalg.det(rotations) < 0
    rotations[improper, :, -1] *= -1
    return piezo_to_irreps(rotate_rank3(piezo_from_irreps(piezo_irreps), rotations))
