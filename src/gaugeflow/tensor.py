"""Tensor-orbit utilities for GaugeFlow."""

from __future__ import annotations

import torch
from e3nn.io import CartesianTensor


PIEZO_IRREPS = CartesianTensor("ijk=ikj")
VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")


def piezo_voigt_to_cartesian(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] != (3, 6):
        raise ValueError(f"Expected [..., 3, 6] Voigt tensor, got {tuple(value.shape)}")
    out = value.new_zeros(*value.shape[:-2], 3, 3, 3)
    out[..., :, 0, 0] = value[..., :, 0]
    out[..., :, 1, 1] = value[..., :, 1]
    out[..., :, 2, 2] = value[..., :, 2]
    out[..., :, 1, 2] = out[..., :, 2, 1] = value[..., :, 3]
    out[..., :, 0, 2] = out[..., :, 2, 0] = value[..., :, 4]
    out[..., :, 0, 1] = out[..., :, 1, 0] = value[..., :, 5]
    return out


def piezo_cartesian_to_voigt(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-3:] != (3, 3, 3):
        raise ValueError(f"Expected [..., 3, 3, 3] tensor, got {tuple(value.shape)}")
    if not torch.allclose(value, value.transpose(-1, -2), atol=1e-6, rtol=1e-6):
        raise ValueError("Piezo tensor must be symmetric in its final two indices")
    return torch.stack(
        (value[..., :, 0, 0], value[..., :, 1, 1], value[..., :, 2, 2],
         value[..., :, 1, 2], value[..., :, 0, 2], value[..., :, 0, 1]),
        dim=-1,
    )


def piezo_to_irreps(value: torch.Tensor) -> torch.Tensor:
    return PIEZO_IRREPS.from_cartesian(value)


def piezo_from_irreps(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != PIEZO_IRREPS.dim:
        raise ValueError(f"Expected {PIEZO_IRREPS.dim} irreps coordinates, got {value.shape[-1]}")
    return PIEZO_IRREPS.to_cartesian(value)


def rotate_rank3(value: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ia,...jb,...kc,...abc->...ijk", rotation, rotation, rotation, value)


def fixed_so3_frames(count: int, *, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministic SO(3) quadrature candidates, including identity."""
    if count < 1:
        raise ValueError("count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrices = torch.randn(count, 3, 3, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrices)
    signs = torch.where(torch.diagonal(r, dim1=-2, dim2=-1) < 0, -1.0, 1.0).to(dtype)
    q = q * signs.unsqueeze(-2)
    negative = torch.linalg.det(q) < 0
    q[negative, :, -1] *= -1
    q[0] = torch.eye(3, dtype=dtype)
    return q


def orbit_irreps(value: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Return a finite SO(3) orbit set with shape [batch, frames, 18]."""
    tensor = piezo_from_irreps(value).unsqueeze(1)
    rotated = rotate_rank3(tensor, rotations.to(value).unsqueeze(0))
    return piezo_to_irreps(rotated)


def response_field(tensor: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Evaluate F_e(n)=e:(n outer n), retaining all tensor degrees of freedom."""
    if tensor.shape[-3:] != (3, 3, 3) or directions.shape[-1] != 3:
        raise ValueError("Expected rank-three tensors and 3D directions")
    return torch.einsum("...ijk,...j,...k->...i", tensor, directions, directions)


def isotypic_slices() -> tuple[slice, slice, slice]:
    """The two l=1 copies are a single isotypic component for scaling."""
    return slice(0, 6), slice(6, 11), slice(11, 18)


def normalize_isotypic(irreps: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Scale 2x1o, 1x2o, and 1x3o components without per-coordinate z-scoring."""
    if scales.shape[-1] != 3:
        raise ValueError("Expected three isotypic scales")
    expanded = torch.cat(
        [scales[..., 0:1].expand(*scales.shape[:-1], 6),
         scales[..., 1:2].expand(*scales.shape[:-1], 5),
         scales[..., 2:3].expand(*scales.shape[:-1], 7)],
        dim=-1,
    )
    return irreps / expanded.to(irreps).clamp_min(torch.finfo(irreps.dtype).eps)


def response_field_error(prediction: torch.Tensor, target: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Mean squared complete vector-field error over directions."""
    delta = prediction - target
    field = torch.einsum("...ijk,mj,mk->...mi", delta, directions, directions)
    return field.square().sum(dim=-1).mean(dim=-1)
