"""Deterministic SO(3) quadrature utilities retained for non-atlas audits."""

from __future__ import annotations

import math

import torch


def _radical_inverse(index: torch.Tensor, base: int) -> torch.Tensor:
    value = torch.zeros_like(index, dtype=torch.float64)
    factor = 1.0 / base
    remaining = index.clone().to(dtype=torch.long)
    while bool((remaining > 0).any()):
        value = value + factor * (remaining % base).to(value)
        remaining = torch.div(remaining, base, rounding_mode="floor")
        factor /= base
    return value


def nested_hopf_so3_grid(
    count: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Nested deterministic Haar-QMC rotations with identity as the first node."""
    if count < 1:
        raise ValueError("SO(3) grid count must be positive")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    index = torch.arange(1, count + 1, device=target_device)
    u = _radical_inverse(index, 2).to(device=target_device)
    v = _radical_inverse(index, 3).to(device=target_device)
    w = _radical_inverse(index, 5).to(device=target_device)
    quaternion = torch.stack(
        (
            (1.0 - u).sqrt() * torch.sin(2.0 * math.pi * v),
            (1.0 - u).sqrt() * torch.cos(2.0 * math.pi * v),
            u.sqrt() * torch.sin(2.0 * math.pi * w),
            u.sqrt() * torch.cos(2.0 * math.pi * w),
        ),
        dim=-1,
    )
    x, y, z, scalar = quaternion.unbind(dim=-1)
    rotation = torch.stack(
        (
            1 - 2 * (y.square() + z.square()), 2 * (x * y - z * scalar), 2 * (x * z + y * scalar),
            2 * (x * y + z * scalar), 1 - 2 * (x.square() + z.square()), 2 * (y * z - x * scalar),
            2 * (x * z - y * scalar), 2 * (y * z + x * scalar), 1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(count, 3, 3).to(dtype=dtype)
    rotation[0] = torch.eye(3, dtype=dtype, device=target_device)
    return rotation
