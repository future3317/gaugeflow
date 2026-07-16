"""Inference-consistent conditioning utilities."""

from __future__ import annotations

import torch
from e3nn import o3

from .tensor import (
    piezo_irrep_blocks,
    tensor_orbit_shape_magnitude,
)


def low_order_orbit_invariants(piezo_irreps: torch.Tensor) -> torch.Tensor:
    """Compact proper-SO(3) invariants with two parity-odd pseudoscalars."""
    first, second, third = piezo_irrep_blocks(piezo_irreps)
    first_gram = torch.einsum("bmi,bni->bmn", first, first)
    quadratic = torch.cat(
        (
            first_gram[:, 0, 0:1],
            first_gram[:, 1, 1:2],
            first_gram[:, 0, 1:2],
            second.square().sum(dim=(-1, -2), keepdim=False).unsqueeze(-1),
            third.square().sum(dim=(-1, -2), keepdim=False).unsqueeze(-1),
        ),
        dim=-1,
    )
    coupling = o3.FullTensorProduct("1x2o", "1x3o").to(device=piezo_irreps.device, dtype=piezo_irreps.dtype)
    axial = coupling(second.flatten(1), third.flatten(1))[:, :3]
    pseudoscalars = torch.einsum("bmi,bi->bm", first, axial)
    magnitude = piezo_irreps.square().sum(dim=-1, keepdim=True).sqrt()
    return torch.cat((quadratic, pseudoscalars, torch.log1p(magnitude)), dim=-1)


def normalized_low_order_orbit_invariants(piezo_irreps: torch.Tensor) -> torch.Tensor:
    """Scale-free orbit shape, log magnitude and an explicit physical-zero flag."""
    raw = low_order_orbit_invariants(piezo_irreps)
    decomposition = tensor_orbit_shape_magnitude(piezo_irreps)
    magnitude = torch.linalg.vector_norm(piezo_irreps, dim=-1, keepdim=True)
    safe = magnitude.clamp_min(torch.finfo(piezo_irreps.dtype).tiny)
    quadratic = raw[:, :5] / safe.square()
    pseudoscalar = raw[:, 5:7] / safe.pow(3)
    normalized = torch.cat(
        (
            quadratic,
            pseudoscalar,
            decomposition.log_magnitude.unsqueeze(-1),
            decomposition.physical_zero.to(piezo_irreps).unsqueeze(-1),
        ),
        dim=-1,
    )
    zero_shape = torch.cat((torch.zeros_like(normalized[:, :7]), normalized[:, 7:]), dim=-1)
    return torch.where(decomposition.physical_zero.unsqueeze(-1), zero_shape, normalized)
