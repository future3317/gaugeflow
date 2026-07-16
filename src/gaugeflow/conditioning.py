"""Inference-consistent conditioning utilities."""

from __future__ import annotations

import torch
from e3nn import o3
from torch import nn

from .tensor import (
    piezo_from_irreps,
    piezo_irrep_blocks,
    piezo_to_irreps,
    rotate_rank3,
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
    coupling = o3.FullTensorProduct("1x2o", "1x3o").to(
        device=piezo_irreps.device, dtype=piezo_irreps.dtype
    )
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


class OrbitInvariantConditionEncoder(nn.Module):
    """Low-order proper-SO(3)-invariant early condition channel."""

    feature_dim = 9

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, piezo_irreps: torch.Tensor) -> torch.Tensor:
        return self.network(normalized_low_order_orbit_invariants(piezo_irreps))


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
