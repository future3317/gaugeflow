"""Full-O(3) piezoelectric compatibility for the symmetry blueprint."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from pymatgen.symmetry.groups import PointGroup, SpaceGroup
from torch import nn

from gaugeflow.harmonic import normalized_low_order_orbit_invariants
from gaugeflow.stabilizer import orthogonal_polar_factor
from gaugeflow.tensor import piezo_from_irreps, piezo_to_irreps, rotate_rank3

from .harmonic_gaugeflow import nested_hopf_so3_grid


@dataclass(frozen=True)
class SpaceGroupCompatibilityRecord:
    number: int
    symbol: str
    point_group: str
    operations: torch.Tensor
    reynolds_irrep: torch.Tensor
    compatible_rank: int


def reynolds_project(tensor: torch.Tensor, operations: torch.Tensor) -> torch.Tensor:
    """Project a polar rank-three tensor with every full-O(3) operation."""
    if tensor.shape[-3:] != (3, 3, 3):
        raise ValueError("piezoelectric tensor must end in [3,3,3]")
    if operations.ndim != 3 or operations.shape[-2:] != (3, 3):
        raise ValueError("point-group operations must have shape [operations,3,3]")
    transformed = rotate_rank3(tensor.unsqueeze(-4), operations.to(tensor))
    return transformed.mean(dim=-4)


def reynolds_irrep_matrix(operations: torch.Tensor) -> torch.Tensor:
    """Return the 18-dimensional Reynolds operator in the e3nn piezo basis."""
    basis = torch.eye(18, dtype=operations.dtype, device=operations.device)
    tensors = piezo_from_irreps(basis)
    projected = reynolds_project(tensors, operations)
    # Rows are projected basis inputs, hence transpose for column-operator
    # diagnostics.  Applying row irreps remains ``x @ matrix.T``.
    return piezo_to_irreps(projected).transpose(0, 1).contiguous()


@lru_cache(maxsize=230)
def cartesian_point_group_operations(space_group_number: int) -> tuple[str, str, torch.Tensor]:
    """Build the conventional Cartesian O(3) point group for a space group."""
    if not 1 <= space_group_number <= 230:
        raise ValueError("space-group number must lie in 1..230")
    space_group = SpaceGroup.from_int_number(space_group_number)
    point_group = PointGroup(space_group.point_group)
    fractional = torch.from_numpy(
        np.stack([operation.rotation_matrix for operation in point_group.symmetry_ops])
    ).to(dtype=torch.float64)
    # Pymatgen exposes crystallographic integer matrices in a conventional
    # fractional basis.  They are not Cartesian-orthogonal for hexagonal and
    # trigonal settings.  Averaging R^T R over the finite group constructs an
    # invariant positive metric M.  With C^T C=M, Q=C R C^-1 is an exact
    # Cartesian O(3) representation and preserves group multiplication.
    invariant_metric = torch.einsum("oji,ojk->ik", fractional, fractional)
    cartesian_map = torch.linalg.cholesky(invariant_metric).transpose(-1, -2)
    inverse_map = torch.linalg.inv(cartesian_map)
    cartesian = cartesian_map.unsqueeze(0) @ fractional @ inverse_map.unsqueeze(0)
    operations = orthogonal_polar_factor(cartesian)
    determinant = torch.linalg.det(operations)
    if not torch.allclose(determinant.abs(), torch.ones_like(determinant), atol=1e-10, rtol=1e-10):
        raise RuntimeError("pymatgen point group did not produce O(3) operations")
    # Preserve improper operations: they are essential to physical
    # compatibility for an odd-rank polar tensor.
    unique: list[torch.Tensor] = []
    for operation in operations:
        if not any(torch.allclose(operation, seen, atol=1e-10, rtol=1e-10) for seen in unique):
            unique.append(operation)
    return space_group.symbol, point_group.symbol, torch.stack(unique)


@lru_cache(maxsize=230)
def compatibility_record(space_group_number: int) -> SpaceGroupCompatibilityRecord:
    symbol, point_group, operations = cartesian_point_group_operations(space_group_number)
    reynolds = reynolds_irrep_matrix(operations)
    singular_values = torch.linalg.svdvals(reynolds)
    rank = int((singular_values > 1e-8).sum())
    return SpaceGroupCompatibilityRecord(
        number=space_group_number,
        symbol=symbol,
        point_group=point_group,
        operations=operations,
        reynolds_irrep=reynolds,
        compatible_rank=rank,
    )


def orbit_compatibility_residual(
    piezo_irreps: torch.Tensor,
    operations: torch.Tensor,
    rotations: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Finite-rule approximation to ``min_R ||rho(R)e-Pi rho(R)e||/||e||``."""
    if piezo_irreps.ndim != 2 or piezo_irreps.shape[-1] != 18:
        raise ValueError("piezo irreps must have shape [batch,18]")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("proper rotations must have shape [frames,3,3]")
    determinant = torch.linalg.det(rotations)
    if not torch.allclose(determinant, torch.ones_like(determinant), atol=2e-5, rtol=2e-5):
        raise ValueError("compatibility minimization uses proper SO(3) frames")
    tensor = piezo_from_irreps(piezo_irreps)
    rotated = rotate_rank3(tensor.unsqueeze(1), rotations.to(tensor).unsqueeze(0))
    projected = reynolds_project(rotated, operations.to(rotated))
    numerator = torch.linalg.vector_norm((rotated - projected).flatten(-3), dim=-1)
    denominator = torch.linalg.vector_norm(tensor.flatten(-3), dim=-1).unsqueeze(-1)
    residual = numerator.amin(dim=-1) / (denominator.squeeze(-1) + epsilon)
    return torch.where(denominator.squeeze(-1) <= epsilon, torch.zeros_like(residual), residual)


class SpaceGroupCompatibilityRouter(nn.Module):
    """Learned invariant prior multiplied by a full-O(3) Reynolds factor."""

    def __init__(
        self,
        space_groups: tuple[int, ...] | list[int],
        *,
        hidden_dim: int = 128,
        rotation_count: int = 240,
        hard_zero_rank: bool = True,
    ) -> None:
        super().__init__()
        selected = tuple(int(value) for value in space_groups)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("router requires unique represented space groups")
        if any(value < 1 or value > 230 for value in selected):
            raise ValueError("space-group number lies outside 1..230")
        self.space_groups = selected
        self.records = tuple(compatibility_record(value) for value in selected)
        self.hard_zero_rank = hard_zero_rank
        self.prior = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, len(selected))
        )
        self.log_beta = nn.Parameter(torch.zeros(len(selected)))
        self.register_buffer("rotations", nested_hopf_so3_grid(rotation_count, dtype=torch.float64))
        self.register_buffer(
            "compatible_rank", torch.tensor([record.compatible_rank for record in self.records], dtype=torch.long)
        )

    def compatibility(self, piezo_irreps: torch.Tensor) -> torch.Tensor:
        values = [
            orbit_compatibility_residual(
                piezo_irreps,
                record.operations.to(piezo_irreps),
                self.rotations.to(piezo_irreps),
            )
            for record in self.records
        ]
        return torch.stack(values, dim=-1)

    def forward(self, piezo_irreps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        invariants = normalized_low_order_orbit_invariants(piezo_irreps)
        prior_logits = self.prior(invariants)
        residual = self.compatibility(piezo_irreps)
        beta = torch.nn.functional.softplus(self.log_beta).to(residual)
        logits = prior_logits - beta * residual.square()
        physical_zero = torch.linalg.vector_norm(piezo_irreps, dim=-1) <= 1e-12
        if self.hard_zero_rank:
            incompatible = (self.compatible_rank == 0).unsqueeze(0) & ~physical_zero.unsqueeze(-1)
            logits = logits.masked_fill(incompatible, -torch.inf)
        return logits, residual
