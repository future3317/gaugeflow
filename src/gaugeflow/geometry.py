"""Periodic metric geometry primitives for the versioned discrete substrate.

The historical vector field used unit edge directions only.  That discards the
metric information needed to distinguish sites in a fixed periodic geometry.
These utilities retain the closest-image displacement, its length and the
integer image shift under GaugeFlow's row-lattice convention ``r = f @ L``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PeriodicEdges:
    """Closest-image directed inter-site edges and their physical geometry."""

    source: torch.Tensor
    target: torch.Tensor
    displacement: torch.Tensor
    direction: torch.Tensor
    distance: torch.Tensor
    image_shift: torch.Tensor


def periodic_closest_image_edges(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    *,
    shifts: torch.Tensor | None = None,
) -> PeriodicEdges:
    """Build all directed inter-site closest-image edges inside each graph.

    ``frac_coords`` are row fractional coordinates and ``lattice[g]`` has real
    basis vectors as rows.  For the directed edge ``i -> j`` we use
    ``(f_j - f_i + s) @ L`` with the minimum-norm integer image ``s`` from the
    supplied finite shell.  The selected shift is returned so representation
    and cell-gauge tests can inspect it rather than treating PBC as opaque.
    """
    if frac_coords.ndim != 2 or frac_coords.shape[-1] != 3:
        raise ValueError("frac_coords must have shape [nodes, 3]")
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError("lattice must have shape [graphs, 3, 3]")
    if batch.ndim != 1 or batch.numel() != frac_coords.shape[0]:
        raise ValueError("batch must contain one graph index per fractional coordinate")
    if batch.numel() and (int(batch.min()) < 0 or int(batch.max()) >= lattice.shape[0]):
        raise ValueError("batch graph index is outside lattice support")
    if not torch.isfinite(frac_coords).all() or not torch.isfinite(lattice).all():
        raise ValueError("periodic geometry must be finite")

    node_count = frac_coords.shape[0]
    same_graph = batch[:, None] == batch[None, :]
    keep = same_graph & ~torch.eye(node_count, dtype=torch.bool, device=batch.device)
    source, target = torch.nonzero(keep, as_tuple=True)
    if source.numel() == 0:
        empty_long = batch.new_empty((0,), dtype=torch.long)
        empty_vec = frac_coords.new_empty((0, 3))
        return PeriodicEdges(empty_long, empty_long, empty_vec, empty_vec, frac_coords.new_empty((0,)), empty_vec)

    if shifts is None:
        axis = torch.arange(-2, 3, device=frac_coords.device, dtype=frac_coords.dtype)
        shifts = torch.cartesian_prod(axis, axis, axis)
    else:
        if shifts.ndim != 2 or shifts.shape[-1] != 3:
            raise ValueError("shifts must have shape [images, 3]")
        shifts = shifts.to(device=frac_coords.device, dtype=frac_coords.dtype)
    if shifts.numel() == 0 or not torch.isfinite(shifts).all():
        raise ValueError("shifts must contain at least one finite periodic image")

    fractional_images = (frac_coords[target] - frac_coords[source]).unsqueeze(1) + shifts.unsqueeze(0)
    cartesian_images = torch.einsum("esi,eij->esj", fractional_images, lattice[batch[source]])
    nearest = cartesian_images.square().sum(dim=-1).argmin(dim=-1)
    rows = torch.arange(source.numel(), device=source.device)
    displacement = cartesian_images[rows, nearest]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    if bool((distance <= 1e-10).any()):
        raise ValueError("distinct periodic sites have a zero closest-image displacement")
    direction = displacement / distance.unsqueeze(-1)
    return PeriodicEdges(source, target, displacement, direction, distance, shifts[nearest])


class GaussianRadialBasis(torch.nn.Module):
    """Smooth finite-cutoff Gaussian RBF for physical periodic distances."""

    def __init__(self, count: int = 16, cutoff: float = 8.0):
        super().__init__()
        if count < 2 or cutoff <= 0:
            raise ValueError("RBF count must be at least two and cutoff positive")
        self.count = int(count)
        self.cutoff = float(cutoff)
        self.register_buffer("centers", torch.linspace(0.0, cutoff, count))
        self.width = float(cutoff / (count - 1))

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        if distance.ndim != 1 or not torch.isfinite(distance).all():
            raise ValueError("distance must be a finite rank-one tensor")
        value = distance.unsqueeze(-1)
        gaussian = torch.exp(-0.5 * ((value - self.centers.to(value)) / self.width).square())
        inside = (value < self.cutoff).to(value)
        envelope = 0.5 * (torch.cos(torch.pi * value.clamp(max=self.cutoff) / self.cutoff) + 1.0)
        return gaussian * envelope * inside
