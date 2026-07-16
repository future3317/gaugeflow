"""Periodic metric geometry primitives for the versioned discrete substrate.

The historical vector field used unit edge directions only.  That discards the
metric information needed to distinguish sites in a fixed periodic geometry.
These utilities retain the closest-image displacement, its length and the
integer image shift under GaugeFlow's row-lattice convention ``r = f @ L``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def _closest_integer_shift(delta: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Solve the three-dimensional closest-vector problem by sphere decoding.

    The discrete image choice is necessarily piecewise constant. Selection is
    therefore performed in detached float64 QR coordinates, while callers
    reconstruct the winning Cartesian displacement from the original tensors
    so gradients inside each Voronoi cell remain exact.
    """
    matrix = lattice.detach().to(dtype=torch.float64, device="cpu").numpy().T
    fractional = delta.detach().to(dtype=torch.float64, device="cpu").numpy()
    if not np.isfinite(matrix).all() or not np.isfinite(fractional).all():
        raise ValueError("closest-image CVP inputs must be finite")
    orthogonal, triangular = np.linalg.qr(matrix)
    if np.min(np.abs(np.diag(triangular))) <= np.finfo(np.float64).tiny:
        raise ValueError("closest-image CVP requires an invertible lattice")
    transformed = orthogonal.T @ (-matrix @ fractional)
    dimension = 3
    babai = np.zeros(dimension, dtype=np.int64)
    for level in range(dimension - 1, -1, -1):
        remainder = transformed[level] - triangular[level, level + 1 :] @ babai[level + 1 :]
        babai[level] = int(np.rint(remainder / triangular[level, level]))
    best = babai.copy()
    best_cost = float(np.square(triangular @ babai - transformed).sum())
    current = np.zeros(dimension, dtype=np.int64)

    def search(level: int, partial_cost: float) -> None:
        nonlocal best, best_cost
        if level < 0:
            if partial_cost < best_cost:
                best_cost = partial_cost
                best = current.copy()
            return
        remainder = transformed[level] - triangular[level, level + 1 :] @ current[level + 1 :]
        diagonal = triangular[level, level]
        center = remainder / diagonal
        radius = np.sqrt(max(best_cost - partial_cost, 0.0)) / abs(diagonal)
        lower = int(np.ceil(center - radius - 16.0 * np.finfo(np.float64).eps))
        upper = int(np.floor(center + radius + 16.0 * np.finfo(np.float64).eps))
        candidates = sorted(range(lower, upper + 1), key=lambda value: abs(value - center))
        for candidate in candidates:
            residual = diagonal * candidate - remainder
            next_cost = partial_cost + float(residual * residual)
            if next_cost <= best_cost + 64.0 * np.finfo(np.float64).eps * max(1.0, best_cost):
                current[level] = candidate
                search(level - 1, next_cost)

    search(dimension - 1, 0.0)
    return torch.tensor(best, dtype=delta.dtype, device=delta.device)


def closest_image_displacement(
    delta: torch.Tensor, lattice: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact closest Cartesian displacement and integer image.

    This public single-vector contract shares the same exact float64 CVP
    solver as the edge builder and is used by production symmetry expansion
    and duplicate checks.  It never falls back to a fixed image cube.
    """
    if delta.shape != (3,) or lattice.shape != (3, 3):
        raise ValueError("delta and lattice must have shapes [3] and [3,3]")
    if not torch.isfinite(delta).all() or not torch.isfinite(lattice).all():
        raise ValueError("closest-image inputs must be finite")
    shift = _closest_integer_shift(delta, lattice)
    return (delta + shift) @ lattice, shift


def periodic_closest_image_edges(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> PeriodicEdges:
    """Build exact directed closest-image edges inside each graph.

    ``frac_coords`` are row fractional coordinates and ``lattice[g]`` has real
    basis vectors as rows.  For the directed edge ``i -> j`` we use
    ``(f_j - f_i + s) @ L`` with the globally minimum-norm integer image
    ``s``. A float64 QR sphere decoder uses a Babai upper bound and exact
    branch-and-bound in the three-dimensional triangular system. This remains
    rigorous for ill-conditioned triclinic cells where a fixed shell or a
    loose singular-value bounding box is not practical.
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

    relative = frac_coords[target] - frac_coords[source]
    edge_lattices = lattice[batch[source]]
    displacements = []
    image_shifts = []
    for edge in range(source.numel()):
        delta = relative[edge]
        edge_lattice = edge_lattices[edge]
        selected_shift = _closest_integer_shift(delta, edge_lattice)
        displacements.append((delta + selected_shift) @ edge_lattice)
        image_shifts.append(selected_shift)
    displacement = torch.stack(displacements)
    selected_shifts = torch.stack(image_shifts)
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    if bool((distance <= 1e-10).any()):
        raise ValueError("distinct periodic sites have a zero closest-image displacement")
    direction = displacement / distance.unsqueeze(-1)
    return PeriodicEdges(source, target, displacement, direction, distance, selected_shifts)


class GaussianRadialBasis(torch.nn.Module):
    """Smooth finite-cutoff Gaussian RBF for physical periodic distances."""

    def __init__(self, count: int = 16, cutoff: float = 8.0):
        super().__init__()
        if count < 2 or cutoff <= 0:
            raise ValueError("RBF count must be at least two and cutoff positive")
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
