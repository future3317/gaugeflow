"""Exact periodic metric geometry for the production crystal graph.

The denoiser uses a radius *multigraph*: every periodic image inside the
physical cutoff is retained, including non-zero self images. Closest-image
queries remain available for crystallographic certification, but are not a
fallback graph representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_cluster import radius


@dataclass(frozen=True)
class PeriodicEdges:
    """Directed periodic image edges and their physical geometry."""

    source: torch.Tensor
    target: torch.Tensor
    displacement: torch.Tensor
    direction: torch.Tensor
    distance: torch.Tensor
    image_shift: torch.Tensor


def _sphere_decode_shift(
    matrix: np.ndarray,
    fractional: np.ndarray,
    orthogonal: np.ndarray,
    triangular: np.ndarray,
) -> np.ndarray:
    """Solve one exact three-dimensional CVP with a shared QR factorization."""
    if not np.isfinite(matrix).all() or not np.isfinite(fractional).all():
        raise ValueError("closest-image CVP inputs must be finite")
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
    return best


def _closest_integer_shift(delta: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Solve the three-dimensional closest-vector problem by sphere decoding.

    The discrete image choice is necessarily piecewise constant. Selection is
    therefore performed in detached float64 QR coordinates, while callers
    reconstruct the winning Cartesian displacement from the original tensors
    so gradients inside each Voronoi cell remain exact.
    """
    matrix = lattice.detach().to(dtype=torch.float64, device="cpu").numpy().T
    fractional = delta.detach().to(dtype=torch.float64, device="cpu").numpy()
    orthogonal, triangular = np.linalg.qr(matrix)
    best = _sphere_decode_shift(matrix, fractional, orthogonal, triangular)
    return torch.tensor(best, dtype=delta.dtype, device=delta.device)


def closest_image_displacements_numpy(
    delta: np.ndarray, lattice: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact closest images for many vectors in one fixed lattice.

    This is the batched, gradient-free companion of
    :func:`closest_image_displacement`.  It reuses one float64 QR
    factorization for every vector while retaining the same exact sphere
    decoder; unlike a fixed image cube, it remains correct for skew cells.
    """
    fractional = np.asarray(delta, dtype=np.float64)
    cell = np.asarray(lattice, dtype=np.float64)
    if fractional.ndim != 2 or fractional.shape[1] != 3 or cell.shape != (3, 3):
        raise ValueError("delta and lattice must have shapes [N,3] and [3,3]")
    if not np.isfinite(fractional).all() or not np.isfinite(cell).all():
        raise ValueError("closest-image inputs must be finite")
    if fractional.shape[0] == 0:
        return fractional.copy(), np.empty((0, 3), dtype=np.int64)
    matrix = cell.T
    orthogonal, triangular = np.linalg.qr(matrix)
    shifts = np.stack(
        [
            _sphere_decode_shift(matrix, value, orthogonal, triangular)
            for value in fractional
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    return (fractional + shifts) @ cell, shifts


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


def periodic_radius_multigraph(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    *,
    cutoff: float,
) -> PeriodicEdges:
    """Build the complete directed periodic radius multigraph on-device.

    For a directed edge ``i -> j`` this returns every integer image ``n`` with
    ``||(f_j-f_i+n)L|| < cutoff``. After wrapping coordinates to one cell, the
    reciprocal-column bound

    ``|n_k| <= ceil(cutoff * ||L^{-1}_{:,k}|| + 1)``

    is complete by Cauchy--Schwarz. Candidate construction and filtering are
    vectorized PyTorch/torch-cluster operations; no edge is sent through a CPU
    closest-vector solver.
    """
    if cutoff <= 0.0:
        raise ValueError("periodic graph cutoff must be positive")
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

    graphs = lattice.shape[0]
    if frac_coords.shape[0] == 0:
        empty_long = batch.new_empty((0,), dtype=torch.long)
        empty_vec = frac_coords.new_empty((0, 3))
        return PeriodicEdges(
            empty_long,
            empty_long,
            empty_vec,
            empty_vec,
            frac_coords.new_empty((0,)),
            empty_long.new_empty((0, 3)),
        )
    if graphs < 1 or not torch.equal(batch, torch.sort(batch).values):
        raise ValueError("periodic graph requires nonempty graph-contiguous batches")
    counts = torch.bincount(batch, minlength=graphs)
    if bool((counts < 1).any()):
        raise ValueError("every lattice graph must contain at least one node")

    wrapped = torch.remainder(frac_coords, 1.0)
    reciprocal_column_norm = torch.linalg.vector_norm(torch.linalg.inv(lattice), dim=1)
    bounds = torch.ceil(cutoff * reciprocal_column_norm + 1.0).to(torch.long)
    maximum = bounds.amax(dim=0).detach().cpu().tolist()
    axes = [
        torch.arange(-int(value), int(value) + 1, device=batch.device)
        for value in maximum
    ]
    shift_grid = torch.cartesian_prod(*axes)
    valid_shift = (shift_grid.abs().unsqueeze(0) <= bounds.unsqueeze(1)).all(dim=-1)
    shift_graph, shift_column = torch.nonzero(valid_shift, as_tuple=True)
    shifts = shift_grid[shift_column]

    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    repeated_counts = counts[shift_graph]
    shift_row = torch.repeat_interleave(
        torch.arange(shifts.shape[0], device=batch.device), repeated_counts
    )
    repeated_starts = torch.repeat_interleave(
        torch.cat((repeated_counts.new_zeros(1), repeated_counts.cumsum(0)[:-1])),
        repeated_counts,
    )
    local_node = torch.arange(shift_row.numel(), device=batch.device) - repeated_starts
    candidate_graph = shift_graph[shift_row]
    candidate_atom = offsets[candidate_graph] + local_node
    candidate_shift = shifts[shift_row]

    candidate_fractional = wrapped[candidate_atom] + candidate_shift.to(wrapped)
    candidate_cartesian = torch.einsum(
        "ni,nij->nj", candidate_fractional, lattice[candidate_graph]
    )
    central_cartesian = torch.einsum("ni,nij->nj", wrapped, lattice[batch])
    maximum_neighbors = int((counts * valid_shift.sum(dim=1)).amax().detach().cpu())
    target, candidate = radius(
        candidate_cartesian,
        central_cartesian,
        cutoff,
        batch_x=candidate_graph,
        batch_y=batch,
        max_num_neighbors=maximum_neighbors,
    )
    source = candidate_atom[candidate]
    selected_shifts = -candidate_shift[candidate]
    nontrivial = (source != target) | (selected_shifts != 0).any(dim=-1)
    source = source[nontrivial]
    target = target[nontrivial]
    selected_shifts = selected_shifts[nontrivial]
    relative = wrapped[target] - wrapped[source] + selected_shifts.to(wrapped)
    displacement = torch.einsum("ni,nij->nj", relative, lattice[batch[target]])
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    inside = distance < cutoff
    source = source[inside]
    target = target[inside]
    selected_shifts = selected_shifts[inside]
    displacement = displacement[inside]
    distance = distance[inside]
    if bool((distance <= 1e-10).any()):
        raise ValueError("periodic graph contains coincident sites")
    # torch-cluster does not promise an enumeration order. Canonical ordering
    # makes reductions reproducible and prevents equivalent wrapped/chart
    # representatives from accumulating the same messages in different FP32
    # orders.
    keys = torch.cat(
        (target.unsqueeze(-1), source.unsqueeze(-1), selected_shifts), dim=-1
    )
    order = torch.arange(source.numel(), device=source.device)
    for column in range(keys.shape[1] - 1, -1, -1):
        order = order[torch.argsort(keys[order, column], stable=True)]
    source = source[order]
    target = target[order]
    selected_shifts = selected_shifts[order]
    displacement = displacement[order]
    distance = distance[order]
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

    def envelope(self, distance: torch.Tensor) -> torch.Tensor:
        if distance.ndim != 1 or not torch.isfinite(distance).all():
            raise ValueError("distance must be a finite rank-one tensor")
        value = distance.unsqueeze(-1)
        inside = (value < self.cutoff).to(value)
        envelope = 0.5 * (torch.cos(torch.pi * value.clamp(max=self.cutoff) / self.cutoff) + 1.0)
        return envelope * inside

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        value = distance.unsqueeze(-1)
        gaussian = torch.exp(-0.5 * ((value - self.centers.to(value)) / self.width).square())
        return gaussian * self.envelope(distance)
