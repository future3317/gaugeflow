"""Minimal standalone crystal product manifold used by GaugeFlow."""

from __future__ import annotations

import torch


def wrap01(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value, 1.0)


def torus_logmap(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Shortest tangent displacement on the unit three-torus."""
    return torch.remainder(end - start + 0.5, 1.0) - 0.5


def project_simplex(value: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of the final axis onto the unit probability simplex.

    This is the exact sorting-based projection, rather than clipping then
    renormalizing. It is the retraction used by the A5 simplex atom-type path.
    """
    if value.shape[-1] < 1:
        raise ValueError("simplex projection requires a non-empty final axis")
    classes = value.shape[-1]
    sorted_value, _ = torch.sort(value, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_value, dim=-1) - 1.0
    index = torch.arange(1, classes + 1, device=value.device, dtype=value.dtype)
    support = sorted_value - cumulative / index > 0
    rho = support.sum(dim=-1).clamp_min(1)
    theta = cumulative.gather(-1, (rho - 1).unsqueeze(-1)).squeeze(-1) / rho.to(value.dtype)
    return (value - theta.unsqueeze(-1)).clamp_min(0.0)


def simplex_tangent(value: torch.Tensor) -> torch.Tensor:
    """Orthogonally project a vector to the simplex tangent hyperplane."""
    return value - value.mean(dim=-1, keepdim=True)


def symmetric_to_vector(value: torch.Tensor) -> torch.Tensor:
    """Kelvin-style vectorization of a symmetric 3x3 matrix."""
    root2 = 2.0**0.5
    return torch.stack(
        (value[..., 0, 0], value[..., 1, 1], value[..., 2, 2],
         root2 * value[..., 1, 2], root2 * value[..., 0, 2], root2 * value[..., 0, 1]),
        dim=-1,
    )


def vector_to_symmetric(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 6:
        raise ValueError("Expected six symmetric-matrix coordinates")
    root2 = 2.0**0.5
    out = value.new_zeros(*value.shape[:-1], 3, 3)
    out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = value.unbind(dim=-1)[:3]
    out[..., 1, 2] = out[..., 2, 1] = value[..., 3] / root2
    out[..., 0, 2] = out[..., 2, 0] = value[..., 4] / root2
    out[..., 0, 1] = out[..., 1, 0] = value[..., 5] / root2
    return out


def spd_log(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    return (eigenvectors * eigenvalues.clamp_min(eps).log().unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


def spd_exp(value: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    return (eigenvectors * eigenvalues.exp().unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


def lattice_to_log_vector(lattice: torch.Tensor) -> torch.Tensor:
    """Map row-vector lattice matrices to an orientation-free SPD log coordinate."""
    metric = lattice @ lattice.transpose(-1, -2)
    return symmetric_to_vector(spd_log(metric))


def log_vector_to_lattice(value: torch.Tensor) -> torch.Tensor:
    """Choose the lower-Cholesky representative of an SPD lattice metric."""
    metric = spd_exp(vector_to_symmetric(value))
    return torch.linalg.cholesky(metric)
