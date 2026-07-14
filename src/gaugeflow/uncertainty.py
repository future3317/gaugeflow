"""Gauge-consistent uncertainty primitives for conditional crystal flows.

The implementation transfers the useful Log-Euclidean lesson from the ICML
tensor-UQ work without importing its rank-2 spherical-harmonic covariance head.
For a generative vector field, a dense covariance over every atom and tensor
component is both prohibitively large and poorly identified.  We instead model
heteroscedastic *tangent* uncertainty in geometrically meaningful blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VelocityUncertainty:
    """Log standard deviations for the three GaugeFlow velocity blocks.

    ``coord_log_std`` is one scalar per atom in Cartesian tangent space, hence
    its covariance is ``sigma^2 I_3`` and remains SO(3)-equivariant.  Type and
    lattice blocks are scalar-coordinate uncertainties in their respective
    Euclidean flow coordinates.
    """

    type_log_std: torch.Tensor
    coord_log_std: torch.Tensor
    lattice_log_std: torch.Tensor


@dataclass
class SampleUncertainty:
    """Euler-propagated diagonal variance proxies for one generated sample."""

    type_variance: torch.Tensor
    coordinate_cartesian_variance: torch.Tensor
    lattice_variance: torch.Tensor
    mean_alignment_entropy: torch.Tensor


def bounded_log_std(raw: torch.Tensor, *, minimum: float = -3.0, maximum: float = 3.0) -> torch.Tensor:
    """Smoothly constrain log standard deviations without a numerical fallback."""
    if not minimum < maximum:
        raise ValueError("minimum must be smaller than maximum")
    midpoint, radius = (minimum + maximum) / 2.0, (maximum - minimum) / 2.0
    return midpoint + radius * torch.tanh(raw)


def scalar_gaussian_nll(residual: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Mean negative log likelihood for independent scalar tangent components."""
    if not (torch.isfinite(residual).all() and torch.isfinite(log_std).all()):
        raise FloatingPointError("Non-finite residual or log standard deviation in uncertainty objective")
    precision = torch.exp(-2.0 * log_std)
    return 0.5 * (residual.square() * precision + 2.0 * log_std).mean()


def cartesian_isotropic_gaussian_nll(
    cartesian_residual: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    """NLL for an SO(3)-isotropic three-vector covariance ``sigma^2 I_3``."""
    if cartesian_residual.shape[-1] != 3:
        raise ValueError("Cartesian residuals must have final dimension three")
    if not (torch.isfinite(cartesian_residual).all() and torch.isfinite(log_std).all()):
        raise FloatingPointError("Non-finite residual or log standard deviation in uncertainty objective")
    precision = torch.exp(-2.0 * log_std.squeeze(-1))
    squared_norm = cartesian_residual.square().sum(dim=-1)
    return 0.5 * (squared_norm * precision + 6.0 * log_std.squeeze(-1)).mean()
