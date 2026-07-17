"""Gauge-safe targets for phonon modes and low-dimensional PES supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional


@dataclass(frozen=True)
class PhononTargets:
    soft: torch.Tensor
    log_magnitude: torch.Tensor


def phonon_targets(omega_squared: torch.Tensor, *, omega0_squared: float) -> PhononTargets:
    """Split unstable-mode classification from source-scale magnitude."""
    if omega0_squared <= 0.0 or not torch.isfinite(omega_squared).all():
        raise ValueError("phonon scale must be positive and targets finite")
    return PhononTargets(
        soft=(omega_squared < 0).to(omega_squared.dtype),
        log_magnitude=torch.log1p(omega_squared.abs() / omega0_squared),
    )


def source_calibrated_frequency_loss(
    predicted_soft_logits: torch.Tensor,
    predicted_shared_magnitude: torch.Tensor,
    source_bias: torch.Tensor,
    targets: PhononTargets,
    *,
    magnitude_weight: float = 1.0,
) -> torch.Tensor:
    """BCE plus Huber magnitude loss with an explicit per-source calibration."""
    shapes = {
        predicted_soft_logits.shape,
        predicted_shared_magnitude.shape,
        source_bias.shape,
        targets.soft.shape,
        targets.log_magnitude.shape,
    }
    if len(shapes) != 1 or magnitude_weight < 0.0:
        raise ValueError("frequency predictions, source calibration and targets must align")
    classification = functional.binary_cross_entropy_with_logits(predicted_soft_logits, targets.soft)
    magnitude = functional.huber_loss(
        predicted_shared_magnitude + source_bias,
        targets.log_magnitude,
    )
    return classification + magnitude_weight * magnitude


def eigenspace_projector(eigenvectors: torch.Tensor, *, tolerance: float = 2e-6) -> torch.Tensor:
    """Return the Hermitian basis-gauge-invariant projector for a mode space."""
    if eigenvectors.ndim != 2 or eigenvectors.shape[1] < 1 or not torch.isfinite(eigenvectors).all():
        raise ValueError("eigenvectors must be a finite [dimension,multiplicity] matrix")
    gram = eigenvectors.mH @ eigenvectors
    identity = torch.eye(eigenvectors.shape[1], dtype=eigenvectors.dtype, device=eigenvectors.device)
    if not torch.allclose(gram, identity, atol=tolerance, rtol=tolerance):
        raise ValueError("mode-space basis must have orthonormal columns")
    return eigenvectors @ eigenvectors.mH


def subspace_projector_loss(predicted_basis: torch.Tensor, target_basis: torch.Tensor) -> torch.Tensor:
    """Frobenius loss insensitive to sign, order and degenerate-space gauge."""
    predicted = eigenspace_projector(predicted_basis)
    target = eigenspace_projector(target_basis)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target mode subspaces must share an ambient dimension")
    return (predicted - target).abs().square().sum().real


def mode_effective_charge(
    born_effective_charge: torch.Tensor,
    mass_weighted_modes: torch.Tensor,
    masses: torch.Tensor,
) -> torch.Tensor:
    """Compute mode effective charge for one or more mass-weighted modes.

    Returns ``[modes,polarization_direction]`` using
    ``sum_{kappa,alpha} Z*_{kappa,i,alpha} Psi_{kappa,alpha,lambda}/sqrt(M_kappa)``.
    """
    atoms = masses.numel()
    if born_effective_charge.shape != (atoms, 3, 3):
        raise ValueError("Born effective charges must have shape [atoms,3,3]")
    if mass_weighted_modes.ndim != 3 or mass_weighted_modes.shape[:2] != (atoms, 3):
        raise ValueError("mass-weighted modes must have shape [atoms,3,modes]")
    if bool((masses <= 0).any()):
        raise ValueError("atomic masses must be positive")
    scaled_modes = mass_weighted_modes / masses.to(mass_weighted_modes).sqrt().view(-1, 1, 1)
    return torch.einsum("kia,kal->li", born_effective_charge.to(scaled_modes), scaled_modes)


def generalized_mode_force(
    cartesian_forces: torch.Tensor,
    masses: torch.Tensor,
    mode_basis: torch.Tensor,
    opd_basis: torch.Tensor,
) -> torch.Tensor:
    """Project a frozen PES teacher force into the sampled OPD coordinates."""
    atoms = masses.numel()
    if cartesian_forces.shape != (atoms, 3) or mode_basis.shape[0] != 3 * atoms:
        raise ValueError("force, mass and mode-basis atom counts must align")
    if opd_basis.ndim != 2 or mode_basis.shape[1] != opd_basis.shape[0]:
        raise ValueError("OPD basis must act in the supplied mode irrep space")
    if bool((masses <= 0).any()):
        raise ValueError("atomic masses must be positive")
    mass_scaled_force = (
        cartesian_forces / masses.to(cartesian_forces).sqrt().unsqueeze(-1)
    ).reshape(-1)
    return opd_basis.to(mass_scaled_force).transpose(0, 1) @ (
        mode_basis.to(mass_scaled_force).transpose(0, 1) @ mass_scaled_force
    )
