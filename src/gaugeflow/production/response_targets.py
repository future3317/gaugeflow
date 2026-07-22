"""Fail-closed canonicalization for Stage-D response supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def kelvin_basis(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return the orthonormal symmetric-tensor Kelvin basis ``[6,3,3]``."""

    basis = torch.zeros(6, 3, 3, dtype=dtype, device=device)
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    basis[2, 2, 2] = 1.0
    inverse_root_two = basis.new_tensor(0.5).sqrt()
    basis[3, 1, 2] = basis[3, 2, 1] = inverse_root_two
    basis[4, 0, 2] = basis[4, 2, 0] = inverse_root_two
    basis[5, 0, 1] = basis[5, 1, 0] = inverse_root_two
    return basis


def engineering_stiffness_to_kelvin(value: torch.Tensor) -> torch.Tensor:
    """Convert engineering-Voigt stiffness to an orthonormal Kelvin matrix."""

    if value.shape[-2:] != (6, 6) or not value.dtype.is_floating_point:
        raise ValueError("engineering stiffness must end in a floating [6,6] matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("engineering stiffness must be finite")
    scale = value.new_ones(6)
    scale[3:] = value.new_tensor(2.0).sqrt()
    return value * scale[..., :, None] * scale[..., None, :]


def kelvin_stiffness_to_cartesian(
    value: torch.Tensor,
    *,
    symmetry_tolerance: float = 1e-5,
) -> torch.Tensor:
    """Decode a major-symmetric Kelvin stiffness into ``C_ijkl``."""

    if value.shape[-2:] != (6, 6) or symmetry_tolerance < 0.0:
        raise ValueError("Kelvin stiffness must end in [6,6]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Kelvin stiffness must be finite")
    residual = (value - value.transpose(-1, -2)).abs().amax()
    scale = value.abs().amax().clamp_min(1.0)
    if float(residual / scale) > symmetry_tolerance:
        raise ValueError("stiffness violates major symmetry")
    symmetric = 0.5 * (value + value.transpose(-1, -2))
    basis = kelvin_basis(dtype=value.dtype, device=value.device)
    return torch.einsum("...ab,aij,bkl->...ijkl", symmetric, basis, basis)


def cartesian_stiffness_to_kelvin(value: torch.Tensor) -> torch.Tensor:
    """Encode a Cartesian stiffness with minor symmetries in Kelvin form."""

    if value.shape[-4:] != (3, 3, 3, 3) or not value.dtype.is_floating_point:
        raise ValueError("Cartesian stiffness must end in floating [3,3,3,3]")
    basis = kelvin_basis(dtype=value.dtype, device=value.device)
    return torch.einsum("aij,...ijkl,bkl->...ab", basis, value, basis)


@dataclass(frozen=True)
class GammaSpectrumTargets:
    soft: torch.Tensor
    log_magnitude: torch.Tensor
    mask: torch.Tensor


def canonical_gamma_spectrum(
    dynamical_eigenvalues: torch.Tensor,
    *,
    maximum_atoms: int,
    eigenvalue_scale: float,
) -> GammaSpectrumTargets:
    """Sort and pad the gauge-invariant Gamma-point dynamical spectrum."""

    if (
        dynamical_eigenvalues.ndim != 1
        or not dynamical_eigenvalues.dtype.is_floating_point
        or dynamical_eigenvalues.numel() % 3 != 0
        or maximum_atoms < 1
        or eigenvalue_scale <= 0.0
    ):
        raise ValueError("invalid Gamma-spectrum target")
    if not bool(torch.isfinite(dynamical_eigenvalues).all()):
        raise ValueError("Gamma-spectrum target must be finite")
    maximum_modes = 3 * maximum_atoms
    if dynamical_eigenvalues.numel() > maximum_modes:
        raise ValueError("Gamma spectrum exceeds the registered atom domain")
    ordered = torch.sort(dynamical_eigenvalues).values
    soft = ordered.new_zeros(maximum_modes)
    magnitude = ordered.new_zeros(maximum_modes)
    mask = torch.zeros(maximum_modes, dtype=torch.bool, device=ordered.device)
    modes = ordered.numel()
    soft[:modes] = (ordered < 0.0).to(ordered)
    magnitude[:modes] = torch.log1p(ordered.abs() / eigenvalue_scale)
    mask[:modes] = True
    return GammaSpectrumTargets(soft=soft, log_magnitude=magnitude, mask=mask)


@dataclass(frozen=True)
class InternalStrainTargets:
    value: torch.Tensor
    mask: torch.Tensor
    maximum_antisymmetric_residual: float
    source_symmetric_within_rounding: bool


def scatter_internal_strain_blocks(
    blocks: torch.Tensor,
    ions: torch.Tensor,
    directions: torch.Tensor,
    *,
    atom_count: int,
    rounding_halfwidth: torch.Tensor | None = None,
    symmetry_tolerance: float = 1e-5,
) -> InternalStrainTargets:
    """Scatter observed ``dF_direction/d(strain)`` blocks without imputation."""

    block_count = blocks.shape[0] if blocks.ndim == 3 else -1
    if (
        blocks.shape != (block_count, 3, 3)
        or ions.shape != (block_count,)
        or directions.shape != (block_count,)
        or ions.dtype != torch.long
        or directions.dtype != torch.long
        or atom_count < 1
        or symmetry_tolerance < 0.0
    ):
        raise ValueError("internal-strain block contract is invalid")
    if not blocks.dtype.is_floating_point or not bool(torch.isfinite(blocks).all()):
        raise ValueError("internal-strain blocks must be finite floating tensors")
    if rounding_halfwidth is not None and (
        rounding_halfwidth.shape != blocks.shape
        or not rounding_halfwidth.dtype.is_floating_point
        or not bool(torch.isfinite(rounding_halfwidth).all())
        or bool((rounding_halfwidth < 0.0).any())
    ):
        raise ValueError("internal-strain rounding half-width is invalid")
    if block_count == 0:
        return InternalStrainTargets(
            value=blocks.new_zeros(atom_count, 3, 3, 3),
            mask=torch.zeros(atom_count, 3, 3, 3, dtype=torch.bool, device=blocks.device),
            maximum_antisymmetric_residual=0.0,
            source_symmetric_within_rounding=True,
        )
    if int(ions.min()) < 0 or int(ions.max()) >= atom_count:
        raise ValueError("internal-strain ion index lies outside the structure")
    if int(directions.min()) < 0 or int(directions.max()) >= 3:
        raise ValueError("internal-strain force direction lies outside Cartesian space")
    keys = ions * 3 + directions
    if torch.unique(keys).numel() != block_count:
        raise ValueError("duplicate internal-strain blocks require source-side resolution")
    residual = (blocks - blocks.transpose(-1, -2)).abs()
    if rounding_halfwidth is None:
        scale = blocks.abs().amax().clamp_min(1.0)
        symmetric_within_source_precision = bool(
            (residual <= symmetry_tolerance * scale).all()
        )
    else:
        uncertainty = rounding_halfwidth + rounding_halfwidth.transpose(-1, -2)
        numerical_slack = torch.finfo(blocks.dtype).eps * blocks.abs().amax().clamp_min(1.0)
        symmetric_within_source_precision = bool(
            (residual <= uncertainty + numerical_slack).all()
        )
    symmetric = 0.5 * (blocks + blocks.transpose(-1, -2))
    value = blocks.new_zeros(atom_count, 3, 3, 3)
    mask = torch.zeros_like(value, dtype=torch.bool)
    value[ions, directions] = symmetric
    mask[ions, directions] = True
    return InternalStrainTargets(
        value=value,
        mask=mask,
        maximum_antisymmetric_residual=float(residual.max()),
        source_symmetric_within_rounding=symmetric_within_source_precision,
    )
