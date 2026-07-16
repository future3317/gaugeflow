"""Deterministic parent-to-child reconstruction on a fixed symmetry branch."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.manifold import wrap01

from .blueprint import DistortionBlueprint, ModeDiffusionState, validate_supercell_hnf


@dataclass(frozen=True)
class ParentCrystal:
    """Ordered parent crystal in row-vector lattice convention."""

    species: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    masses: torch.Tensor

    def __post_init__(self) -> None:
        nodes = self.species.numel()
        if self.species.shape != (nodes,) or self.species.dtype != torch.long:
            raise ValueError("parent species must be an int64 vector")
        if self.fractional_coordinates.shape != (nodes, 3):
            raise ValueError("parent fractional coordinates must have shape [nodes,3]")
        if self.lattice.shape != (3, 3) or float(torch.linalg.det(self.lattice)) <= 0.0:
            raise ValueError("parent lattice must be a right-handed 3x3 row matrix")
        if self.masses.shape != (nodes,) or bool((self.masses <= 0).any()):
            raise ValueError("parent masses must be a positive vector")
        if not all(torch.isfinite(value).all() for value in (self.fractional_coordinates, self.lattice, self.masses)):
            raise ValueError("parent crystal contains non-finite values")


@dataclass(frozen=True)
class ChildCrystal:
    species: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    mode_displacement: torch.Tensor
    residual_displacement: torch.Tensor
    child_operation_indices: torch.Tensor
    supercell_translations: torch.Tensor


@dataclass(frozen=True)
class HierarchicalSample:
    parent_structure: ParentCrystal
    distortion_blueprint: DistortionBlueprint
    child_structure: ChildCrystal


def supercell_coset_translations(
    supercell_matrix: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Enumerate the finite quotient ``Z^3 / B Z^3`` without duplicates."""
    index = validate_supercell_hnf(supercell_matrix)
    selected_device = torch.device(device) if device is not None else supercell_matrix.device
    integer_adjugate = (index * torch.linalg.inv(supercell_matrix.to(dtype=torch.float64))).round().to(torch.int64)
    representatives: list[torch.Tensor] = []
    keys: set[tuple[int, int, int]] = set()
    for i in range(index):
        for j in range(index):
            for k in range(index):
                parent_translation = torch.tensor([i, j, k], dtype=dtype, device=selected_device)
                integer_translation = torch.tensor([i, j, k], dtype=torch.int64)
                numerator = integer_translation @ integer_adjugate.cpu()
                key = (
                    int(numerator[0]) % index,
                    int(numerator[1]) % index,
                    int(numerator[2]) % index,
                )
                if key not in keys:
                    keys.add(key)
                    representatives.append(parent_translation)
                if len(representatives) == index:
                    return torch.stack(representatives)
    raise RuntimeError("failed to enumerate every supercell coset representative")


def expand_parent_supercell(
    parent: ParentCrystal,
    supercell_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand the parent into ``B L`` and return child-cell fractional sites."""
    translations = supercell_coset_translations(
        supercell_matrix,
        dtype=parent.fractional_coordinates.dtype,
        device=parent.fractional_coordinates.device,
    )
    matrix = supercell_matrix.to(parent.fractional_coordinates)
    inverse = torch.linalg.inv(matrix)
    expanded_fractional = wrap01(
        (parent.fractional_coordinates.unsqueeze(0) + translations.unsqueeze(1)) @ inverse
    ).reshape(-1, 3)
    species = parent.species.repeat(translations.shape[0])
    masses = parent.masses.repeat(translations.shape[0])
    lattice = matrix @ parent.lattice
    return species, masses, expanded_fractional, lattice, translations


def reynolds_project_displacements(
    displacements: torch.Tensor,
    representations: torch.Tensor,
) -> torch.Tensor:
    """Project a Cartesian displacement field into a finite-group fixed space."""
    if displacements.ndim != 2 or displacements.shape[-1] != 3:
        raise ValueError("displacements must have shape [nodes,3]")
    dimension = displacements.numel()
    if representations.ndim != 3 or representations.shape[1:] != (dimension, dimension):
        raise ValueError("displacement representations must have shape [group,3N,3N]")
    if representations.shape[0] < 1:
        raise ValueError("displacement representation must contain the identity")
    flattened = displacements.reshape(-1)
    projected = torch.einsum("gij,j->gi", representations.to(flattened), flattened).mean(0)
    return projected.reshape_as(displacements)


def _remove_mass_weighted_translation(displacements: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    translation = (masses.unsqueeze(-1) * displacements).sum(0) / masses.sum()
    return displacements - translation


class ChildReconstructor:
    """Reconstruct an ordered commensurate child after the OPD branch is fixed.

    This class does not infer a parent, a mode catalogue, or a paired target
    mapping.  Those are offline/catalogue responsibilities.  It fails closed if
    the supplied mode basis is not invariant under the declared child subgroup
    or if the residual branch exceeds its registered RMS budget.
    """

    def __init__(self, *, residual_rms_limit_angstrom: float = 0.10, tolerance: float = 2e-5) -> None:
        if residual_rms_limit_angstrom <= 0.0 or tolerance <= 0.0:
            raise ValueError("reconstruction tolerances must be positive")
        self.residual_rms_limit_angstrom = float(residual_rms_limit_angstrom)
        self.tolerance = float(tolerance)

    def reconstruct(
        self,
        parent: ParentCrystal,
        blueprint: DistortionBlueprint,
        state: ModeDiffusionState,
        *,
        parent_fractional_rotations: torch.Tensor,
        parent_cartesian_operations: torch.Tensor,
        displacement_representations: torch.Tensor,
        invariant_strain_basis: torch.Tensor,
    ) -> ChildCrystal:
        species, masses, parent_fractional, parent_lattice, translations = expand_parent_supercell(
            parent, blueprint.supercell_matrix
        )
        nodes = species.numel()
        if len(state.mode_amplitudes) != len(blueprint.modes):
            raise ValueError("mode amplitude tuple must align with the sampled modes")
        if state.residual_displacements.shape != (nodes, 3):
            raise ValueError("residual displacement count does not match the expanded parent")
        if invariant_strain_basis.ndim != 3 or invariant_strain_basis.shape[1:] != (3, 3):
            raise ValueError("invariant strain basis must have shape [strain_dim,3,3]")
        if state.child_strain.shape != (invariant_strain_basis.shape[0],):
            raise ValueError("child strain coordinates do not match the invariant basis")
        if not torch.allclose(
            invariant_strain_basis,
            invariant_strain_basis.transpose(-1, -2),
            atol=self.tolerance,
            rtol=self.tolerance,
        ):
            raise ValueError("child strain basis must be symmetric")

        child_indices = blueprint.child_operation_indices(parent_fractional_rotations)
        operation_count = parent_fractional_rotations.shape[0]
        if parent_cartesian_operations.shape != (operation_count, 3, 3):
            raise ValueError("Cartesian operations must align with parent fractional operations")
        if displacement_representations.shape != (operation_count, 3 * nodes, 3 * nodes):
            raise ValueError("displacement representations must align with parent operations and nodes")
        child_representations = displacement_representations[child_indices]
        child_cartesian = parent_cartesian_operations[child_indices].to(invariant_strain_basis)
        transformed_strain_basis = torch.einsum(
            "gik,akl,gjl->gaij",
            child_cartesian,
            invariant_strain_basis,
            child_cartesian,
        )
        if not torch.allclose(
            transformed_strain_basis,
            invariant_strain_basis.unsqueeze(0).expand_as(transformed_strain_basis),
            atol=self.tolerance,
            rtol=self.tolerance,
        ):
            raise ValueError("child strain basis is not invariant under the declared child subgroup")

        mode_displacement = torch.zeros((nodes, 3), dtype=parent.lattice.dtype, device=parent.lattice.device)
        inverse_sqrt_mass = masses.to(mode_displacement).rsqrt().repeat_interleave(3)
        for selected_mode, amplitude in zip(blueprint.modes, state.mode_amplitudes, strict=True):
            branch = selected_mode.branch
            if amplitude.shape != (branch.basis.shape[1],):
                raise ValueError("mode amplitude dimension does not match its OPD branch")
            if not selected_mode.active:
                if not torch.allclose(amplitude, torch.zeros_like(amplitude), atol=self.tolerance, rtol=0.0):
                    raise ValueError("inactive mode must have exactly zero continuous amplitude")
                continue
            basis = selected_mode.entry.mode_basis.to(mode_displacement)
            if basis.shape[0] != 3 * nodes:
                raise ValueError("mode basis node count does not match the expanded parent")
            order_parameter = branch.basis.to(amplitude) @ amplitude
            contribution = (inverse_sqrt_mass * (basis @ order_parameter).to(mode_displacement)).reshape(nodes, 3)
            mode_displacement = mode_displacement + contribution

        mode_displacement = _remove_mass_weighted_translation(mode_displacement, masses.to(mode_displacement))
        flattened_mode = mode_displacement.reshape(-1)
        transformed_mode = torch.einsum(
            "gij,j->gi", child_representations.to(flattened_mode), flattened_mode
        )
        if not torch.allclose(
            transformed_mode,
            flattened_mode.expand_as(transformed_mode),
            atol=self.tolerance,
            rtol=self.tolerance,
        ):
            raise ValueError("selected mode/OPD is not invariant under its declared child subgroup")

        residual = _remove_mass_weighted_translation(state.residual_displacements.to(mode_displacement), masses)
        residual = reynolds_project_displacements(residual, child_representations)
        residual = _remove_mass_weighted_translation(residual, masses)
        flattened_residual = residual.reshape(-1)
        transformed_residual = torch.einsum(
            "gij,j->gi", child_representations.to(flattened_residual), flattened_residual
        )
        if not torch.allclose(
            transformed_residual,
            flattened_residual.expand_as(transformed_residual),
            atol=self.tolerance,
            rtol=self.tolerance,
        ):
            raise ValueError("projected residual is not fixed by the declared child subgroup")
        residual_rms = torch.linalg.vector_norm(residual, dim=-1).square().mean().sqrt()
        if float(residual_rms) > self.residual_rms_limit_angstrom:
            raise ValueError("projected residual exceeds the registered reconstruction budget")

        strain = torch.einsum(
            "a,aij->ij",
            state.child_strain.to(parent.lattice),
            invariant_strain_basis.to(parent.lattice),
        )
        child_lattice = parent_lattice @ torch.matrix_exp(strain)
        parent_cartesian = parent_fractional @ parent_lattice
        child_cartesian = parent_cartesian + mode_displacement + residual
        child_fractional = wrap01(child_cartesian @ torch.linalg.inv(child_lattice))
        if not all(torch.isfinite(value).all() for value in (child_fractional, child_lattice)):
            raise ValueError("child reconstruction produced non-finite state")
        return ChildCrystal(
            species=species,
            fractional_coordinates=child_fractional,
            lattice=child_lattice,
            mode_displacement=mode_displacement,
            residual_displacement=residual,
            child_operation_indices=child_indices,
            supercell_translations=translations,
        )
