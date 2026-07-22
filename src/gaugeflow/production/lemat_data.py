"""Typed LeMat-BulkUnique records with explicit physical-label policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, cast

import torch
from pymatgen.core import Element

from gaugeflow.vocabulary import atomic_numbers_to_tokens

from .data_splitting import deterministic_iid_split
from .matpes_data import MatPESPhysicalRecord
from .physical_pretraining import symmetric_cartesian_to_kelvin

LeMatPhysicalLabelPolicy = Literal["compatible_only", "all_with_functional"]
_ALEX_WRAPPER = re.compile(r"^alex<([^<>]+)>$", re.IGNORECASE)


def normalize_external_material_id(material_id: str) -> str:
    """Normalize the known Alex wrapper without guessing unrelated ID aliases."""

    value = material_id.strip().lower()
    match = _ALEX_WRAPPER.fullmatch(value)
    return match.group(1) if match is not None else value


lemat_iid_split = deterministic_iid_split


@dataclass(frozen=True)
class _LeMatGeometry:
    material_id: str
    functional: str
    element_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor


def _parse_lemat_geometry(row: Mapping[str, Any]) -> _LeMatGeometry:
    material_id = row.get("immutable_id")
    functional = row.get("functional")
    if not isinstance(material_id, str) or not material_id or not isinstance(functional, str):
        raise ValueError("LeMat row lacks immutable ID or functional")
    if row.get("nperiodic_dimensions") != 3 or list(row.get("dimension_types", [])) != [1, 1, 1]:
        raise ValueError("GaugeFlow structure pretraining accepts only three-periodic bulk rows")
    lattice = torch.as_tensor(row.get("lattice_vectors"), dtype=torch.float64)
    cartesian = torch.as_tensor(row.get("cartesian_site_positions"), dtype=torch.float64)
    species = row.get("species_at_sites")
    node_count = int(row.get("nsites", 0))
    if lattice.shape != (3, 3) or not bool(torch.isfinite(lattice).all()):
        raise ValueError("LeMat lattice must be finite 3x3")
    if float(torch.linalg.det(lattice)) <= 0.0:
        raise ValueError("LeMat lattice must have positive row-basis volume")
    if cartesian.shape != (node_count, 3) or not bool(torch.isfinite(cartesian).all()):
        raise ValueError("LeMat Cartesian positions disagree with nsites")
    if not isinstance(species, list) or len(species) != node_count or not all(
        isinstance(symbol, str) for symbol in species
    ):
        raise ValueError("LeMat species_at_sites disagrees with nsites")
    fractional = torch.linalg.solve(lattice.T, cartesian.T).T.remainder(1.0)
    atomic_numbers = torch.tensor([Element(symbol).Z for symbol in species], dtype=torch.long)
    return _LeMatGeometry(
        material_id=material_id,
        functional=functional.lower(),
        element_tokens=atomic_numbers_to_tokens(atomic_numbers),
        fractional_coordinates=fractional.float(),
        lattice=lattice.float(),
    )


def lemat_stress_kbar_to_kelvin_gpa(stress: Sequence[Sequence[float]]) -> torch.Tensor:
    """Convert full compressive-positive kbar stress to tensile-positive Kelvin GPa."""

    value = torch.as_tensor(stress, dtype=torch.float64)
    if value.shape != (3, 3) or not bool(torch.isfinite(value).all()):
        raise ValueError("LeMat stress must be finite 3x3 kbar")
    symmetric = 0.5 * (value + value.T)
    return symmetric_cartesian_to_kelvin((-0.1 * symmetric).unsqueeze(0)).squeeze(0)


def parse_lemat_row(
    row: Mapping[str, Any],
    *,
    physical_label_policy: LeMatPhysicalLabelPolicy = "compatible_only",
) -> MatPESPhysicalRecord:
    """Parse one ordered bulk row; missing labels are masked, never imputed."""

    if physical_label_policy not in {"compatible_only", "all_with_functional"}:
        raise ValueError("unknown LeMat physical-label policy")
    geometry = _parse_lemat_geometry(row)
    node_count = geometry.element_tokens.numel()

    compatible = bool(row.get("cross_compatibility"))
    labels_allowed = compatible or physical_label_policy == "all_with_functional"
    energy_value = row.get("energy")
    energy = torch.zeros((), dtype=torch.float32)
    energy_present = False
    if labels_allowed and energy_value is not None:
        energy = torch.as_tensor(float(energy_value) / node_count, dtype=torch.float32)
        if not bool(torch.isfinite(energy)):
            raise ValueError("LeMat energy is non-finite")
        energy_present = True

    force_value = row.get("forces")
    forces = torch.zeros((node_count, 3), dtype=torch.float32)
    forces_present = False
    if labels_allowed and force_value is not None:
        candidate = torch.as_tensor(force_value, dtype=torch.float32)
        if candidate.shape == (node_count, 3):
            if not bool(torch.isfinite(candidate).all()):
                raise ValueError("LeMat forces are non-finite")
            forces = candidate
            forces_present = True

    stress_value = row.get("stress_tensor")
    stress = torch.zeros(6, dtype=torch.float32)
    stress_present = False
    if labels_allowed and stress_value is not None:
        stress = lemat_stress_kbar_to_kelvin_gpa(stress_value).float()
        stress_present = True

    return MatPESPhysicalRecord(
        material_id=geometry.material_id,
        functional=geometry.functional,
        element_tokens=geometry.element_tokens,
        fractional_coordinates=geometry.fractional_coordinates,
        lattice=geometry.lattice,
        energy_per_atom_ev=energy,
        forces_ev_per_angstrom=forces,
        stress_kelvin_gpa=stress,
        energy_present=energy_present,
        forces_present=forces_present,
        stress_present=stress_present,
    )


def parse_lemat_structure_row(row: Mapping[str, Any]) -> MatPESPhysicalRecord:
    """Parse only geometry for LeMat generative replay and mask all labels."""

    geometry = _parse_lemat_geometry(row)
    node_count = geometry.element_tokens.numel()
    return MatPESPhysicalRecord(
        material_id=geometry.material_id,
        functional=geometry.functional,
        element_tokens=geometry.element_tokens,
        fractional_coordinates=geometry.fractional_coordinates,
        lattice=geometry.lattice,
        energy_per_atom_ev=torch.zeros((), dtype=torch.float32),
        forces_ev_per_angstrom=torch.zeros((node_count, 3), dtype=torch.float32),
        stress_kelvin_gpa=torch.zeros(6, dtype=torch.float32),
        energy_present=False,
        forces_present=False,
        stress_present=False,
    )


def lemat_split_group(row: Mapping[str, Any]) -> str:
    """Prefer the cross-source structural fingerprint and fail if both keys are absent."""

    fingerprint = row.get("entalpic_fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        return fingerprint
    material_id = row.get("immutable_id")
    if isinstance(material_id, str) and material_id:
        return normalize_external_material_id(material_id)
    raise ValueError("LeMat row has no split grouping key")


def validate_lemat_physical_label_policy(value: str) -> LeMatPhysicalLabelPolicy:
    if value not in {"compatible_only", "all_with_functional"}:
        raise ValueError("unknown LeMat physical-label policy")
    return cast(LeMatPhysicalLabelPolicy, value)
