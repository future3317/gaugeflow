"""Typed MatPES records for post-A1 physical representation training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from pymatgen.core import Element

from gaugeflow.vocabulary import atomic_numbers_to_tokens

from .physical_pretraining import symmetric_cartesian_to_kelvin


@dataclass(frozen=True)
class MatPESPhysicalRecord:
    material_id: str
    functional: str
    element_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    energy_per_atom_ev: torch.Tensor
    forces_ev_per_angstrom: torch.Tensor
    stress_kelvin_gpa: torch.Tensor
    energy_present: bool
    forces_present: bool
    stress_present: bool


def matpes_stress_kbar_to_kelvin_gpa(stress: Sequence[float]) -> torch.Tensor:
    """Convert compressive-positive MatPES Voigt kbar to tensile-positive Kelvin GPa."""

    value = torch.as_tensor(stress, dtype=torch.float64)
    if value.shape != (6,) or not bool(torch.isfinite(value).all()):
        raise ValueError("MatPES stress must be finite [xx,yy,zz,yz,xz,xy] kbar")
    xx, yy, zz, yz, xz, xy = (-0.1 * value).unbind()
    full = torch.stack(
        (
            torch.stack((xx, xy, xz)),
            torch.stack((xy, yy, yz)),
            torch.stack((xz, yz, zz)),
        )
    ).unsqueeze(0)
    return symmetric_cartesian_to_kelvin(full).squeeze(0)


def _optional_finite_tensor(value: object, shape: tuple[int, ...]) -> tuple[torch.Tensor, bool]:
    if value is None:
        return torch.zeros(shape, dtype=torch.float32), False
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != shape or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"physical label must be finite with shape {shape}")
    return tensor, True


def parse_matpes_row(row: Mapping[str, Any]) -> MatPESPhysicalRecord:
    """Parse one ordered, fully occupied MatPES JSONL row without target imputation."""

    structure = row.get("structure")
    if not isinstance(structure, Mapping):
        raise ValueError("MatPES row lacks a structure object")
    lattice_value = structure.get("lattice")
    sites = structure.get("sites")
    if not isinstance(lattice_value, Mapping) or not isinstance(sites, list) or not sites:
        raise ValueError("MatPES structure lacks lattice or sites")
    lattice = torch.as_tensor(lattice_value.get("matrix"), dtype=torch.float32)
    if lattice.shape != (3, 3) or not bool(torch.isfinite(lattice).all()):
        raise ValueError("MatPES lattice must be finite 3x3")
    if float(torch.linalg.det(lattice)) <= 0.0:
        raise ValueError("MatPES lattice must have positive row-basis volume")
    atomic_numbers: list[int] = []
    fractional: list[list[float]] = []
    for site in sites:
        if not isinstance(site, Mapping):
            raise ValueError("MatPES site must be an object")
        species = site.get("species")
        coordinates = site.get("abc")
        if not isinstance(species, list) or len(species) != 1 or not isinstance(species[0], Mapping):
            raise ValueError("physical pretraining accepts only one fully occupied species per site")
        occupancy = float(species[0].get("occu", float("nan")))
        symbol = species[0].get("element")
        if occupancy != 1.0 or not isinstance(symbol, str):
            raise ValueError("physical pretraining excludes partial occupancy and disorder")
        coordinate = torch.as_tensor(coordinates, dtype=torch.float32)
        if coordinate.shape != (3,) or not bool(torch.isfinite(coordinate).all()):
            raise ValueError("MatPES fractional coordinate must be finite length three")
        atomic_numbers.append(int(Element(symbol).Z))
        fractional.append(coordinate.tolist())
    node_count = len(atomic_numbers)
    declared_count = int(row.get("nsites", node_count))
    if declared_count != node_count:
        raise ValueError("MatPES declared and structural site counts disagree")

    energy_value = row.get("energy")
    if energy_value is None:
        energy = torch.zeros((), dtype=torch.float32)
        energy_present = False
    else:
        energy = torch.as_tensor(float(energy_value) / node_count, dtype=torch.float32)
        energy_present = bool(torch.isfinite(energy))
        if not energy_present:
            raise ValueError("MatPES energy is non-finite")
    forces, forces_present = _optional_finite_tensor(row.get("forces"), (node_count, 3))
    stress_value = row.get("stress")
    if stress_value is None:
        stress = torch.zeros(6, dtype=torch.float32)
        stress_present = False
    else:
        stress = matpes_stress_kbar_to_kelvin_gpa(stress_value).float()
        stress_present = True
    functional = row.get("functional")
    material_id = row.get("matpes_id")
    if not isinstance(functional, str) or not isinstance(material_id, str):
        raise ValueError("MatPES row lacks functional or stable material ID")
    return MatPESPhysicalRecord(
        material_id=material_id,
        functional=functional,
        element_tokens=atomic_numbers_to_tokens(torch.tensor(atomic_numbers, dtype=torch.long)),
        fractional_coordinates=torch.tensor(fractional, dtype=torch.float32),
        lattice=lattice,
        energy_per_atom_ev=energy,
        forces_ev_per_angstrom=forces,
        stress_kelvin_gpa=stress,
        energy_present=energy_present,
        forces_present=forces_present,
        stress_present=stress_present,
    )
