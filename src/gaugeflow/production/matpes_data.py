"""Typed MatPES records for post-A1 physical representation training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
from pymatgen.core import Element

from gaugeflow.vocabulary import atomic_numbers_to_tokens

from .data_splitting import DataSplit, deterministic_iid_split
from .physical_pretraining import (
    FunctionalPhysicalNormalizer,
    PhysicalTargets,
    symmetric_cartesian_to_kelvin,
)


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
    teacher_features: torch.Tensor | None = None


@dataclass(frozen=True)
class MatPESPhysicalBatch:
    """Packed model inputs and explicitly masked physical targets."""

    element_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    batch: torch.Tensor
    functional_index: torch.Tensor
    targets: PhysicalTargets

    def pin_memory(self) -> MatPESPhysicalBatch:
        target = self.targets
        return MatPESPhysicalBatch(
            element_tokens=self.element_tokens.pin_memory(),
            fractional_coordinates=self.fractional_coordinates.pin_memory(),
            lattice=self.lattice.pin_memory(),
            batch=self.batch.pin_memory(),
            functional_index=self.functional_index.pin_memory(),
            targets=PhysicalTargets(
                energy_per_atom=target.energy_per_atom.pin_memory(),
                forces=target.forces.pin_memory(),
                stress_kelvin=target.stress_kelvin.pin_memory(),
                teacher_features=target.teacher_features.pin_memory(),
                energy_mask=target.energy_mask.pin_memory(),
                force_mask=target.force_mask.pin_memory(),
                stress_mask=target.stress_mask.pin_memory(),
                teacher_mask=target.teacher_mask.pin_memory(),
            ),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> MatPESPhysicalBatch:
        target = self.targets
        return MatPESPhysicalBatch(
            element_tokens=self.element_tokens.to(device, non_blocking=non_blocking),
            fractional_coordinates=self.fractional_coordinates.to(
                device, non_blocking=non_blocking
            ),
            lattice=self.lattice.to(device, non_blocking=non_blocking),
            batch=self.batch.to(device, non_blocking=non_blocking),
            functional_index=self.functional_index.to(device, non_blocking=non_blocking),
            targets=PhysicalTargets(
                energy_per_atom=target.energy_per_atom.to(device, non_blocking=non_blocking),
                forces=target.forces.to(device, non_blocking=non_blocking),
                stress_kelvin=target.stress_kelvin.to(device, non_blocking=non_blocking),
                teacher_features=target.teacher_features.to(device, non_blocking=non_blocking),
                energy_mask=target.energy_mask.to(device, non_blocking=non_blocking),
                force_mask=target.force_mask.to(device, non_blocking=non_blocking),
                stress_mask=target.stress_mask.to(device, non_blocking=non_blocking),
                teacher_mask=target.teacher_mask.to(device, non_blocking=non_blocking),
            ),
        )


MatPESEnergyTarget = Literal[
    "total_energy_per_atom",
    "cohesive_energy_per_atom",
    "formation_energy_per_atom",
]
MatPESSplit = DataSplit
matpes_iid_split = deterministic_iid_split


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


def parse_matpes_row(
    row: Mapping[str, Any],
    *,
    energy_target: MatPESEnergyTarget = "total_energy_per_atom",
) -> MatPESPhysicalRecord:
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

    if energy_target == "total_energy_per_atom":
        energy_value = row.get("energy")
        energy_divisor = node_count
    elif energy_target in {"cohesive_energy_per_atom", "formation_energy_per_atom"}:
        energy_value = row.get(energy_target)
        energy_divisor = 1
    else:
        raise ValueError(f"unsupported MatPES energy target {energy_target!r}")
    if energy_value is None:
        energy = torch.zeros((), dtype=torch.float32)
        energy_present = False
    else:
        energy = torch.as_tensor(float(energy_value) / energy_divisor, dtype=torch.float32)
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


def collate_matpes_records(
    records: Sequence[MatPESPhysicalRecord],
    *,
    functional_vocabulary: Mapping[str, int],
    teacher_dim: int,
) -> MatPESPhysicalBatch:
    """Pack variable-size records without exposing material identifiers to the model."""

    if not records:
        raise ValueError("cannot collate an empty MatPES batch")
    if teacher_dim < 1:
        raise ValueError("teacher feature dimension must be positive")
    expected_indices = set(range(len(functional_vocabulary)))
    if not functional_vocabulary or set(functional_vocabulary.values()) != expected_indices:
        raise ValueError("functional vocabulary indices must be contiguous from zero")
    counts = torch.tensor([record.element_tokens.numel() for record in records], dtype=torch.long)
    functional: list[int] = []
    for record in records:
        if record.functional not in functional_vocabulary:
            raise ValueError(f"unregistered MatPES functional {record.functional!r}")
        if record.teacher_features is not None:
            expected = (record.element_tokens.numel(), teacher_dim)
            if record.teacher_features.shape != expected or not bool(
                torch.isfinite(record.teacher_features).all()
            ):
                raise ValueError(
                    "teacher features must be finite with one vector per MatPES node"
                )
        functional.append(functional_vocabulary[record.functional])
    graph_count = len(records)
    return MatPESPhysicalBatch(
        element_tokens=torch.cat([record.element_tokens for record in records]),
        fractional_coordinates=torch.cat([record.fractional_coordinates for record in records]),
        lattice=torch.stack([record.lattice for record in records]),
        batch=torch.repeat_interleave(torch.arange(graph_count), counts),
        functional_index=torch.tensor(functional, dtype=torch.long),
        targets=PhysicalTargets(
            energy_per_atom=torch.stack([record.energy_per_atom_ev for record in records]),
            forces=torch.cat([record.forces_ev_per_angstrom for record in records]),
            stress_kelvin=torch.stack([record.stress_kelvin_gpa for record in records]),
            teacher_features=torch.cat(
                [
                    (
                        record.teacher_features
                        if record.teacher_features is not None
                        else torch.zeros(record.element_tokens.numel(), teacher_dim)
                    )
                    for record in records
                ]
            ),
            energy_mask=torch.tensor([record.energy_present for record in records]),
            force_mask=torch.repeat_interleave(
                torch.tensor([record.forces_present for record in records]), counts
            ),
            stress_mask=torch.tensor([record.stress_present for record in records]),
            teacher_mask=torch.cat(
                [
                    torch.full(
                        (record.element_tokens.numel(),),
                        record.teacher_features is not None,
                        dtype=torch.bool,
                    )
                    for record in records
                ]
            ),
        ),
    )


def fit_functional_physical_normalizer(
    records: Iterable[MatPESPhysicalRecord],
    *,
    functional_vocabulary: Mapping[str, int],
    minimum_scale: float = 1.0e-6,
) -> FunctionalPhysicalNormalizer:
    """Fit train-only streaming moments with covariance-preserving parameterization."""

    functionals = len(functional_vocabulary)
    if functionals < 1 or set(functional_vocabulary.values()) != set(range(functionals)):
        raise ValueError("functional vocabulary indices must be contiguous from zero")
    if minimum_scale <= 0.0:
        raise ValueError("minimum physical scale must be positive")
    energy_count = torch.zeros(functionals, dtype=torch.float64)
    energy_sum = torch.zeros_like(energy_count)
    energy_square_sum = torch.zeros_like(energy_count)
    force_component_count = torch.zeros_like(energy_count)
    force_square_sum = torch.zeros_like(energy_count)
    stress_count = torch.zeros_like(energy_count)
    stress_trace_sum = torch.zeros_like(energy_count)
    stress_norm_square_sum = torch.zeros_like(energy_count)
    for record in records:
        if record.functional not in functional_vocabulary:
            raise ValueError(f"unregistered MatPES functional {record.functional!r}")
        index = functional_vocabulary[record.functional]
        if record.energy_present:
            energy = record.energy_per_atom_ev.double()
            energy_count[index] += 1.0
            energy_sum[index] += energy
            energy_square_sum[index] += energy.square()
        if record.forces_present:
            forces = record.forces_ev_per_angstrom.double()
            force_component_count[index] += forces.numel()
            force_square_sum[index] += forces.square().sum()
        if record.stress_present:
            stress = record.stress_kelvin_gpa.double()
            stress_count[index] += 1.0
            stress_trace_sum[index] += stress[:3].sum()
            stress_norm_square_sum[index] += stress.square().sum()
    return _normalizer_from_sufficient_statistics(
        energy_count,
        energy_sum,
        energy_square_sum,
        force_component_count,
        force_square_sum,
        stress_count,
        stress_trace_sum,
        stress_norm_square_sum,
        minimum_scale=minimum_scale,
    )


def fit_functional_physical_normalizer_from_batches(
    batches: Iterable[MatPESPhysicalBatch],
    *,
    functional_vocabulary: Mapping[str, int],
    minimum_scale: float = 1.0e-6,
) -> FunctionalPhysicalNormalizer:
    """Fit the same moments with vectorized graph/node segment reductions."""

    functionals = len(functional_vocabulary)
    if functionals < 1 or set(functional_vocabulary.values()) != set(range(functionals)):
        raise ValueError("functional vocabulary indices must be contiguous from zero")
    if minimum_scale <= 0.0:
        raise ValueError("minimum physical scale must be positive")
    statistics = [torch.zeros(functionals, dtype=torch.float64) for _ in range(8)]
    for packed in batches:
        targets = packed.targets
        functional_index = packed.functional_index
        if functional_index.ndim != 1 or functional_index.dtype != torch.long:
            raise ValueError("batched physical statistics require graph functional indices")
        energy_index = functional_index[targets.energy_mask]
        energy = targets.energy_per_atom[targets.energy_mask].double()
        statistics[0] += torch.bincount(energy_index, minlength=functionals)
        statistics[1] += torch.bincount(energy_index, weights=energy, minlength=functionals)
        statistics[2] += torch.bincount(
            energy_index,
            weights=energy.square(),
            minlength=functionals,
        )

        force_index = functional_index[packed.batch][targets.force_mask]
        force_square = targets.forces[targets.force_mask].double().square().sum(dim=-1)
        statistics[3] += 3.0 * torch.bincount(force_index, minlength=functionals)
        statistics[4] += torch.bincount(
            force_index,
            weights=force_square,
            minlength=functionals,
        )

        stress_index = functional_index[targets.stress_mask]
        stress = targets.stress_kelvin[targets.stress_mask].double()
        statistics[5] += torch.bincount(stress_index, minlength=functionals)
        statistics[6] += torch.bincount(
            stress_index,
            weights=stress[:, :3].sum(dim=-1),
            minlength=functionals,
        )
        statistics[7] += torch.bincount(
            stress_index,
            weights=stress.square().sum(dim=-1),
            minlength=functionals,
        )
    return _normalizer_from_sufficient_statistics(
        *statistics,
        minimum_scale=minimum_scale,
    )


def _normalizer_from_sufficient_statistics(
    energy_count: torch.Tensor,
    energy_sum: torch.Tensor,
    energy_square_sum: torch.Tensor,
    force_component_count: torch.Tensor,
    force_square_sum: torch.Tensor,
    stress_count: torch.Tensor,
    stress_trace_sum: torch.Tensor,
    stress_norm_square_sum: torch.Tensor,
    *,
    minimum_scale: float,
) -> FunctionalPhysicalNormalizer:
    if bool((energy_count == 0).any()) or bool((force_component_count == 0).any()) or bool(
        (stress_count == 0).any()
    ):
        raise ValueError("each functional requires energy, force and stress support")
    energy_location = energy_sum / energy_count
    energy_variance = energy_square_sum / energy_count - energy_location.square()
    force_variance = force_square_sum / force_component_count
    stress_location = stress_trace_sum / (3.0 * stress_count)
    stress_residual_square = (
        stress_norm_square_sum
        - 2.0 * stress_location * stress_trace_sum
        + 3.0 * stress_count * stress_location.square()
    )
    stress_variance = stress_residual_square / (6.0 * stress_count)
    clamp = minimum_scale**2
    return FunctionalPhysicalNormalizer(
        energy_location=energy_location.float(),
        energy_scale=energy_variance.clamp_min(clamp).sqrt().float(),
        force_scale=force_variance.clamp_min(clamp).sqrt().float(),
        stress_isotropic_location=stress_location.float(),
        stress_scale=stress_variance.clamp_min(clamp).sqrt().float(),
    )
