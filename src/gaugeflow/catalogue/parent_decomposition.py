"""Concrete parent realization and compact mode decomposition for H0-E.

This module is an offline data compiler.  It does not expose paired child
metadata to the generative model and never materializes a dense 3N by 3N
displacement representation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from math import prod
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from gaugeflow.catalogue.affine_quotient import (
    AffineQuotient,
    CompactDisplacementAction,
    PrimitiveSpaceGroup,
    build_compact_displacement_action,
    real_irrep_multiplicity,
)
from gaugeflow.catalogue.finite_group import (
    RealIrrep,
    canonical_stabilizer_key,
    enumerate_opd_classes,
    enumerate_real_irreps,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class StandardCrystal:
    lattice: FloatArray
    fractional: FloatArray
    species: IntArray
    space_group: int
    rotations: IntArray
    translations: FloatArray


@dataclass(frozen=True)
class ParentCandidate:
    parent: StandardCrystal
    child: StandardCrystal
    supercell_hnf: IntArray
    expanded_parent_fractional: FloatArray
    child_fractional_aligned: FloatArray
    child_lattice_aligned: FloatArray
    expanded_species: IntArray
    construction: str
    source_max_displacement_angstrom: float
    source_hencky_norm: float
    symprec: float


@dataclass(frozen=True)
class ComponentResult:
    sector: str
    irrep_key: str
    dimension: int
    multiplicity: int
    energy: float
    branch_key: str | None
    stabilizer_size: int
    stabilizer: tuple[int, ...]
    values: FloatArray


@dataclass(frozen=True)
class DecompositionResult:
    parent_space_group: int
    child_space_group: int
    supercell_hnf: IntArray
    supercell_index: int
    parent_construction: str
    source_max_displacement_angstrom: float
    source_hencky_norm: float
    symprec: float
    periodic_rms_angstrom: float
    residual_rms_angstrom: float
    top2_energy_fraction: float
    terminal_space_group: int
    terminal_space_group_agrees: bool
    structure_matcher_agrees: bool
    occurrence_integral: bool
    opd_mapping_complete: bool
    stabilizer_size: int
    physical_class_key: str
    active_components: tuple[ComponentResult, ...]


def _right_handed(
    lattice: FloatArray, fractional: FloatArray
) -> tuple[FloatArray, FloatArray]:
    if np.linalg.det(lattice) > 0:
        return lattice, fractional
    changed_lattice = lattice.copy()
    changed_fractional = fractional.copy()
    changed_lattice[0] *= -1.0
    changed_fractional[:, 0] *= -1.0
    changed_fractional %= 1.0
    return changed_lattice, changed_fractional


def standardize_crystal(
    lattice: FloatArray,
    fractional: FloatArray,
    species: IntArray,
    *,
    symprec: float,
    angle_tolerance: float,
    no_idealize: bool,
) -> StandardCrystal:
    """Return one primitive standardized crystal and its actual operations."""
    import spglib

    standardized = spglib.standardize_cell(
        (
            np.asarray(lattice, dtype=np.float64),
            np.asarray(fractional, dtype=np.float64),
            np.asarray(species, dtype=np.int32),
        ),
        to_primitive=True,
        no_idealize=no_idealize,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if standardized is None:
        raise ValueError("spglib failed to standardize the crystal")
    selected_lattice, selected_fractional, selected_species = standardized
    selected_lattice, selected_fractional = _right_handed(
        np.asarray(selected_lattice, dtype=np.float64),
        np.asarray(selected_fractional, dtype=np.float64),
    )
    dataset = spglib.get_symmetry_dataset(
        (selected_lattice, selected_fractional, selected_species),
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if dataset is None:
        raise ValueError("spglib failed to identify standardized symmetry")
    return StandardCrystal(
        lattice=selected_lattice,
        fractional=selected_fractional % 1.0,
        species=np.asarray(selected_species, dtype=np.int64),
        space_group=int(dataset.number),
        rotations=np.asarray(dataset.rotations, dtype=np.int64),
        translations=np.asarray(dataset.translations, dtype=np.float64),
    )


def _as_structure(crystal: StandardCrystal):
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice(crystal.lattice),
        crystal.species.tolist(),
        crystal.fractional,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def _composition_index(child: IntArray, parent: IntArray) -> int | None:
    child_counts = Counter(map(int, child))
    parent_counts = Counter(map(int, parent))
    if set(child_counts) != set(parent_counts):
        return None
    ratios = {
        child_counts[number] / parent_counts[number]
        for number in child_counts
        if parent_counts[number] > 0
    }
    if len(ratios) != 1:
        return None
    ratio = ratios.pop()
    rounded = int(round(ratio))
    return rounded if 1 <= rounded <= 4 and np.isclose(ratio, rounded) else None


def _matcher(settings: dict[str, float | bool], *, attempt_supercell: bool):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(
        ltol=float(settings["ltol"]),
        stol=float(settings["stol"]),
        angle_tol=float(settings["angle_tol"]),
        primitive_cell=False,
        scale=bool(settings["scale"]),
        attempt_supercell=attempt_supercell,
        allow_subset=False,
    )


def _expanded_parent(
    parent: StandardCrystal, quotient: AffineQuotient
) -> tuple[FloatArray, IntArray]:
    inverse = np.linalg.inv(quotient.translations.supercell_matrix.astype(np.float64))
    fractional = (
        (
            parent.fractional[None, :, :]
            + quotient.translations.representatives[:, None, :]
        )
        @ inverse
    ).reshape(-1, 3)
    species = np.tile(parent.species, quotient.translations.order)
    return fractional % 1.0, species


def _candidate_from_parent(
    child: StandardCrystal,
    parent: StandardCrystal,
    *,
    matcher_settings: dict[str, float | bool],
    construction: str,
    symprec: float,
) -> ParentCandidate | None:
    """Certify one proposed parent with exact composition, HNF and site mapping."""
    from hsnf import row_style_hermite_normal_form
    from pymatgen.core import Lattice, Structure

    if not (
        len(parent.rotations) > len(child.rotations)
        or parent.species.size < child.species.size
    ):
        return None
    composition_index = _composition_index(child.species, parent.species)
    if composition_index is None:
        return None
    child_structure = _as_structure(child)
    try:
        transformation = _matcher(
            matcher_settings, attempt_supercell=True
        ).get_transformation(child_structure, _as_structure(parent))
    except (ValueError, TypeError):
        return None
    if transformation is None:
        return None
    supercell = np.rint(np.asarray(transformation[0], dtype=np.float64)).astype(
        np.int64
    )
    if not np.allclose(transformation[0], supercell, atol=1e-8, rtol=0.0):
        return None
    hnf, _ = row_style_hermite_normal_form(supercell)
    hnf = np.asarray(hnf, dtype=np.int64)
    determinant = int(round(np.linalg.det(hnf)))
    if determinant != composition_index or not 1 <= determinant <= 4:
        return None
    try:
        primitive_group = PrimitiveSpaceGroup.from_operations(
            parent.rotations, parent.translations
        )
        quotient = AffineQuotient.build(primitive_group, hnf)
    except (ValueError, RuntimeError):
        return None
    expanded_fractional, expanded_species = _expanded_parent(parent, quotient)
    expanded_structure = Structure(
        Lattice(hnf @ parent.lattice),
        expanded_species.tolist(),
        expanded_fractional,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )
    try:
        aligned_child = _matcher(
            matcher_settings, attempt_supercell=False
        ).get_s2_like_s1(expanded_structure, child_structure)
    except (ValueError, TypeError):
        return None
    if aligned_child is None or len(aligned_child) != len(expanded_structure):
        return None
    aligned_species = np.asarray(
        [int(site.specie.Z) for site in aligned_child], dtype=np.int64
    )
    if not np.array_equal(aligned_species, expanded_species):
        return None
    parent_supercell_lattice = hnf @ parent.lattice
    aligned_fractional = np.asarray(aligned_child.frac_coords, dtype=np.float64) % 1.0
    mapped_displacement = translation_quotient_displacement(
        expanded_fractional,
        aligned_fractional,
        parent_supercell_lattice,
        _atomic_masses(expanded_species),
    )
    source_max_displacement = float(
        np.linalg.norm(mapped_displacement, axis=1).max(initial=0.0)
    )
    source_hencky_norm = float(
        np.linalg.norm(
            _logarithmic_strain(
                parent_supercell_lattice,
                np.asarray(aligned_child.lattice.matrix, dtype=np.float64),
            )
        )
    )
    return ParentCandidate(
        parent=parent,
        child=child,
        supercell_hnf=hnf,
        expanded_parent_fractional=expanded_fractional,
        child_fractional_aligned=aligned_fractional,
        child_lattice_aligned=np.asarray(
            aligned_child.lattice.matrix, dtype=np.float64
        ),
        expanded_species=expanded_species,
        construction=construction,
        source_max_displacement_angstrom=source_max_displacement,
        source_hencky_norm=source_hencky_norm,
        symprec=float(symprec),
    )


def _candidate_key(candidate: ParentCandidate) -> tuple[object, ...]:
    return (
        candidate.parent.space_group,
        tuple(int(value) for value in candidate.supercell_hnf.ravel()),
        tuple(int(value) for value in candidate.parent.species),
        tuple(np.round(candidate.parent.fractional, 7).ravel()),
    )


def find_parent_candidates(
    lattice: FloatArray,
    fractional: FloatArray,
    species: IntArray,
    *,
    child_symprec: float,
    symprec_ladder: Iterable[float],
    angle_tolerance: float,
    matcher_settings: dict[str, float | bool],
) -> tuple[StandardCrystal, tuple[ParentCandidate, ...]]:
    """Find idealized higher-symmetry parents without a tensor condition."""
    child = standardize_crystal(
        lattice,
        fractional,
        species,
        symprec=child_symprec,
        angle_tolerance=angle_tolerance,
        no_idealize=True,
    )
    candidates: dict[tuple[object, ...], ParentCandidate] = {}
    for symprec in symprec_ladder:
        try:
            parent = standardize_crystal(
                lattice,
                fractional,
                species,
                symprec=float(symprec),
                angle_tolerance=angle_tolerance,
                no_idealize=False,
            )
        except ValueError:
            continue
        candidate = _candidate_from_parent(
            child,
            parent,
            matcher_settings=matcher_settings,
            construction="spglib_tolerance_ladder",
            symprec=float(symprec),
        )
        if candidate is None:
            continue
        key = _candidate_key(candidate)
        previous = candidates.get(key)
        if previous is None or candidate.symprec < previous.symprec:
            candidates[key] = candidate
    return child, tuple(
        sorted(
            candidates.values(),
            key=lambda value: (
                value.symprec,
                int(round(np.linalg.det(value.supercell_hnf))),
                value.parent.space_group,
            ),
        )
    )


@lru_cache(maxsize=118)
def _atomic_mass(atomic_number: int) -> float:
    """Return one immutable periodic-table mass without repeated Element parsing."""
    from pymatgen.core import Element

    return float(Element.from_Z(atomic_number).atomic_mass)


def _atomic_masses(species: IntArray) -> FloatArray:
    """Return masses in node order using the exact cached element lookup."""
    return np.fromiter(
        (_atomic_mass(int(number)) for number in species),
        dtype=np.float64,
        count=int(species.size),
    )


def translation_quotient_displacement(
    parent_fractional: FloatArray,
    child_fractional: FloatArray,
    lattice: FloatArray,
    masses: FloatArray,
) -> FloatArray:
    """Return exact closest-image displacements with zero mass-weighted mean."""
    from gaugeflow.geometry import closest_image_displacements_numpy

    parent = np.asarray(parent_fractional, dtype=np.float64)
    child = np.asarray(child_fractional, dtype=np.float64)
    cell = np.asarray(lattice, dtype=np.float64)
    inverse = np.linalg.inv(cell)
    translation_cartesian = np.zeros(3, dtype=np.float64)
    displacements = np.zeros_like(parent)
    for _ in range(12):
        translation_fractional = translation_cartesian @ inverse
        displacements, _ = closest_image_displacements_numpy(
            child - parent - translation_fractional, cell
        )
        correction = (masses[:, None] * displacements).sum(axis=0) / masses.sum()
        translation_cartesian += correction
        if np.linalg.norm(correction) <= 1e-12:
            break
    displacements -= (masses[:, None] * displacements).sum(axis=0) / masses.sum()
    return displacements


def _irrep_key(group_keys: tuple[str, ...], irrep: RealIrrep) -> str:
    payload = "|".join(
        f"{key}:{character:.10g}"
        for key, character in zip(group_keys, irrep.character_key, strict=True)
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _isotypic_component(
    action: CompactDisplacementAction,
    irrep: RealIrrep,
    transformed: FloatArray,
) -> FloatArray:
    character = np.trace(irrep.matrices, axis1=1, axis2=2)
    norm = float(np.mean(character * character))
    coefficient = irrep.dimension / (norm * action.group.order)
    return coefficient * np.einsum(
        "g,gnc->nc", character[action.group.inverses], transformed, optimize=True
    )


def _kelvin_basis() -> FloatArray:
    basis = np.zeros((6, 3, 3), dtype=np.float64)
    basis[0, 0, 0] = basis[1, 1, 1] = basis[2, 2, 2] = 1.0
    scale = 1.0 / np.sqrt(2.0)
    basis[3, 1, 2] = basis[3, 2, 1] = scale
    basis[4, 0, 2] = basis[4, 2, 0] = scale
    basis[5, 0, 1] = basis[5, 1, 0] = scale
    return basis


def _strain_representation(action: CompactDisplacementAction) -> FloatArray:
    basis = _kelvin_basis()
    rotations = action.cartesian_rotations
    transformed = np.einsum(
        "gik,bkl,gjl->gbij", rotations, basis, rotations, optimize=True
    )
    representation = np.einsum("aij,gbij->gab", basis, transformed, optimize=True)
    generators = np.asarray(action.group.generators(), dtype=np.int64)
    products = np.einsum(
        "aij,hjk->ahik",
        representation[generators],
        representation,
        optimize=True,
    )
    if not np.allclose(
        products,
        representation[action.group.table[generators]],
        atol=1e-9,
        rtol=1e-9,
    ):
        raise RuntimeError("Kelvin strain action violates the affine quotient group law")
    return representation


def _representation_multiplicity(
    character: FloatArray, irrep: RealIrrep
) -> int:
    irrep_character = np.trace(irrep.matrices, axis1=1, axis2=2)
    norm = float(np.mean(irrep_character * irrep_character))
    inner = float(np.mean(character * irrep_character))
    multiplicity = int(round(inner / norm))
    if multiplicity < 0 or not np.isclose(inner, multiplicity * norm, atol=1e-8, rtol=1e-8):
        raise RuntimeError("Cartesian strain character has nonintegral irrep occurrence")
    return multiplicity


def _vector_isotypic_component(
    group_order: int,
    inverses: IntArray,
    irrep: RealIrrep,
    transformed: FloatArray,
) -> FloatArray:
    character = np.trace(irrep.matrices, axis1=1, axis2=2)
    norm = float(np.mean(character * character))
    coefficient = irrep.dimension / (norm * group_order)
    return coefficient * np.einsum(
        "g,gi->i", character[inverses], transformed, optimize=True
    )


def _vector_stabilizer(
    group_identity: int,
    representation: FloatArray,
    values: FloatArray,
    *,
    tolerance: float = 1e-8,
) -> tuple[int, ...]:
    transformed = np.einsum("gij,j->gi", representation, values, optimize=True)
    error = np.linalg.norm(transformed - values[None, :], axis=1)
    selected = tuple(int(value) for value in np.flatnonzero(error <= tolerance))
    if group_identity not in selected:
        raise RuntimeError("strain-component stabilizer lost the affine identity")
    return selected


def _logarithmic_strain(parent_lattice: FloatArray, child_lattice: FloatArray) -> FloatArray:
    """Return the symmetric Hencky strain in the parent Cartesian frame.

    Lattices use row vectors.  If ``P A = C``, the column-vector deformation
    gradient is ``F=A.T`` and the right Cauchy--Green tensor is ``A A.T``.
    The logarithmic strain is therefore ``0.5 log(A A.T)``.  The spectral
    implementation is exact for the symmetric positive-definite tensor and
    avoids a general complex-valued matrix logarithm.
    """
    deformation = np.linalg.solve(parent_lattice, child_lattice)
    metric = deformation @ deformation.T
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if np.min(eigenvalues) <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("parent-to-child metric must be positive definite")
    return 0.5 * np.einsum(
        "ia,a,ja->ij", eigenvectors, np.log(eigenvalues), eigenvectors, optimize=True
    )


def _metric_invariant(
    rotations: FloatArray,
    metric: FloatArray,
    selected: tuple[int, ...],
    *,
    relative_tolerance: float,
) -> bool:
    """Test a declared subgroup against the full continuous lattice strain."""
    indices = np.asarray(selected, dtype=np.int64)
    changed = np.einsum(
        "gik,kl,gjl->gij",
        rotations[indices],
        metric,
        rotations[indices],
        optimize=True,
    )
    relative = np.linalg.norm(changed - metric[None, :, :], axis=(1, 2)) / max(
        np.linalg.norm(metric), 1e-12
    )
    return bool(np.all(relative <= relative_tolerance))


def _terminal_evaluation(
    candidate: ParentCandidate,
    parent_lattice: FloatArray,
    parent_lattice_inverse: FloatArray,
    masses: FloatArray,
    predicted: FloatArray,
    *,
    terminal_symprec: float,
    angle_tolerance: float,
    matcher_settings: dict[str, float | bool],
) -> tuple[float, int, bool, bool]:
    import spglib
    from pymatgen.core import Lattice, Structure

    difference_parent = translation_quotient_displacement(
        candidate.expanded_parent_fractional
        + predicted @ parent_lattice_inverse,
        candidate.child_fractional_aligned,
        parent_lattice,
        masses,
    )
    difference_fractional = difference_parent @ parent_lattice_inverse
    difference_child = difference_fractional @ candidate.child_lattice_aligned
    periodic_rms = float(np.sqrt(np.mean(np.sum(difference_child**2, axis=1))))
    predicted_fractional = (
        candidate.expanded_parent_fractional
        + predicted @ parent_lattice_inverse
    ) % 1.0
    terminal_dataset = spglib.get_symmetry_dataset(
        (
            candidate.child_lattice_aligned,
            predicted_fractional,
            candidate.expanded_species.astype(np.int32),
        ),
        symprec=terminal_symprec,
        angle_tolerance=angle_tolerance,
    )
    terminal_space_group = int(terminal_dataset.number) if terminal_dataset else 0
    predicted_structure = Structure(
        Lattice(candidate.child_lattice_aligned),
        candidate.expanded_species.tolist(),
        predicted_fractional,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )
    child_structure = Structure(
        Lattice(candidate.child_lattice_aligned),
        candidate.expanded_species.tolist(),
        candidate.child_fractional_aligned,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )
    structure_agrees = bool(
        _matcher(matcher_settings, attempt_supercell=False).fit(
            predicted_structure, child_structure
        )
    )
    return (
        periodic_rms,
        terminal_space_group,
        bool(terminal_space_group == candidate.child.space_group),
        structure_agrees,
    )


def _compact_reynolds_residual(
    transformed: FloatArray,
    displacement_orbits: dict[str, FloatArray],
    subset: tuple[ComponentResult, ...],
    selected: tuple[int, ...],
    masses: FloatArray,
) -> FloatArray:
    """Project the unmodelled displacement without a dense representation.

    Linearity of the Reynolds operator gives

    ``mean_g (g.d - sum_j g.d_j) = mean_g(g.d) - sum_j mean_g(g.d_j)``.

    Evaluating those reductions directly is mathematically identical to
    copying a full ``[|G|, N, 3]`` residual orbit for every one/two-mode
    subset, but avoids that repeated allocation.
    """
    indices = np.asarray(selected, dtype=np.int64)
    residual = transformed[indices].mean(axis=0)
    selected_orbits = [
        displacement_orbits[value.irrep_key][indices]
        for value in subset
        if value.sector == "displacement"
    ]
    if selected_orbits:
        residual -= np.stack(selected_orbits, axis=0).mean(axis=1).sum(axis=0)
    residual -= (masses[:, None] * residual).sum(axis=0) / masses.sum()
    return residual


def decompose_parent_candidate(
    candidate: ParentCandidate,
    *,
    residual_rms_limit: float,
    stabilizer_rms_tolerance: float,
    stabilizer_metric_tolerance: float,
    displacement_energy_floor: float,
    strain_energy_floor: float,
    terminal_symprec: float,
    angle_tolerance: float,
    matcher_settings: dict[str, float | bool],
) -> DecompositionResult:
    """Decompose one child into at most two displacement-or-strain OPDs.

    The occurrence space is the direct sum of the compact atomic-displacement
    representation and the six-dimensional Kelvin representation of symmetric
    logarithmic strain.  Candidate subsets use the exact intersection of their
    component stabilizers; the observed child group is never used to project a
    residual and therefore cannot leak the answer into the decomposition.
    """
    primitive_group = PrimitiveSpaceGroup.from_operations(
        candidate.parent.rotations, candidate.parent.translations
    )
    quotient = AffineQuotient.build(primitive_group, candidate.supercell_hnf)
    action = build_compact_displacement_action(
        quotient,
        candidate.parent.lattice,
        candidate.parent.fractional,
        candidate.parent.species,
    )
    if not np.array_equal(candidate.expanded_species, np.tile(
        candidate.parent.species, quotient.translations.order
    )):
        raise RuntimeError("candidate node order does not match compact displacement action")
    parent_lattice = candidate.supercell_hnf @ candidate.parent.lattice
    parent_lattice_inverse = np.linalg.inv(parent_lattice)
    masses = _atomic_masses(candidate.expanded_species)
    displacement = translation_quotient_displacement(
        candidate.expanded_parent_fractional,
        candidate.child_fractional_aligned,
        parent_lattice,
        masses,
    )
    transformed = action.apply(displacement)
    component_stabilizer_tolerance = min(float(stabilizer_rms_tolerance), 1e-8)
    if component_stabilizer_tolerance <= 0.0:
        raise ValueError("component stabilizer tolerance must be positive")
    irreps = enumerate_real_irreps(action.group)
    components: list[ComponentResult] = []
    displacement_orbits: dict[str, FloatArray] = {}
    branch_maps: dict[str, dict[str, str]] = {}

    def branch_key(irrep: RealIrrep, stabilizer: tuple[int, ...]) -> str | None:
        key = _irrep_key(action.group.element_keys, irrep)
        mapping = branch_maps.get(key)
        if mapping is None:
            mapping = {
                canonical_stabilizer_key(action.group, branch.stabilizer): sha256(
                    branch.physical_key.encode("utf-8")
                ).hexdigest()
                for branch in enumerate_opd_classes(action.group, irrep)
            }
            branch_maps[key] = mapping
        return mapping.get(canonical_stabilizer_key(action.group, stabilizer))

    reconstructed_displacement = np.zeros_like(displacement)

    # Atomic displacement sector.  Central idempotents are evaluated from the
    # one already-vectorized orbit tensor ``transformed``; no dense 3N x 3N
    # representation is ever materialized.
    for irrep in irreps:
        multiplicity = real_irrep_multiplicity(action, irrep)
        if multiplicity <= 0:
            continue
        component = _isotypic_component(action, irrep, transformed)
        reconstructed_displacement += component
        energy = float(np.sum(masses[:, None] * component * component))
        if energy <= displacement_energy_floor:
            continue
        orbit = action.apply(component)
        component_rms = np.sqrt(
            np.mean((orbit - component[None, :, :]) ** 2, axis=(1, 2))
        )
        stabilizer = tuple(
            int(value)
            for value in np.flatnonzero(
                component_rms <= component_stabilizer_tolerance
            )
        )
        irrep_key = _irrep_key(action.group.element_keys, irrep)
        displacement_orbits[irrep_key] = orbit
        components.append(
            ComponentResult(
                sector="displacement",
                irrep_key=irrep_key,
                dimension=irrep.dimension,
                multiplicity=multiplicity,
                energy=energy,
                branch_key=branch_key(irrep, stabilizer),
                stabilizer_size=len(stabilizer),
                stabilizer=stabilizer,
                values=component,
            )
        )

    # Symmetric strain sector in an orthonormal Kelvin basis.  This is the
    # minimal six-coordinate Cartesian representation of a homogeneous metric
    # distortion and is mathematically equivalent to a symmetric 3 x 3 tensor.
    strain_tensor = _logarithmic_strain(
        parent_lattice, candidate.child_lattice_aligned
    )
    kelvin_basis = _kelvin_basis()
    strain = np.einsum("aij,ij->a", kelvin_basis, strain_tensor, optimize=True)
    strain_representation = _strain_representation(action)
    transformed_strain = np.einsum(
        "gij,j->gi", strain_representation, strain, optimize=True
    )
    strain_character = np.trace(strain_representation, axis1=1, axis2=2)
    reconstructed_strain = np.zeros_like(strain)
    for irrep in irreps:
        multiplicity = _representation_multiplicity(strain_character, irrep)
        if multiplicity <= 0:
            continue
        component = _vector_isotypic_component(
            action.group.order,
            action.group.inverses,
            irrep,
            transformed_strain,
        )
        reconstructed_strain += component
        energy = float(component @ component)
        if energy <= strain_energy_floor:
            continue
        stabilizer = _vector_stabilizer(
            action.group.identity, strain_representation, component
        )
        components.append(
            ComponentResult(
                sector="strain",
                irrep_key=_irrep_key(action.group.element_keys, irrep),
                dimension=irrep.dimension,
                multiplicity=multiplicity,
                energy=energy,
                branch_key=branch_key(irrep, stabilizer),
                stabilizer_size=len(stabilizer),
                stabilizer=stabilizer,
                values=component,
            )
        )

    displacement_error = float(
        np.sqrt(np.mean((reconstructed_displacement - displacement) ** 2))
    )
    strain_error = float(np.linalg.norm(reconstructed_strain - strain))
    occurrence_integral = bool(displacement_error <= 1e-8 and strain_error <= 1e-8)
    components.sort(key=lambda value: (value.sector, -value.energy, value.irrep_key))
    eligible = tuple(value for value in components if value.branch_key is not None)

    sector_totals = {
        sector: sum(value.energy for value in components if value.sector == sector)
        for sector in ("displacement", "strain")
    }
    deformation = np.linalg.solve(parent_lattice, candidate.child_lattice_aligned)
    metric = deformation @ deformation.T
    evaluated: list[
        tuple[
            tuple[float | int | str, ...],
            tuple[ComponentResult, ...],
            tuple[int, ...],
            float,
            int,
            bool,
            bool,
            float,
            float,
        ]
    ] = []
    for count in (1, 2):
        for subset in combinations(eligible, count):
            declared = tuple(
                sorted(set(subset[0].stabilizer).intersection(*(value.stabilizer for value in subset[1:])))
            )
            if not declared or action.group.identity not in declared:
                continue
            if not _metric_invariant(
                action.cartesian_rotations,
                metric,
                declared,
                relative_tolerance=stabilizer_metric_tolerance,
            ):
                continue
            displacement_modes = [
                value.values for value in subset if value.sector == "displacement"
            ]
            mode = sum(displacement_modes, start=np.zeros_like(displacement))
            residual = _compact_reynolds_residual(
                transformed,
                displacement_orbits,
                subset,
                declared,
                masses,
            )
            residual_rms = float(
                np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
            )
            predicted = mode + residual
            periodic_rms, terminal_group, terminal_agrees, structure_agrees = (
                _terminal_evaluation(
                    candidate,
                    parent_lattice,
                    parent_lattice_inverse,
                    masses,
                    predicted,
                    terminal_symprec=terminal_symprec,
                    angle_tolerance=angle_tolerance,
                    matcher_settings=matcher_settings,
                )
            )
            fractions = [
                sum(value.energy for value in subset if value.sector == sector)
                / total
                for sector, total in sector_totals.items()
                if total
                > (
                    displacement_energy_floor
                    if sector == "displacement"
                    else strain_energy_floor
                )
            ]
            explained = float(np.mean(fractions)) if fractions else 1.0
            deterministic = "|".join(
                f"{value.sector}:{value.irrep_key}:{value.branch_key}" for value in subset
            )
            ordering: tuple[float | int | str, ...] = (
                0 if residual_rms <= residual_rms_limit else 1,
                periodic_rms,
                0 if terminal_agrees else 1,
                residual_rms,
                -explained,
                count,
                deterministic,
            )
            evaluated.append(
                (
                    ordering,
                    subset,
                    declared,
                    periodic_rms,
                    terminal_group,
                    terminal_agrees,
                    structure_agrees,
                    explained,
                    residual_rms,
                )
            )

    if evaluated:
        (
            _,
            active,
            declared_stabilizer,
            periodic_rms,
            terminal_space_group,
            terminal_agrees,
            structure_agrees,
            top_fraction,
            residual_rms,
        ) = min(evaluated, key=lambda value: value[0])
        opd_complete = all(value.branch_key is not None for value in active)
    else:
        active = ()
        declared_stabilizer = tuple(range(action.group.order))
        residual = transformed.mean(axis=0)
        residual -= (masses[:, None] * residual).sum(axis=0) / masses.sum()
        residual_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        periodic_rms, terminal_space_group, terminal_agrees, structure_agrees = (
            _terminal_evaluation(
                candidate,
                parent_lattice,
                parent_lattice_inverse,
                masses,
                residual,
                terminal_symprec=terminal_symprec,
                angle_tolerance=angle_tolerance,
                matcher_settings=matcher_settings,
            )
        )
        top_fraction = 0.0
        opd_complete = False

    physical_payload = (
        f"parent={candidate.parent.space_group}|child={candidate.child.space_group}"
        f"|B={','.join(map(str, candidate.supercell_hnf.ravel()))}"
        f"|sectors={','.join(value.sector for value in active)}"
        f"|irreps={','.join(value.irrep_key for value in active)}"
        f"|branches={','.join(str(value.branch_key) for value in active)}"
        f"|H={canonical_stabilizer_key(action.group, declared_stabilizer)}"
    )
    if residual_rms > residual_rms_limit + 1e-12:
        structure_agrees = False
    return DecompositionResult(
        parent_space_group=candidate.parent.space_group,
        child_space_group=candidate.child.space_group,
        supercell_hnf=candidate.supercell_hnf,
        supercell_index=int(round(np.linalg.det(candidate.supercell_hnf))),
        parent_construction=candidate.construction,
        source_max_displacement_angstrom=candidate.source_max_displacement_angstrom,
        source_hencky_norm=candidate.source_hencky_norm,
        symprec=candidate.symprec,
        periodic_rms_angstrom=periodic_rms,
        residual_rms_angstrom=residual_rms,
        top2_energy_fraction=top_fraction,
        terminal_space_group=terminal_space_group,
        terminal_space_group_agrees=bool(terminal_agrees),
        structure_matcher_agrees=bool(structure_agrees),
        occurrence_integral=bool(occurrence_integral),
        opd_mapping_complete=bool(opd_complete),
        stabilizer_size=len(declared_stabilizer),
        physical_class_key=sha256(physical_payload.encode("utf-8")).hexdigest(),
        active_components=active,
    )


def crystal_system(space_group: int) -> str:
    boundaries = (
        (2, "triclinic"),
        (15, "monoclinic"),
        (74, "orthorhombic"),
        (142, "tetragonal"),
        (167, "trigonal"),
        (194, "hexagonal"),
        (230, "cubic"),
    )
    for upper, label in boundaries:
        if space_group <= upper:
            return label
    raise ValueError("space-group number must lie in 1..230")


def site_count_bin(site_count: int, boundaries: Iterable[int]) -> str:
    limits = tuple(int(value) for value in boundaries)
    if site_count < 1 or any(left >= right for left, right in zip(limits, limits[1:])):
        raise ValueError("site-count bins must be increasing and count positive")
    for limit in limits:
        if site_count <= limit:
            return f"le_{limit}"
    return f"gt_{limits[-1]}"


def balanced_selection(
    records: Iterable[dict[str, object]],
    *,
    split_counts: dict[str, int],
    seed: int,
    site_boundaries: Iterable[int],
) -> tuple[dict[str, object], ...]:
    """Deterministically cycle over strata inside each frozen split quota."""
    by_split: dict[str, dict[tuple[str, str], list[dict[str, object]]]] = {}
    for record in records:
        split = str(record["gaugeflow_split"])
        if split not in split_counts:
            continue
        stratum = (
            crystal_system(int(record["space_group_number"])),
            site_count_bin(int(record["primitive_sites"]), site_boundaries),
        )
        by_split.setdefault(split, {}).setdefault(stratum, []).append(dict(record))
    selected: list[dict[str, object]] = []
    for split, quota in split_counts.items():
        strata = by_split.get(split, {})
        if not strata:
            raise ValueError(f"split {split!r} has no pilot candidates")
        for values in strata.values():
            values.sort(
                key=lambda value: sha256(
                    f"{seed}|{value['material_id']}".encode("utf-8")
                ).hexdigest()
            )
        ordered_strata = sorted(strata)
        offsets = {key: 0 for key in ordered_strata}
        split_selected = 0
        while split_selected < quota:
            progress = False
            for key in ordered_strata:
                offset = offsets[key]
                if offset >= len(strata[key]):
                    continue
                selected.append(strata[key][offset])
                offsets[key] += 1
                split_selected += 1
                progress = True
                if split_selected == quota:
                    break
            if not progress:
                raise ValueError(f"split {split!r} cannot satisfy frozen pilot quota")
    if len(selected) != sum(split_counts.values()):
        raise RuntimeError("balanced pilot selection returned the wrong size")
    if len({str(value["material_id"]) for value in selected}) != len(selected):
        raise RuntimeError("balanced pilot selection duplicated a material ID")
    return tuple(selected)


def hnf_key(matrix: IntArray) -> str:
    if matrix.shape != (3, 3) or prod(np.diag(matrix)) < 1:
        raise ValueError("HNF key requires a positive 3x3 matrix")
    return ",".join(str(int(value)) for value in matrix.ravel())
