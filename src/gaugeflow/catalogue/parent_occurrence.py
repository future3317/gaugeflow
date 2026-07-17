"""Setting-exact maximal-subgroup parent occurrence for offline H0-E data."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from gaugeflow.catalogue.parent_decomposition import (
    ParentCandidate,
    StandardCrystal,
    certify_parent_candidate,
)
from gaugeflow.catalogue.parent_projection import (
    GeometryParentProjection,
    ParentProjection,
    conjugate_embedding_to_primitive,
    conventional_to_primitive_structure,
    project_geometry_klassengleiche_parent,
    project_geometry_translationengleiche_parent,
    project_klassengleiche_parent,
    project_translationengleiche_parent,
)
from gaugeflow.catalogue.subgroup_embeddings import RationalAffineTransform
from gaugeflow.production.blueprint import OccupationalPattern
from gaugeflow.vocabulary import atomic_numbers_to_tokens

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PrimitiveSetting:
    space_group: int
    hall_number: int
    primitive_basis: FloatArray
    rotations: IntArray
    translations: FloatArray


@dataclass(frozen=True)
class EmbeddingParentOccurrence:
    embedding_key: str
    parent_space_group: int
    cell_index: int
    full_action_order: int
    parent_site_count: int
    projection: ParentProjection
    candidate: ParentCandidate


@dataclass(frozen=True)
class OccupationalParentOccurrence:
    """Species-free parent geometry plus an exact ordered child coloring."""

    embedding_key: str
    parent_space_group: int
    child_space_group: int
    cell_index: int
    full_action_order: int
    parent_site_count: int
    projection: GeometryParentProjection
    occupational_pattern: OccupationalPattern
    occupational_stabilizer_indices: IntArray
    child_operation_order: int
    child_atomic_numbers: IntArray

    @property
    def exact_coloring_reconstruction(self) -> bool:
        reconstructed = self.occupational_pattern.tokens.detach().cpu().numpy() + 1
        return bool(np.array_equal(reconstructed, self.child_atomic_numbers))

    @property
    def stabilizer_order_matches_child(self) -> bool:
        return int(self.occupational_stabilizer_indices.size) == self.child_operation_order


def _strict_geometry_parent_reidentified(
    projection: GeometryParentProjection,
    *,
    expected_space_group: int,
    angle_tolerance: float,
) -> bool:
    """Identify a carrier geometry without assigning it a physical element."""
    import spglib

    carrier_classes = np.ones(projection.fractional.shape[0], dtype=np.int32)
    identified = spglib.get_symmetry_dataset(
        (projection.lattice, projection.fractional, carrier_classes),
        symprec=1e-5,
        angle_tolerance=float(angle_tolerance),
    )
    return identified is not None and int(identified.number) == int(expected_space_group)


def _occupational_occurrence(
    child: StandardCrystal,
    record: Mapping[str, object],
    projection: GeometryParentProjection,
    *,
    full_action_order: int,
    maximum_source_hencky_norm: float,
    angle_tolerance: float,
) -> OccupationalParentOccurrence | None:
    """Attach the terminal coloring only after the full geometry action exists."""
    if projection.source_hencky_norm > maximum_source_hencky_norm:
        return None
    parent_space_group = int(record["parent_space_group"])
    if not _strict_geometry_parent_reidentified(
        projection,
        expected_space_group=parent_space_group,
        angle_tolerance=angle_tolerance,
    ):
        return None
    if projection.permutations.shape != (full_action_order, child.species.size):
        raise RuntimeError("geometry action does not cover the terminal coloring sites")
    tokens = atomic_numbers_to_tokens(torch.from_numpy(child.species.copy()))
    pattern = OccupationalPattern.from_tokens(tokens)
    permutations = torch.from_numpy(projection.permutations.copy())
    stabilizer = pattern.stabilizer_indices(permutations)
    if not pattern.stabilizer_is_subgroup(permutations, stabilizer):
        raise RuntimeError("occupational stabilizer is not a subgroup of the parent action")
    child_operation_order = int(child.rotations.shape[0])
    if stabilizer.numel() != child_operation_order:
        return None
    occurrence = OccupationalParentOccurrence(
        embedding_key=str(record["embedding_key"]),
        parent_space_group=parent_space_group,
        child_space_group=child.space_group,
        cell_index=int(record["cell_index"]),
        full_action_order=full_action_order,
        parent_site_count=int(projection.fractional.shape[0]),
        projection=projection,
        occupational_pattern=pattern,
        occupational_stabilizer_indices=stabilizer.detach().cpu().numpy(),
        child_operation_order=child_operation_order,
        child_atomic_numbers=child.species.copy(),
    )
    if not occurrence.exact_coloring_reconstruction:
        raise RuntimeError("occupational class encoding did not reconstruct terminal elements")
    return occurrence


@lru_cache(maxsize=230)
def pyxtal_primitive_setting(space_group: int) -> PrimitiveSetting:
    """Return the exact primitive form of the setting used by the E0 source.

    PyXtal is consulted only for its frozen standard-setting operations.  No
    per-material supergroup search, random generation or tolerance ladder is
    invoked.
    """
    import spglib
    from pyxtal.symmetry import Group
    from spgrep.symmetry.transform import (
        get_primitive_transformation_matrix,
        transform_symmetry_and_kpoint,
        unique_primitive_symmetry,
    )

    if not 1 <= int(space_group) <= 230:
        raise ValueError("space group must lie in 1..230")
    operations = [np.asarray(value.affine_matrix, dtype=np.float64) for value in Group(int(space_group))[0].ops]
    rotations = np.rint([value[:3, :3] for value in operations]).astype(np.int64)
    translations = np.asarray([value[:3, 3] for value in operations], dtype=np.float64)
    identified = spglib.get_spacegroup_type_from_symmetry(
        rotations,
        translations,
        np.eye(3),
        symprec=1e-6,
    )
    if identified is None or int(identified.number) != int(space_group):
        raise RuntimeError("PyXtal setting was not reidentified by spglib")
    hall_number = int(identified.hall_number)
    primitive_basis = get_primitive_transformation_matrix(hall_number)
    primitive_rotations, primitive_translations, _ = transform_symmetry_and_kpoint(
        primitive_basis,
        rotations,
        translations,
        np.zeros(3),
    )
    primitive_rotations, primitive_translations, _ = unique_primitive_symmetry(
        primitive_rotations, primitive_translations
    )
    return PrimitiveSetting(
        space_group=int(space_group),
        hall_number=hall_number,
        primitive_basis=np.asarray(primitive_basis, dtype=np.float64),
        rotations=np.asarray(primitive_rotations, dtype=np.int64),
        translations=np.asarray(primitive_translations, dtype=np.float64),
    )


def standardize_child_to_e0_setting(
    lattice: FloatArray,
    cartesian_positions: FloatArray,
    species: IntArray,
    *,
    expected_space_group: int,
    expected_primitive_sites: int,
    symprec: float,
    angle_tolerance: float,
) -> StandardCrystal:
    """Standardize one observed child in the exact E0 affine setting."""
    import spglib

    cell = np.asarray(lattice, dtype=np.float64)
    cartesian = np.asarray(cartesian_positions, dtype=np.float64)
    numbers = np.asarray(species, dtype=np.int64)
    if (
        cell.shape != (3, 3)
        or cartesian.ndim != 2
        or cartesian.shape[1] != 3
        or numbers.shape != (cartesian.shape[0],)
        or symprec <= 0.0
    ):
        raise ValueError("raw child arrays or tolerances are invalid")
    fractional = cartesian @ np.linalg.inv(cell)
    setting = pyxtal_primitive_setting(int(expected_space_group))
    dataset = spglib.get_symmetry_dataset(
        (cell, fractional, numbers.astype(np.int32)),
        hall_number=setting.hall_number,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    if dataset is None or int(dataset.number) != int(expected_space_group):
        raise ValueError("raw child does not match its frozen space-group setting")
    primitive_lattice, primitive_fractional, primitive_species = conventional_to_primitive_structure(
        np.asarray(dataset.std_lattice, dtype=np.float64),
        np.asarray(dataset.std_positions, dtype=np.float64),
        np.asarray(dataset.std_types, dtype=np.int64),
        setting.primitive_basis,
    )
    if primitive_species.size != int(expected_primitive_sites):
        raise ValueError("setting conversion changed the frozen primitive site count")
    strict = spglib.get_symmetry_dataset(
        (
            primitive_lattice,
            primitive_fractional,
            primitive_species.astype(np.int32),
        ),
        symprec=1e-5,
        angle_tolerance=float(angle_tolerance),
    )
    if strict is None or int(strict.number) != int(expected_space_group):
        raise ValueError("standardized primitive child failed strict reidentification")
    return StandardCrystal(
        lattice=primitive_lattice,
        fractional=primitive_fractional,
        species=primitive_species,
        space_group=int(expected_space_group),
        rotations=setting.rotations,
        translations=setting.translations,
    )


def _embedding_transform(record: Mapping[str, object]) -> RationalAffineTransform:
    denominator = int(record["transform_denominator"])
    numerators = np.asarray(record["transform_numerators"], dtype=np.int64)
    if denominator <= 0 or numerators.shape != (12,):
        raise ValueError("embedding record has invalid compact affine storage")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3] = numerators.reshape(3, 4) / denominator
    return RationalAffineTransform.from_array(matrix)


def project_maximal_t_embedding(
    child: StandardCrystal,
    record: Mapping[str, object],
    *,
    maximum_source_displacement_angstrom: float,
    matcher_settings: dict[str, float | bool],
    angle_tolerance: float,
) -> EmbeddingParentOccurrence | None:
    """Project and independently StructureMatcher-certify one E0 t edge."""
    import spglib

    if str(record["kind"]) != "t" or int(record["cell_index"]) != 1:
        raise ValueError("E1a accepts maximal translationengleiche records only")
    if int(record["child_space_group"]) != child.space_group:
        raise ValueError("embedding child group does not match the material")
    parent_space_group = int(record["parent_space_group"])
    parent_setting = pyxtal_primitive_setting(parent_space_group)
    child_setting = pyxtal_primitive_setting(child.space_group)
    primitive_embedding = conjugate_embedding_to_primitive(
        _embedding_transform(record),
        parent_setting.primitive_basis,
        child_setting.primitive_basis,
    )
    projection = project_translationengleiche_parent(
        child.lattice,
        child.fractional,
        child.species,
        parent_setting.rotations,
        parent_setting.translations,
        primitive_embedding,
        maximum_source_displacement_angstrom=(maximum_source_displacement_angstrom),
    )
    if projection is None:
        return None
    identified = spglib.get_symmetry_dataset(
        (
            projection.lattice,
            projection.fractional,
            projection.species.astype(np.int32),
        ),
        symprec=1e-5,
        angle_tolerance=float(angle_tolerance),
    )
    if identified is None or int(identified.number) != parent_space_group:
        return None
    parent = StandardCrystal(
        lattice=projection.lattice,
        fractional=projection.fractional,
        species=projection.species,
        space_group=parent_space_group,
        rotations=parent_setting.rotations,
        translations=parent_setting.translations,
    )
    candidate = certify_parent_candidate(
        child,
        parent,
        matcher_settings=matcher_settings,
        construction="maximal_t_embedding_v2",
        symprec=1e-5,
    )
    if candidate is None:
        return None
    return EmbeddingParentOccurrence(
        embedding_key=str(record["embedding_key"]),
        parent_space_group=parent_space_group,
        cell_index=1,
        full_action_order=int(parent_setting.rotations.shape[0]),
        parent_site_count=int(projection.species.size),
        projection=projection,
        candidate=candidate,
    )


def project_maximal_k_embedding(
    child: StandardCrystal,
    record: Mapping[str, object],
    *,
    maximum_source_displacement_angstrom: float,
    matcher_settings: dict[str, float | bool],
    angle_tolerance: float,
) -> EmbeddingParentOccurrence | None:
    """Project and certify one exact index-2..4 maximal-k embedding."""
    import spglib

    cell_index = int(record["cell_index"])
    if str(record["kind"]) != "k" or not 2 <= cell_index <= 4:
        raise ValueError("K0 accepts maximal klassengleiche records of index 2..4 only")
    if int(record["child_space_group"]) != child.space_group:
        raise ValueError("embedding child group does not match the material")
    parent_space_group = int(record["parent_space_group"])
    parent_setting = pyxtal_primitive_setting(parent_space_group)
    child_setting = pyxtal_primitive_setting(child.space_group)
    primitive_embedding = conjugate_embedding_to_primitive(
        _embedding_transform(record),
        parent_setting.primitive_basis,
        child_setting.primitive_basis,
    )
    projected = project_klassengleiche_parent(
        child.lattice,
        child.fractional,
        child.species,
        parent_setting.rotations,
        parent_setting.translations,
        primitive_embedding,
        maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
        maximum_index=4,
    )
    if projected is None:
        return None
    projection, full_action_order = projected
    identified = spglib.get_symmetry_dataset(
        (
            projection.lattice,
            projection.fractional,
            projection.species.astype(np.int32),
        ),
        symprec=1e-5,
        angle_tolerance=float(angle_tolerance),
    )
    if identified is None or int(identified.number) != parent_space_group:
        return None
    parent = StandardCrystal(
        lattice=projection.lattice,
        fractional=projection.fractional,
        species=projection.species,
        space_group=parent_space_group,
        rotations=parent_setting.rotations,
        translations=parent_setting.translations,
    )
    candidate = certify_parent_candidate(
        child,
        parent,
        matcher_settings=matcher_settings,
        construction="maximal_k_embedding_v3",
        symprec=1e-5,
    )
    if (
        candidate is None
        or abs(int(round(np.linalg.det(candidate.supercell_hnf)))) != cell_index
    ):
        return None
    return EmbeddingParentOccurrence(
        embedding_key=str(record["embedding_key"]),
        parent_space_group=parent_space_group,
        cell_index=cell_index,
        full_action_order=full_action_order,
        parent_site_count=int(projection.species.size),
        projection=projection,
        candidate=candidate,
    )


def project_occupational_maximal_t_embedding(
    child: StandardCrystal,
    record: Mapping[str, object],
    *,
    maximum_source_displacement_angstrom: float,
    maximum_source_hencky_norm: float,
    angle_tolerance: float,
) -> OccupationalParentOccurrence | None:
    """Project a t-parent geometry, then reduce it by ordered coloring."""
    if str(record["kind"]) != "t" or int(record["cell_index"]) != 1:
        raise ValueError("occupational t projection accepts maximal-t records only")
    if int(record["child_space_group"]) != child.space_group:
        raise ValueError("embedding child group does not match the material")
    if maximum_source_hencky_norm <= 0.0:
        raise ValueError("source Hencky bound must be positive")
    parent_space_group = int(record["parent_space_group"])
    parent_setting = pyxtal_primitive_setting(parent_space_group)
    child_setting = pyxtal_primitive_setting(child.space_group)
    primitive_embedding = conjugate_embedding_to_primitive(
        _embedding_transform(record),
        parent_setting.primitive_basis,
        child_setting.primitive_basis,
    )
    projection = project_geometry_translationengleiche_parent(
        child.lattice,
        child.fractional,
        int(child.species.size),
        parent_setting.rotations,
        parent_setting.translations,
        primitive_embedding,
        maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
    )
    if projection is None:
        return None
    return _occupational_occurrence(
        child,
        record,
        projection,
        full_action_order=int(parent_setting.rotations.shape[0]),
        maximum_source_hencky_norm=maximum_source_hencky_norm,
        angle_tolerance=angle_tolerance,
    )


def project_occupational_maximal_k_embedding(
    child: StandardCrystal,
    record: Mapping[str, object],
    *,
    maximum_source_displacement_angstrom: float,
    maximum_source_hencky_norm: float,
    angle_tolerance: float,
) -> OccupationalParentOccurrence | None:
    """Project a k-parent supercell geometry, then reduce it by coloring."""
    cell_index = int(record["cell_index"])
    if str(record["kind"]) != "k" or not 2 <= cell_index <= 4:
        raise ValueError("occupational k projection accepts index-2..4 maximal-k records only")
    if int(record["child_space_group"]) != child.space_group:
        raise ValueError("embedding child group does not match the material")
    if maximum_source_hencky_norm <= 0.0:
        raise ValueError("source Hencky bound must be positive")
    parent_space_group = int(record["parent_space_group"])
    parent_setting = pyxtal_primitive_setting(parent_space_group)
    child_setting = pyxtal_primitive_setting(child.space_group)
    primitive_embedding = conjugate_embedding_to_primitive(
        _embedding_transform(record),
        parent_setting.primitive_basis,
        child_setting.primitive_basis,
    )
    projected = project_geometry_klassengleiche_parent(
        child.lattice,
        child.fractional,
        int(child.species.size),
        parent_setting.rotations,
        parent_setting.translations,
        primitive_embedding,
        maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
        maximum_index=4,
    )
    if projected is None:
        return None
    projection, full_action_order = projected
    if full_action_order != int(parent_setting.rotations.shape[0]) * cell_index:
        raise RuntimeError("klassengleiche geometry action has the wrong quotient order")
    return _occupational_occurrence(
        child,
        record,
        projection,
        full_action_order=full_action_order,
        maximum_source_hencky_norm=maximum_source_hencky_norm,
        angle_tolerance=angle_tolerance,
    )


def search_maximal_t_parents(
    child: StandardCrystal,
    records: Sequence[Mapping[str, object]],
    *,
    maximum_source_displacement_angstrom: float,
    matcher_settings: dict[str, float | bool],
    angle_tolerance: float,
) -> tuple[EmbeddingParentOccurrence, ...]:
    """Enumerate the complete frozen maximal-t fiber for one child group."""
    selected = []
    for record in records:
        occurrence = project_maximal_t_embedding(
            child,
            record,
            maximum_source_displacement_angstrom=(maximum_source_displacement_angstrom),
            matcher_settings=matcher_settings,
            angle_tolerance=angle_tolerance,
        )
        if occurrence is not None:
            selected.append(occurrence)
    return tuple(selected)


def search_maximal_k_parents(
    child: StandardCrystal,
    records: Sequence[Mapping[str, object]],
    *,
    maximum_source_displacement_angstrom: float,
    matcher_settings: dict[str, float | bool],
    angle_tolerance: float,
) -> tuple[EmbeddingParentOccurrence, ...]:
    """Enumerate the complete frozen maximal-k fiber for one child group."""
    selected = []
    for record in records:
        occurrence = project_maximal_k_embedding(
            child,
            record,
            maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
            matcher_settings=matcher_settings,
            angle_tolerance=angle_tolerance,
        )
        if occurrence is not None:
            selected.append(occurrence)
    return tuple(selected)


def search_occupational_maximal_t_parents(
    child: StandardCrystal,
    records: Sequence[Mapping[str, object]],
    *,
    maximum_source_displacement_angstrom: float,
    maximum_source_hencky_norm: float,
    angle_tolerance: float,
) -> tuple[OccupationalParentOccurrence, ...]:
    """Enumerate the frozen maximal-t geometry fiber with ordered coloring."""
    selected = []
    for record in records:
        occurrence = project_occupational_maximal_t_embedding(
            child,
            record,
            maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
            maximum_source_hencky_norm=maximum_source_hencky_norm,
            angle_tolerance=angle_tolerance,
        )
        if occurrence is not None:
            selected.append(occurrence)
    return tuple(selected)


def search_occupational_maximal_k_parents(
    child: StandardCrystal,
    records: Sequence[Mapping[str, object]],
    *,
    maximum_source_displacement_angstrom: float,
    maximum_source_hencky_norm: float,
    angle_tolerance: float,
) -> tuple[OccupationalParentOccurrence, ...]:
    """Enumerate the frozen maximal-k geometry fiber with ordered coloring."""
    selected = []
    for record in records:
        occurrence = project_occupational_maximal_k_embedding(
            child,
            record,
            maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
            maximum_source_hencky_norm=maximum_source_hencky_norm,
            angle_tolerance=angle_tolerance,
        )
        if occurrence is not None:
            selected.append(occurrence)
    return tuple(selected)
