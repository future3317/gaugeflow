from __future__ import annotations

import numpy as np
import spglib

from gaugeflow.catalogue.affine_quotient import primitive_space_group_from_hall
from gaugeflow.catalogue.parent_projection import (
    _operation_table,
    conjugate_embedding_to_primitive,
    conventional_to_primitive_structure,
    project_lattice_metric,
    project_translationengleiche_parent,
)
from gaugeflow.catalogue.subgroup_embeddings import RationalAffineTransform


def test_conventional_embedding_conjugates_to_unimodular_primitive_action():
    parent_primitive = np.diag([0.5, 1.0, 1.0])
    child_primitive = np.eye(3)
    conventional = RationalAffineTransform.from_array(
        np.array([[0.5, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    )
    primitive = conjugate_embedding_to_primitive(conventional, parent_primitive, child_primitive)
    assert primitive.denominator == 1
    assert np.array_equal(primitive.as_float(), np.eye(4))


def test_f_centering_quotient_is_an_exact_vectorized_coordinate_change():
    conventional_lattice = np.diag([4.0, 4.0, 4.0])
    conventional_fractional = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    primitive_basis = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    lattice, fractional, species = conventional_to_primitive_structure(
        conventional_lattice,
        conventional_fractional,
        np.full(4, 29, dtype=np.int64),
        primitive_basis,
    )
    assert fractional.shape == (1, 3)
    assert np.array_equal(species, np.array([29], dtype=np.int64))
    assert np.isclose(
        abs(np.linalg.det(lattice)),
        abs(np.linalg.det(conventional_lattice)) / 4.0,
        atol=1e-12,
        rtol=0.0,
    )


def test_metric_reynolds_projection_is_invariant_under_axis_exchange():
    lattice = np.diag([2.0, 3.0, 4.0])
    rotations = np.array(
        [
            np.eye(3, dtype=np.int64),
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        ]
    )
    projected = project_lattice_metric(lattice, rotations)
    metric = projected @ projected.T
    changed = np.einsum("gji,jk,gkl->gil", rotations, metric, rotations, optimize=True)
    assert np.allclose(changed, metric[None], atol=1e-12, rtol=0.0)
    assert np.isclose(metric[0, 0], metric[1, 1], atol=1e-12, rtol=0.0)


def test_translationengleiche_projection_recovers_inversion_parent():
    lattice = np.diag([5.0, 6.0, 7.0])
    fractional = np.array([[0.10, 0.20, 0.30], [0.91, 0.79, 0.71]])
    species = np.array([8, 8], dtype=np.int64)
    rotations = np.stack([np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)])
    translations = np.zeros((2, 3), dtype=np.float64)
    result = project_translationengleiche_parent(
        lattice,
        fractional,
        species,
        rotations,
        translations,
        RationalAffineTransform.from_array(np.eye(4)),
        maximum_source_displacement_angstrom=0.2,
    )
    assert result is not None
    assert result.source_max_displacement_angstrom < 0.06
    assert result.projected_group_max_error_angstrom <= 1e-12
    assert np.allclose(
        (result.fractional[0] + result.fractional[1]) % 1.0,
        0.0,
        atol=1e-12,
        rtol=0.0,
    )


def test_translationengleiche_projection_rejects_species_incompatible_inversion():
    result = project_translationengleiche_parent(
        np.diag([5.0, 6.0, 7.0]),
        np.array([[0.10, 0.20, 0.30], [0.91, 0.79, 0.71]]),
        np.array([8, 14], dtype=np.int64),
        np.stack([np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)]),
        np.zeros((2, 3), dtype=np.float64),
        RationalAffineTransform.from_array(np.eye(4)),
        maximum_source_displacement_angstrom=0.2,
    )
    assert result is None


def test_vectorized_operation_table_matches_direct_seitz_products():
    lattice = np.diag([4.0, 4.0, 4.0])
    fractional = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    species = np.array([56, 22, 8, 8, 8], dtype=np.int32)
    dataset = spglib.get_symmetry_dataset((lattice, fractional, species), symprec=1e-5)
    assert dataset is not None
    rotations = np.asarray(dataset.rotations, dtype=np.int64)
    translations = np.asarray(dataset.translations, dtype=np.float64)
    table = _operation_table(rotations, translations)
    for left in range(rotations.shape[0]):
        for right in range(rotations.shape[0]):
            selected = int(table[left, right])
            assert np.array_equal(rotations[left] @ rotations[right], rotations[selected])
            product_translation = rotations[left] @ translations[right] + translations[left]
            difference = product_translation - translations[selected]
            difference -= np.rint(difference)
            assert np.max(np.abs(difference)) <= 1e-12


def test_batio3_positive_control_projects_p4mm_distortion_to_pm3m():
    parent_lattice = np.diag([4.0, 4.0, 4.0])
    parent_fractional = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    species = np.array([56, 22, 8, 8, 8], dtype=np.int64)
    dataset = spglib.get_symmetry_dataset((parent_lattice, parent_fractional, species), symprec=1e-5)
    assert dataset is not None and int(dataset.number) == 221
    distorted_lattice = np.diag([4.0, 4.0, 4.08])
    distorted_fractional = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.515], [0.5, 0.5, 0.035], [0.5, 0.0, 0.535], [0.0, 0.5, 0.535]]
    )
    result = project_translationengleiche_parent(
        distorted_lattice,
        distorted_fractional,
        species,
        np.asarray(dataset.rotations, dtype=np.int64),
        np.asarray(dataset.translations, dtype=np.float64),
        RationalAffineTransform.from_array(np.eye(4)),
        maximum_source_displacement_angstrom=0.3,
    )
    assert result is not None
    projected_dataset = spglib.get_symmetry_dataset((result.lattice, result.fractional, result.species), symprec=1e-5)
    assert projected_dataset is not None
    assert int(projected_dataset.number) == 221
    assert result.projected_group_max_error_angstrom <= 1e-12


def test_nonsymmorphic_general_orbit_projection_preserves_sg62():
    parent = primitive_space_group_from_hall(292)
    rotations = parent.rotations
    translations = parent.translation_numerators.astype(np.float64) / parent.translation_denominator
    lattice = project_lattice_metric(
        np.array([[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [1.1, 0.0, 7.0]]),
        rotations,
    )
    seed = np.array([0.137, 0.219, 0.371])
    fractional = (np.einsum("i,gji->gj", seed, rotations, optimize=True) + translations) % 1.0
    species = np.full(fractional.shape[0], 14, dtype=np.int64)
    generator = np.random.default_rng(162)
    distorted_fractional = (fractional + generator.normal(scale=0.002, size=fractional.shape)) % 1.0
    distorted_lattice = lattice @ np.diag([1.002, 0.999, 1.001])
    result = project_translationengleiche_parent(
        distorted_lattice,
        distorted_fractional,
        species,
        rotations,
        translations,
        RationalAffineTransform.from_array(np.eye(4)),
        maximum_source_displacement_angstrom=0.2,
    )
    assert result is not None
    dataset = spglib.get_symmetry_dataset((result.lattice, result.fractional, result.species), symprec=1e-5)
    assert dataset is not None
    assert int(dataset.number) == 62
    assert result.projected_group_max_error_angstrom <= 1e-12
