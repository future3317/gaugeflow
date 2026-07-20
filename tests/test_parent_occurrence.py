from __future__ import annotations

import numpy as np

from gaugeflow.catalogue.affine_quotient import integer_lattice_coset_representatives
from gaugeflow.catalogue.assignment_carrier import canonicalize_assignment_carrier
from gaugeflow.catalogue.parent_decomposition import StandardCrystal
from gaugeflow.catalogue.parent_occurrence import (
    project_geometry_complete_occupational_embedding,
    project_maximal_k_embedding,
    project_maximal_t_embedding,
    project_occupational_maximal_k_embedding,
    project_occupational_maximal_t_embedding,
    standardize_child_to_e0_setting,
)


def _p4mmm_to_p4mm_record() -> dict[str, object]:
    return {
        "cell_index": 1,
        "child_space_group": 99,
        "embedding_key": "synthetic-p4mmm-to-p4mm",
        "kind": "t",
        "parent_space_group": 123,
        "subgroup_index": 2,
        "transform_denominator": 1,
        "transform_numerators": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    }


def test_setting_exact_e1a_chain_recovers_batio3_p4mmm_parent():
    lattice = np.diag([4.0, 4.0, 4.08])
    fractional = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.515],
            [0.5, 0.5, 0.035],
            [0.5, 0.0, 0.535],
            [0.0, 0.5, 0.535],
        ]
    )
    species = np.array([56, 22, 8, 8, 8], dtype=np.int64)
    child = standardize_child_to_e0_setting(
        lattice,
        fractional @ lattice,
        species,
        expected_space_group=99,
        expected_primitive_sites=5,
        symprec=0.01,
        angle_tolerance=5.0,
    )
    occurrence = project_maximal_t_embedding(
        child,
        _p4mmm_to_p4mm_record(),
        maximum_source_displacement_angstrom=0.2,
        matcher_settings={
            "ltol": 0.2,
            "stol": 0.3,
            "angle_tol": 5.0,
            "scale": True,
        },
        angle_tolerance=5.0,
    )
    assert occurrence is not None
    assert occurrence.parent_space_group == 123
    assert np.isclose(
        occurrence.projection.source_max_displacement_angstrom,
        0.1428,
        atol=1e-10,
        rtol=0.0,
    )
    assert occurrence.projection.projected_group_max_error_angstrom <= 1e-12


def test_e1a_uses_one_sided_displacement_threshold_not_orbit_defect():
    lattice = np.diag([4.0, 4.0, 4.08])
    fractional = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.515],
            [0.5, 0.5, 0.035],
            [0.5, 0.0, 0.535],
            [0.0, 0.5, 0.535],
        ]
    )
    child = standardize_child_to_e0_setting(
        lattice,
        fractional @ lattice,
        np.array([56, 22, 8, 8, 8], dtype=np.int64),
        expected_space_group=99,
        expected_primitive_sites=5,
        symprec=0.01,
        angle_tolerance=5.0,
    )
    rejected = project_maximal_t_embedding(
        child,
        _p4mmm_to_p4mm_record(),
        maximum_source_displacement_angstrom=0.1,
        matcher_settings={
            "ltol": 0.2,
            "stol": 0.3,
            "angle_tol": 5.0,
            "scale": True,
        },
        angle_tolerance=5.0,
    )
    assert rejected is None


def test_setting_exact_k0_recovers_index_two_translation_parent():
    parent_lattice = np.diag([3.0, 4.0, 5.0])
    basis = np.diag([2, 1, 1])
    child_lattice = basis.T @ parent_lattice
    parent_fractional = np.array([[0.13, 0.21, 0.37], [0.31, 0.07, 0.62]])
    parent_species = np.array([14, 8], dtype=np.int64)
    child_fractional = np.concatenate(
        [
            parent_fractional @ np.linalg.inv(basis),
            (parent_fractional + np.array([1.0, 0.0, 0.0])) @ np.linalg.inv(basis),
        ]
    )
    child_fractional = (
        child_fractional
        + np.array(
            [
                [0.006, -0.004, 0.002],
                [-0.003, 0.005, -0.004],
                [-0.005, 0.003, -0.002],
                [0.004, -0.006, 0.003],
            ]
        )
    ) % 1.0
    child = StandardCrystal(
        lattice=child_lattice,
        fractional=child_fractional,
        species=np.tile(parent_species, 2),
        space_group=1,
        rotations=np.eye(3, dtype=np.int64)[None],
        translations=np.zeros((1, 3), dtype=np.float64),
    )
    occurrence = project_maximal_k_embedding(
        child,
        {
            "cell_index": 2,
            "child_space_group": 1,
            "embedding_key": "synthetic-index-two-p1",
            "kind": "k",
            "parent_space_group": 1,
            "subgroup_index": 2,
            "transform_denominator": 1,
            "transform_numerators": [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        },
        maximum_source_displacement_angstrom=0.2,
        matcher_settings={
            "ltol": 0.2,
            "stol": 0.3,
            "angle_tol": 5.0,
            "scale": True,
        },
        angle_tolerance=5.0,
    )
    assert occurrence is not None
    assert occurrence.cell_index == 2
    assert occurrence.full_action_order == 2
    assert occurrence.parent_site_count == 2
    assert abs(int(round(np.linalg.det(occurrence.candidate.supercell_hnf)))) == 2


def test_occupational_t_projection_recovers_geometry_rejected_by_species_assignment():
    lattice = np.array(
        [[4.1, 0.0, 0.0], [0.3, 5.2, 0.0], [0.2, 0.4, 6.3]],
        dtype=np.float64,
    )
    position = np.array([0.137, 0.219, 0.371])
    fractional = np.stack([position, (-position) % 1.0])
    child = StandardCrystal(
        lattice=lattice,
        fractional=fractional,
        species=np.array([14, 8], dtype=np.int64),
        space_group=1,
        rotations=np.eye(3, dtype=np.int64)[None],
        translations=np.zeros((1, 3), dtype=np.float64),
    )
    record = {
        "cell_index": 1,
        "child_space_group": 1,
        "embedding_key": "synthetic-p-1-to-colored-p1",
        "kind": "t",
        "parent_space_group": 2,
        "subgroup_index": 2,
        "transform_denominator": 1,
        "transform_numerators": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    }

    legacy = project_maximal_t_embedding(
        child,
        record,
        maximum_source_displacement_angstrom=0.2,
        matcher_settings={"ltol": 0.2, "stol": 0.3, "angle_tol": 5.0, "scale": True},
        angle_tolerance=5.0,
    )
    occurrence = project_occupational_maximal_t_embedding(
        child,
        record,
        maximum_source_displacement_angstrom=0.2,
        maximum_source_hencky_norm=0.15,
        angle_tolerance=5.0,
    )

    assert legacy is None
    assert occurrence is not None
    assert not hasattr(occurrence.projection, "species")
    assert occurrence.full_action_order == 2
    assert occurrence.occupational_stabilizer_indices.size == 1
    assert occurrence.stabilizer_order_matches_child
    assert occurrence.exact_coloring_reconstruction


def test_occupational_off_diagonal_k_projection_uses_full_translation_quotient():
    parent_lattice = np.array(
        [[3.2, 0.0, 0.0], [0.4, 4.3, 0.0], [0.2, 0.7, 5.4]],
        dtype=np.float64,
    )
    basis = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.int64)
    parent_fractional = np.array(
        [[0.113, 0.217, 0.319], [0.347, 0.071, 0.613], [0.729, 0.443, 0.157]],
        dtype=np.float64,
    )
    cosets = integer_lattice_coset_representatives(basis)
    child_fractional = np.concatenate([(parent_fractional + coset) @ np.linalg.inv(basis).T for coset in cosets]) % 1.0
    child = StandardCrystal(
        lattice=basis.T @ parent_lattice,
        fractional=child_fractional,
        species=np.array([6, 7, 8, 13, 14, 15], dtype=np.int64),
        space_group=1,
        rotations=np.eye(3, dtype=np.int64)[None],
        translations=np.zeros((1, 3), dtype=np.float64),
    )
    record = {
        "cell_index": 2,
        "child_space_group": 1,
        "embedding_key": "synthetic-off-diagonal-index-two-colored-p1",
        "kind": "k",
        "parent_space_group": 1,
        "subgroup_index": 2,
        "transform_denominator": 1,
        "transform_numerators": [1, -1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    }
    occurrence = project_occupational_maximal_k_embedding(
        child,
        record,
        maximum_source_displacement_angstrom=0.2,
        maximum_source_hencky_norm=0.15,
        angle_tolerance=5.0,
    )

    assert occurrence is not None
    assert occurrence.cell_index == 2
    assert occurrence.full_action_order == 2
    assert occurrence.parent_site_count == 3
    assert occurrence.occupational_stabilizer_indices.tolist() == [0]
    assert occurrence.stabilizer_order_matches_child
    assert occurrence.exact_coloring_reconstruction

    complete = project_geometry_complete_occupational_embedding(
        child,
        record,
        maximum_source_displacement_angstrom=0.2,
        maximum_source_hencky_norm=0.15,
        angle_tolerance=5.0,
    )
    assert complete is not None
    aligned = canonicalize_assignment_carrier(complete)
    carrier = aligned.carrier
    assert carrier.expanded_parent_fractional.shape == (6, 3)
    assert abs(round(np.linalg.det(carrier.supercell_hnf))) == 2
    assert carrier.translation_cosets.shape == (2, 3)
    assert carrier.parent_action_permutations.shape == (2, 6)
    assert aligned.maximum_periodic_alignment_error_angstrom <= 1e-8
