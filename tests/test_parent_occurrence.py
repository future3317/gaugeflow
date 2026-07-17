from __future__ import annotations

import numpy as np

from gaugeflow.catalogue.parent_occurrence import (
    project_maximal_t_embedding,
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
