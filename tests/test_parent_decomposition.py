from __future__ import annotations

import numpy as np
import torch

from gaugeflow.catalogue.parent_decomposition import (
    balanced_selection,
    decompose_parent_candidate,
    find_parent_candidates,
    translation_quotient_displacement,
)
from gaugeflow.geometry import (
    closest_image_displacement,
    closest_image_displacements_numpy,
)
from scripts.build_h0_e_parent_decomposition_pilot import _records_table

MATCHER = {
    "ltol": 0.2,
    "stol": 0.3,
    "angle_tol": 5.0,
    "primitive_cell": False,
    "scale": True,
    "attempt_supercell": True,
}


def test_balanced_selection_preserves_split_quota_and_is_deterministic():
    records = []
    for split, space_group in (("train", 221), ("val", 62), ("test", 15)):
        for index in range(8):
            records.append(
                {
                    "material_id": f"{split}-{index}",
                    "gaugeflow_split": split,
                    "space_group_number": space_group,
                    "primitive_sites": 1 + index,
                }
            )
    kwargs = {
        "split_counts": {"train": 4, "val": 2, "test": 2},
        "seed": 20260717,
        "site_boundaries": [4, 8, 16],
    }
    left = balanced_selection(records, **kwargs)
    right = balanced_selection(records[::-1], **kwargs)
    assert left == right
    assert len(left) == 8
    assert {value["gaugeflow_split"] for value in left} == {"train", "val", "test"}


def test_sparse_builder_records_use_the_union_of_all_result_fields():
    table = _records_table(
        [
            {"material_id": "first", "qualified_nontrivial": False},
            {
                "material_id": "second",
                "qualified_nontrivial": True,
                "active_sectors_json": '["strain"]',
            },
        ]
    )
    assert "active_sectors_json" in table.schema.names
    assert table.column("active_sectors_json").to_pylist() == [None, '["strain"]']


def test_translation_quotient_removes_global_shift_without_erasing_distortion():
    lattice = np.diag([2.0, 1.0, 1.0])
    parent = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    child = np.array([[0.11, 0.0, 0.0], [0.59, 0.0, 0.0]])
    displacement = translation_quotient_displacement(
        parent, child, lattice, np.ones(2)
    )
    assert np.allclose(displacement[:, 0], [0.02, -0.02], atol=1e-10, rtol=0.0)
    assert np.allclose(displacement.mean(axis=0), 0.0, atol=1e-12, rtol=0.0)


def test_batched_numpy_cvp_matches_exact_single_vector_solver_for_skew_cell():
    lattice = np.array([[1.0, 0.0, 0.0], [0.91, 0.37, 0.0], [0.22, 0.18, 0.43]])
    delta = np.array(
        [[0.49, -0.51, 1.37], [-1.9, 2.2, -0.41], [0.5, 0.5, 0.5]],
        dtype=np.float64,
    )
    batched, shifts = closest_image_displacements_numpy(delta, lattice)
    singles = []
    single_shifts = []
    for value in delta:
        displacement, shift = closest_image_displacement(
            torch.as_tensor(value), torch.as_tensor(lattice)
        )
        singles.append(displacement.numpy())
        single_shifts.append(shift.numpy())
    assert np.allclose(batched, np.stack(singles), atol=1e-12, rtol=0.0)
    assert np.array_equal(shifts, np.stack(single_shifts))


def test_known_doubled_cell_finds_and_decomposes_nontrivial_parent():
    lattice = np.diag([2.0, 1.0, 1.0])
    fractional = np.array([[0.01, 0.0, 0.0], [0.49, 0.0, 0.0]])
    species = np.array([14, 14], dtype=np.int64)
    child, candidates = find_parent_candidates(
        lattice,
        fractional,
        species,
        child_symprec=0.005,
        symprec_ladder=[0.05, 0.1],
        angle_tolerance=5.0,
        matcher_settings=MATCHER,
    )
    assert candidates
    result = decompose_parent_candidate(
        candidates[0],
        residual_rms_limit=0.1,
        stabilizer_rms_tolerance=0.015,
        stabilizer_metric_tolerance=0.002,
        displacement_energy_floor=1e-12,
        strain_energy_floor=1e-12,
        terminal_symprec=0.005,
        angle_tolerance=5.0,
        matcher_settings=MATCHER,
    )
    assert result.child_space_group == child.space_group
    assert result.supercell_index in (1, 2)
    assert result.occurrence_integral
    assert result.periodic_rms_angstrom <= 0.1
    assert 1 <= len(result.active_components) <= 2


def test_homogeneous_metric_distortion_is_identified_as_a_strain_opd():
    lattice = np.diag([1.0, 1.02, 1.04])
    child, candidates = find_parent_candidates(
        lattice,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([14], dtype=np.int64),
        child_symprec=0.001,
        symprec_ladder=[0.05, 0.1],
        angle_tolerance=5.0,
        matcher_settings=MATCHER,
    )
    assert child.space_group == 47
    assert candidates and candidates[0].parent.space_group == 221
    result = decompose_parent_candidate(
        candidates[0],
        residual_rms_limit=0.1,
        stabilizer_rms_tolerance=0.015,
        stabilizer_metric_tolerance=0.002,
        displacement_energy_floor=1e-12,
        strain_energy_floor=1e-12,
        terminal_symprec=0.001,
        angle_tolerance=5.0,
        matcher_settings=MATCHER,
    )
    assert result.periodic_rms_angstrom == 0.0
    assert result.terminal_space_group_agrees
    assert [value.sector for value in result.active_components] == ["strain"]
