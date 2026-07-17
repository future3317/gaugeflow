import numpy as np

from scripts.audit_phonondb_h0_b import (
    _deterministic_unitary,
    _frequency_clusters,
    _select_stratified_materials,
    _symmetrized_dielectric,
    _translation_basis,
)
from scripts.build_phonondb_force_constants_v2 import _project_full_hessian


def test_mass_weighted_translation_basis_is_orthonormal():
    basis = _translation_basis(np.asarray([1.0, 4.0, 9.0]))
    assert np.allclose(basis.conj().T @ basis, np.eye(3), atol=1e-14)
    reshaped = basis.reshape(3, 3, 3)
    assert np.allclose(reshaped[1], 2.0 * reshaped[0])
    assert np.allclose(reshaped[2], 3.0 * reshaped[0])


def test_frequency_clusters_use_frozen_adjacent_gap_rule():
    frequencies = np.asarray([-1.0, -0.999999, 0.0, 0.1, 0.100004, 2.0])
    clusters = _frequency_clusters(frequencies, 1e-5)
    assert [cluster.tolist() for cluster in clusters] == [[0, 1], [2], [3, 4], [5]]


def test_deterministic_complex_unitary_preserves_projector():
    unitary = _deterministic_unitary(3, "material:q:cluster")
    assert np.allclose(unitary.conj().T @ unitary, np.eye(3), atol=1e-14)
    basis = np.linalg.qr(
        np.asarray(
            [[1.0, 2.0j, 0.0], [0.0, 1.0, 1.0j], [1.0j, 0.0, 2.0], [2.0, 1.0, 0.0j]]
        )
    )[0]
    projector = basis @ basis.conj().T
    transformed = basis @ unitary
    assert np.allclose(projector, transformed @ transformed.conj().T, atol=1e-14)


def test_full_hessian_projection_enforces_permutation_and_bilateral_asr():
    force_constants = np.asarray(
        [
            [[[2.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 3.0]],
             [[-2.0, -0.2, 0.0], [0.0, -1.0, -0.1], [0.0, 0.0, -3.0]]],
            [[[1.0, 0.0, 0.0], [0.3, 2.0, 0.0], [0.0, 0.0, 1.0]],
             [[-1.0, 0.0, 0.0], [-0.3, -2.0, 0.0], [0.0, 0.0, -1.0]]],
        ],
        dtype=np.float64,
    )
    assert np.max(np.abs(force_constants.sum(axis=1))) == 0.0
    assert np.max(np.abs(force_constants - force_constants.transpose(1, 0, 3, 2))) > 0.0
    projected = _project_full_hessian(force_constants)
    assert np.max(np.abs(projected.sum(axis=1))) < 1e-12
    assert np.max(np.abs(projected.sum(axis=0))) < 1e-12
    assert np.max(np.abs(projected - projected.transpose(1, 0, 3, 2))) < 1e-12


def test_dielectric_symmetrization_is_explicit_and_idempotent():
    raw = np.asarray([[4.0, 0.2, 0.0], [0.1, 3.0, 0.3], [0.0, 0.2, 2.0]])
    production = _symmetrized_dielectric(raw)
    assert np.allclose(production, production.T)
    assert np.allclose(_symmetrized_dielectric(production), production)
    assert not np.allclose(raw, production)


def test_stratified_selection_is_deterministic_and_keeps_mandatory_tails():
    force_rows = {}
    mode_rows = {}
    v1_metrics = {}
    for index in range(80):
        material_id = f"mp-{index}"
        force_rows[material_id] = {
            "projection_relative_l2": 0.1 if index == 1 else 0.0,
            "n_primitive_atoms": 4 + index,
        }
        mode_rows[material_id] = {
            "has_born_effective_charge": index % 2 == 0,
            "has_dielectric_constant": index % 2 == 0,
        }
        v1_metrics[material_id] = {
            "translation_max_abs_frequency_thz": 0.1 if index == 0 else 0.0,
            "translation_subspace_min_singular": 1.0,
            "translation_dynamical_residual": 0.0,
            "nac_dielectric_symmetry_error": float(index),
        }
    first, evidence = _select_stratified_materials(
        force_rows, mode_rows, v1_metrics, sample_size=70
    )
    second, _ = _select_stratified_materials(
        force_rows, mode_rows, v1_metrics, sample_size=70
    )
    assert first == second
    assert len(first) == 70
    assert {"mp-0", "mp-1", "mp-79"} <= first
    assert evidence["mandatory_acoustic_outliers"] == 1
    assert evidence["mandatory_projection_relative_l2_gt_0_05"] == 1
