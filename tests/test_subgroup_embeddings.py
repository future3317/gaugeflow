from __future__ import annotations

import numpy as np

from gaugeflow.catalogue.subgroup_embeddings import (
    RationalAffineTransform,
    certify_affine_subgroup_inclusion,
    normalized_relation_variant,
    species_wyckoff_exact_cover,
    wyckoff_multiset_has_exact_cover,
)


def test_rational_affine_transform_is_compact_and_quotients_integer_origin_shifts():
    transform = RationalAffineTransform.from_array(
        np.array(
            [
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 1.0, -0.25],
            ]
        )
    )
    assert transform.denominator == 4
    assert np.array_equal(
        transform.compact_numerators(),
        np.array([[8, 0, 0, 0], [0, 4, 0, 2], [0, 0, 4, 3]]),
    )
    assert np.allclose(
        transform.as_float()[:3, 3], np.array([0.0, 0.5, 0.75])
    )


def test_vectorized_affine_inclusion_accepts_p1_inside_p_minus_one():
    parent_rotations = np.stack(
        [np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)]
    )
    parent_translations = np.zeros((2, 3), dtype=np.float64)
    child_rotations = np.eye(3, dtype=np.int64)[None, :, :]
    child_translations = np.zeros((1, 3), dtype=np.float64)
    transform = RationalAffineTransform.from_array(np.eye(4))
    certificate = certify_affine_subgroup_inclusion(
        parent_rotations,
        parent_translations,
        child_rotations,
        child_translations,
        transform,
    )
    assert certificate.passed
    assert certificate.parent_order == 2
    assert certificate.child_order == 1
    assert certificate.representative_image_order == 1
    assert certificate.representative_kernel_size == 1
    assert certificate.maximum_rotation_error == 0.0
    assert certificate.maximum_periodic_translation_error == 0.0


def test_vectorized_affine_inclusion_rejects_supergroup_as_subgroup():
    parent_rotations = np.eye(3, dtype=np.int64)[None, :, :]
    parent_translations = np.zeros((1, 3), dtype=np.float64)
    child_rotations = np.stack(
        [np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)]
    )
    child_translations = np.zeros((2, 3), dtype=np.float64)
    certificate = certify_affine_subgroup_inclusion(
        parent_rotations,
        parent_translations,
        child_rotations,
        child_translations,
        RationalAffineTransform.from_array(np.eye(4)),
    )
    assert not certificate.passed
    assert not certificate.contained


def test_centered_cell_representative_kernel_is_not_mistaken_for_noninclusion():
    parent_rotations = np.repeat(np.eye(3, dtype=np.int64)[None, :, :], 2, axis=0)
    parent_translations = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    child_rotations = np.repeat(np.eye(3, dtype=np.int64)[None, :, :], 4, axis=0)
    child_translations = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]
    )
    transform = RationalAffineTransform.from_array(
        np.array([[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    )
    certificate = certify_affine_subgroup_inclusion(
        parent_rotations,
        parent_translations,
        child_rotations,
        child_translations,
        transform,
    )
    assert certificate.passed
    assert certificate.representative_image_order == 2
    assert certificate.representative_kernel_size == 2


def test_relation_variant_ignores_child_orbit_enumeration_order_only():
    left = normalized_relation_variant([("2b", "1a"), ("4c",)])
    right = normalized_relation_variant([("1a", "2b"), ("4c",)])
    assert left == right
    assert left != normalized_relation_variant([("1a",), ("2b", "4c")])


def test_wyckoff_multiset_cover_allows_repeated_parent_orbit_types():
    relation = [("1a", "1a"), ("2b",), ("4c", "4c")]
    assert wyckoff_multiset_has_exact_cover(["1a"] * 4, relation)
    assert wyckoff_multiset_has_exact_cover(["2b", "2b"], relation)
    assert not wyckoff_multiset_has_exact_cover(["1a"] * 3, relation)


def test_species_wyckoff_cover_cannot_mix_labels_across_elements():
    relation = [("1a", "2b")]
    assert species_wyckoff_exact_cover([("Na", "1a"), ("Na", "2b")], relation)
    assert not species_wyckoff_exact_cover([("Na", "1a"), ("Cl", "2b")], relation)
