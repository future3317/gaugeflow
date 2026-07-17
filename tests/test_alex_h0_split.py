import numpy as np

from scripts.audit_alex_h0_split import _select_candidates
from scripts.build_alex_h0_split import assign_components, prototype_signature


def test_anonymous_primitive_prototype_is_translation_permutation_and_species_name_invariant():
    cell = [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.2]]
    positions = [[0.0, 0.0, 0.0], [2.1, 2.1, 2.1]]
    reference = prototype_signature(
        positions,
        cell,
        [11, 17],
        symprec=0.01,
        angle_tolerance=5.0,
    )
    translated_permuted = prototype_signature(
        [[3.15, 3.15, 3.15], [1.05, 1.05, 1.05]],
        cell,
        [35, 19],
        symprec=0.01,
        angle_tolerance=5.0,
    )
    assert reference == translated_permuted
    assert reference[1] > 0
    assert reference[2] == 2


def test_component_assignment_is_input_order_invariant_and_balanced():
    components = {f"component-{index}": count for index, count in enumerate([40, 20, 15, 10, 8, 4, 2, 1])}
    reversed_components = dict(reversed(list(components.items())))
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    first = assign_components(components, fractions=fractions, seed=20260717)
    second = assign_components(reversed_components, fractions=fractions, seed=20260717)
    assert first == second
    assert set(first.values()) == {"train", "val", "test"}


def test_structure_matcher_candidate_selection_is_reorder_invariant():
    table = {
        "material_id": ["a", "b", "c", "d", "e", "f"],
        "anonymous_stoichiometry": ["1:1"] * 6,
        "primitive_sites": [2] * 6,
        "gaugeflow_split": ["train", "train", "val", "val", "test", "test"],
    }
    first = _select_candidates(
        table,
        seed=17,
        representatives_per_bucket_split=2,
        maximum_pairs=12,
    )
    permutation = np.array([5, 2, 0, 4, 1, 3])
    shuffled = {key: [values[index] for index in permutation] for key, values in table.items()}
    second = _select_candidates(
        shuffled,
        seed=17,
        representatives_per_bucket_split=2,
        maximum_pairs=12,
    )
    assert first == second
    assert len(first) == 12
