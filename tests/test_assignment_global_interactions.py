import torch

from scripts.audit_h1a_assignment_global_interactions import (
    _relabel_action,
    action_pair_descriptors,
    audit_carrier,
    relabel_invariance_check,
    unary_collision_class_size,
)


def _klein_action() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ],
        dtype=torch.long,
    )


def test_pair_descriptors_are_exactly_relabeling_equivariant() -> None:
    action = _klein_action()
    assert relabel_invariance_check([4, 7, 4, 7], action, seed=19, maximum_sites=20)
    relabel = torch.tensor([2, 0, 3, 1])
    transformed = _relabel_action(action, relabel)
    assert action_pair_descriptors(action)[2] == action_pair_descriptors(transformed)[2]


def test_pair_interaction_resolves_a_unary_coloring_collision() -> None:
    action = torch.arange(4, dtype=torch.long).unsqueeze(0)
    result = audit_carrier(
        [4, 7, 4, 7],
        action,
        maximum_sites=20,
        maximum_collision_class=100,
        chunk_size=8,
    )
    assert result["unary_collision_class_size"] == 6
    assert result["exact_enumerated"]
    assert result["target_orbit_size"] == 1
    assert result["orbital_pair_resolved"]
    assert result["target_orbit_containment_failure"] is False


def test_unary_collision_size_is_product_of_signature_multinomials() -> None:
    signatures = ((1,), (1,), (2,), (2,))
    assert unary_collision_class_size([0, 1, 0, 1], signatures) == 4
