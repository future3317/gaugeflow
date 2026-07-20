import torch

from scripts.audit_h1a_assignment_global_interactions import _relabel_action
from scripts.audit_h1a_assignment_orbital_set import (
    audit_orbital_set_carrier,
    orbital_set_signature,
)


def _cyclic_action() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]],
        dtype=torch.long,
    )


def test_orbital_set_signature_is_node_relabeling_invariant() -> None:
    action = _cyclic_action()
    assignment = torch.tensor([2, 5, 2, 5])
    relabel = torch.tensor([2, 0, 3, 1])
    assert orbital_set_signature(assignment.tolist(), action) == orbital_set_signature(
        assignment[relabel].tolist(), _relabel_action(action, relabel)
    )


def test_orbital_set_audit_contains_the_complete_target_orbit() -> None:
    result = audit_orbital_set_carrier(
        [2, 5, 2, 5],
        _cyclic_action(),
        maximum_sites=20,
        maximum_collision_class=100,
        chunk_size=8,
    )
    assert result["exact_enumerated"]
    assert result["target_orbit_containment_failure"] is False
    assert 0.0 < result["orbital_set_target_ceiling"] <= 1.0


def test_identity_action_orbital_set_cannot_invent_site_identity() -> None:
    action = torch.arange(4, dtype=torch.long).unsqueeze(0)
    assert orbital_set_signature([2, 2, 5, 5], action) == orbital_set_signature(
        [2, 5, 2, 5], action
    )
