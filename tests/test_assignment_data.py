from __future__ import annotations

import copy

import numpy as np
import torch

from gaugeflow.production.assignment_data import (
    pack_assignment_carriers,
    prepare_assignment_carrier_example,
)
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _candidate() -> dict[str, object]:
    return {
        "embedding_key": "synthetic",
        "cell_index": 1,
        "parent_space_group": 1,
        "carrier": {
            "expanded_parent_fractional": [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            "expanded_parent_lattice": [
                [2.0, 0.0, 0.0],
                [0.8, 1.7, 0.0],
                [0.6, 0.3, 1.5],
            ],
            "parent_action_permutations": [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
            ],
            "supercell_hnf": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "target": {
            "active_species_tokens": [2, 5],
            "active_species_counts": [2, 2],
            "assignment_tokens": [2, 5, 5, 2],
        },
    }


def _prepare(candidate: dict[str, object]):
    return prepare_assignment_carrier_example(
        candidate,
        embedding_key="synthetic",
        material_id_audit_only="synthetic-material",
        evidence_role_audit_only="test",
        radial_channels=6,
    )


def test_carrier_features_are_gl3z_invariant_and_target_free() -> None:
    source = _candidate()
    reference = _prepare(source)
    changed = copy.deepcopy(source)
    unimodular = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    carrier = changed["carrier"]
    assert isinstance(carrier, dict)
    fractional = np.asarray(carrier["expanded_parent_fractional"], dtype=np.float64)
    lattice = np.asarray(carrier["expanded_parent_lattice"], dtype=np.float64)
    carrier["expanded_parent_fractional"] = (fractional @ np.linalg.inv(unimodular) % 1.0).tolist()
    carrier["expanded_parent_lattice"] = (unimodular @ lattice).tolist()
    transformed = _prepare(changed)
    assert torch.allclose(transformed.graph_features, reference.graph_features, atol=1e-6, rtol=1e-6)
    assert torch.allclose(transformed.edge_rbf, reference.edge_rbf, atol=1e-6, rtol=1e-6)
    assert torch.equal(transformed.site_features, reference.site_features)

    changed_target = copy.deepcopy(source)
    target = changed_target["target"]
    assert isinstance(target, dict)
    target["assignment_tokens"] = [5, 2, 2, 5]
    alternate = _prepare(changed_target)
    assert torch.equal(alternate.site_features, reference.site_features)
    assert torch.equal(alternate.graph_features, reference.graph_features)
    assert torch.equal(alternate.edge_rbf, reference.edge_rbf)


def test_assignment_carrier_packing_preserves_graph_boundaries() -> None:
    example = _prepare(_candidate())
    packed = pack_assignment_carriers([example, example], device="cpu")
    packed.validate(vocabulary_size=CHEMICAL_ELEMENT_COUNT)
    assert packed.site_features.shape[0] == 8
    assert packed.edge_rbf.shape[1] == 6 + 6 * 7 // 2
    assert torch.equal(torch.bincount(packed.batch), torch.tensor([4, 4]))
    assert bool((packed.edge_source[:12] < 4).all())
    assert bool((packed.edge_source[12:] >= 4).all())
