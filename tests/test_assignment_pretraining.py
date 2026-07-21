from __future__ import annotations

import numpy as np
import pytest
import torch

from gaugeflow.geometry import closest_image_displacements_numpy
from gaugeflow.production.assignment_data import prepare_assignment_carrier_example
from gaugeflow.production.assignment_pretraining import (
    compile_masked_assignment_batch,
    ddp_global_mean_loss,
    exact_periodic_pair_distances,
    rank_shard_of_global_batch,
    sample_rank_sharded_reveal_ranks,
)
from gaugeflow.production.assignment_training import sample_uniform_reveal_ranks
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _candidate(fractional: torch.Tensor, lattice: torch.Tensor, tokens: torch.Tensor) -> dict:
    active, counts = torch.unique(tokens, sorted=True, return_counts=True)
    sites = tokens.numel()
    return {
        "cell_index": 1,
        "parent_space_group": 1,
        "carrier": {
            "expanded_parent_fractional": fractional.tolist(),
            "expanded_parent_lattice": lattice.tolist(),
            "parent_action_permutations": [list(range(sites))],
            "supercell_hnf": torch.eye(3, dtype=torch.long).tolist(),
        },
        "target": {
            "assignment_tokens": tokens.tolist(),
            "active_species_tokens": active.tolist(),
            "active_species_counts": counts.tolist(),
        },
    }


def test_masked_compiler_matches_identity_parent_reference() -> None:
    fractional_parts = [
        torch.tensor(
            [[0.02, 0.10, 0.20], [0.48, 0.11, 0.75], [0.77, 0.62, 0.24], [0.35, 0.83, 0.91]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.12, 0.22, 0.32], [0.66, 0.15, 0.54], [0.41, 0.79, 0.86]],
            dtype=torch.float32,
        ),
    ]
    lattice = torch.tensor(
        [
            [[3.1, 0.0, 0.0], [0.4, 3.4, 0.0], [0.2, 0.3, 4.0]],
            [[4.2, 0.0, 0.0], [0.3, 3.7, 0.0], [0.1, 0.2, 3.2]],
        ],
        dtype=torch.float32,
    )
    token_parts = [torch.tensor([2, 5, 2, 7]), torch.tensor([3, 3, 8])]
    batch = torch.repeat_interleave(torch.arange(2), torch.tensor([4, 3]))
    compiled = compile_masked_assignment_batch(
        torch.cat(fractional_parts),
        lattice,
        batch,
        torch.cat(token_parts),
    )
    assert compiled.maximum_candidate_count <= 27
    assert not bool(compiled.refined_edge_mask.any())
    node_offset = 0
    edge_offset = 0
    for graph, (fractional, tokens) in enumerate(zip(fractional_parts, token_parts)):
        reference = prepare_assignment_carrier_example(
            _candidate(fractional, lattice[graph], tokens),
            embedding_key=f"identity-{graph}",
            material_id_audit_only=f"material-{graph}",
            evidence_role_audit_only="pretrain",
        )
        nodes = tokens.numel()
        edges = nodes * (nodes - 1)
        assert torch.equal(
            compiled.carrier.edge_source[edge_offset : edge_offset + edges] - node_offset,
            reference.edge_source,
        )
        assert torch.equal(
            compiled.carrier.edge_target[edge_offset : edge_offset + edges] - node_offset,
            reference.edge_target,
        )
        assert torch.allclose(
            compiled.carrier.edge_rbf[edge_offset : edge_offset + edges],
            reference.edge_rbf,
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            compiled.carrier.site_features[node_offset : node_offset + nodes],
            reference.site_features,
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            compiled.carrier.graph_features[graph],
            reference.graph_features,
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.equal(
            compiled.carrier.composition_counts[graph],
            torch.bincount(tokens, minlength=CHEMICAL_ELEMENT_COUNT),
        )
        node_offset += nodes
        edge_offset += edges


def test_dual_bound_cvp_matches_exact_solver_beyond_a_fixed_image_cube() -> None:
    fractional = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.49, 0.0]], dtype=torch.float64)
    lattice = torch.tensor(
        [[[1.0, 0.0, 0.0], [5.0, 0.1, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    batch = torch.zeros(2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="registered candidate cap"):
        exact_periodic_pair_distances(fractional, lattice, batch)
    source, target, _, _, _, distance, candidate_count, refined = exact_periodic_pair_distances(
        fractional,
        lattice,
        batch,
        maximum_refinement_candidates=20_000,
    )
    delta = (fractional[target] - fractional[source]).numpy()
    exact, _ = closest_image_displacements_numpy(delta, lattice[0].numpy())
    assert np.allclose(distance.numpy(), np.linalg.norm(exact, axis=1), atol=1e-12, rtol=1e-12)
    assert bool(refined.all())
    assert int(candidate_count.max()) > 27


def test_dual_bound_refines_elongated_cells_without_rejecting_valid_data() -> None:
    fractional = torch.tensor(
        [[0.0, 0.0, 0.0], [0.49, 0.49, 0.49], [0.2, 0.8, 0.1]],
        dtype=torch.float32,
    )
    lattice = torch.tensor(
        [[[3.55, 0.0, 0.0], [-1.775, 3.074, 0.0], [0.0, 0.0, 40.1]]],
        dtype=torch.float32,
    )
    batch = torch.zeros(3, dtype=torch.long)
    source, target, _, _, _, distance, candidate_count, refined = exact_periodic_pair_distances(
        fractional,
        lattice,
        batch,
    )
    delta = (fractional[target] - fractional[source]).to(torch.float64).numpy()
    exact, _ = closest_image_displacements_numpy(delta, lattice[0].to(torch.float64).numpy())
    assert np.allclose(distance.numpy(), np.linalg.norm(exact, axis=1), atol=2e-5, rtol=2e-5)
    assert bool(refined.any())
    assert int(candidate_count.max()) <= 4096


def test_global_batch_shards_cover_one_exact_pass_without_padding() -> None:
    permutation = torch.tensor([9, 2, 8, 4, 7, 3, 1, 6, 5, 0, 10])
    observed: list[int] = []
    for update in range(3):
        shards = [
            rank_shard_of_global_batch(
                permutation,
                update=update,
                global_batch_size=4,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        ]
        start = update * 4
        expected = permutation[start : start + 4]
        assert torch.equal(torch.sort(torch.cat(shards)).values, torch.sort(expected).values)
        observed.extend(torch.cat(shards).tolist())
    assert sorted(observed) == sorted(permutation.tolist())


def test_ddp_loss_scaling_matches_one_global_mean_with_uneven_ranks() -> None:
    parameter = torch.tensor(1.7, requires_grad=True)
    rank_zero = (parameter * torch.tensor([1.0, 2.0, 4.0])).square()
    rank_one = (parameter * torch.tensor([3.0, 5.0])).square()
    distributed = 0.5 * (
        ddp_global_mean_loss(rank_zero, global_count=5, world_size=2)
        + ddp_global_mean_loss(rank_one, global_count=5, world_size=2)
    )
    (distributed - torch.cat((rank_zero, rank_one)).mean()).backward()
    assert float(parameter.grad.abs()) <= 1e-6


def test_rank_sharded_reveal_order_is_world_size_invariant() -> None:
    counts = torch.tensor([3, 5, 2, 4, 1], dtype=torch.long)
    generator = torch.Generator().manual_seed(5705)
    graph = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    expected = sample_uniform_reveal_ranks(graph, generator=generator)
    observed: list[torch.Tensor] = []
    for rank in range(2):
        observed.append(
            sample_rank_sharded_reveal_ranks(
                counts,
                rank=rank,
                world_size=2,
                generator=torch.Generator().manual_seed(5705),
                device="cpu",
            )
        )
    for rank, local in enumerate(observed):
        assert torch.equal(local, expected[graph.remainder(2) == rank])
