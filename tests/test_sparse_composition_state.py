from __future__ import annotations

import itertools
import math

import pytest
import torch

from gaugeflow.production.composition_state import (
    IntegerPartitionCatalogue,
    SparseCompositionState,
    StoichiometryFirstCompositionModel,
    fit_integer_partition_log_prior,
)


def _positive_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    return [
        (first, *tail)
        for first in range(1, total - parts + 2)
        for tail in _positive_compositions(total - first, parts - 1)
    ]


def _enumerate_states(
    node_count: int,
    *,
    vocabulary_size: int,
    maximum_species: int,
) -> SparseCompositionState:
    species_rows: list[list[int]] = []
    count_rows: list[list[int]] = []
    lengths: list[int] = []
    for support in range(1, min(node_count, vocabulary_size, maximum_species) + 1):
        for species in itertools.combinations(range(vocabulary_size), support):
            for counts in _positive_compositions(node_count, support):
                species_rows.append([*species, *([-1] * (maximum_species - support))])
                count_rows.append([*counts, *([0] * (maximum_species - support))])
                lengths.append(support)
    graphs = len(species_rows)
    return SparseCompositionState(
        species=torch.tensor(species_rows, dtype=torch.long),
        counts=torch.tensor(count_rows, dtype=torch.long),
        length=torch.tensor(lengths, dtype=torch.long),
        node_count=torch.full((graphs,), node_count, dtype=torch.long),
    )


def test_sparse_composition_roundtrip_quotients_site_order() -> None:
    tokens = torch.tensor([4, 1, 4, 7, 1, 3, 3, 3], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    first = SparseCompositionState.from_element_tokens(
        tokens,
        batch,
        graphs=2,
        maximum_species=4,
        vocabulary_size=9,
    )
    permutation = torch.tensor([4, 1, 2, 0, 3, 7, 5, 6])
    second = SparseCompositionState.from_element_tokens(
        tokens[permutation],
        batch,
        graphs=2,
        maximum_species=4,
        vocabulary_size=9,
    )
    assert torch.equal(first.species, second.species)
    assert torch.equal(first.counts, second.counts)
    assert torch.equal(first.to_dense(9), second.to_dense(9))
    assert torch.equal(
        first.to_dense(9),
        torch.tensor(
            [[0, 2, 0, 0, 2, 0, 0, 1, 0], [0, 0, 0, 3, 0, 0, 0, 0, 0]],
            dtype=torch.long,
        ),
    )


def test_packed_sparse_composition_matches_batched_dense_encoding() -> None:
    tokens = torch.tensor([4, 1, 4, 7, 1, 3, 3, 3], dtype=torch.long)
    offsets = torch.tensor([0, 5, 8], dtype=torch.long)
    packed = SparseCompositionState.from_packed_element_tokens(
        tokens,
        offsets,
        maximum_species=4,
        vocabulary_size=9,
    )
    batch = torch.repeat_interleave(torch.arange(2), offsets.diff())
    batched = SparseCompositionState.from_element_tokens(
        tokens,
        batch,
        graphs=2,
        maximum_species=4,
        vocabulary_size=9,
    )
    assert torch.equal(packed.species, batched.species)
    assert torch.equal(packed.counts, batched.counts)
    selected = packed.index_select(torch.tensor([1, 0])).to("cpu")
    assert torch.equal(selected.to_dense(9), packed.to_dense(9).flip(0))


def test_sparse_state_rejects_noncanonical_or_nonconserving_values() -> None:
    noncanonical = SparseCompositionState(
        species=torch.tensor([[4, 2, -1]]),
        counts=torch.tensor([[1, 1, 0]]),
        length=torch.tensor([2]),
        node_count=torch.tensor([2]),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        noncanonical.validate(vocabulary_size=6)
    nonconserving = SparseCompositionState(
        species=torch.tensor([[2, -1, -1]]),
        counts=torch.tensor([[1, 0, 0]]),
        length=torch.tensor([1]),
        node_count=torch.tensor([2]),
    )
    with pytest.raises(ValueError, match="sum to node_count"):
        nonconserving.validate(vocabulary_size=6)


def _uniform_partition_prior(
    *, maximum_atoms: int, maximum_species: int
) -> tuple[IntegerPartitionCatalogue, torch.Tensor]:
    catalogue = IntegerPartitionCatalogue.build(
        maximum_atoms=maximum_atoms,
        maximum_species=maximum_species,
    )
    log_prior = torch.empty(catalogue.size, dtype=torch.float64)
    for atoms in range(1, maximum_atoms + 1):
        valid = catalogue.node_count == atoms
        log_prior[valid] = -math.log(int(valid.sum()))
    return catalogue, log_prior


@pytest.mark.parametrize("node_count", range(1, 7))
def test_stoichiometry_first_model_is_exactly_normalized(node_count: int) -> None:
    torch.manual_seed(37)
    _, log_prior = _uniform_partition_prior(maximum_atoms=6, maximum_species=3)
    model = StoichiometryFirstCompositionModel(
        4,
        9,
        log_prior,
        maximum_atoms=6,
        maximum_species=3,
        vocabulary_size=5,
    ).double()
    state = _enumerate_states(node_count, vocabulary_size=5, maximum_species=3)
    context = torch.tensor([[0.3, -0.2, 0.1, 0.8]], dtype=torch.float64).expand(state.graphs, -1)
    log_probability = model.log_prob(context, state.node_count, state).total
    assert torch.allclose(
        torch.logsumexp(log_probability, dim=0),
        torch.zeros((), dtype=torch.float64),
        atol=2e-12,
        rtol=0.0,
    )


def test_stoichiometry_first_sampling_closes_and_recomputes_probability() -> None:
    torch.manual_seed(41)
    _, log_prior = _uniform_partition_prior(maximum_atoms=8, maximum_species=4)
    active = torch.tensor([True, False, True, True, False, True])
    model = StoichiometryFirstCompositionModel(
        5,
        12,
        log_prior,
        maximum_atoms=8,
        maximum_species=4,
        vocabulary_size=6,
        active_vocabulary_mask=active,
    )
    context = torch.randn(256, 5)
    node_count = torch.randint(1, 9, (256,))
    sample = model.sample(
        context,
        node_count,
        generator=torch.Generator().manual_seed(43),
    )
    sample.state.validate(vocabulary_size=6, maximum_atoms=8)
    assert torch.equal(sample.state.to_dense(6).sum(dim=1), node_count)
    recomputed = model.log_prob(context, node_count, sample.state).total
    assert torch.allclose(sample.log_probability, recomputed, atol=2e-5, rtol=2e-5)


def test_stoichiometry_first_slot_likelihood_and_fixed_partition_sampling_close() -> None:
    torch.manual_seed(47)
    catalogue, log_prior = _uniform_partition_prior(maximum_atoms=8, maximum_species=4)
    model = StoichiometryFirstCompositionModel(
        5,
        12,
        log_prior,
        maximum_atoms=8,
        maximum_species=4,
        vocabulary_size=6,
    )
    context = torch.randn(128, 5)
    partition_index = torch.randint(catalogue.size, (128,))
    sampled = model.sample_species_given_partition(
        context,
        partition_index,
        generator=torch.Generator().manual_seed(53),
    )
    expected_count = catalogue.counts.index_select(0, partition_index)
    assert torch.equal(
        torch.sort(sampled.state.counts, dim=1, descending=True).values,
        expected_count,
    )
    node_count = catalogue.node_count.index_select(0, partition_index)
    by_slot = model.species_log_probability_by_slot(context, node_count, sampled.state)
    active = torch.arange(4).unsqueeze(0) < sampled.state.length.unsqueeze(1)
    assert torch.equal(by_slot[~active], torch.zeros_like(by_slot[~active]))
    assert torch.allclose(sampled.log_probability, by_slot.sum(dim=1), atol=2e-5, rtol=2e-5)


def test_empirical_partition_prior_is_normalized_and_permutation_free() -> None:
    dense = torch.tensor(
        [
            [2, 0, 1, 0, 1],
            [0, 1, 0, 2, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 2, 0, 2],
        ],
        dtype=torch.long,
    )
    state = SparseCompositionState.from_dense(dense, maximum_species=4)
    catalogue = IntegerPartitionCatalogue.build(maximum_atoms=6, maximum_species=4)
    log_prior = fit_integer_partition_log_prior(
        state,
        catalogue,
        maximum_atoms=6,
        smoothing=0.5,
    )
    for atoms in range(1, 7):
        valid = catalogue.node_count == atoms
        assert torch.allclose(
            torch.logsumexp(log_prior[valid], dim=0),
            torch.zeros((), dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )
    encoded = catalogue.encode(state, maximum_atoms=6)
    permuted_counts = state.counts.gather(
        1,
        torch.tensor([[2, 0, 1, 3]]).expand(state.graphs, -1),
    )
    permuted = SparseCompositionState(
        species=state.species,
        counts=permuted_counts,
        length=state.length,
        node_count=state.node_count,
    )
    # Catalogue encoding quotients count order; it does not inspect species order.
    assert torch.equal(
        encoded,
        catalogue.encode(permuted, maximum_atoms=6),
    )
