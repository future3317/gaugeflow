from __future__ import annotations

import torch

from gaugeflow.production.blueprint import EmpiricalNodeCountPrior
from gaugeflow.production.generation_law import (
    CrystalGenerationState,
    FactorizedGenerationLogProbability,
    LearnedNodeCountLaw,
    ParentDeltaNodeCountLaw,
)


def test_empirical_node_count_prior_has_an_explicit_probability_law() -> None:
    law = EmpiricalNodeCountPrior.fit(torch.tensor([1, 2, 2, 4]))
    actual = law.log_prob(torch.tensor([1, 2, 4, 3]))
    expected = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64).log()
    assert torch.allclose(actual[:3], expected)
    assert torch.isneginf(actual[3])


def test_learned_and_parent_conditioned_node_count_laws_are_normalized() -> None:
    torch.manual_seed(13)
    learned = LearnedNodeCountLaw(5, context_dim=3).double()
    context = torch.randn(2, 3, dtype=torch.float64)
    support = torch.arange(1, 6).repeat_interleave(2)
    repeated = context.repeat(5, 1)
    log_probability = learned.log_prob(support, repeated).reshape(5, 2)
    assert torch.allclose(torch.logsumexp(log_probability, dim=0), torch.zeros(2, dtype=torch.float64))
    parent = ParentDeltaNodeCountLaw(torch.tensor([2, 4]))
    sampled, logp = parent.sample()
    assert torch.equal(sampled, torch.tensor([2, 4]))
    assert torch.equal(logp, torch.zeros(2))
    assert torch.isneginf(parent.log_prob(torch.tensor([2, 3]))[1])


def test_generation_state_closes_n_c_a_l_f_exactly() -> None:
    state = CrystalGenerationState(
        node_count=torch.tensor([2, 3]),
        composition_counts=torch.tensor(
            [[1, 1, 0, 0], [0, 1, 0, 2]], dtype=torch.long
        ),
        assignment=torch.tensor([0, 1, 3, 1, 3]),
        batch=torch.tensor([0, 0, 1, 1, 1]),
        lattice=torch.eye(3).expand(2, -1, -1).clone(),
        fractional_coordinates=torch.rand(5, 3),
    )
    state.validate(vocabulary_size=4)
    logp = FactorizedGenerationLogProbability(
        *(torch.tensor([-1.0, -2.0]) for _ in range(5))
    )
    assert torch.equal(logp.total, torch.tensor([-5.0, -10.0]))
