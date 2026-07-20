from __future__ import annotations

import math

import torch

from gaugeflow.production.composition_qualification import (
    evaluate_species_slots,
    fit_count_slot_reference,
    graph_mean,
    pair_calibration_metrics,
    structure_bootstrap_mean,
)
from gaugeflow.production.composition_state import (
    IntegerPartitionCatalogue,
    SparseCompositionState,
    StoichiometryFirstCompositionModel,
)


def _model() -> StoichiometryFirstCompositionModel:
    catalogue = IntegerPartitionCatalogue.build(maximum_atoms=4, maximum_species=3)
    log_prior = torch.empty(catalogue.size, dtype=torch.float64)
    for atoms in range(1, 5):
        selected = catalogue.node_count == atoms
        log_prior[selected] = -math.log(int(selected.sum()))
    return StoichiometryFirstCompositionModel(
        1,
        8,
        log_prior,
        maximum_atoms=4,
        maximum_species=3,
        vocabulary_size=5,
    ).float()


def _state() -> SparseCompositionState:
    return SparseCompositionState(
        species=torch.tensor([[0, 1, -1], [0, 2, -1], [1, 2, -1], [0, -1, -1]]),
        counts=torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [2, 0, 0]]),
        length=torch.tensor([2, 2, 2, 1]),
        node_count=torch.tensor([2, 2, 2, 2]),
    )


def test_legal_count_slot_reference_and_species_metrics_are_finite() -> None:
    model = _model()
    state = _state()
    reference = fit_count_slot_reference(model, state, smoothing=0.5)
    metrics = evaluate_species_slots(
        model,
        state,
        reference,
        batch_size=2,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    assert metrics["model_nll"].numel() == int(state.length.sum())
    assert torch.isfinite(metrics["model_nll"]).all()
    assert torch.isfinite(metrics["empirical_nll"]).all()
    assert torch.isfinite(metrics["uniform_nll"]).all()
    reduced = graph_mean(
        metrics["model_nll"], metrics["graph_index"], state.graphs
    )
    assert reduced.shape == (state.graphs,)


def test_structure_bootstrap_and_pair_calibration_are_deterministic() -> None:
    values = torch.tensor([-0.3, -0.2, 0.1, -0.4])
    first = structure_bootstrap_mean(values, seed=41, replicates=200)
    second = structure_bootstrap_mean(values, seed=41, replicates=200)
    assert first == second
    state = _state()
    mask = torch.zeros((5, 5), dtype=torch.bool)
    mask[0, 1] = mask[0, 2] = mask[1, 2] = True
    pair = pair_calibration_metrics(state, state, mask, vocabulary_size=5)
    assert pair["qualified_pairs"] == 3
    assert pair["pair_probability_rmse"] == 0.0
    assert pair["pair_identity_recall"] == 1.0
