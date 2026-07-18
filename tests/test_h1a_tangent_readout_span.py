from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.audit_h1a_tangent_readout_span import (
    DesignBlock,
    classify_attribution,
    evaluate_readout,
    solve_minimum_norm,
    tensor_candidate_count,
)


def test_minimum_norm_readout_recovers_exact_full_rank_target() -> None:
    generator = torch.Generator().manual_seed(23)
    design = torch.randn((128, 4), dtype=torch.float64, generator=generator)
    expected = torch.tensor([0.5, -1.0, 0.25, 2.0], dtype=torch.float64)
    target = design @ expected
    weight, span = solve_minimum_norm(design, target, rcond=1.0e-10)
    metrics = evaluate_readout(
        DesignBlock(design=design, target=target, graph_observations=32),
        weight,
    )
    assert span["rank"] == 4
    assert torch.allclose(weight, expected, atol=1.0e-10, rtol=1.0e-10)
    assert metrics["loss"] < 1.0e-20
    assert metrics["explained_fraction"] > 1.0 - 1.0e-12


def test_attribution_classification_obeys_preregistered_order() -> None:
    thresholds = {
        "generalizable_head_train_relative_max": 0.5,
        "generalizable_head_validation_relative_max": 0.75,
        "validation_oracle_explained_fraction_min": 0.75,
    }
    assert (
        classify_attribution(
            train_relative=0.4,
            validation_relative=0.7,
            validation_oracle_explained=0.2,
            thresholds=thresholds,
        )
        == "head_optimization_limited"
    )
    assert (
        classify_attribution(
            train_relative=0.8,
            validation_relative=0.9,
            validation_oracle_explained=0.7,
            thresholds=thresholds,
        )
        == "backbone_span_limited"
    )
    assert (
        classify_attribution(
            train_relative=0.4,
            validation_relative=0.9,
            validation_oracle_explained=0.9,
            thresholds=thresholds,
        )
        == "split_specific_readout"
    )
    assert (
        classify_attribution(
            train_relative=0.8,
            validation_relative=0.9,
            validation_oracle_explained=0.9,
            thresholds=thresholds,
        )
        == "distributed_nonlinear_optimization"
    )


def test_tensor_candidate_count_uses_null_condition_contract() -> None:
    atlas = SimpleNamespace(effective_frame_count=torch.tensor([0, 0, 0]))
    assert tensor_candidate_count(atlas) == 0
