"""Graph-equal sufficient statistics for Stage-B physical validation."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .physical_pretraining import PhysicalPredictions, PhysicalTargets, sorted_segment_sum

PHYSICAL_STATISTIC_DIM = 9


def physical_metric_sums(
    prediction: PhysicalPredictions,
    target: PhysicalTargets,
    batch: torch.Tensor,
    functional_index: torch.Tensor,
    functional_count: int,
) -> torch.Tensor:
    """Return per-functional sums/counts without retaining graph predictions."""

    graph_count = functional_index.numel()
    if functional_count < 1 or functional_index.shape != (graph_count,):
        raise ValueError("physical evaluation functional indices are invalid")
    if batch.ndim != 1 or batch.numel() != prediction.forces.shape[0]:
        raise ValueError("physical evaluation batch is invalid")
    if int(functional_index.min()) < 0 or int(functional_index.max()) >= functional_count:
        raise ValueError("physical evaluation functional index is out of range")
    statistics = prediction.energy_per_atom.new_zeros(
        (functional_count, PHYSICAL_STATISTIC_DIM), dtype=torch.float64
    )

    def add(values: torch.Tensor, mask: torch.Tensor, sum_column: int, count_column: int) -> None:
        selected_functional = functional_index[mask]
        statistics[:, sum_column].index_add_(0, selected_functional, values[mask].double())
        statistics[:, count_column].index_add_(
            0,
            selected_functional,
            torch.ones(selected_functional.numel(), dtype=torch.float64, device=values.device),
        )

    energy_error = (prediction.energy_per_atom - target.energy_per_atom).square()
    add(energy_error, target.energy_mask, 0, 1)

    force_node_mse = (prediction.forces - target.forces).square().mean(dim=-1)
    force_node_cosine = torch.nn.functional.cosine_similarity(
        prediction.forces.float(), target.forces.float(), dim=-1, eps=1.0e-8
    )
    force_count = torch.bincount(batch[target.force_mask], minlength=graph_count)
    force_mask = force_count > 0
    force_mse = sorted_segment_sum(
        force_node_mse * target.force_mask.to(force_node_mse), batch, graph_count
    ) / force_count.clamp_min(1).to(force_node_mse)
    force_cosine = sorted_segment_sum(
        force_node_cosine * target.force_mask.to(force_node_cosine), batch, graph_count
    ) / force_count.clamp_min(1).to(force_node_cosine)
    add(force_mse, force_mask, 2, 4)
    statistics[:, 3].index_add_(
        0,
        functional_index[force_mask],
        force_cosine[force_mask].double(),
    )

    stress_error = (prediction.stress_kelvin - target.stress_kelvin).square().mean(dim=-1)
    add(stress_error, target.stress_mask, 5, 6)

    feature_node_cosine = torch.nn.functional.cosine_similarity(
        prediction.teacher_features.float(),
        target.teacher_features.float(),
        dim=-1,
        eps=1.0e-8,
    )
    feature_count = torch.bincount(batch[target.teacher_mask], minlength=graph_count)
    feature_mask = feature_count > 0
    feature_cosine = sorted_segment_sum(
        feature_node_cosine * target.teacher_mask.to(feature_node_cosine),
        batch,
        graph_count,
    ) / feature_count.clamp_min(1).to(feature_node_cosine)
    statistics[:, 7].index_add_(
        0,
        functional_index[feature_mask],
        feature_cosine[feature_mask].double(),
    )
    statistics[:, 8].index_add_(
        0,
        functional_index[feature_mask],
        torch.ones(int(feature_mask.sum()), dtype=torch.float64, device=batch.device),
    )
    return statistics


def finalize_physical_metrics(
    statistics: torch.Tensor,
    functional_vocabulary: Mapping[str, int],
) -> dict[str, Any]:
    """Convert additive statistics into separately labelled and aggregate metrics."""

    if statistics.shape != (len(functional_vocabulary), PHYSICAL_STATISTIC_DIM):
        raise ValueError("physical evaluation statistic shape is invalid")
    if set(functional_vocabulary.values()) != set(range(len(functional_vocabulary))):
        raise ValueError("functional vocabulary must be contiguous")

    def metrics(row: torch.Tensor) -> dict[str, float | int | None]:
        energy_count = int(row[1])
        force_count = int(row[4])
        stress_count = int(row[6])
        feature_count = int(row[8])
        energy_mse = float(row[0] / energy_count) if energy_count else None
        force_mse = float(row[2] / force_count) if force_count else None
        stress_mse = float(row[5] / stress_count) if stress_count else None
        feature_cosine = float(row[7] / feature_count) if feature_count else None
        supervised_losses = [value for value in (energy_mse, force_mse, stress_mse) if value is not None]
        if feature_cosine is not None:
            supervised_losses.append(1.0 - feature_cosine)
        return {
            "normalized_energy_rmse": energy_mse**0.5 if energy_mse is not None else None,
            "normalized_force_rmse": force_mse**0.5 if force_mse is not None else None,
            "force_cosine": float(row[3] / force_count) if force_count else None,
            "normalized_kelvin_stress_rmse": stress_mse**0.5 if stress_mse is not None else None,
            "teacher_feature_cosine": feature_cosine,
            "equal_head_composite_loss": sum(supervised_losses),
            "energy_graphs": energy_count,
            "force_graphs": force_count,
            "stress_graphs": stress_count,
            "feature_graphs": feature_count,
        }

    per_functional = {
        name: metrics(statistics[index])
        for name, index in sorted(functional_vocabulary.items(), key=lambda item: item[1])
    }
    return {
        "aggregate": metrics(statistics.sum(dim=0)),
        "per_functional": per_functional,
    }
