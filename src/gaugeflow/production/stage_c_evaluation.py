"""Deterministic helpers for held-out Stage-C structure evaluation."""

from __future__ import annotations

import math

import torch

from .hybrid_diffusion import HybridLossOutput
from .orderless_product_state import orderless_next_reveal_nll


def balanced_functional_panel(
    functional_group_index: torch.Tensor,
    *,
    functional_count: int,
    graphs_per_functional: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Select a target-independent, source-balanced held-out panel.

    Returned indices retain random order within each functional.  The same
    tuple can therefore be reused across checkpoints for paired noise and row
    exposure without material identifiers entering a model batch.
    """

    if (
        functional_group_index.ndim != 1
        or functional_group_index.dtype != torch.long
        or functional_group_index.numel() < 1
        or functional_count < 1
        or graphs_per_functional < 1
        or int(functional_group_index.min()) < 0
        or int(functional_group_index.max()) >= functional_count
    ):
        raise ValueError("functional panel inputs are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    panels: list[torch.Tensor] = []
    for functional in range(functional_count):
        candidates = torch.nonzero(
            functional_group_index == functional,
            as_tuple=False,
        ).squeeze(1)
        if candidates.numel() < graphs_per_functional:
            raise ValueError("functional lacks the declared calibration support")
        permutation = torch.randperm(candidates.numel(), generator=generator)
        panels.append(candidates[permutation[:graphs_per_functional]].contiguous())
    return tuple(panels)


def graphwise_structure_replay_loss(output: HybridLossOutput) -> torch.Tensor:
    """Recover the graphwise product-space objective used by Stage-C replay."""

    occupation = output.noisy.orderless_occupation
    if occupation is None or output.noisy.composition_counts is None:
        raise ValueError("Stage-C structure evaluation requires the exact-count orderless path")
    element, _ = orderless_next_reveal_nll(
        output.prediction.clean_element_logits,
        occupation,
    )
    coordinate = output.graph_coordinate_loss / 3.0
    volume = (
        output.prediction.clean_volume_latent - output.noisy.clean_volume_latent_target
    ).square()
    if volume.ndim > 1:
        volume = volume.flatten(start_dim=1).mean(dim=1)
    shape = (
        output.prediction.clean_shape_latent - output.noisy.clean_shape_latent_target
    ).square().flatten(start_dim=1).mean(dim=1)
    graph_loss = element + coordinate + volume + shape
    if not torch.isfinite(graph_loss).all():
        raise FloatingPointError("graphwise Stage-C structure loss is non-finite")
    if not torch.allclose(
        graph_loss.mean(),
        output.loss.float(),
        atol=5.0e-5,
        rtol=5.0e-5,
    ):
        raise AssertionError("graphwise Stage-C loss does not reconstruct the training objective")
    return graph_loss


def select_pareto_minimax_checkpoint(
    objectives: dict[int, dict[str, float]],
) -> dict[str, object]:
    """Select a balanced checkpoint without collapsing objectives before audit."""

    if len(objectives) < 2:
        raise ValueError("checkpoint selection needs at least two eligible candidates")
    metric_names = tuple(next(iter(objectives.values())))
    if not metric_names or any(tuple(values) != metric_names for values in objectives.values()):
        raise ValueError("checkpoint candidates have inconsistent objectives")
    if any(
        not math.isfinite(float(value))
        for values in objectives.values()
        for value in values.values()
    ):
        raise ValueError("checkpoint objective is non-finite")

    steps = sorted(objectives)
    pareto: list[int] = []
    for candidate in steps:
        current = objectives[candidate]
        dominated = any(
            other != candidate
            and all(objectives[other][name] <= current[name] for name in metric_names)
            and any(objectives[other][name] < current[name] for name in metric_names)
            for other in steps
        )
        if not dominated:
            pareto.append(candidate)

    minima = {
        name: min(objectives[step][name] for step in steps) for name in metric_names
    }
    maxima = {
        name: max(objectives[step][name] for step in steps) for name in metric_names
    }
    regrets: dict[int, dict[str, float]] = {}
    for step in pareto:
        regrets[step] = {
            name: 0.0
            if maxima[name] == minima[name]
            else (objectives[step][name] - minima[name]) / (maxima[name] - minima[name])
            for name in metric_names
        }
    selected = min(
        pareto,
        key=lambda step: (
            max(regrets[step].values()),
            sum(regrets[step].values()) / len(metric_names),
            step,
        ),
    )
    return {
        "selected_stage_c_step": selected,
        "pareto_stage_c_steps": pareto,
        "normalized_regrets": regrets,
        "maximum_regret": max(regrets[selected].values()),
        "mean_regret": sum(regrets[selected].values()) / len(metric_names),
    }
