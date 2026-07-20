"""Shared evaluation kernels for exact stochastic composition models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch

from .composition_state import SparseCompositionSample, SparseCompositionState


class ExactCompositionModel(Protocol):
    maximum_atoms: int

    def log_prob(
        self,
        context: torch.Tensor,
        node_count: torch.Tensor,
        state: SparseCompositionState,
    ) -> Any: ...

    def sample(
        self,
        context: torch.Tensor,
        node_count: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        mode: bool = False,
    ) -> SparseCompositionSample: ...

    def eval(self) -> Any: ...


def load_compositions(
    path: Path,
    *,
    maximum_species: int,
    vocabulary_size: int,
) -> SparseCompositionState:
    packed = torch.load(path, map_location="cpu", weights_only=True)
    return SparseCompositionState.from_packed_element_tokens(
        packed["atom_tokens"].long(),
        packed["offsets"].long(),
        maximum_species=maximum_species,
        vocabulary_size=vocabulary_size,
    )


@torch.no_grad()
def evaluate_nll(
    model: ExactCompositionModel,
    state: SparseCompositionState,
    *,
    batch_size: int,
    device: torch.device,
    context_dim: int,
    use_bf16: bool,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "support": 0.0, "species": 0.0, "counts": 0.0}
    for start in range(0, state.graphs, batch_size):
        stop = min(start + batch_size, state.graphs)
        selected = state.index_select(torch.arange(start, stop, dtype=torch.long)).to(device)
        context = torch.ones((stop - start, context_dim), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            output = model.log_prob(context, selected.node_count, selected)
        for name in totals:
            totals[name] += float(-getattr(output, name).float().sum())
    return {f"mean_{name}_nll": value / state.graphs for name, value in totals.items()}


def jensen_shannon(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.double() / first.sum()
    second = second.double() / second.sum()
    midpoint = 0.5 * (first + second)
    first_term = torch.where(first > 0, first * (first.log() - midpoint.log()), 0.0)
    second_term = torch.where(second > 0, second * (second.log() - midpoint.log()), 0.0)
    return float(0.5 * (first_term.sum() + second_term.sum()))


def categorical_total_variation(first: torch.Tensor, second: torch.Tensor) -> float:
    keys = torch.cat((first, second))
    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    first_counts = torch.bincount(inverse[: first.numel()])
    second_counts = torch.bincount(inverse[first.numel() :], minlength=first_counts.numel())
    size = max(first_counts.numel(), second_counts.numel())
    first_counts = torch.nn.functional.pad(first_counts, (0, size - first_counts.numel()))
    second_counts = torch.nn.functional.pad(second_counts, (0, size - second_counts.numel()))
    first_probability = first_counts.double() / first.numel()
    second_probability = second_counts.double() / second.numel()
    return float(0.5 * (first_probability - second_probability).abs().sum())


def partition_key(state: SparseCompositionState) -> torch.Tensor:
    ordered = torch.sort(state.counts, dim=1, descending=True).values
    powers = 21 ** torch.arange(state.maximum_species, dtype=torch.long)
    return (ordered * powers.unsqueeze(0)).sum(dim=1)


@torch.no_grad()
def sample_validation(
    model: ExactCompositionModel,
    reference: SparseCompositionState,
    *,
    batch_size: int,
    device: torch.device,
    context_dim: int,
    vocabulary_size: int,
    seed: int,
    minimum_reference_atoms: int,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    samples: list[SparseCompositionState] = []
    failures = 0
    for start in range(0, reference.graphs, batch_size):
        stop = min(start + batch_size, reference.graphs)
        node_count = reference.node_count[start:stop].to(device)
        context = torch.ones((stop - start, context_dim), device=device)
        try:
            samples.append(model.sample(context, node_count, generator=generator).state.to("cpu"))
        except (RuntimeError, ValueError, FloatingPointError):
            failures += stop - start
    if failures or sum(sample.graphs for sample in samples) != reference.graphs:
        return {"sampling_failures": failures, "successful_graphs": 0}
    sampled = SparseCompositionState(
        torch.cat([sample.species for sample in samples]),
        torch.cat([sample.counts for sample in samples]),
        torch.cat([sample.length for sample in samples]),
        torch.cat([sample.node_count for sample in samples]),
    )
    sampled.validate(vocabulary_size=vocabulary_size, maximum_atoms=model.maximum_atoms)
    sampled_dense = sampled.to_dense(vocabulary_size)
    reference_dense = reference.to_dense(vocabulary_size)
    sampled_element = sampled_dense.sum(dim=0)
    reference_element = reference_dense.sum(dim=0)
    supported = reference_element >= minimum_reference_atoms
    overlap = torch.minimum(sampled_dense, reference_dense).sum(dim=1) / reference.node_count
    exact = (sampled_dense == reference_dense).all(dim=1)
    return {
        "sampling_failures": 0,
        "successful_graphs": sampled.graphs,
        "atom_count_preservation": float((sampled_dense.sum(dim=1) == reference.node_count).float().mean()),
        "invalid_compositions": int((sampled_dense.sum(dim=1) != reference.node_count).sum()),
        "element_marginal_jsd": jensen_shannon(sampled_element, reference_element),
        "support_size_total_variation": categorical_total_variation(sampled.length, reference.length),
        "count_partition_total_variation": categorical_total_variation(
            partition_key(sampled), partition_key(reference)
        ),
        "supported_element_recall": float((sampled_element[supported] > 0).float().mean()),
        "supported_elements": int(supported.sum()),
        "paired_exact_composition_accuracy_diagnostic": float(exact.float().mean()),
        "paired_composition_overlap_diagnostic": float(overlap.float().mean()),
        "sampled_formula_unique_fraction": float(torch.unique(sampled_dense, dim=0).shape[0] / sampled.graphs),
    }
