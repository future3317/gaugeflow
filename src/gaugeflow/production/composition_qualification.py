"""Qualification statistics for exact stoichiometry-first composition laws."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .composition_metrics import jensen_shannon
from .composition_state import (
    SparseCompositionState,
    StoichiometryFirstCompositionModel,
)


@dataclass(frozen=True)
class CountSlotReference:
    """Fit-only legal categorical reference indexed by ``(N, slot, count)``."""

    frequency: torch.Tensor
    smoothing: float

    def validate(self, model: StoichiometryFirstCompositionModel) -> None:
        rows = (
            (model.maximum_atoms + 1)
            * model.maximum_species
            * (model.maximum_atoms + 1)
        )
        if self.frequency.shape != (rows, model.vocabulary_size):
            raise ValueError("count-slot reference table has the wrong shape")
        if self.frequency.dtype != torch.float64 or bool((self.frequency < 0).any()):
            raise ValueError("count-slot reference frequencies must be nonnegative FP64")
        if self.smoothing <= 0:
            raise ValueError("count-slot reference smoothing must be positive")


def _slot_bucket(
    node_count: torch.Tensor,
    positions: torch.Tensor,
    counts: torch.Tensor,
    model: StoichiometryFirstCompositionModel,
) -> torch.Tensor:
    return (
        (node_count.unsqueeze(1) * model.maximum_species + positions)
        * (model.maximum_atoms + 1)
        + counts
    )


def fit_count_slot_reference(
    model: StoichiometryFirstCompositionModel,
    fit: SparseCompositionState,
    *,
    smoothing: float,
) -> CountSlotReference:
    """Fit the lower-context empirical law without reading any panel row."""

    if smoothing <= 0:
        raise ValueError("count-slot smoothing must be positive")
    species, counts = model.count_first_order(fit)
    positions = torch.arange(model.maximum_species).unsqueeze(0).expand(fit.graphs, -1)
    active = positions < fit.length.unsqueeze(1)
    bucket = _slot_bucket(fit.node_count, positions, counts, model)
    flat = bucket[active] * model.vocabulary_size + species[active]
    rows = (
        (model.maximum_atoms + 1)
        * model.maximum_species
        * (model.maximum_atoms + 1)
    )
    frequency = torch.bincount(
        flat, minlength=rows * model.vocabulary_size
    ).reshape(rows, model.vocabulary_size)
    reference = CountSlotReference(frequency.double(), smoothing)
    reference.validate(model)
    return reference


@torch.no_grad()
def evaluate_species_slots(
    model: StoichiometryFirstCompositionModel,
    state: SparseCompositionState,
    reference: CountSlotReference,
    *,
    batch_size: int,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, torch.Tensor]:
    """Return decision-level model and legal-baseline NLLs with audit keys."""

    reference.validate(model)
    model.eval()
    names = (
        "model_nll",
        "empirical_nll",
        "uniform_nll",
        "species",
        "count",
        "slot",
        "support",
        "node_count",
        "partition_key",
        "fit_event_frequency",
        "graph_index",
    )
    output: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    frequency = reference.frequency.to(device)
    positions_template = torch.arange(model.maximum_species, device=device)
    for start in range(0, state.graphs, batch_size):
        stop = min(start + batch_size, state.graphs)
        selected = state.index_select(torch.arange(start, stop)).to(device)
        context = torch.ones((selected.graphs, model.context_dim), device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            selected_log = model.species_log_probability_by_slot(
                context, selected.node_count, selected
            )
        species, counts = model.count_first_order(selected)
        positions = positions_template.unsqueeze(0).expand(selected.graphs, -1)
        active = positions < selected.length.unsqueeze(1)
        valid = model.species_validity_by_slot(selected)
        uniform_nll = valid.sum(dim=2).to(torch.float64).log()
        bucket = _slot_bucket(selected.node_count, positions, counts, model)
        raw_frequency = frequency.index_select(0, bucket.reshape(-1)).reshape(
            selected.graphs,
            model.maximum_species,
            model.vocabulary_size,
        )
        weights = (raw_frequency + reference.smoothing).masked_fill(~valid, 0.0)
        empirical_log = weights.gather(
            2, species.clamp_min(0).unsqueeze(2)
        ).squeeze(2).log() - weights.sum(dim=2).log()
        chosen_frequency = raw_frequency.gather(
            2, species.clamp_min(0).unsqueeze(2)
        ).squeeze(2)
        ordered_counts = torch.sort(selected.counts, dim=1, descending=True).values
        powers = 21 ** torch.arange(model.maximum_species, device=device)
        key = (ordered_counts * powers.unsqueeze(0)).sum(dim=1)
        values = {
            "model_nll": -selected_log.float(),
            "empirical_nll": -empirical_log,
            "uniform_nll": uniform_nll,
            "species": species,
            "count": counts,
            "slot": positions,
            "support": selected.length.unsqueeze(1).expand_as(counts),
            "node_count": selected.node_count.unsqueeze(1).expand_as(counts),
            "partition_key": key.unsqueeze(1).expand_as(counts),
            "fit_event_frequency": chosen_frequency,
            "graph_index": torch.arange(start, stop, device=device)
            .unsqueeze(1)
            .expand_as(counts),
        }
        for name, value in values.items():
            output[name].append(value[active].detach().cpu())
    return {name: torch.cat(parts) for name, parts in output.items()}


def graph_mean(
    values: torch.Tensor,
    graph_index: torch.Tensor,
    graphs: int,
) -> torch.Tensor:
    """Reduce decision values to equal-weight structure means."""

    if values.ndim != 1 or graph_index.shape != values.shape:
        raise ValueError("graph mean requires aligned rank-one values and indices")
    total = torch.zeros(graphs, dtype=values.dtype)
    total.scatter_add_(0, graph_index.long(), values)
    count = torch.bincount(graph_index.long(), minlength=graphs).clamp_min(1)
    return total / count.to(values)


def structure_bootstrap_mean(
    values: torch.Tensor,
    *,
    seed: int,
    replicates: int,
    chunk: int = 50,
) -> dict[str, float]:
    """Structure bootstrap for a paired scalar difference vector."""

    if values.ndim != 1 or values.numel() < 2 or replicates < 100 or chunk < 1:
        raise ValueError("bootstrap input or budget is invalid")
    generator = torch.Generator().manual_seed(seed)
    draws: list[torch.Tensor] = []
    for start in range(0, replicates, chunk):
        current = min(chunk, replicates - start)
        index = torch.randint(
            values.numel(),
            (current, values.numel()),
            generator=generator,
        )
        draws.append(values[index].double().mean(dim=1))
    samples = torch.cat(draws)
    quantile = torch.quantile(samples, torch.tensor([0.025, 0.5, 0.975]))
    return {
        "mean": float(values.double().mean()),
        "bootstrap_95_low": float(quantile[0]),
        "bootstrap_median": float(quantile[1]),
        "bootstrap_95_high": float(quantile[2]),
    }


@torch.no_grad()
def sample_fixed_partitions(
    model: StoichiometryFirstCompositionModel,
    reference: SparseCompositionState,
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> SparseCompositionState:
    """Sample the species kernel while holding only the audit partition fixed."""

    catalogue = model._catalogue()
    partition_index = catalogue.encode(reference, maximum_atoms=model.maximum_atoms)
    generator = torch.Generator(device=device).manual_seed(seed)
    sampled: list[SparseCompositionState] = []
    for start in range(0, reference.graphs, batch_size):
        stop = min(start + batch_size, reference.graphs)
        context = torch.ones((stop - start, model.context_dim), device=device)
        sampled.append(
            model.sample_species_given_partition(
                context,
                partition_index[start:stop].to(device),
                generator=generator,
            ).state.to("cpu")
        )
    return SparseCompositionState(
        species=torch.cat([value.species for value in sampled]),
        counts=torch.cat([value.counts for value in sampled]),
        length=torch.cat([value.length for value in sampled]),
        node_count=torch.cat([value.node_count for value in sampled]),
    )


def pair_count_matrix(
    state: SparseCompositionState,
    vocabulary_size: int,
) -> torch.Tensor:
    """Count graph-level unordered element-pair co-occurrence."""

    presence = state.to_dense(vocabulary_size) > 0
    return presence.long().T @ presence.long()


def pair_calibration_metrics(
    sampled: SparseCompositionState,
    reference: SparseCompositionState,
    qualified_pair_mask: torch.Tensor,
    *,
    vocabulary_size: int,
) -> dict[str, float | int]:
    """Evaluate fixed-partition species co-occurrence on a frozen pair mask."""

    expected_shape = (vocabulary_size, vocabulary_size)
    if qualified_pair_mask.shape != expected_shape or qualified_pair_mask.dtype != torch.bool:
        raise ValueError("qualified pair mask has the wrong shape or dtype")
    mask = torch.triu(qualified_pair_mask, diagonal=1)
    if not bool(mask.any()) or sampled.graphs != reference.graphs:
        raise ValueError("pair calibration requires paired nonempty graph panels")
    sampled_count = pair_count_matrix(sampled, vocabulary_size)[mask].double()
    reference_count = pair_count_matrix(reference, vocabulary_size)[mask].double()
    sampled_probability = sampled_count / sampled.graphs
    reference_probability = reference_count / reference.graphs
    return {
        "qualified_pairs": int(mask.sum()),
        "pair_distribution_jsd": jensen_shannon(sampled_count, reference_count),
        "pair_probability_rmse": float(
            (sampled_probability - reference_probability).square().mean().sqrt()
        ),
        "pair_probability_mae": float(
            (sampled_probability - reference_probability).abs().mean()
        ),
        "pair_identity_recall": float((sampled_count > 0).double().mean()),
    }
