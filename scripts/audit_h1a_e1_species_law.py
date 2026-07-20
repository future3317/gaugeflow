"""Attribute the frozen E1 composition NLL miss without taking an optimizer step."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.composition_metrics import jensen_shannon, load_compositions
from gaugeflow.production.composition_state import (
    IntegerPartitionCatalogue,
    SparseCompositionState,
    StoichiometryFirstCompositionModel,
)


def _index_hash(index: torch.Tensor) -> str:
    return hashlib.sha256(index.contiguous().numpy().tobytes()).hexdigest()


def _load_model(
    checkpoint_path: Path,
    model_config: dict[str, Any],
    *,
    device: torch.device,
) -> StoichiometryFirstCompositionModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    log_prior = state["partition_log_prior"].double().clone()
    catalogue = IntegerPartitionCatalogue.build(
        maximum_atoms=int(model_config["maximum_atoms"]),
        maximum_species=int(model_config["maximum_species"]),
    )
    for atoms in range(1, int(model_config["maximum_atoms"]) + 1):
        valid = catalogue.node_count == atoms
        log_prior[valid] -= torch.logsumexp(log_prior[valid], dim=0)
    model = StoichiometryFirstCompositionModel(
        int(model_config["context_dim"]),
        int(model_config["hidden_dim"]),
        log_prior,
        maximum_atoms=int(model_config["maximum_atoms"]),
        maximum_species=int(model_config["maximum_species"]),
        vocabulary_size=int(model_config["vocabulary_size"]),
        active_vocabulary_mask=state["active_vocabulary_mask"].bool(),
    )
    model.load_state_dict(state, strict=True)
    return model.float().to(device).eval()


def _split_train(
    complete: SparseCompositionState,
    *,
    split_seed: int,
    fit_graphs: int,
) -> tuple[SparseCompositionState, SparseCompositionState, str, str]:
    permutation = torch.randperm(complete.graphs, generator=torch.Generator().manual_seed(split_seed))
    fit_index = permutation[:fit_graphs]
    calibration_index = permutation[fit_graphs:]
    return (
        complete.index_select(fit_index),
        complete.index_select(calibration_index),
        _index_hash(fit_index),
        _index_hash(calibration_index),
    )


def _frequency_table(
    model: StoichiometryFirstCompositionModel,
    fit: SparseCompositionState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    species, counts = model.count_first_order(fit)
    positions = torch.arange(model.maximum_species).unsqueeze(0).expand(fit.graphs, -1)
    active = positions < fit.length.unsqueeze(1)
    bucket = (
        (fit.node_count.unsqueeze(1) * model.maximum_species + positions) * (model.maximum_atoms + 1)
        + counts
    )
    flat = bucket[active] * model.vocabulary_size + species[active]
    rows = (model.maximum_atoms + 1) * model.maximum_species * (model.maximum_atoms + 1)
    table = torch.bincount(flat, minlength=rows * model.vocabulary_size).reshape(rows, model.vocabulary_size)
    element_events = torch.bincount(species[active], minlength=model.vocabulary_size)
    catalogue = model._catalogue()  # exact buffer view; no duplicated partition encoding
    partition_events = torch.bincount(
        catalogue.encode(fit, maximum_atoms=model.maximum_atoms),
        minlength=catalogue.size,
    )
    return table.double(), element_events.long(), partition_events.long()


@torch.no_grad()
def _evaluate_slots(
    initial: StoichiometryFirstCompositionModel,
    final: StoichiometryFirstCompositionModel,
    calibration: SparseCompositionState,
    frequency: torch.Tensor,
    *,
    smoothing: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "initial_nll",
            "final_nll",
            "uniform_nll",
            "empirical_nll",
            "species",
            "count",
            "slot",
            "support",
            "node_count",
            "tie",
            "partition_index",
        )
    }
    frequency_device = frequency.to(device)
    catalogue = final._catalogue()
    for start in range(0, calibration.graphs, batch_size):
        stop = min(start + batch_size, calibration.graphs)
        state = calibration.index_select(torch.arange(start, stop, dtype=torch.long)).to(device)
        context = torch.ones((state.graphs, final.context_dim), device=device)
        initial_log = initial.species_log_probability_by_slot(context, state.node_count, state)
        final_log = final.species_log_probability_by_slot(context, state.node_count, state)
        species, counts = final.count_first_order(state)
        positions = torch.arange(final.maximum_species, device=device).unsqueeze(0).expand(state.graphs, -1)
        active = positions < state.length.unsqueeze(1)
        valid = final.species_validity_by_slot(state)
        uniform_log = -valid.sum(dim=2).to(torch.float64).log()
        bucket = (
            (state.node_count.unsqueeze(1) * final.maximum_species + positions) * (final.maximum_atoms + 1)
            + counts
        )
        weights = frequency_device.index_select(0, bucket.reshape(-1)).reshape(
            state.graphs,
            final.maximum_species,
            final.vocabulary_size,
        )
        weights = (weights + smoothing).masked_fill(~valid, 0.0)
        empirical_log = weights.gather(2, species.clamp_min(0).unsqueeze(2)).squeeze(2).log() - weights.sum(
            dim=2
        ).log()
        previous_count = torch.cat((counts.new_zeros((state.graphs, 1)), counts[:, :-1]), dim=1)
        tie = active & (positions > 0) & (counts == previous_count)
        partition_index = catalogue.encode(state, maximum_atoms=final.maximum_atoms).unsqueeze(1).expand_as(counts)
        values = {
            "initial_nll": -initial_log,
            "final_nll": -final_log,
            "uniform_nll": -uniform_log,
            "empirical_nll": -empirical_log,
            "species": species,
            "count": counts,
            "slot": positions,
            "support": state.length.unsqueeze(1).expand_as(counts),
            "node_count": state.node_count.unsqueeze(1).expand_as(counts),
            "tie": tie.long(),
            "partition_index": partition_index,
        }
        for name, value in values.items():
            output[name].append(value[active].detach().cpu())
    return {name: torch.cat(parts) for name, parts in output.items()}


@torch.no_grad()
def _sample_fixed_partitions(
    model: StoichiometryFirstCompositionModel,
    reference: SparseCompositionState,
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> SparseCompositionState:
    catalogue = model._catalogue()
    partition_index = catalogue.encode(reference, maximum_atoms=model.maximum_atoms)
    generator = torch.Generator(device=device).manual_seed(seed)
    sampled: list[SparseCompositionState] = []
    for start in range(0, reference.graphs, batch_size):
        stop = min(start + batch_size, reference.graphs)
        context = torch.ones((stop - start, model.context_dim), device=device)
        selected = partition_index[start:stop].to(device)
        sampled.append(
            model.sample_species_given_partition(
                context,
                selected,
                generator=generator,
            ).state.to("cpu")
        )
    return SparseCompositionState(
        species=torch.cat([state.species for state in sampled]),
        counts=torch.cat([state.counts for state in sampled]),
        length=torch.cat([state.length for state in sampled]),
        node_count=torch.cat([state.node_count for state in sampled]),
    )


def _cooccurrence_metrics(
    sampled: SparseCompositionState,
    reference: SparseCompositionState,
    *,
    vocabulary_size: int,
    active_mask: torch.Tensor,
    frequent_pair_graphs: int,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    sampled_dense = sampled.to_dense(vocabulary_size).double()
    reference_dense = reference.to_dense(vocabulary_size).double()
    sampled_presence = (sampled_dense > 0).double()
    reference_presence = (reference_dense > 0).double()
    sampled_pair = sampled_presence.T @ sampled_presence
    reference_pair = reference_presence.T @ reference_presence
    upper = torch.triu(torch.ones_like(reference_pair, dtype=torch.bool), diagonal=1)
    pair_mask = upper & active_mask.unsqueeze(0) & active_mask.unsqueeze(1)
    sampled_pair_vector = sampled_pair[pair_mask]
    reference_pair_vector = reference_pair[pair_mask]
    sampled_probability = sampled_pair_vector / sampled.graphs
    reference_probability = reference_pair_vector / reference.graphs
    sampled_element = sampled_dense.sum(dim=0)
    reference_element = reference_dense.sum(dim=0)
    sampled_presence_count = sampled_presence.sum(dim=0)
    reference_presence_count = reference_presence.sum(dim=0)
    sampled_covariance = sampled_pair / sampled.graphs - torch.outer(
        sampled_presence.mean(dim=0), sampled_presence.mean(dim=0)
    )
    reference_covariance = reference_pair / reference.graphs - torch.outer(
        reference_presence.mean(dim=0), reference_presence.mean(dim=0)
    )
    sampled_covariance_vector = sampled_covariance[pair_mask]
    reference_covariance_vector = reference_covariance[pair_mask]
    covariance_norm = torch.linalg.vector_norm(reference_covariance_vector).clamp_min(1e-12)
    frequent = reference_pair_vector >= frequent_pair_graphs
    pair_i, pair_j = torch.nonzero(pair_mask, as_tuple=True)
    absolute_error = (sampled_probability - reference_probability).abs()
    top = torch.topk(absolute_error, k=min(25, absolute_error.numel())).indices
    top_rows = [
        {
            "atomic_number_i": int(pair_i[index]) + 1,
            "atomic_number_j": int(pair_j[index]) + 1,
            "reference_probability": float(reference_probability[index]),
            "sampled_probability": float(sampled_probability[index]),
            "absolute_error": float(absolute_error[index]),
        }
        for index in top
    ]
    metrics = {
        "element_count_jsd": jensen_shannon(sampled_element, reference_element),
        "element_presence_jsd": jensen_shannon(sampled_presence_count, reference_presence_count),
        "pair_distribution_jsd": jensen_shannon(sampled_pair_vector, reference_pair_vector),
        "pair_probability_rmse": float(torch.mean((sampled_probability - reference_probability) ** 2).sqrt()),
        "pair_probability_mae": float(torch.mean(absolute_error)),
        "frequent_pair_recall": float((sampled_pair_vector[frequent] > 0).double().mean()),
        "frequent_pairs": int(frequent.sum()),
        "covariance_relative_frobenius_error": float(
            torch.linalg.vector_norm(sampled_covariance_vector - reference_covariance_vector) / covariance_norm
        ),
        "covariance_cosine": float(
            torch.nn.functional.cosine_similarity(
                sampled_covariance_vector.unsqueeze(0),
                reference_covariance_vector.unsqueeze(0),
            )
        ),
    }
    return metrics, top_rows


def _aggregate(
    events: dict[str, torch.Tensor],
    group: str,
    values: torch.Tensor,
    labels: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in torch.unique(values, sorted=True):
        selected = values == value
        initial = float(events["initial_nll"][selected].mean())
        final = float(events["final_nll"][selected].mean())
        uniform = float(events["uniform_nll"][selected].mean())
        empirical = float(events["empirical_nll"][selected].mean())
        key = int(value)
        rows.append(
            {
                "group": group,
                "value": labels[key] if labels is not None else str(key),
                "decisions": int(selected.sum()),
                "initial_nll_per_decision": initial,
                "final_nll_per_decision": final,
                "ratio": final / initial,
                "uniform_nll_per_decision": uniform,
                "empirical_nll_per_decision": empirical,
                "final_minus_empirical": final - empirical,
                "learning_gain": initial - final,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_e1_species_law_attribution_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen species-law audit protocol")
    prerequisites = protocol["prerequisites"]
    if sha256_file(args.cache_root / "manifest.json") != prerequisites["cache_manifest_sha256"]:
        raise ValueError("cache manifest mismatch")
    source_result = Path(protocol["source_result"])
    if sha256_file(source_result) != prerequisites["source_result_sha256"]:
        raise ValueError("source result mismatch")
    initial_path = args.run_root / protocol["checkpoints"]["initial"]
    final_path = args.run_root / protocol["checkpoints"]["final"]
    if sha256_file(initial_path) != prerequisites["initial_checkpoint_sha256"]:
        raise ValueError("initial checkpoint mismatch")
    if sha256_file(final_path) != prerequisites["final_checkpoint_sha256"]:
        raise ValueError("final checkpoint mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("species-law audit requires the WSL CUDA environment")

    model_config = protocol["model"]
    data_config = protocol["data"]
    evaluation = protocol["evaluation"]
    device = torch.device("cuda")
    complete = load_compositions(
        args.cache_root / "train.pt",
        maximum_species=int(model_config["maximum_species"]),
        vocabulary_size=int(model_config["vocabulary_size"]),
    )
    fit, calibration, fit_hash, calibration_hash = _split_train(
        complete,
        split_seed=int(data_config["split_seed"]),
        fit_graphs=int(data_config["iid_fit_graphs"]),
    )
    if fit_hash != prerequisites["fit_index_sha256"] or calibration_hash != prerequisites["calibration_index_sha256"]:
        raise ValueError("IID split hashes do not reproduce the frozen source screen")
    initial = _load_model(initial_path, model_config, device=device)
    final = _load_model(final_path, model_config, device=device)
    frequency, element_events, partition_events = _frequency_table(final, fit)
    events = _evaluate_slots(
        initial,
        final,
        calibration,
        frequency,
        smoothing=float(evaluation["empirical_smoothing"]),
        batch_size=int(evaluation["batch_size"]),
        device=device,
    )
    events["partition_frequency"] = partition_events.index_select(0, events["partition_index"])
    events["element_frequency"] = element_events.index_select(0, events["species"])
    partition_bin = torch.bucketize(events["partition_frequency"], torch.tensor([5, 20, 100]))
    active_frequencies = element_events[element_events > 0].double()
    quantiles = torch.quantile(active_frequencies, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)).long()
    element_bin = torch.bucketize(events["element_frequency"], quantiles)

    rows: list[dict[str, Any]] = []
    rows.extend(_aggregate(events, "slot", events["slot"]))
    rows.extend(_aggregate(events, "count", events["count"]))
    rows.extend(_aggregate(events, "support", events["support"]))
    rows.extend(_aggregate(events, "node_count", events["node_count"]))
    rows.extend(_aggregate(events, "equal_count_tie", events["tie"], {0: "false", 1: "true"}))
    rows.extend(
        _aggregate(
            events,
            "partition_fit_frequency",
            partition_bin,
            {0: "1-5", 1: "6-20", 2: "21-100", 3: ">100"},
        )
    )
    rows.extend(
        _aggregate(
            events,
            "element_fit_frequency_quartile",
            element_bin,
            {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"},
        )
    )

    initial_sample = _sample_fixed_partitions(
        initial,
        calibration,
        seed=int(evaluation["sampling_seed"]),
        batch_size=int(evaluation["batch_size"]),
        device=device,
    )
    final_sample = _sample_fixed_partitions(
        final,
        calibration,
        seed=int(evaluation["sampling_seed"]),
        batch_size=int(evaluation["batch_size"]),
        device=device,
    )
    active_mask = final.active_vocabulary_mask.detach().cpu()
    initial_cooccurrence, _ = _cooccurrence_metrics(
        initial_sample,
        calibration,
        vocabulary_size=final.vocabulary_size,
        active_mask=active_mask,
        frequent_pair_graphs=int(evaluation["frequent_pair_graphs"]),
    )
    final_cooccurrence, top_errors = _cooccurrence_metrics(
        final_sample,
        calibration,
        vocabulary_size=final.vocabulary_size,
        active_mask=active_mask,
        frequent_pair_graphs=int(evaluation["frequent_pair_graphs"]),
    )

    decisions = events["final_nll"].numel()
    graphs = calibration.graphs
    summary = {
        "decisions": decisions,
        "mean_species_per_graph": decisions / graphs,
        "initial_nll_per_graph": float(events["initial_nll"].sum() / graphs),
        "final_nll_per_graph": float(events["final_nll"].sum() / graphs),
        "final_initial_ratio": float(events["final_nll"].sum() / events["initial_nll"].sum()),
        "initial_nll_per_decision": float(events["initial_nll"].mean()),
        "final_nll_per_decision": float(events["final_nll"].mean()),
        "uniform_nll_per_decision": float(events["uniform_nll"].mean()),
        "empirical_count_slot_nll_per_decision": float(events["empirical_nll"].mean()),
        "initial_minus_uniform": float((events["initial_nll"] - events["uniform_nll"]).mean()),
        "final_minus_uniform": float((events["final_nll"] - events["uniform_nll"]).mean()),
        "final_minus_empirical": float((events["final_nll"] - events["empirical_nll"]).mean()),
        "uniform_to_final_gain": float((events["uniform_nll"] - events["final_nll"]).mean()),
        "uniform_to_empirical_headroom": float((events["uniform_nll"] - events["empirical_nll"]).mean()),
    }
    headroom = summary["uniform_to_empirical_headroom"]
    summary["empirical_headroom_fraction_captured"] = (
        summary["uniform_to_final_gain"] / headroom if headroom > 0 else float("nan")
    )
    acceptance = protocol["diagnostic_thresholds"]
    checks = {
        "conditional_nll": summary["final_minus_empirical"] <= acceptance["final_minus_empirical_nll_per_decision_max"],
        "pair_jsd": final_cooccurrence["pair_distribution_jsd"] <= acceptance["pair_distribution_jsd_max"],
        "pair_rmse": final_cooccurrence["pair_probability_rmse"] <= acceptance["pair_probability_rmse_max"],
        "frequent_pair_recall": final_cooccurrence["frequent_pair_recall"] >= acceptance["frequent_pair_recall_min"],
    }
    if all(checks.values()):
        attribution = "initial_ratio_criterion_is_too_indirect_for_species_law_calibration"
    elif not checks["conditional_nll"] and not checks["pair_jsd"]:
        attribution = "conditional_species_law_underfit"
    else:
        attribution = "mixed_nll_and_cooccurrence_evidence"
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "source_result_sha256": sha256_file(source_result),
        "cache_manifest_sha256": sha256_file(args.cache_root / "manifest.json"),
        "checkpoints": {
            "initial_sha256": sha256_file(initial_path),
            "final_sha256": sha256_file(final_path),
        },
        "iid_split": {
            "fit_graphs": fit.graphs,
            "calibration_graphs": calibration.graphs,
            "fit_index_sha256": fit_hash,
            "calibration_index_sha256": calibration_hash,
        },
        "nll_attribution": summary,
        "initial_fixed_partition_cooccurrence": initial_cooccurrence,
        "final_fixed_partition_cooccurrence": final_cooccurrence,
        "diagnostic_checks": checks,
        "attribution": attribution,
        "decision": protocol["decision_rule"][attribution],
        "boundary": protocol["decision_rule"]["boundary"],
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "optimizer_steps": 0,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "conditional_nll_groups.csv", rows)
    _write_csv(args.output_dir / "cooccurrence_top_errors.csv", top_errors)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
