"""Numerically qualify the exact stoichiometry-first composition law."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.composition_metrics import load_compositions
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
        torch.tensor(species_rows, dtype=torch.long),
        torch.tensor(count_rows, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.full((graphs,), node_count, dtype=torch.long),
    )


def _uniform_small_prior(maximum_atoms: int, maximum_species: int) -> tuple[IntegerPartitionCatalogue, torch.Tensor]:
    catalogue = IntegerPartitionCatalogue.build(
        maximum_atoms=maximum_atoms,
        maximum_species=maximum_species,
    )
    log_prior = torch.empty(catalogue.size, dtype=torch.float64)
    for atoms in range(1, maximum_atoms + 1):
        valid = catalogue.node_count == atoms
        log_prior[valid] = -math.log(int(valid.sum()))
    return catalogue, log_prior


def _exhaustive_normalization(seed: int) -> float:
    torch.manual_seed(seed)
    _, log_prior = _uniform_small_prior(6, 3)
    model = StoichiometryFirstCompositionModel(
        4,
        9,
        log_prior,
        maximum_atoms=6,
        maximum_species=3,
        vocabulary_size=5,
    ).double()
    maximum_error = 0.0
    for atoms in range(1, 7):
        state = _enumerate_states(atoms, vocabulary_size=5, maximum_species=3)
        context = torch.tensor([[0.2, -0.1, 0.5, 0.7]], dtype=torch.float64).expand(state.graphs, -1)
        log_probability = model.log_prob(context, state.node_count, state).total
        error = abs(float(torch.logsumexp(log_probability, dim=0)))
        maximum_error = max(maximum_error, error)
    return maximum_error


def _categorical_tv(sampled: torch.Tensor, probability: torch.Tensor) -> float:
    empirical = torch.bincount(sampled.cpu(), minlength=probability.numel()).double()
    empirical /= sampled.numel()
    return float(0.5 * (empirical - probability.cpu().double()).abs().sum())


@torch.no_grad()
def _precision_audit(
    model: StoichiometryFirstCompositionModel,
    state: SparseCompositionState,
    context: torch.Tensor,
) -> dict[str, float]:
    model64 = model.double().cpu()
    state64 = state.to("cpu")
    reference = model64.log_prob(context.double().cpu(), state64.node_count, state64).total
    model32 = model64.float()
    fp32 = model32.log_prob(context.float().cpu(), state64.node_count, state64).total.double()
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 Q1 path requires CUDA")
    model32 = model32.cuda()
    state_cuda = state.to("cuda")
    context_cuda = context.float().cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bf16 = model32.log_prob(context_cuda, state_cuda.node_count, state_cuda).total.float()
    return {
        "fp32_fp64_max_absolute_error": float((fp32 - reference).abs().max()),
        "bf16_fp32_mean_absolute_error": float((bf16.cpu() - fp32.float()).abs().mean()),
    }


def _latency(
    function: Any,
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_e1_stoichiometry_first_factorized_q2_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen factorized stoichiometry-first Q2 protocol")
    if sha256_file(args.calibration_audit) != protocol["prerequisites"]["calibration_audit_sha256"]:
        raise ValueError("calibration audit hash mismatch")
    if sha256_file(args.cache_root / "manifest.json") != protocol["prerequisites"]["cache_manifest_sha256"]:
        raise ValueError("cache manifest hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("Q1 requires the WSL CUDA environment")

    model_config = protocol["model"]
    evaluation = protocol["evaluation"]
    acceptance = protocol["acceptance"]
    seed = int(evaluation["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train = load_compositions(
        args.cache_root / "train.pt",
        maximum_species=int(model_config["maximum_species"]),
        vocabulary_size=int(model_config["vocabulary_size"]),
    )
    catalogue = IntegerPartitionCatalogue.build(
        maximum_atoms=int(model_config["maximum_atoms"]),
        maximum_species=int(model_config["maximum_species"]),
    )
    log_prior = fit_integer_partition_log_prior(
        train,
        catalogue,
        maximum_atoms=int(model_config["maximum_atoms"]),
        smoothing=float(model_config["partition_smoothing"]),
    )
    normalization_error = max(
        abs(float(torch.logsumexp(log_prior[catalogue.node_count == atoms], dim=0)))
        for atoms in range(1, int(model_config["maximum_atoms"]) + 1)
    )
    active_mask = torch.zeros(int(model_config["vocabulary_size"]), dtype=torch.bool)
    active_mask[train.species[train.species >= 0].unique()] = True
    model = StoichiometryFirstCompositionModel(
        int(model_config["context_dim"]),
        int(model_config["hidden_dim"]),
        log_prior,
        maximum_atoms=int(model_config["maximum_atoms"]),
        maximum_species=int(model_config["maximum_species"]),
        vocabulary_size=int(model_config["vocabulary_size"]),
        active_vocabulary_mask=active_mask,
    )
    draws = int(evaluation["draws"])
    valid_partition = catalogue.node_count == 12
    expected = log_prior[valid_partition].exp()
    partition_sample = torch.multinomial(
        expected.float(),
        num_samples=draws,
        replacement=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    partition_tv = _categorical_tv(partition_sample, expected)

    batch_size = int(evaluation["batch_size"])
    selected = train.index_select(torch.arange(batch_size)).to("cpu")
    context = torch.randn(batch_size, int(model_config["context_dim"]))
    sampled = model.sample(
        context,
        selected.node_count,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    recomputed = model.log_prob(context, selected.node_count, sampled.state).total
    sample_probability_error = float((sampled.log_probability - recomputed).abs().max())
    invalid_samples = int(
        (sampled.state.to_dense(int(model_config["vocabulary_size"])).sum(dim=1) != selected.node_count).sum()
    )
    precision = _precision_audit(model, selected, context)

    model = model.float().cuda()
    selected_cuda = selected.to("cuda")
    context_cuda = context.float().cuda()
    model.zero_grad(set_to_none=True)
    loss = -model.log_prob(context_cuda, selected_cuda.node_count, selected_cuda).total.mean()
    loss.backward()
    gradient_norm = float(
        torch.stack(
            [parameter.grad.float().square().sum() for parameter in model.parameters() if parameter.grad is not None]
        )
        .sum()
        .sqrt()
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    teacher_latency = _latency(
        lambda: model.log_prob(context_cuda, selected_cuda.node_count, selected_cuda),
        warmup=int(evaluation["latency_warmup"]),
        repeats=int(evaluation["latency_repeats"]),
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 3)
    sampling_latency = _latency(
        lambda: model.sample(context_cuda, selected_cuda.node_count, generator=generator),
        warmup=int(evaluation["latency_warmup"]),
        repeats=int(evaluation["latency_repeats"]),
    )
    memory = torch.cuda.max_memory_allocated() / (1024.0**2)
    metrics = {
        "catalogue_partitions": catalogue.size,
        "partition_prior_normalization_error": normalization_error,
        "exhaustive_small_law_normalization_error": _exhaustive_normalization(seed),
        "sample_partition_tv": partition_tv,
        "sample_log_probability_error": sample_probability_error,
        "invalid_samples": invalid_samples,
        **precision,
        "gradient_norm": gradient_norm,
        "teacher_forced_latency_ms": teacher_latency,
        "sampling_latency_ms": sampling_latency,
        "incremental_cuda_memory_mib": memory,
    }
    checks = {
        "catalogue": metrics["catalogue_partitions"] == acceptance["catalogue_partitions"],
        "prior_normalization": metrics["partition_prior_normalization_error"]
        <= acceptance["partition_prior_normalization_error_max"],
        "law_normalization": metrics["exhaustive_small_law_normalization_error"]
        <= acceptance["exhaustive_small_law_normalization_error_max"],
        "partition_sampling": metrics["sample_partition_tv"] <= acceptance["sample_partition_tv_max"],
        "sample_probability": metrics["sample_log_probability_error"] <= acceptance["sample_log_probability_error_max"],
        "validity": metrics["invalid_samples"] == acceptance["invalid_samples"],
        "fp32": metrics["fp32_fp64_max_absolute_error"] <= acceptance["fp32_fp64_log_probability_error_max"],
        "bf16": metrics["bf16_fp32_mean_absolute_error"] <= acceptance["bf16_fp32_mean_log_probability_error_max"],
        "teacher_latency": metrics["teacher_forced_latency_ms"] <= acceptance["teacher_forced_latency_ms_max"],
        "sampling_latency": metrics["sampling_latency_ms"] <= acceptance["sampling_latency_ms_max"],
        "memory": metrics["incremental_cuda_memory_mib"] <= acceptance["incremental_cuda_memory_mib_max"],
        "gradients": math.isfinite(gradient_norm) and gradient_norm > 0.0,
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "cache_manifest_sha256": sha256_file(args.cache_root / "manifest.json"),
        "calibration_audit_sha256": sha256_file(args.calibration_audit),
        "seed": seed,
        "active_vocabulary_size": int(active_mask.sum()),
        "metrics": metrics,
        "checks": checks,
        "qualified": qualified,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
        "environment": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
