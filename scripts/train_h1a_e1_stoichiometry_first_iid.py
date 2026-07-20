"""Train the stoichiometry-first composition law on one IID calibration pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.composition_metrics import (
    categorical_total_variation,
    evaluate_nll,
    load_compositions,
    partition_key,
    sample_validation,
)
from gaugeflow.production.composition_state import (
    IntegerPartitionCatalogue,
    StoichiometryFirstCompositionModel,
    fit_integer_partition_log_prior,
)


def _index_hash(index: torch.Tensor) -> str:
    return hashlib.sha256(index.contiguous().numpy().tobytes()).hexdigest()


def _save_checkpoint(
    path: Path,
    *,
    model: StoichiometryFirstCompositionModel,
    optimizer: torch.optim.Optimizer,
    protocol_hash: str,
    fit_index_hash: str,
    calibration_index_hash: str,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            "protocol": "h1a_e1_stoichiometry_first_factorized_iid_v1",
            "protocol_sha256": protocol_hash,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "fit_index_sha256": fit_index_hash,
            "calibration_index_sha256": calibration_index_hash,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--q2-result", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_e1_stoichiometry_first_factorized_iid_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen stoichiometry-first IID protocol")
    if sha256_file(args.q2_result) != protocol["prerequisites"]["q2_result_sha256"]:
        raise ValueError("factorized stoichiometry-first Q2 result mismatch")
    if sha256_file(args.cache_root / "manifest.json") != protocol["prerequisites"]["cache_manifest_sha256"]:
        raise ValueError("cache manifest mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("IID composition screen requires the WSL CUDA environment")

    data_config = protocol["data"]
    model_config = protocol["model"]
    training = protocol["training"]
    evaluation = protocol["evaluation"]
    acceptance = protocol["acceptance"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    maximum_species = int(model_config["maximum_species"])
    vocabulary_size = int(model_config["vocabulary_size"])
    complete_train = load_compositions(
        args.cache_root / "train.pt",
        maximum_species=maximum_species,
        vocabulary_size=vocabulary_size,
    )
    if complete_train.graphs != int(data_config["graphs"]):
        raise ValueError("qualified child-train row count changed")
    split_generator = torch.Generator().manual_seed(int(data_config["split_seed"]))
    permutation = torch.randperm(complete_train.graphs, generator=split_generator)
    fit_graphs = int(data_config["iid_fit_graphs"])
    fit_index = permutation[:fit_graphs]
    calibration_index = permutation[fit_graphs:]
    if calibration_index.numel() != int(data_config["iid_calibration_graphs"]):
        raise ValueError("frozen IID split sizes do not cover the child-train cache")
    fit = complete_train.index_select(fit_index)
    calibration = complete_train.index_select(calibration_index)
    ood = load_compositions(
        args.cache_root / f"{evaluation['ood_split']}.pt",
        maximum_species=maximum_species,
        vocabulary_size=vocabulary_size,
    )
    fit_index_hash = _index_hash(fit_index)
    calibration_index_hash = _index_hash(calibration_index)
    empirical = {
        "partition_tv": categorical_total_variation(partition_key(fit), partition_key(calibration)),
        "support_tv": categorical_total_variation(fit.length, calibration.length),
    }
    if empirical["partition_tv"] > acceptance["iid_empirical_partition_tv_max"]:
        raise RuntimeError("frozen IID calibration reference has excessive partition drift")
    if empirical["support_tv"] > acceptance["iid_empirical_support_tv_max"]:
        raise RuntimeError("frozen IID calibration reference has excessive support drift")

    active_mask = torch.zeros(vocabulary_size, dtype=torch.bool)
    active_mask[fit.species[fit.species >= 0].unique()] = True
    if bool((~active_mask[calibration.species.clamp_min(0)])[calibration.species >= 0].any()):
        raise ValueError("IID calibration contains an element outside fit support")
    catalogue = IntegerPartitionCatalogue.build(
        maximum_atoms=int(model_config["maximum_atoms"]),
        maximum_species=maximum_species,
    )
    log_prior = fit_integer_partition_log_prior(
        fit,
        catalogue,
        maximum_atoms=int(model_config["maximum_atoms"]),
        smoothing=float(model_config["partition_smoothing"]),
    )
    device = torch.device("cuda")
    model = (
        StoichiometryFirstCompositionModel(
            int(model_config["context_dim"]),
            int(model_config["hidden_dim"]),
            log_prior,
            maximum_atoms=int(model_config["maximum_atoms"]),
            maximum_species=maximum_species,
            vocabulary_size=vocabulary_size,
            active_vocabulary_mask=active_mask,
        )
        .float()
        .to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    protocol_hash = canonical_json_hash(protocol)
    run_directory = args.run_root / protocol["protocol"] / f"seed_{seed}"
    initial_checkpoint = run_directory / "step_000000.pt"
    _save_checkpoint(
        initial_checkpoint,
        model=model,
        optimizer=optimizer,
        protocol_hash=protocol_hash,
        fit_index_hash=fit_index_hash,
        calibration_index_hash=calibration_index_hash,
        step=0,
    )
    initial = evaluate_nll(
        model,
        calibration,
        batch_size=int(evaluation["batch_size"]),
        device=device,
        context_dim=int(model_config["context_dim"]),
        use_bf16=training["precision"] == "bf16",
    )

    training_order = torch.randperm(fit.graphs, generator=torch.Generator().manual_seed(seed))
    batch_size = int(training["batch_size"])
    expected_steps = math.ceil(fit.graphs / batch_size)
    if expected_steps != int(training["steps"]):
        raise ValueError("frozen step count is not one exact IID fit pass")
    metrics_path = run_directory / "training_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    clip_count = 0
    final_loss = float("nan")
    for step, start in enumerate(range(0, fit.graphs, batch_size), start=1):
        index = training_order[start : start + batch_size]
        state = fit.index_select(index).to(device)
        context = torch.ones((state.graphs, int(model_config["context_dim"])), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=training["precision"] == "bf16",
        ):
            output = model.log_prob(context, state.node_count, state)
            loss = -output.species.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("stoichiometry-first species loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("stoichiometry-first gradient is non-finite")
        clip_count += int(float(gradient_norm) > float(training["gradient_clip_norm"]))
        optimizer.step()
        final_loss = float(loss.detach())
        if step == 1 or step % int(training["log_every"]) == 0 or step == expected_steps:
            elapsed = time.perf_counter() - started
            row = {
                "step": step,
                "graphs": min(step * batch_size, fit.graphs),
                "species_nll": final_loss,
                "gradient_norm": float(gradient_norm),
                "graphs_per_second": min(step * batch_size, fit.graphs) / elapsed,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    final_checkpoint = run_directory / f"step_{expected_steps:06d}.pt"
    _save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        protocol_hash=protocol_hash,
        fit_index_hash=fit_index_hash,
        calibration_index_hash=calibration_index_hash,
        step=expected_steps,
    )
    final = evaluate_nll(
        model,
        calibration,
        batch_size=int(evaluation["batch_size"]),
        device=device,
        context_dim=int(model_config["context_dim"]),
        use_bf16=training["precision"] == "bf16",
    )
    sampled = sample_validation(
        model,
        calibration,
        batch_size=int(evaluation["batch_size"]),
        device=device,
        context_dim=int(model_config["context_dim"]),
        vocabulary_size=vocabulary_size,
        seed=int(evaluation["sampling_seed"]),
        minimum_reference_atoms=int(evaluation["minimum_reference_atoms_for_element_recall"]),
    )
    ood_nll = evaluate_nll(
        model,
        ood,
        batch_size=int(evaluation["batch_size"]),
        device=device,
        context_dim=int(model_config["context_dim"]),
        use_bf16=training["precision"] == "bf16",
    )
    ood_sampled = sample_validation(
        model,
        ood,
        batch_size=int(evaluation["batch_size"]),
        device=device,
        context_dim=int(model_config["context_dim"]),
        vocabulary_size=vocabulary_size,
        seed=int(evaluation["sampling_seed"]) + 1,
        minimum_reference_atoms=int(evaluation["minimum_reference_atoms_for_element_recall"]),
    )
    ratio = final["mean_total_nll"] / initial["mean_total_nll"]
    throughput = fit.graphs / elapsed
    checks = {
        "iid_empirical_partition": empirical["partition_tv"] <= acceptance["iid_empirical_partition_tv_max"],
        "iid_empirical_support": empirical["support_tv"] <= acceptance["iid_empirical_support_tv_max"],
        "calibration_nll": ratio <= acceptance["final_initial_calibration_nll_ratio_max"],
        "element_marginal": sampled["element_marginal_jsd"] <= acceptance["element_marginal_jsd_max"],
        "support_size": sampled["support_size_total_variation"] <= acceptance["support_size_total_variation_max"],
        "count_partition": sampled["count_partition_total_variation"]
        <= acceptance["count_partition_total_variation_max"],
        "element_recall": sampled["supported_element_recall"] >= acceptance["supported_element_recall_min"],
        "atom_count": sampled["atom_count_preservation"] == acceptance["atom_count_preservation"],
        "invalid_compositions": sampled["invalid_compositions"] == acceptance["invalid_compositions"],
        "sampling_failures": sampled["sampling_failures"] == acceptance["sampling_failures"],
        "throughput": throughput >= acceptance["training_graphs_per_second_min"],
        "memory": peak_memory <= acceptance["peak_cuda_memory_mib_max"],
        "finite_gradients": math.isfinite(final_loss) is acceptance["finite_gradients"],
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_hash,
        "cache_manifest_sha256": sha256_file(args.cache_root / "manifest.json"),
        "q2_result_sha256": sha256_file(args.q2_result),
        "seed": seed,
        "iid_split": {
            "fit_graphs": fit.graphs,
            "calibration_graphs": calibration.graphs,
            "fit_index_sha256": fit_index_hash,
            "calibration_index_sha256": calibration_index_hash,
            **empirical,
        },
        "active_vocabulary_size": int(active_mask.sum()),
        "initial_calibration": initial,
        "final_calibration": final,
        "final_initial_calibration_nll_ratio": ratio,
        "calibration_sampling": sampled,
        "ood_formula_prototype_disjoint_diagnostic": {
            "split": evaluation["ood_split"],
            "nll": ood_nll,
            "sampling": ood_sampled,
            "qualification_role": "none",
        },
        "training": {
            "steps": expected_steps,
            "graph_presentations": fit.graphs,
            "data_passes": 1.0,
            "final_species_loss": final_loss,
            "clip_fraction": clip_count / expected_steps,
            "graphs_per_second": throughput,
            "peak_cuda_memory_mib": peak_memory,
            "checkpoint": str(final_checkpoint),
            "checkpoint_sha256": sha256_file(final_checkpoint),
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "checks": checks,
        "qualified": qualified,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
