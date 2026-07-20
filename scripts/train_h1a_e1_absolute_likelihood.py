"""Run the preregistered absolute-likelihood E1 composition qualification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.composition_metrics import (
    load_compositions,
    partition_key,
    sample_validation,
)
from gaugeflow.production.composition_qualification import (
    evaluate_species_slots,
    fit_count_slot_reference,
    graph_mean,
    pair_calibration_metrics,
    pair_count_matrix,
    sample_fixed_partitions,
    structure_bootstrap_mean,
)
from gaugeflow.production.composition_state import (
    IntegerPartitionCatalogue,
    SparseCompositionState,
    StoichiometryFirstCompositionModel,
    fit_integer_partition_log_prior,
)


def _normalized_source_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _save_checkpoint(
    path: Path,
    *,
    model: StoichiometryFirstCompositionModel,
    optimizer: torch.optim.Optimizer,
    protocol_hash: str,
    split_manifest_hash: str,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            "protocol": "h1a_e1_absolute_likelihood_v1",
            "protocol_sha256": protocol_hash,
            "split_manifest_sha256": split_manifest_hash,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _load_panel(
    complete: SparseCompositionState,
    split_root: Path,
    name: str,
    expected_hash: str,
) -> SparseCompositionState:
    path = split_root / f"{name}_index.pt"
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{name} index hash mismatch")
    index = torch.load(path, map_location="cpu", weights_only=True).long()
    return complete.index_select(index)


def _slot_summary(
    metrics: dict[str, torch.Tensor],
    *,
    graphs: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    difference = metrics["model_nll"] - metrics["empirical_nll"]
    uniform_difference = metrics["model_nll"] - metrics["uniform_nll"]
    graph_difference = graph_mean(difference, metrics["graph_index"], graphs)
    return {
        "decisions": int(metrics["model_nll"].numel()),
        "mean_species_per_graph": float(metrics["model_nll"].numel() / graphs),
        "model_nll_per_decision": float(metrics["model_nll"].double().mean()),
        "empirical_nll_per_decision": float(metrics["empirical_nll"].double().mean()),
        "uniform_nll_per_decision": float(metrics["uniform_nll"].double().mean()),
        "model_minus_empirical_per_decision": float(difference.double().mean()),
        "model_minus_uniform_per_decision": float(uniform_difference.double().mean()),
        "structure_paired_model_minus_empirical": structure_bootstrap_mean(
            graph_difference,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        ),
    }


def _stratum_rows(
    panel: str,
    metrics: dict[str, torch.Tensor],
    *,
    minimum_decisions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frequency_tier = torch.bucketize(
        metrics["fit_event_frequency"].double(),
        torch.tensor([10.0, 100.0]),
        right=False,
    )
    groups = {
        "node_count_N": metrics["node_count"],
        "species_support_S": metrics["support"],
        "partition_lambda": metrics["partition_key"],
        "fit_event_frequency_tier": frequency_tier,
        "element_token": metrics["species"],
    }
    tier_name = {0: "rare_lt_10", 1: "mid_10_99", 2: "frequent_ge_100"}
    for group_name, values in groups.items():
        unique, counts = torch.unique(values, sorted=True, return_counts=True)
        for value, count in zip(unique.tolist(), counts.tolist()):
            if count < minimum_decisions:
                continue
            selected = values == value
            label: str | int = int(value)
            if group_name == "fit_event_frequency_tier":
                label = tier_name[int(value)]
            rows.append(
                {
                    "panel": panel,
                    "group": group_name,
                    "value": label,
                    "graphs": int(torch.unique(metrics["graph_index"][selected]).numel()),
                    "decisions": int(count),
                    "model_nll": float(metrics["model_nll"][selected].double().mean()),
                    "empirical_nll": float(
                        metrics["empirical_nll"][selected].double().mean()
                    ),
                    "uniform_nll": float(metrics["uniform_nll"][selected].double().mean()),
                    "model_minus_empirical": float(
                        (
                            metrics["model_nll"][selected]
                            - metrics["empirical_nll"][selected]
                        )
                        .double()
                        .mean()
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty qualification table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pair_rows(
    qualified: torch.Tensor,
    reference: dict[str, torch.Tensor],
    sampled: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    left, right = torch.nonzero(torch.triu(qualified, diagonal=1), as_tuple=True)
    rows: list[dict[str, Any]] = []
    for index in range(left.numel()):
        i = int(left[index])
        j = int(right[index])
        rows.append(
            {
                "element_i_token": i,
                "element_j_token": j,
                "calibration_reference_graphs": int(reference["calibration"][i, j]),
                "calibration_sampled_graphs": int(sampled["calibration"][i, j]),
                "test_reference_graphs": int(reference["test"][i, j]),
                "test_sampled_graphs": int(sampled["test"][i, j]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--q2-result", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_e1_absolute_likelihood_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen absolute-likelihood E1 protocol")
    prerequisites = protocol["prerequisites"]
    identities = {
        args.q2_result: prerequisites["q2_result_sha256"],
        args.cache_root / "manifest.json": prerequisites["cache_manifest_sha256"],
        args.split_root / "manifest.json": prerequisites["split_manifest_sha256"],
        args.split_audit: prerequisites["split_audit_sha256"],
    }
    for path, expected in identities.items():
        if sha256_file(path) != expected:
            raise ValueError(f"E1 prerequisite identity changed: {path}")
    if _normalized_source_sha256(Path(__file__)) != prerequisites["trainer_sha256"]:
        raise ValueError("E1 trainer does not match the preregistered implementation")
    split_audit = load_json_object(args.split_audit)
    if not split_audit.get("qualified"):
        raise ValueError("IID split independent audit did not qualify")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite an E1 result: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("absolute-likelihood E1 requires CUDA")

    model_config = protocol["model"]
    training = protocol["training"]
    evaluation = protocol["evaluation"]
    acceptance = protocol["acceptance"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    complete = load_compositions(
        args.cache_root / "train.pt",
        maximum_species=int(model_config["maximum_species"]),
        vocabulary_size=int(model_config["vocabulary_size"]),
    )
    split_hashes = prerequisites["split_index_sha256"]
    fit = _load_panel(complete, args.split_root, "fit", split_hashes["fit"])
    calibration = _load_panel(
        complete, args.split_root, "calibration", split_hashes["calibration"]
    )
    test = _load_panel(complete, args.split_root, "test", split_hashes["test"])
    if (fit.graphs, calibration.graphs, test.graphs) != tuple(protocol["data"]["graphs"]):
        raise ValueError("E1 split graph counts changed")

    catalogue = IntegerPartitionCatalogue.build(
        maximum_atoms=int(model_config["maximum_atoms"]),
        maximum_species=int(model_config["maximum_species"]),
    )
    log_prior = fit_integer_partition_log_prior(
        fit,
        catalogue,
        maximum_atoms=int(model_config["maximum_atoms"]),
        smoothing=float(model_config["partition_smoothing"]),
    )
    active_mask = torch.zeros(int(model_config["vocabulary_size"]), dtype=torch.bool)
    active_mask[fit.species[fit.species >= 0].unique()] = True
    for name, panel in (("calibration", calibration), ("test", test)):
        unknown = (~active_mask[panel.species.clamp_min(0)])[panel.species >= 0]
        if bool(unknown.any()):
            raise ValueError(f"{name} contains an element outside fit support")

    device = torch.device("cuda")
    model = StoichiometryFirstCompositionModel(
        int(model_config["context_dim"]),
        int(model_config["hidden_dim"]),
        log_prior,
        maximum_atoms=int(model_config["maximum_atoms"]),
        maximum_species=int(model_config["maximum_species"]),
        vocabulary_size=int(model_config["vocabulary_size"]),
        active_vocabulary_mask=active_mask,
    ).float().to(device)
    empirical_reference = fit_count_slot_reference(
        model.cpu(), fit, smoothing=float(evaluation["empirical_smoothing"])
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    protocol_hash = canonical_json_hash(protocol)
    split_manifest_hash = sha256_file(args.split_root / "manifest.json")
    run_directory = args.run_root / protocol["protocol"] / f"seed_{seed}"
    if run_directory.exists():
        raise FileExistsError(f"refusing to reuse E1 run directory: {run_directory}")
    _save_checkpoint(
        run_directory / "step_000000.pt",
        model=model,
        optimizer=optimizer,
        protocol_hash=protocol_hash,
        split_manifest_hash=split_manifest_hash,
        step=0,
    )

    order = torch.randperm(fit.graphs, generator=torch.Generator().manual_seed(seed))
    batch_size = int(training["batch_size"])
    expected_steps = math.ceil(fit.graphs / batch_size)
    if expected_steps != int(training["steps"]):
        raise ValueError("frozen E1 step count is not one exact fit pass")
    metrics_path = run_directory / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_loss = float("nan")
    clip_count = 0
    for step, start in enumerate(range(0, fit.graphs, batch_size), start=1):
        state = fit.index_select(order[start : start + batch_size]).to(device)
        context = torch.ones((state.graphs, int(model_config["context_dim"])), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=training["precision"] == "bf16",
        ):
            loss = -model.log_prob(context, state.node_count, state).species.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("E1 species loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("E1 gradient is non-finite")
        clip_count += int(float(gradient_norm) > float(training["gradient_clip_norm"]))
        optimizer.step()
        final_loss = float(loss.detach())
        if step == 1 or step % int(training["log_every"]) == 0 or step == expected_steps:
            elapsed = time.perf_counter() - started
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "graphs": min(step * batch_size, fit.graphs),
                            "species_nll": final_loss,
                            "gradient_norm": float(gradient_norm),
                            "graphs_per_second": min(step * batch_size, fit.graphs) / elapsed,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    checkpoint = run_directory / f"step_{expected_steps:06d}.pt"
    _save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        protocol_hash=protocol_hash,
        split_manifest_hash=split_manifest_hash,
        step=expected_steps,
    )

    panel_states = {"calibration": calibration, "test": test}
    slot_metrics: dict[str, dict[str, torch.Tensor]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    strata: list[dict[str, Any]] = []
    for offset, (name, panel) in enumerate(panel_states.items()):
        metrics = evaluate_species_slots(
            model,
            panel,
            empirical_reference,
            batch_size=int(evaluation["batch_size"]),
            device=device,
            use_bf16=training["precision"] == "bf16",
        )
        slot_metrics[name] = metrics
        summaries[name] = _slot_summary(
            metrics,
            graphs=panel.graphs,
            bootstrap_seed=int(evaluation["bootstrap_seed"]) + offset,
            bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
        )
        strata.extend(
            _stratum_rows(
                name,
                metrics,
                minimum_decisions=int(evaluation["stratum_minimum_decisions"]),
            )
        )

    fixed_samples: dict[str, SparseCompositionState] = {}
    free_samples: dict[str, dict[str, Any]] = {}
    for offset, (name, panel) in enumerate(panel_states.items()):
        fixed_samples[name] = sample_fixed_partitions(
            model,
            panel,
            seed=int(evaluation["fixed_partition_sampling_seed"]) + offset,
            batch_size=int(evaluation["batch_size"]),
            device=device,
        )
        free_samples[name] = sample_validation(
            model,
            panel,
            batch_size=int(evaluation["batch_size"]),
            device=device,
            context_dim=int(model_config["context_dim"]),
            vocabulary_size=int(model_config["vocabulary_size"]),
            seed=int(evaluation["free_sampling_seed"]) + offset,
            minimum_reference_atoms=int(evaluation["minimum_reference_atoms"]),
        )

    vocabulary_size = int(model_config["vocabulary_size"])
    complete_pair = pair_count_matrix(complete, vocabulary_size)
    reference_pair = {
        name: pair_count_matrix(panel, vocabulary_size)
        for name, panel in panel_states.items()
    }
    sampled_pair = {
        name: pair_count_matrix(fixed_samples[name], vocabulary_size)
        for name in panel_states
    }
    qualified_pair = torch.triu(
        complete_pair >= int(evaluation["eligible_pair_source_graphs_min"]),
        diagonal=1,
    )
    for name in panel_states:
        qualified_pair &= reference_pair[name] >= int(
            evaluation["pair_panel_graphs_min"]
        )
    pair_metrics = {
        name: pair_calibration_metrics(
            fixed_samples[name],
            panel,
            qualified_pair,
            vocabulary_size=vocabulary_size,
        )
        for name, panel in panel_states.items()
    }

    output_directory = args.output.parent
    strata_path = output_directory / "conditional_nll_strata.csv"
    pair_path = output_directory / "pair_calibration.csv"
    _write_csv(strata_path, strata)
    _write_csv(
        pair_path,
        _pair_rows(qualified_pair, reference_pair, sampled_pair),
    )
    checks: dict[str, bool] = {}
    for name in panel_states:
        checks[f"{name}_absolute_nll"] = (
            summaries[name]["model_nll_per_decision"]
            <= float(acceptance["species_nll_per_decision_max"])
        )
        checks[f"{name}_empirical_noninferiority"] = (
            summaries[name]["structure_paired_model_minus_empirical"][
                "bootstrap_95_high"
            ]
            <= float(acceptance["model_minus_empirical_bootstrap_95_high_max"])
        )
        checks[f"{name}_uniform_gain"] = (
            summaries[name]["model_minus_uniform_per_decision"]
            <= -float(acceptance["uniform_gain_per_decision_min"])
        )
        checks[f"{name}_pair_jsd"] = (
            pair_metrics[name]["pair_distribution_jsd"]
            <= float(acceptance["pair_distribution_jsd_max"])
        )
        checks[f"{name}_pair_rmse"] = (
            pair_metrics[name]["pair_probability_rmse"]
            <= float(acceptance["pair_probability_rmse_max"])
        )
        checks[f"{name}_pair_recall"] = (
            pair_metrics[name]["pair_identity_recall"]
            >= float(acceptance["pair_identity_recall_min"])
        )
        checks[f"{name}_atom_count"] = (
            free_samples[name].get("atom_count_preservation", 0.0)
            == float(acceptance["atom_count_preservation"])
        )
        checks[f"{name}_invalid_compositions"] = (
            free_samples[name].get("invalid_compositions", panel_states[name].graphs)
            == int(acceptance["invalid_compositions"])
        )
        checks[f"{name}_sampling_failures"] = (
            free_samples[name]["sampling_failures"]
            == int(acceptance["sampling_failures"])
        )
    checks["pair_mask_matches_split_manifest"] = int(qualified_pair.sum()) == int(
        split_audit["profile"]["pair_calibration_eligible_pairs"]
    )
    checks["throughput"] = fit.graphs / elapsed >= float(
        acceptance["training_graphs_per_second_min"]
    )
    checks["memory"] = peak_memory <= float(acceptance["peak_cuda_memory_mib_max"])
    checks["finite_gradient"] = math.isfinite(final_loss)
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_hash,
        "trainer_sha256": _normalized_source_sha256(Path(__file__)),
        "prerequisite_hashes": {
            "cache_manifest": sha256_file(args.cache_root / "manifest.json"),
            "q2_result": sha256_file(args.q2_result),
            "split_manifest": split_manifest_hash,
            "split_audit": sha256_file(args.split_audit),
        },
        "split_graphs": {
            "fit": fit.graphs,
            "calibration": calibration.graphs,
            "test": test.graphs,
        },
        "active_vocabulary_size": int(active_mask.sum()),
        "conditional_species_likelihood": summaries,
        "fixed_partition_pair_calibration": pair_metrics,
        "free_composition_sampling": free_samples,
        "tables": {
            "conditional_nll_strata": strata_path.name,
            "conditional_nll_strata_sha256": sha256_file(strata_path),
            "pair_calibration": pair_path.name,
            "pair_calibration_sha256": sha256_file(pair_path),
        },
        "training": {
            "seed": seed,
            "steps": expected_steps,
            "graph_presentations": fit.graphs,
            "data_passes": 1.0,
            "final_species_loss": final_loss,
            "clip_fraction": clip_count / expected_steps,
            "graphs_per_second": fit.graphs / elapsed,
            "peak_cuda_memory_mib": peak_memory,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
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
