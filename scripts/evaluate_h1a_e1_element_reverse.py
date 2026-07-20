"""Evaluate the frozen E1 element-only reverse qualification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from gaugeflow.file_utils import (
    canonical_json_hash,
    load_json_object,
    numeric_tree_is_finite,
    sha256_file,
)
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset, collate_packed_alex
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.composition_assignment import (
    composition_counts_from_tokens,
    count_constrained_assignment,
    rounded_graph_composition,
)
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.runtime import load_tensor_free_ema_runtime


def _chunks(indices: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return list(indices.split(batch_size))


@torch.no_grad()
def _teacher_forced_metrics(
    checkpoint: Path,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    element_times: list[float],
    *,
    batch_size: int,
    noise_seed: int,
    device: torch.device,
    protocol_name: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=protocol_name,
        protocol_sha256=protocol_sha256,
    )
    if runtime.model.modality_time_conditioning != "separate":
        raise ValueError("E1 checkpoint does not use the unified separate-clock backbone")
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
        categorical_path=str(runtime.training_config.get("categorical_path", "absorbing_mask")),
    )
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    rows: list[dict[str, float]] = []
    tensor_candidates = 0
    for element_time_value in element_times:
        nll_sum = torch.zeros((), device=device)
        top1_sum = torch.zeros((), device=device)
        top5_sum = torch.zeros((), device=device)
        masked_count = torch.zeros((), device=device)
        exact_composition = 0
        composition_count_l1 = 0
        composition_count_overlap = 0
        input_count_overlap = 0
        oracle_count_overlap = 0
        oracle_exact_composition = 0
        oracle_site_correct = 0
        graph_count = 0
        node_count = 0
        for chunk in _chunks(indices, batch_size):
            batch_data = collate_packed_alex([dataset[int(index)] for index in chunk]).to(
                device,
                non_blocking=True,
            )
            graphs = int(batch_data.num_graphs)
            counts = torch.bincount(batch_data.batch, minlength=graphs)
            blueprint = ParentBlueprintBatch.from_node_counts(
                counts,
                dtype=batch_data.frac_coords.dtype,
                device=device,
            )
            clean_time = torch.zeros((graphs,), dtype=batch_data.frac_coords.dtype, device=device)
            element_time = torch.full_like(clean_time, element_time_value)
            output = diffusion(
                batch_data.atom_types,
                batch_data.frac_coords,
                batch_data.lattice,
                batch_data.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=clean_time,
                element_time=element_time,
                lattice_time=clean_time,
                generator=generator,
            )
            mask = output.noisy.element_was_masked
            logits = output.prediction.clean_element_logits
            target = batch_data.atom_types
            target_counts = composition_counts_from_tokens(target, batch_data.batch, graphs)
            predicted_counts = rounded_graph_composition(
                output.prediction.clean_composition_logits,
                counts,
            )
            input_counts = composition_counts_from_tokens(
                output.noisy.element_tokens,
                batch_data.batch,
                graphs,
            )
            condition = torch.zeros((graphs, 18), dtype=clean_time.dtype, device=device)
            condition_present = torch.zeros(
                (graphs, 1), dtype=torch.bool, device=device
            )
            oracle_prediction = runtime.model(
                target,
                output.noisy.fractional_coordinates,
                output.noisy.log_volume,
                output.noisy.log_shape,
                batch_data.batch,
                clean_time,
                condition,
                condition_present,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                element_time=element_time,
                lattice_time=clean_time,
            )
            oracle_counts = rounded_graph_composition(
                oracle_prediction.clean_composition_logits,
                counts,
            )
            nll = F.cross_entropy(logits, target, reduction="none")
            top = logits.topk(5, dim=-1).indices
            nll_sum += nll[mask].sum()
            top1_sum += (top[:, 0] == target)[mask].sum()
            top5_sum += (top == target.unsqueeze(-1)).any(dim=-1)[mask].sum()
            masked_count += mask.sum()
            exact_composition += int((predicted_counts == target_counts).all(dim=-1).sum())
            composition_count_l1 += int((predicted_counts - target_counts).abs().sum())
            composition_count_overlap += int(torch.minimum(predicted_counts, target_counts).sum())
            input_count_overlap += int(torch.minimum(input_counts, target_counts).sum())
            oracle_count_overlap += int(torch.minimum(oracle_counts, target_counts).sum())
            oracle_exact_composition += int((oracle_counts == target_counts).all(dim=-1).sum())
            oracle_site_correct += int(
                (oracle_prediction.clean_element_logits.argmax(dim=-1) == target).sum()
            )
            graph_count += graphs
            node_count += int(target.numel())
            tensor_candidates += int(output.prediction.gauge_atlas.effective_frame_count.sum())
        denominator = masked_count.clamp_min(1)
        rows.append(
            {
                "element_time": element_time_value,
                "masked_fraction": float(masked_count / node_count),
                "masked_tokens": float(masked_count),
                "nll": float(nll_sum / denominator),
                "top1_accuracy": float(top1_sum / denominator),
                "top5_accuracy": float(top5_sum / denominator),
                "exact_composition_accuracy": exact_composition / max(graph_count, 1),
                "mean_composition_count_l1_per_graph": composition_count_l1
                / max(graph_count, 1),
                "composition_count_overlap_fraction": composition_count_overlap
                / max(node_count, 1),
                "input_count_overlap_fraction": input_count_overlap / max(node_count, 1),
                "clean_token_oracle_count_overlap_fraction": oracle_count_overlap
                / max(node_count, 1),
                "clean_token_oracle_exact_composition_accuracy": oracle_exact_composition
                / max(graph_count, 1),
                "clean_token_oracle_site_accuracy": oracle_site_correct
                / max(node_count, 1),
            }
        )
    return {"by_time": rows, "tensor_candidate_count": tensor_candidates}


@torch.no_grad()
def _reverse_metrics(
    checkpoint: Path,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    *,
    batch_size: int,
    steps: int,
    sampling_seed: int,
    device: torch.device,
    protocol_name: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=protocol_name,
        protocol_sha256=protocol_sha256,
    )
    sampler = TensorFreeReverseSampler(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
        categorical_path=str(runtime.training_config.get("categorical_path", "absorbing_mask")),
    )
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    exact_composition = 0
    exact_assignment = 0
    correct_sites = 0
    terminal_argmax_correct_sites = 0
    oracle_count_correct_sites = 0
    oracle_count_exact_assignment = 0
    composition_count_l1 = 0
    composition_count_overlap = 0
    nodes = 0
    failures = 0
    terminal_masks = 0
    masked_curves: list[torch.Tensor] = []
    started = time.perf_counter()
    for chunk in _chunks(indices, batch_size):
        batch_data = collate_packed_alex([dataset[int(index)] for index in chunk]).to(
            device,
            non_blocking=True,
        )
        graphs = int(batch_data.num_graphs)
        counts = torch.bincount(batch_data.batch, minlength=graphs)
        blueprint = ParentBlueprintBatch.from_node_counts(
            counts,
            dtype=batch_data.frac_coords.dtype,
            device=device,
        )
        try:
            generated = sampler.sample_elements(
                blueprint,
                batch_data.frac_coords,
                batch_data.lattice,
                steps=steps,
                categorical_generator=generator,
            )
        except SamplingFailure:
            failures += graphs
            continue
        target = batch_data.atom_types
        predicted = generated.element_tokens
        target_counts = composition_counts_from_tokens(target, batch_data.batch, graphs)
        predicted_counts = generated.predicted_composition_counts
        oracle_count_assignment = count_constrained_assignment(
            generated.terminal_clean_element_logits,
            batch_data.batch,
            target_counts,
        )
        terminal_argmax = generated.terminal_clean_element_logits.argmax(dim=-1)
        correct_sites += int((predicted == target).sum())
        terminal_argmax_correct_sites += int((terminal_argmax == target).sum())
        oracle_count_correct_sites += int((oracle_count_assignment == target).sum())
        composition_count_l1 += int((predicted_counts - target_counts).abs().sum())
        composition_count_overlap += int(torch.minimum(predicted_counts, target_counts).sum())
        nodes += int(target.numel())
        terminal_masks += int(generated.diagnostics.masked_count[-1])
        masked_curves.append(generated.diagnostics.masked_count.float() / target.numel())
        for graph in range(graphs):
            selected = batch_data.batch == graph
            exact_assignment += int(torch.equal(predicted[selected], target[selected]))
            oracle_count_exact_assignment += int(
                torch.equal(oracle_count_assignment[selected], target[selected])
            )
            exact_composition += int(
                torch.equal(
                    predicted[selected].sort().values,
                    target[selected].sort().values,
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    successful_graphs = int(indices.numel()) - failures
    curve = torch.stack(masked_curves).mean(dim=0) if masked_curves else torch.ones(steps)
    return {
        "graphs": int(indices.numel()),
        "successful_graphs": successful_graphs,
        "sampling_failures": failures,
        "terminal_masks": terminal_masks,
        "site_accuracy": correct_sites / max(nodes, 1),
        "terminal_argmax_site_accuracy": terminal_argmax_correct_sites / max(nodes, 1),
        "oracle_count_site_accuracy": oracle_count_correct_sites / max(nodes, 1),
        "oracle_count_exact_assignment_accuracy": oracle_count_exact_assignment
        / max(successful_graphs, 1),
        "exact_assignment_accuracy": exact_assignment / max(successful_graphs, 1),
        "exact_composition_accuracy": exact_composition / max(successful_graphs, 1),
        "mean_composition_count_l1_per_graph": composition_count_l1
        / max(successful_graphs, 1),
        "composition_count_overlap_fraction": composition_count_overlap / max(nodes, 1),
        "atom_count_preservation": 1.0 if failures == 0 else successful_graphs / int(indices.numel()),
        "latency_seconds": elapsed,
        "graphs_per_second": successful_graphs / max(elapsed, 1.0e-12),
        "masked_fraction_curve": [float(value) for value in curve],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    training = protocol.get("training")
    if (
        protocol.get("protocol")
        not in {
            "h1a_e1_element_reverse_v1",
            "h1a_e1_uniform_count_projection_v1",
            "h1a_e1_graph_composition_field_v1",
            "h1a_e1_exchangeable_histogram_residual_v1",
        }
        or protocol.get("status_before_run") != "frozen_not_run"
        or not isinstance(training, dict)
        or training.get("objective") != "element"
        or training.get("modality_time_mode") != "element_only"
        or training.get("seeds") != [5705]
    ):
        raise ValueError("unexpected or unfrozen E1 protocol")
    prerequisites = protocol["prerequisites"]
    hash_contract = {
        args.cache_root / "manifest.json": prerequisites["cache_manifest_sha256"],
        Path("reports")
        / str(prerequisites["qualified_mechanism"])
        / "result.json": prerequisites["qualification_result_sha256"],
    }
    for path, expected in hash_contract.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen prerequisite hash mismatch: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    protocol_sha256 = canonical_json_hash(protocol)
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    evaluation = protocol["evaluation"]
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    run = args.run_root / "seed_5705"
    final_step = int(training["steps"])
    checkpoints = {
        "initial": run / "checkpoint_step_00000000.pt",
        "final": run / f"checkpoint_step_{final_step:08d}.pt",
    }
    teacher_forced = {
        name: _teacher_forced_metrics(
            checkpoint,
            dataset,
            indices,
            [float(value) for value in evaluation["teacher_forced_times"]],
            batch_size=int(evaluation["batch_size"]),
            noise_seed=int(evaluation["teacher_forced_noise_seed"]),
            device=device,
            protocol_name=str(protocol["protocol"]),
            protocol_sha256=protocol_sha256,
        )
        for name, checkpoint in checkpoints.items()
    }
    reverse = _reverse_metrics(
        checkpoints["final"],
        dataset,
        indices,
        batch_size=int(evaluation["batch_size"]),
        steps=int(evaluation["reverse_steps"]),
        sampling_seed=int(evaluation["sampling_seed"]),
        device=device,
        protocol_name=str(protocol["protocol"]),
        protocol_sha256=protocol_sha256,
    )
    initial_rows = {row["element_time"]: row for row in teacher_forced["initial"]["by_time"]}
    final_rows = {row["element_time"]: row for row in teacher_forced["final"]["by_time"]}
    mean_initial_nll = sum(float(row["nll"]) for row in initial_rows.values()) / len(initial_rows)
    mean_final_nll = sum(float(row["nll"]) for row in final_rows.values()) / len(final_rows)
    records = [
        json.loads(line)
        for line in (run / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    final_log = records[-1]
    acceptance = protocol["acceptance"]
    t05 = final_rows[0.5]
    t09 = final_rows[0.9]
    checks = {
        "finite": numeric_tree_is_finite(teacher_forced)
        and numeric_tree_is_finite(reverse)
        and numeric_tree_is_finite(records),
        "nll_ratio": mean_final_nll / mean_initial_nll <= float(acceptance["mean_nll_ratio_max"]),
        "t05_top1": float(t05["top1_accuracy"]) >= float(acceptance["t05_top1_min"]),
        "t05_top5": float(t05["top5_accuracy"]) >= float(acceptance["t05_top5_min"]),
        "t09_top1": float(t09["top1_accuracy"]) >= float(acceptance["t09_top1_min"]),
        "composition": float(reverse["exact_composition_accuracy"])
        >= float(acceptance["exact_composition_accuracy_min"]),
        "terminal_masks": int(reverse["terminal_masks"]) == int(acceptance["terminal_masks"]),
        "sampling_failures": int(reverse["sampling_failures"])
        == int(acceptance["sampling_failures"]),
        "atom_count": float(reverse["atom_count_preservation"])
        == float(acceptance["atom_count_preservation"]),
        "tensor_bypass": int(teacher_forced["final"]["tensor_candidate_count"])
        == int(acceptance["tensor_candidates"]),
        "throughput": float(final_log["graphs_per_second"])
        >= float(acceptance["training_graphs_per_second_min"]),
        "memory": float(final_log["peak_cuda_memory_mib"])
        <= float(acceptance["peak_cuda_memory_mib_max"]),
        "element_head_gradient": float(
            final_log["clipped_module_gradient_norms"]["element_readout"]
        )
        > 0.0,
        "element_time_gradient": float(
            final_log["modality_time_gradient_norms"]["element"]
        )
        > 0.0,
    }
    qualified = all(checks.values())
    decision = "pass" if qualified else "fail"
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha256,
        "seed": 5705,
        "checkpoint_sha256": sha256_file(checkpoints["final"]),
        "validation_indices_sha256": canonical_json_hash(indices.tolist()),
        "mean_initial_nll": mean_initial_nll,
        "mean_final_nll": mean_final_nll,
        "mean_nll_ratio": mean_final_nll / mean_initial_nll,
        "teacher_forced": teacher_forced,
        "reverse": reverse,
        "training": {
            "final_loss": float(final_log["loss"]),
            "graphs_per_second": float(final_log["graphs_per_second"]),
            "peak_cuda_memory_mib": float(final_log["peak_cuda_memory_mib"]),
            "clip_fraction": float(final_log["clip_fraction"]),
            "final_modality_time_gradient_norms": final_log["modality_time_gradient_norms"],
        },
        "checks": checks,
        "qualified": qualified,
        "decision": decision,
        "decision_text": protocol["decision_rule"][decision],
        "l1_authorized": qualified,
        "m1_authorized": False,
        "historical_h1a_status_changed": False,
        "tensor_work_authorized": False,
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
