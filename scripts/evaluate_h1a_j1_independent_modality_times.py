"""Evaluate the frozen J1 independent-modality-time attribution gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.modality_time_diagnostics import (
    CORNER_NAMES,
    corner_graph_losses,
    paired_bootstrap_ratio,
)
from gaugeflow.production.runtime import load_tensor_free_ema_runtime


def _finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    training = protocol["training"]
    if (
        protocol.get("protocol") != "h1a_j1_independent_modality_times_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
        or training.get("seeds") != [5705]
        or training.get("steps") != 2111
        or training.get("modality_time_mode") != "independent_corner_mixture"
        or training.get("coordinate_clean_side_information") is not False
    ):
        raise ValueError("unexpected or unfrozen J1 protocol")
    prerequisites = protocol["prerequisites"]
    hash_contract = {
        args.cache_root / "manifest.json": prerequisites["cache_manifest_sha256"],
        Path("reports/h1a_j0_side_information_sensitivity_v1/result.json"): prerequisites[
            "qualification_result_sha256"
        ],
        Path("configs/gates/h1a_j0_side_information_sensitivity_v1.json"): prerequisites[
            "j0_protocol_sha256"
        ],
    }
    for path, expected in hash_contract.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen prerequisite hash mismatch: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    evaluation = protocol["evaluation"]
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    graph_count = indices.numel()
    reference = torch.zeros(1, device=device)
    time_diffusion_checkpoint = args.run_root / "checkpoint_step_00000000.pt"
    protocol_hash = canonical_json_hash(protocol)
    time_runtime = load_tensor_free_ema_runtime(
        time_diffusion_checkpoint,
        device,
        protocol_name=protocol["protocol"],
        protocol_sha256=protocol_hash,
    )
    time_diffusion = TensorFreeHybridDiffusion(
        time_runtime.model,
        time_runtime.lattice_standardizer,
        minimum_time=float(training["minimum_time"]),
        maximum_time=float(training["maximum_time"]),
    )
    time_generator = torch.Generator(device=device).manual_seed(int(evaluation["validation_noise_seed"]) - 1)
    coordinate_time = time_diffusion.sample_time(graph_count, reference, generator=time_generator)
    interior_element_time = time_diffusion.sample_time(graph_count, reference, generator=time_generator)
    interior_lattice_time = time_diffusion.sample_time(graph_count, reference, generator=time_generator)
    del time_runtime, time_diffusion

    losses_by_step: dict[str, dict[str, torch.Tensor]] = {}
    candidates = 0
    for step in (0, int(training["steps"])):
        losses, count = corner_graph_losses(
            args.run_root / f"checkpoint_step_{step:08d}.pt",
            dataset,
            indices,
            coordinate_time,
            interior_element_time,
            interior_lattice_time,
            device=device,
            noise_seed=int(evaluation["validation_noise_seed"]),
            protocol_name=protocol["protocol"],
            protocol_sha256=protocol_hash,
        )
        losses_by_step[str(step)] = losses
        candidates += count

    corner_results: dict[str, Any] = {}
    for index, name in enumerate(CORNER_NAMES):
        initial = losses_by_step["0"][name]
        final = losses_by_step[str(training["steps"])][name]
        ratio = float(final.mean() / initial.mean())
        corner_results[name] = {
            "initial_coordinate_mse": float(initial.mean()),
            "final_coordinate_mse": float(final.mean()),
            "validation_ratio": ratio,
            "bootstrap_ratio": paired_bootstrap_ratio(
                initial,
                final,
                seed=int(evaluation["bootstrap_seed"]) + index,
                replicates=int(evaluation["bootstrap_replicates"]),
            ),
        }

    records = [
        json.loads(line)
        for line in (args.run_root / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    final_log = records[-1]
    time_gradient_rows = [row.get("modality_time_gradient_norms", {}) for row in records]
    positive_time_gradients = all(
        set(row) == {"coordinate", "element", "lattice", "fusion"}
        and all(math.isfinite(float(value)) and float(value) > 0.0 for value in row.values())
        for row in time_gradient_rows
    )
    acceptance = protocol["acceptance"]
    checks = {
        "clean_clean_retention": corner_results["clean_clean"]["validation_ratio"]
        <= float(acceptance["clean_clean_validation_ratio_max"]),
        "diagonal_improvement": corner_results["diagonal"]["validation_ratio"]
        <= float(acceptance["diagonal_validation_ratio_max"]),
        "modality_time_gradients": positive_time_gradients,
        "finite_corner_losses": _finite_tree(corner_results),
        "tensor_bypass": candidates == int(acceptance["tensor_candidates"]),
        "throughput": float(final_log["graphs_per_second"])
        >= float(acceptance["training_graphs_per_second_min"]),
        "memory": float(final_log["peak_cuda_memory_mib"])
        <= float(acceptance["peak_cuda_memory_mib_max"]),
        "target_leakage_fields": int(acceptance["target_leakage_fields"]) == 0,
    }
    qualified = all(checks.values())
    decision = "pass" if qualified else "fail"
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_hash,
        "seed": 5705,
        "checkpoint_sha256": sha256_file(
            args.run_root / f"checkpoint_step_{int(training['steps']):08d}.pt"
        ),
        "validation_indices_sha256": canonical_json_hash(indices.tolist()),
        "corner_results": corner_results,
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
        "historical_h1a_status_changed": False,
        "j2_authorized": qualified,
        "acf_authorized": False,
        "tensor_work_authorized": False,
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
