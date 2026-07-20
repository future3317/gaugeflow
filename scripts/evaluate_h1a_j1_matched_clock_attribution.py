"""Evaluate the frozen parameter-matched C0/C1/C2 modality-clock attribution."""

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
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from scripts.evaluate_h1a_j1_independent_modality_times import (
    CORNER_NAMES,
    _corner_graph_losses,
    _paired_bootstrap_mean_difference,
    _paired_bootstrap_ratio,
)

ARM_FILES = {
    "C0": Path("configs/gates/h1a_j1_c0_single_clock_control_v1.json"),
    "C1": Path("configs/gates/h1a_j1_c1_side_summary_control_v1.json"),
    "C2": Path("configs/gates/h1a_j1_independent_modality_times_v1.json"),
}


def _finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(float(value))


def _load_training_record(run_root: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_root / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if not rows:
        raise ValueError(f"training log is empty: {run_root}")
    return rows[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_j1_matched_clock_attribution_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen matched-clock protocol")
    prerequisites = protocol["prerequisites"]
    expected_hashes = {
        args.cache_root / "manifest.json": prerequisites["cache_manifest_sha256"],
        ARM_FILES["C0"]: prerequisites["c0_protocol_file_sha256"],
        ARM_FILES["C1"]: prerequisites["c1_protocol_file_sha256"],
        ARM_FILES["C2"]: prerequisites["c2_protocol_file_sha256"],
        Path("reports/h1a_j1_independent_modality_times_v1/result.json"): prerequisites[
            "c2_result_file_sha256"
        ],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"matched-clock prerequisite hash mismatch: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    evaluation = protocol["evaluation"]
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    graph_count = indices.numel()

    arm_protocols = {name: load_json_object(path) for name, path in ARM_FILES.items()}
    arm_roots = {
        name: args.runs_root / str(arm_protocols[name]["protocol"]) / "seed_5705"
        for name in ARM_FILES
    }
    c2_hash = canonical_json_hash(arm_protocols["C2"])
    c2_step0 = arm_roots["C2"] / "checkpoint_step_00000000.pt"
    c2_runtime = load_tensor_free_ema_runtime(
        c2_step0,
        device,
        protocol_name=arm_protocols["C2"]["protocol"],
        protocol_sha256=c2_hash,
    )
    time_diffusion = TensorFreeHybridDiffusion(
        c2_runtime.model,
        c2_runtime.lattice_standardizer,
        minimum_time=float(arm_protocols["C2"]["training"]["minimum_time"]),
        maximum_time=float(arm_protocols["C2"]["training"]["maximum_time"]),
    )
    reference = torch.zeros(1, device=device)
    time_generator = torch.Generator(device=device).manual_seed(
        int(evaluation["validation_noise_seed"]) - 1
    )
    coordinate_time = time_diffusion.sample_time(graph_count, reference, generator=time_generator)
    interior_element_time = time_diffusion.sample_time(
        graph_count, reference, generator=time_generator
    )
    interior_lattice_time = time_diffusion.sample_time(
        graph_count, reference, generator=time_generator
    )
    del c2_runtime, time_diffusion

    arm_losses: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    arm_results: dict[str, Any] = {}
    total_candidates = 0
    parameter_counts: dict[str, int] = {}
    for arm_index, (arm, arm_protocol) in enumerate(arm_protocols.items()):
        training = arm_protocol["training"]
        if (
            training["seeds"] != [5705]
            or int(training["steps"]) != 2111
            or int(training["batch_size"]) != 64
            or training["modality_time_mode"] != "independent_corner_mixture"
        ):
            raise ValueError(f"{arm} violates the matched training contract")
        arm_hash = canonical_json_hash(arm_protocol)
        runtime = load_tensor_free_ema_runtime(
            arm_roots[arm] / "checkpoint_step_00002111.pt",
            device,
            protocol_name=arm_protocol["protocol"],
            protocol_sha256=arm_hash,
        )
        parameter_counts[arm] = sum(parameter.numel() for parameter in runtime.model.parameters())
        expected_mode = protocol["arms"][arm]["time_conditioning"]
        if runtime.model.modality_time_conditioning != expected_mode:
            raise ValueError(f"{arm} checkpoint has the wrong time-conditioning mode")
        del runtime

        by_step: dict[str, dict[str, torch.Tensor]] = {}
        for step in (0, 2111):
            losses, candidates = _corner_graph_losses(
                arm_roots[arm] / f"checkpoint_step_{step:08d}.pt",
                dataset,
                indices,
                coordinate_time,
                interior_element_time,
                interior_lattice_time,
                device=device,
                noise_seed=int(evaluation["validation_noise_seed"]),
                protocol_name=arm_protocol["protocol"],
                protocol_sha256=arm_hash,
                batch_size=int(evaluation["batch_size"]),
            )
            by_step[str(step)] = losses
            total_candidates += candidates
        arm_losses[arm] = by_step
        corners: dict[str, Any] = {}
        for corner_index, corner in enumerate(CORNER_NAMES):
            initial = by_step["0"][corner]
            final = by_step["2111"][corner]
            corners[corner] = {
                "initial_coordinate_mse": float(initial.mean()),
                "final_coordinate_mse": float(final.mean()),
                "validation_ratio": float(final.mean() / initial.mean()),
                "bootstrap_ratio": _paired_bootstrap_ratio(
                    initial,
                    final,
                    seed=int(evaluation["bootstrap_seed"]) + 100 * arm_index + corner_index,
                    replicates=int(evaluation["bootstrap_replicates"]),
                ),
            }
        final_log = _load_training_record(arm_roots[arm])
        arm_results[arm] = {
            "protocol": arm_protocol["protocol"],
            "protocol_sha256": arm_hash,
            "checkpoint_sha256": sha256_file(
                arm_roots[arm] / "checkpoint_step_00002111.pt"
            ),
            "parameter_count": parameter_counts[arm],
            "corners": corners,
            "training": {
                "final_loss": float(final_log["loss"]),
                "clip_fraction": float(final_log["clip_fraction"]),
                "graphs_per_second": float(final_log["graphs_per_second"]),
                "peak_cuda_memory_mib": float(final_log["peak_cuda_memory_mib"]),
            },
        }

    paired: dict[str, Any] = {}
    for comparator_index, comparator in enumerate(("C0", "C1")):
        paired["C2_minus_" + comparator] = {
            corner: _paired_bootstrap_mean_difference(
                arm_losses["C2"]["2111"][corner],
                arm_losses[comparator]["2111"][corner],
                seed=int(evaluation["bootstrap_seed"])
                + 1000
                + 100 * comparator_index
                + corner_index,
                replicates=int(evaluation["bootstrap_replicates"]),
            )
            for corner_index, corner in enumerate(CORNER_NAMES)
        }
    c2_adjacent = {}
    for pair_index, (left, right) in enumerate(zip(CORNER_NAMES[1:], CORNER_NAMES[:-1], strict=True)):
        c2_adjacent[f"{left}_minus_{right}"] = _paired_bootstrap_mean_difference(
            arm_losses["C2"]["2111"][left],
            arm_losses["C2"]["2111"][right],
            seed=int(evaluation["bootstrap_seed"]) + 2000 + pair_index,
            replicates=int(evaluation["bootstrap_replicates"]),
        )

    acceptance = protocol["acceptance"]
    clean_ratio = (
        arm_results["C2"]["corners"]["clean_clean"]["final_coordinate_mse"]
        / arm_results["C0"]["corners"]["clean_clean"]["final_coordinate_mse"]
    )
    checks = {
        "c2_diagonal_paired_improvement": paired["C2_minus_C0"]["diagonal"]["q975"]
        < float(acceptance["c2_minus_c0_diagonal_paired_difference_q975_max"]),
        "c2_interior_paired_improvement": paired["C2_minus_C0"]["interior"]["q975"]
        < float(acceptance["c2_minus_c0_interior_paired_difference_q975_max"]),
        "clean_corner_retention": clean_ratio
        <= float(acceptance["c2_clean_final_mse_over_c0_clean_final_mse_max"]),
        "equal_parameter_count": len(set(parameter_counts.values())) == 1
        and next(iter(parameter_counts.values())) == int(protocol["matched_contract"]["parameter_count"]),
        "finite_losses": _finite_tree(arm_results) and _finite_tree(paired),
        "tensor_bypass": total_candidates == int(acceptance["tensor_candidates"]),
    }
    qualified = all(checks.values())
    decision = "pass" if qualified else "fail"
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "validation_indices_sha256": canonical_json_hash(indices.tolist()),
        "arms": arm_results,
        "paired_final_mse_differences": paired,
        "c2_adjacent_regime_differences": c2_adjacent,
        "c2_clean_over_c0_clean": clean_ratio,
        "checks": checks,
        "qualified": qualified,
        "decision": decision,
        "decision_text": protocol["decision_rule"][decision],
        "free_joint_generation_qualified": False,
        "e1_l1_authorized": False,
        "j2_authorized": False,
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
