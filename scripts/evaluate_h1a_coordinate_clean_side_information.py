"""Evaluate the frozen clean-side-information coordinate screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from diagnose_h1a_coordinate_generator import _score_calibration
from evaluate_h1a_coordinate_pretraining import _rollout_closure
from evaluate_h1a_p1_protocol import _validation_losses

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.checkpointing import read_production_checkpoint_metadata


def _finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(float(value))


@torch.no_grad()
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
        protocol.get("protocol") != "h1a_coordinate_clean_side_information_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
        or not isinstance(training, dict)
        or training.get("objective") != "coordinate"
        or training.get("coordinate_clean_side_information") is not True
        or training.get("exposure_mode") != "prefix_screen"
        or training.get("seeds") != [5705]
    ):
        raise ValueError("unexpected or unfrozen clean-side-information protocol")
    prerequisites = protocol["prerequisites"]
    hash_contract = {
        args.cache_root / "manifest.json": prerequisites["cache_manifest_sha256"],
        Path("reports/h1a_dynamic_persistent_edge_v1/result.json"): prerequisites[
            "qualification_result_sha256"
        ],
        Path("configs/gates/h1a_dynamic_persistent_edge_coordinate_pretraining_v1.json"): prerequisites[
            "source_architecture_protocol_sha256"
        ],
        Path("reports/h1a_fixed_dynamic_coordinate_learning_curve_v1/result.json"): prerequisites[
            "historical_learning_curve_result_sha256"
        ],
    }
    for path, expected in hash_contract.items():
        if sha256_file(path) != str(expected):
            raise ValueError(f"frozen prerequisite hash mismatch: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    protocol_hash = canonical_json_hash(protocol)
    seed = int(training["seeds"][0])
    run = args.run_root / f"seed_{seed}"
    final_step = int(training["steps"])
    checkpoint = run / f"checkpoint_step_{final_step:08d}.pt"
    records = [json.loads(line) for line in (run / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records or int(records[-1]["step"]) != final_step or not _finite_tree(records):
        raise ValueError("training log is incomplete or non-finite")

    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    evaluation = protocol["evaluation"]
    validation_indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(evaluation["validation_seed"]))
    )[: int(evaluation["validation_graphs"])]
    validation = {
        str(step): _validation_losses(
            run / f"checkpoint_step_{int(step):08d}.pt",
            dataset,
            validation_indices,
            device=device,
            seed=int(evaluation["validation_noise_seed"]),
            protocol_name=str(protocol["protocol"]),
            protocol_sha256=protocol_hash,
        )
        for step in training["checkpoint_steps"]
    }
    initial = validation["0"]["coordinate"]
    final = validation[str(final_step)]["coordinate"]
    ratio = final / initial

    from gaugeflow.production.runtime import load_tensor_free_ema_runtime

    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=str(protocol["protocol"]),
        protocol_sha256=protocol_hash,
    )
    score_indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(evaluation["score_seed"]))
    )[: int(evaluation["score_graphs"])]
    score = _score_calibration(
        runtime,
        dataset,
        score_indices,
        {
            "batch_size": 16,
            "noise_seed": int(evaluation["score_noise_seed"]),
            "times": evaluation["score_times"],
        },
        device=device,
    )
    del runtime
    rollout_indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(evaluation["rollout_seed"]))
    )[: int(evaluation["rollout_graphs"])]
    rollout = _rollout_closure(
        checkpoint,
        dataset,
        rollout_indices,
        evaluation,
        device=device,
        protocol_name=str(protocol["protocol"]),
        protocol_sha256=protocol_hash,
    )

    score_by_time = {float(row["time"]): row for row in score}
    rollout_by_time = {float(row["start_time"]): row for row in rollout}
    acceptance = protocol["acceptance"]
    historical = float(prerequisites["historical_025_validation_ratio"])
    final_log = records[-1]
    checks = {
        "finite_training": _finite_tree(records),
        "coordinate_contract_recorded": bool(
            read_production_checkpoint_metadata(checkpoint)["training_config"][
                "coordinate_clean_side_information"
            ]
        ),
        "validation_coordinate_ratio": ratio <= float(acceptance["validation_coordinate_ratio_max"]),
        "material_improvement": historical - ratio
        >= float(acceptance["material_ratio_improvement_over_historical_min"]),
        "t005_endpoint": score_by_time[0.005]["endpoint_rms_angstrom"]
        <= float(acceptance["t005_endpoint_rms_angstrom_max"]),
        "t01_endpoint": score_by_time[0.1]["endpoint_rms_angstrom"]
        <= float(acceptance["t01_endpoint_rms_angstrom_max"]),
        "t06_explained_fraction": score_by_time[0.6]["score_explained_fraction"]
        >= float(acceptance["t06_score_explained_fraction_min"]),
        "t01_rollout": rollout_by_time[0.1]["mean_endpoint_rms_angstrom"]
        <= float(acceptance["t01_rollout_endpoint_rms_angstrom_max"]),
        "t02_rollout": rollout_by_time[0.2]["mean_endpoint_rms_angstrom"]
        <= float(acceptance["t02_rollout_endpoint_rms_angstrom_max"]),
        "sampling_failures": sum(int(row["sampling_failures"]) for row in rollout)
        == int(acceptance["sampling_failures"]),
        "tensor_bypass": validation[str(final_step)]["tensor_candidate_count"]
        == float(acceptance["tensor_candidates"]),
        "throughput": float(final_log["graphs_per_second"])
        >= float(acceptance["training_graphs_per_second_min"]),
        "memory": float(final_log["peak_cuda_memory_mib"])
        <= float(acceptance["peak_cuda_memory_mib_max"]),
    }
    qualified = all(checks.values())
    key = "pass" if qualified else "fail"
    result: dict[str, Any] = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_hash,
        "seed": seed,
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_indices_sha256": canonical_json_hash(validation_indices.tolist()),
        "historical_025_validation_ratio": historical,
        "validation_coordinate_ratio": ratio,
        "absolute_ratio_improvement": historical - ratio,
        "validation": validation,
        "score_calibration": score,
        "conditional_rollout": rollout,
        "training_throughput_graphs_per_second": float(final_log["graphs_per_second"]),
        "peak_cuda_memory_mib": float(final_log["peak_cuda_memory_mib"]),
        "checks": checks,
        "qualified": qualified,
        "decision": key,
        "decision_text": protocol["decision_rule"][key],
        "historical_h1a_status_changed": False,
        "topology_module_authorized": False,
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
