"""Evaluate the frozen H1a fixed-architecture 0.25/0.5/1/2-pass curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.runtime import load_tensor_free_ema_runtime


def _classify_learning_curve(
    one_pass_loss: float,
    two_pass_loss: float,
    one_pass_ratio: float,
    reference_ratio: float,
    specification: dict[str, Any],
) -> tuple[str, float, bool]:
    if one_pass_loss <= 0.0 or two_pass_loss < 0.0:
        raise ValueError("learning-curve losses must be nonnegative with positive baseline")
    improvement = (one_pass_loss - two_pass_loss) / one_pass_loss
    reference_matches = abs(one_pass_ratio - reference_ratio) <= float(
        specification["one_pass_reference_ratio_absolute_tolerance"]
    )
    if not reference_matches:
        decision = "reference_mismatch"
    elif improvement >= float(specification["one_to_two_pass_relative_validation_improvement_undertraining_min"]):
        decision = "undertraining"
    elif improvement <= float(specification["one_to_two_pass_relative_validation_improvement_plateau_max"]):
        decision = "representation_ceiling"
    else:
        decision = "ambiguous"
    return decision, improvement, reference_matches


def _is_finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_is_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_is_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    from diagnose_h1a_coordinate_generator import _score_calibration
    from evaluate_h1a_p1_protocol import _validation_losses

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_fixed_dynamic_coordinate_learning_curve_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen fixed-architecture protocol")
    training = protocol["training"]
    if (
        training.get("objective") != "coordinate"
        or training.get("seeds") != [5705]
        or float(training.get("data_passes", -1.0)) != 2.0
        or not bool(training.get("from_scratch_required"))
    ):
        raise ValueError("learning-curve training contract changed")
    prerequisites = protocol["prerequisites"]
    hash_contract = {
        args.cache_root / "manifest.json": prerequisites["cache_manifest_sha256"],
        Path("configs/gates/h1a_dynamic_persistent_edge_coordinate_pretraining_v1.json"): prerequisites[
            "source_architecture_protocol_sha256"
        ],
        Path("reports/h1a_dynamic_persistent_edge_v1/result.json"): prerequisites["qualification_result_sha256"],
        Path("reports/h1a_dynamic_persistent_edge_coordinate_pretraining_v1/result.json"): prerequisites[
            "reference_one_pass_result_sha256"
        ],
        Path("reports/h1a_all_pair_clean_topology_attribution_v2/result.json"): prerequisites[
            "all_pair_topology_result_sha256"
        ],
    }
    for path, expected in hash_contract.items():
        if sha256_file(path) != str(expected):
            raise ValueError(f"frozen prerequisite hash mismatch: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    protocol_sha256 = canonical_json_hash(protocol)
    seed = int(training["seeds"][0])
    run = args.run_root / f"seed_{seed}"
    records = [json.loads(line) for line in (run / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records or int(records[-1]["step"]) != int(training["steps"]) or not _is_finite_tree(records):
        raise ValueError("training log is incomplete or non-finite")
    records_by_step = {int(record["step"]): record for record in records}

    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    evaluation = protocol["evaluation"]
    validation_indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    score_indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["score_seed"])),
    )[: int(evaluation["score_graphs"])]
    checkpoints = [int(value) for value in training["checkpoint_steps"]]
    pass_labels = training["nominal_data_passes_by_checkpoint"]
    validation_by_step: dict[int, dict[str, float]] = {}
    score_by_step: dict[int, list[dict[str, float]]] = {}
    checkpoint_hashes: dict[int, str] = {}
    learning_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []

    for step in checkpoints:
        checkpoint = run / f"checkpoint_step_{step:08d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[step] = sha256_file(checkpoint)
        validation = _validation_losses(
            checkpoint,
            dataset,
            validation_indices,
            device=device,
            seed=int(evaluation["validation_noise_seed"]),
            protocol_name=str(protocol["protocol"]),
            protocol_sha256=protocol_sha256,
        )
        validation_by_step[step] = validation
        runtime = load_tensor_free_ema_runtime(
            checkpoint,
            device,
            protocol_name=str(protocol["protocol"]),
            protocol_sha256=protocol_sha256,
        )
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
        score_by_step[step] = score
        del runtime
        if device.type == "cuda":
            torch.cuda.empty_cache()
        initial = validation_by_step[checkpoints[0]]["coordinate"]
        learning_rows.append(
            {
                "step": step,
                "nominal_data_passes": float(pass_labels[str(step)]),
                "graphs_seen": 0 if step == 0 else int(records_by_step[step]["graphs_seen_this_invocation"]),
                "validation_coordinate_loss": validation["coordinate"],
                "validation_coordinate_ratio": validation["coordinate"] / initial,
                "training_coordinate_loss": "" if step == 0 else records_by_step[step]["coordinate_loss"],
                "graphs_per_second": "" if step == 0 else records_by_step[step]["graphs_per_second"],
                "peak_cuda_memory_mib": "" if step == 0 else records_by_step[step]["peak_cuda_memory_mib"],
            }
        )
        for score_record in score:
            score_rows.append(
                {
                    "step": step,
                    "nominal_data_passes": float(pass_labels[str(step)]),
                    **score_record,
                }
            )
        gradients = (
            {name: "" for name in evaluation["gradient_groups"]}
            if step == 0
            else records_by_step[step]["clipped_module_gradient_norms"]
        )
        gradient_rows.append(
            {
                "step": step,
                "nominal_data_passes": float(pass_labels[str(step)]),
                "global_preclip_gradient_norm": "" if step == 0 else records_by_step[step]["gradient_norm"],
                **gradients,
            }
        )

    initial_loss = validation_by_step[0]["coordinate"]
    one_pass_loss = validation_by_step[int(training["steps_per_complete_pass"])]["coordinate"]
    two_pass_loss = validation_by_step[int(training["steps"])]["coordinate"]
    one_pass_ratio = one_pass_loss / initial_loss
    two_pass_ratio = two_pass_loss / initial_loss
    decision, improvement, reference_matches = _classify_learning_curve(
        one_pass_loss,
        two_pass_loss,
        one_pass_ratio,
        float(prerequisites["reference_one_pass_validation_ratio"]),
        protocol["classification"],
    )
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "checkpoint_sha256": {str(step): value for step, value in checkpoint_hashes.items()},
        "validation_indices_sha256": canonical_json_hash(validation_indices.tolist()),
        "score_indices_sha256": canonical_json_hash(score_indices.tolist()),
        "initial_validation_coordinate_loss": initial_loss,
        "one_pass_validation_coordinate_loss": one_pass_loss,
        "two_pass_validation_coordinate_loss": two_pass_loss,
        "one_pass_validation_ratio": one_pass_ratio,
        "two_pass_validation_ratio": two_pass_ratio,
        "reference_one_pass_validation_ratio": float(prerequisites["reference_one_pass_validation_ratio"]),
        "one_pass_reference_matches": reference_matches,
        "one_to_two_pass_relative_validation_improvement": improvement,
        "decision": decision,
        "decision_text": protocol["classification"][decision],
        "historical_h1a_status_changed": False,
        "production_architecture_changed": False,
        "tensor_condition_enabled": False,
        "decision_boundary": protocol["decision_boundary"],
        "validation_by_step": {str(step): value for step, value in validation_by_step.items()},
        "score_by_step": {str(step): value for step, value in score_by_step.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "learning_curve.csv", learning_rows)
    _write_csv(args.output_dir / "score_endpoint_curve.csv", score_rows)
    _write_csv(args.output_dir / "module_gradient_curve.csv", gradient_rows)
    (args.output_dir / "summary.md").write_text(
        "\n".join(
            (
                "# H1a fixed dynamic-architecture learning curve v1",
                "",
                f"Decision: **{decision}**.",
                "",
                f"- one-pass validation ratio: `{one_pass_ratio:.6f}`",
                f"- two-pass validation ratio: `{two_pass_ratio:.6f}`",
                f"- one-to-two-pass relative improvement: `{improvement:.6f}`",
                f"- archived one-pass ratio reproduced: `{reference_matches}`",
                "- historical H1a status changed: `False`",
                "- production architecture changed: `False`",
                "",
                protocol["decision_boundary"],
                "",
            )
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
