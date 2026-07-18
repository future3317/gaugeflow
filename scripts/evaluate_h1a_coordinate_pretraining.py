"""Evaluate one frozen coordinate-only H1a pretraining checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from diagnose_h1a_coordinate_generator import (
    _score_calibration,
    _translation_aligned_endpoint_rms,
)
from evaluate_h1a_p1_protocol import (
    _validation_losses,
    _validation_losses_for_runtime,
)
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.reverse_sampler import quotient_coordinate_reverse_step
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.schedules import ExponentialTorusNoiseSchedule
from gaugeflow.production.state_projection import project_translation_state


@torch.no_grad()
def _rollout_closure(
    checkpoint: Path,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    specification: dict[str, Any],
    *,
    device: torch.device,
    protocol_name: str,
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=protocol_name,
        protocol_sha256=protocol_sha256,
    )
    packed = Batch.from_data_list([dataset[int(index)] for index in indices]).to(device)
    graphs = int(packed.num_graphs)
    counts = torch.bincount(packed.batch, minlength=graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=packed.frac_coords.dtype, device=device
    )
    clean = project_translation_state(packed.frac_coords, packed.batch, graphs)
    lattice_state = LatticeVolumeShape.from_lattice(
        packed.lattice, blueprint.fractional_to_cartesian
    )
    condition = packed.lattice.new_zeros((graphs, 18))
    present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
    schedule = ExponentialTorusNoiseSchedule(
        sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
    )
    use_bf16 = runtime.training_config["precision"] == "bf16" and device.type == "cuda"
    results: list[dict[str, Any]] = []
    for offset, start_time in enumerate(specification["rollout_start_times"]):
        generator = torch.Generator(device=device).manual_seed(
            int(specification["rollout_noise_seed"]) + offset
        )
        start = packed.lattice.new_full((graphs,), float(start_time))
        noise = torch.randn(
            clean.shape,
            dtype=clean.dtype,
            device=device,
            generator=generator,
        )
        noise = project_translation_state(noise, packed.batch, graphs)
        coordinates = clean + schedule.sigma(start)[packed.batch, None] * noise
        steps = int(specification["rollout_steps"])
        times = torch.linspace(
            float(start_time), 0.0, steps + 1, dtype=clean.dtype, device=device
        )
        failure: str | None = None
        try:
            for step in range(steps):
                time_from = times[step].expand(graphs)
                time_to = times[step + 1].expand(graphs)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    prediction = runtime.model(
                        packed.atom_types,
                        coordinates,
                        lattice_state.log_volume,
                        lattice_state.log_shape,
                        packed.batch,
                        time_from,
                        condition,
                        present,
                        blueprint.shape_projector,
                        blueprint.fractional_to_cartesian,
                    )
                coordinates = quotient_coordinate_reverse_step(
                    coordinates,
                    prediction.coordinate_fractional_scaled_score.float(),
                    schedule.variance(time_from),
                    schedule.variance(time_to),
                    packed.batch,
                    graphs,
                    generator=generator,
                    stochastic=bool(specification["rollout_stochastic"])
                    and float(times[step + 1]) > 0.0,
                )
            if not bool(torch.isfinite(coordinates).all()):
                failure = "nonfinite_terminal_coordinate"
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failure = f"{type(error).__name__}: {error}"
        if failure is None:
            rms = _translation_aligned_endpoint_rms(
                coordinates, clean, packed.lattice, packed.batch
            )
            mean_rms = float(rms.mean())
            quantiles = torch.quantile(
                rms.double(),
                torch.tensor(
                    [0.0, 0.5, 0.9, 0.95, 1.0],
                    dtype=torch.float64,
                    device=device,
                ),
            ).cpu().tolist()
        else:
            mean_rms = float("inf")
            quantiles = []
        results.append(
            {
                "start_time": float(start_time),
                "steps": steps,
                "mean_endpoint_rms_angstrom": mean_rms,
                "endpoint_rms_quantiles_angstrom": quantiles,
                "sampling_failures": int(failure is not None),
                "failure": failure,
            }
        )
    return results


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
        not isinstance(training, dict)
        or training.get("objective") != "coordinate"
        or len(training.get("seeds", [])) != 1
        or float(training.get("data_passes", -1.0)) != 1.0
    ):
        raise ValueError("evaluation requires one exact-pass coordinate protocol")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate-pretraining cache manifest mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    protocol_sha256 = canonical_json_hash(protocol)
    seed = int(protocol["training"]["seeds"][0])
    checkpoint = (
        args.run_root
        / f"seed_{seed}"
        / f"checkpoint_step_{int(protocol['training']['steps']):08d}.pt"
    )
    run = args.run_root / f"seed_{seed}"
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    specification = protocol["evaluation"]
    validation_indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(specification["validation_seed"])),
    )[: int(specification["validation_graphs"])]
    validation_curve = {
        str(step): _validation_losses(
            run / f"checkpoint_step_{int(step):08d}.pt",
            dataset,
            validation_indices,
            device=device,
            seed=int(specification["validation_noise_seed"]),
            protocol_name=str(protocol["protocol"]),
            protocol_sha256=protocol_sha256,
        )
        for step in protocol["training"]["checkpoint_steps"]
    }
    validation = validation_curve[str(int(protocol["training"]["steps"]))]
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=str(protocol["protocol"]),
        protocol_sha256=protocol_sha256,
    )
    raw_runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=str(protocol["protocol"]),
        protocol_sha256=protocol_sha256,
    )
    checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=True)
    raw_runtime.model.load_state_dict(checkpoint_payload["model"], strict=True)
    raw_validation = _validation_losses_for_runtime(
        raw_runtime,
        dataset,
        validation_indices,
        device=device,
        seed=int(specification["validation_noise_seed"]),
    )
    train_dataset = PackedAlexP1Dataset(args.cache_root, "train")
    train_indices = torch.randperm(
        len(train_dataset), generator=torch.Generator().manual_seed(8610)
    )[: int(specification["validation_graphs"])]
    ema_train = _validation_losses_for_runtime(
        runtime,
        train_dataset,
        train_indices,
        device=device,
        seed=8612,
    )
    raw_train = _validation_losses_for_runtime(
        raw_runtime,
        train_dataset,
        train_indices,
        device=device,
        seed=8612,
    )
    score_indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(specification["score_seed"])),
    )[: int(specification["score_graphs"])]
    score = _score_calibration(
        runtime,
        dataset,
        score_indices,
        {
            "batch_size": 16,
            "noise_seed": int(specification["score_noise_seed"]),
            "times": specification["score_times"],
        },
        device=device,
    )
    # Post-hoc attribution only: acceptance remains bound to ``score_times``
    # above.  These high-noise points test whether a learned score fails to
    # vanish as the wrapped heat kernel approaches the uniform torus law.
    high_noise_score = _score_calibration(
        runtime,
        dataset,
        score_indices,
        {
            "batch_size": 16,
            "noise_seed": int(specification["score_noise_seed"]),
            "times": [0.2, 0.5, 0.9],
        },
        device=device,
    )
    rollout_indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(specification["rollout_seed"])),
    )[: int(specification["rollout_graphs"])]
    rollout = _rollout_closure(
        checkpoint,
        dataset,
        rollout_indices,
        specification,
        device=device,
        protocol_name=str(protocol["protocol"]),
        protocol_sha256=protocol_sha256,
    )
    score_by_time = {value["time"]: value for value in score}
    rollout_by_time = {value["start_time"]: value for value in rollout}
    acceptance = protocol["acceptance"]
    initial_coordinate_validation = validation_curve["0"]["coordinate"]
    coordinate_validation_ratio = (
        validation["coordinate"] / initial_coordinate_validation
    )
    checks = {
        "coordinate_validation_reduction": coordinate_validation_ratio
        <= float(acceptance["final_over_initial_coordinate_validation_max"]),
        "t005_teacher_forced": score_by_time[0.005]["endpoint_rms_angstrom"]
        <= float(acceptance["t005_teacher_forced_endpoint_rms_angstrom_max"]),
        "t01_teacher_forced": score_by_time[0.1]["endpoint_rms_angstrom"]
        <= float(acceptance["t01_teacher_forced_endpoint_rms_angstrom_max"]),
        "t01_rollout": rollout_by_time[0.1]["mean_endpoint_rms_angstrom"]
        <= float(acceptance["t01_rollout_endpoint_rms_angstrom_max"]),
        "t02_rollout": rollout_by_time[0.2]["mean_endpoint_rms_angstrom"]
        <= float(acceptance["t02_rollout_endpoint_rms_angstrom_max"]),
        "sampling_failures": sum(value["sampling_failures"] for value in rollout)
        == int(acceptance["sampling_failures"]),
        "tensor_candidates": validation["tensor_candidate_count"]
        == float(acceptance["tensor_candidates"]),
    }
    training_records = [
        json.loads(line)
        for line in (run / "training_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    training_log_finite = bool(training_records) and all(
        math.isfinite(float(value))
        for record in training_records
        for key, value in record.items()
        if key != "step"
    )
    if not training_log_finite or int(training_records[-1]["step"]) != int(
        protocol["training"]["steps"]
    ):
        raise ValueError("coordinate-pretraining log is incomplete or non-finite")
    result = {
        "protocol": protocol["protocol"],
        "seed": seed,
        "checkpoint": str(checkpoint),
        "training": {
            "steps": int(protocol["training"]["steps"]),
            "graph_presentations": int(protocol["training"]["graph_presentations"]),
            "data_passes": float(protocol["training"]["data_passes"]),
            "final_logged_coordinate_loss": float(
                training_records[-1]["coordinate_loss"]
            ),
            "peak_cuda_memory_mib": max(
                float(record["peak_cuda_memory_mib"]) for record in training_records
            ),
            "training_log_finite": training_log_finite,
        },
        "validation": validation,
        "validation_curve": validation_curve,
        "final_over_initial_coordinate_validation": coordinate_validation_ratio,
        "score_calibration": score,
        "posthoc_high_noise_score_diagnostic": high_noise_score,
        "posthoc_fit_diagnostic": {
            "graphs_per_split": int(specification["validation_graphs"]),
            "train_index_seed": 8610,
            "noise_seed": 8612,
            "ema_train_coordinate": ema_train["coordinate"],
            "ema_validation_coordinate": validation["coordinate"],
            "raw_train_coordinate": raw_train["coordinate"],
            "raw_validation_coordinate": raw_validation["coordinate"],
        },
        "rollout_closure": rollout,
        "checks": checks,
        "qualified": all(checks.values()),
        "decision": (
            protocol["decision_rule"]["pass"]
            if all(checks.values())
            else protocol["decision_rule"]["fail"]
        ),
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
