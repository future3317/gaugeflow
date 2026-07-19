"""Run the frozen zero-training coordinate SDE/probability-flow comparison."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from benchmark_h1a_tensor_free import _minimum_distances, _wasserstein
from diagnose_h1a_coordinate_generator import _translation_aligned_endpoint_rms
from torch_geometric.data import Batch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.reverse_sampler import (
    ContinuousReverseMode,
    quotient_coordinate_reverse_step,
    reverse_time_grid,
)
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.schedules import CosineNoiseSchedule, ExponentialTorusNoiseSchedule
from gaugeflow.production.state_projection import project_translation_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("sampler audit produced no rows")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _warmup_model(
    model: torch.nn.Module,
    packed: Any,
    blueprint: ParentBlueprintBatch,
    log_volume: torch.Tensor,
    log_shape: torch.Tensor,
    time: torch.Tensor,
    calls: int,
    *,
    use_bf16: bool,
) -> None:
    condition = time.new_zeros((time.numel(), 18))
    present = torch.zeros((time.numel(), 1), dtype=torch.bool, device=time.device)
    coordinates = project_translation_state(
        packed.frac_coords, packed.batch, int(packed.num_graphs)
    )
    for _ in range(calls):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            model(
                packed.atom_types,
                coordinates,
                log_volume,
                log_shape,
                packed.batch,
                time,
                condition,
                present,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
            )
    torch.cuda.synchronize(time.device)


def _rollout(
    model: torch.nn.Module,
    coordinates: torch.Tensor,
    packed: Any,
    blueprint: ParentBlueprintBatch,
    log_volume: torch.Tensor,
    log_shape: torch.Tensor,
    coordinate_schedule: ExponentialTorusNoiseSchedule,
    vp_schedule: CosineNoiseSchedule,
    maximum_time: float,
    steps: int,
    spacing: str,
    mode: ContinuousReverseMode,
    generator: torch.Generator,
    *,
    use_bf16: bool,
) -> torch.Tensor:
    graphs = int(packed.num_graphs)
    times = reverse_time_grid(
        vp_schedule,
        maximum_time,
        steps,
        dtype=coordinates.dtype,
        device=coordinates.device,
        spacing=spacing,
    )
    condition = coordinates.new_zeros((graphs, 18))
    present = torch.zeros((graphs, 1), dtype=torch.bool, device=coordinates.device)
    state = coordinates.clone()
    for index in range(steps):
        time_from = times[index].expand(graphs)
        time_to = times[index + 1].expand(graphs)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            prediction = model(
                packed.atom_types,
                state,
                log_volume,
                log_shape,
                packed.batch,
                time_from,
                condition,
                present,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
            )
        state = quotient_coordinate_reverse_step(
            state,
            prediction.coordinate_fractional_scaled_score.float(),
            coordinate_schedule.variance(time_from),
            coordinate_schedule.variance(time_to),
            packed.batch,
            graphs,
            generator=generator,
            mode=mode,
        )
    return state


@torch.no_grad()
def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_sde_probability_flow_sampler_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen sampler protocol")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the formal sampler audit requires CUDA")
    source = protocol["source"]
    for path, expected in (
        (args.checkpoint, source["checkpoint_sha256"]),
        (args.cache_root / "manifest.json", source["cache_manifest_sha256"]),
    ):
        if sha256_file(path) != str(expected):
            raise ValueError(f"frozen input hash mismatch: {path}")
    runtime = load_tensor_free_ema_runtime(
        args.checkpoint,
        device,
        protocol_name=str(source["protocol"]),
        protocol_sha256=str(source["protocol_sha256"]),
    )
    if runtime.training_config.get("objective") != source["checkpoint_objective"]:
        raise ValueError("checkpoint objective does not match coordinate-only audit")

    evaluation = protocol["evaluation"]
    dataset = PackedAlexP1Dataset(args.cache_root, str(evaluation["split"]))
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["graphs"])]
    conditions: list[tuple[ContinuousReverseMode, int]] = [
        (mode, int(steps))
        for mode in evaluation["modes"]
        for steps in evaluation["step_counts"]
    ]
    accumulators: dict[tuple[str, int], dict[str, Any]] = {
        condition: {
            "distances": [],
            "endpoint_rms": [],
            "elapsed": 0.0,
            "peak_memory": 0,
            "failures": 0,
        }
        for condition in conditions
    }
    clean_distances: list[torch.Tensor] = []
    coordinate_schedule = ExponentialTorusNoiseSchedule(
        sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
    )
    vp_schedule = CosineNoiseSchedule()
    maximum_time = float(runtime.training_config["maximum_time"])
    use_bf16 = runtime.training_config["precision"] == "bf16"
    batch_size = int(evaluation["batch_size"])
    did_warmup = False

    for batch_index, start in enumerate(range(0, indices.numel(), batch_size)):
        selected = indices[start : start + batch_size]
        packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
        graphs = int(packed.num_graphs)
        counts = torch.bincount(packed.batch, minlength=graphs)
        blueprint = ParentBlueprintBatch.from_node_counts(
            counts, dtype=packed.frac_coords.dtype, device=device
        )
        lattice_state = LatticeVolumeShape.from_lattice(
            packed.lattice, blueprint.fractional_to_cartesian
        )
        clean = project_translation_state(packed.frac_coords, packed.batch, graphs)
        clean_distances.append(
            _minimum_distances(clean, packed.lattice, packed.batch).cpu()
        )
        initial_generator = torch.Generator(device=device).manual_seed(
            int(evaluation["initialization_seed"]) + batch_index
        )
        initial = torch.rand(
            clean.shape, dtype=clean.dtype, device=device, generator=initial_generator
        )
        initial = project_translation_state(initial, packed.batch, graphs)
        if not did_warmup:
            _warmup_model(
                runtime.model,
                packed,
                blueprint,
                lattice_state.log_volume,
                lattice_state.log_shape,
                clean.new_full((graphs,), maximum_time),
                int(evaluation["warmup_model_calls"]),
                use_bf16=use_bf16,
            )
            did_warmup = True
        ordered = conditions if batch_index % 2 == 0 else list(reversed(conditions))
        for mode, steps in ordered:
            noise_generator = torch.Generator(device=device).manual_seed(
                int(evaluation["continuous_noise_seed"]) + batch_index
            )
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                terminal = _rollout(
                    runtime.model,
                    initial,
                    packed,
                    blueprint,
                    lattice_state.log_volume,
                    lattice_state.log_shape,
                    coordinate_schedule,
                    vp_schedule,
                    maximum_time,
                    steps,
                    str(evaluation["time_grid"]),
                    mode,
                    noise_generator,
                    use_bf16=use_bf16,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                peak_memory = torch.cuda.max_memory_allocated(device)
                if not bool(torch.isfinite(terminal).all()):
                    raise FloatingPointError("coordinate rollout produced non-finite state")
                values = accumulators[(mode, steps)]
                values["distances"].append(
                    _minimum_distances(terminal, packed.lattice, packed.batch).cpu()
                )
                values["endpoint_rms"].append(
                    _translation_aligned_endpoint_rms(
                        terminal, clean, packed.lattice, packed.batch
                    ).cpu()
                )
                values["elapsed"] += elapsed
                values["peak_memory"] = max(
                    int(values["peak_memory"]), peak_memory
                )
            except (RuntimeError, ValueError, FloatingPointError):
                accumulators[(mode, steps)]["failures"] += graphs
                torch.cuda.synchronize(device)

    clean_distance = torch.cat(clean_distances).double()
    clean_iqr = float(
        torch.quantile(clean_distance, 0.75) - torch.quantile(clean_distance, 0.25)
    )
    if clean_iqr <= 0.0:
        raise RuntimeError("clean nearest-neighbour IQR is not positive")
    points = int(evaluation["wasserstein_quantile_points"])
    threshold = float(evaluation["minimum_distance_angstrom"])
    rows: list[dict[str, object]] = []
    for mode, steps in conditions:
        values = accumulators[(mode, steps)]
        failures = int(values["failures"])
        if failures or not values["distances"]:
            raise RuntimeError(f"{mode}/{steps} NFE produced {failures} failures")
        distances = torch.cat(values["distances"]).double()
        endpoint = torch.cat(values["endpoint_rms"]).double()
        elapsed = float(values["elapsed"])
        rows.append(
            {
                "mode": mode,
                "nfe": steps,
                "graphs": int(distances.numel()),
                "sampling_failures": failures,
                "terminal_finite_fraction": 1.0,
                "nearest_neighbour_wasserstein_angstrom": _wasserstein(
                    distances, clean_distance, points
                ),
                "nearest_neighbour_wasserstein_normalized": _wasserstein(
                    distances, clean_distance, points
                )
                / clean_iqr,
                "minimum_distance_valid_fraction": float(
                    (distances >= threshold).double().mean()
                ),
                "endpoint_rms_mean_angstrom": float(endpoint.mean()),
                "endpoint_rms_median_angstrom": float(endpoint.median()),
                "latency_seconds": elapsed,
                "graphs_per_second": int(distances.numel()) / elapsed,
                "peak_cuda_memory_mib": int(values["peak_memory"]) / (1024.0**2),
                "composition_exact_identity_fraction": 1.0,
                "lattice_exact_identity_fraction": 1.0,
            }
        )

    lookup = {(str(row["mode"]), int(row["nfe"])): row for row in rows}
    reference = lookup[("reverse_sde", 100)]
    acceptance = protocol["acceptance"]
    candidates: list[dict[str, object]] = []
    for steps in acceptance["candidate_step_counts"]:
        candidate = lookup[("probability_flow", int(steps))]
        checks = {
            "nearest_neighbour_noninferior": float(
                candidate["nearest_neighbour_wasserstein_normalized"]
            )
            - float(reference["nearest_neighbour_wasserstein_normalized"])
            <= float(
                acceptance[
                    "nearest_neighbour_normalized_wasserstein_additive_degradation_max"
                ]
            ),
            "minimum_distance_noninferior": float(
                reference["minimum_distance_valid_fraction"]
            )
            - float(candidate["minimum_distance_valid_fraction"])
            <= float(acceptance["minimum_distance_valid_fraction_degradation_max"]),
            "finite": float(candidate["terminal_finite_fraction"])
            >= float(acceptance["terminal_finite_fraction"]),
            "failures": int(candidate["sampling_failures"])
            == int(acceptance["sampling_failures"]),
            "controlled_composition": float(
                candidate["composition_exact_identity_fraction"]
            )
            == float(acceptance["composition_exact_identity_fraction"]),
            "controlled_lattice": float(candidate["lattice_exact_identity_fraction"])
            == float(acceptance["lattice_exact_identity_fraction"]),
            "latency": float(candidate["latency_seconds"])
            / float(reference["latency_seconds"])
            <= float(acceptance["latency_ratio_to_reverse_sde_100_max"]),
        }
        candidates.append(
            {
                "nfe": int(steps),
                "checks": checks,
                "qualified": all(checks.values()),
                "latency_ratio_to_reverse_sde_100": float(candidate["latency_seconds"])
                / float(reference["latency_seconds"]),
            }
        )
    retain = any(bool(candidate["qualified"]) for candidate in candidates)
    result = {
        "protocol": protocol["protocol"],
        "protocol_file_sha256": sha256_file(args.protocol),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "graphs": int(indices.numel()),
        "clean_nearest_neighbour_iqr_angstrom": clean_iqr,
        "rows": rows,
        "candidate_checks": candidates,
        "qualified": retain,
        "decision": protocol["decision_rule"][
            "retain_probability_flow" if retain else "retain_reverse_sde_only"
        ],
        "scientific_scope": protocol["decision_rule"]["full_hybrid_follow_up"],
        "optimizer_steps": 0,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "sampler_metrics.csv", rows)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
