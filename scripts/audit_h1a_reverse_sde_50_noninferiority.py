"""Qualify reverse-SDE-50 against 100 NFE on a disjoint coordinate panel."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from audit_h1a_sde_probability_flow_sampler import _warmup_model
from benchmark_h1a_tensor_free import _minimum_distances
from diagnose_h1a_coordinate_generator import _translation_aligned_endpoint_rms
from torch_geometric.data import Batch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.reverse_sampler import quotient_coordinate_reverse_step, reverse_time_grid
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.schedules import CosineNoiseSchedule, ExponentialTorusNoiseSchedule
from gaugeflow.production.state_projection import project_translation_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prior-protocol", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("SDE non-inferiority audit produced no rows")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _held_out_indices(length: int, specification: dict[str, Any]) -> torch.Tensor:
    prior = torch.randperm(
        length,
        generator=torch.Generator().manual_seed(int(specification["prior_panel_seed"])),
    )[: int(specification["prior_panel_graphs"])]
    candidates = torch.randperm(
        length,
        generator=torch.Generator().manual_seed(
            int(specification["candidate_permutation_seed"])
        ),
    )
    selected = candidates[~torch.isin(candidates, prior)][: int(specification["graphs"])]
    overlap = int(torch.isin(selected, prior).sum())
    if selected.numel() != int(specification["graphs"]):
        raise ValueError("validation split is too small for the held-out panel")
    if overlap != int(specification["required_overlap_with_prior_panel"]):
        raise ValueError("held-out panel overlaps the prior sampler panel")
    return selected


def _nested_bridge_noises(
    shape: torch.Size,
    reference: torch.Tensor,
    fine_times: torch.Tensor,
    schedule: ExponentialTorusNoiseSchedule,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if fine_times.numel() != 101:
        raise ValueError("nested bridge coupling requires the frozen 100-step grid")
    fine = [
        torch.randn(shape, dtype=reference.dtype, device=reference.device, generator=generator)
        for _ in range(100)
    ]
    variance = schedule.variance(fine_times)
    coarse: list[torch.Tensor] = []
    for index in range(50):
        high = variance[2 * index]
        middle = variance[2 * index + 1]
        low = variance[2 * index + 2]
        if float(low) == 0.0:
            coarse.append(torch.zeros_like(fine[2 * index]))
            continue
        denominator = middle * (high - low)
        first_weight = (low * (high - middle) / denominator).sqrt()
        second_weight = (high * (middle - low) / denominator).sqrt()
        if not torch.allclose(
            first_weight.square() + second_weight.square(),
            torch.ones_like(first_weight),
            atol=2.0e-6,
            rtol=2.0e-6,
        ):
            raise FloatingPointError("nested bridge weights lost unit variance")
        coarse.append(
            first_weight * fine[2 * index] + second_weight * fine[2 * index + 1]
        )
    return coarse, fine


def _rollout(
    model: torch.nn.Module,
    initial: torch.Tensor,
    packed: Any,
    blueprint: ParentBlueprintBatch,
    lattice_state: LatticeVolumeShape,
    coordinate_schedule: ExponentialTorusNoiseSchedule,
    times: torch.Tensor,
    noises: list[torch.Tensor],
    *,
    use_bf16: bool,
) -> torch.Tensor:
    graphs = int(packed.num_graphs)
    if len(noises) != times.numel() - 1:
        raise ValueError("reverse grid and nested noise sequence do not match")
    condition = initial.new_zeros((graphs, 18))
    present = torch.zeros((graphs, 1), dtype=torch.bool, device=initial.device)
    state = initial.clone()
    for index, noise in enumerate(noises):
        time_from = times[index].expand(graphs)
        time_to = times[index + 1].expand(graphs)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            prediction = model(
                packed.atom_types,
                state,
                lattice_state.log_volume,
                lattice_state.log_shape,
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
            generator=None,
            mode="reverse_sde",
            standard_noise=noise,
        )
    return state


def _empirical_w1(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.sort(dim=-1).values - right.sort(dim=-1).values).abs().mean(dim=-1)


def _bootstrap(
    clean: torch.Tensor,
    candidate: torch.Tensor,
    reference: torch.Tensor,
    endpoint_candidate: torch.Tensor,
    endpoint_reference: torch.Tensor,
    specification: dict[str, Any],
    quantiles: list[float],
) -> dict[str, object]:
    if not (
        clean.shape
        == candidate.shape
        == reference.shape
        == endpoint_candidate.shape
        == endpoint_reference.shape
    ):
        raise ValueError("structure bootstrap inputs must be paired")
    generator = torch.Generator().manual_seed(int(specification["seed"]))
    samples = torch.randint(
        clean.numel(),
        (int(specification["replicates"]), clean.numel()),
        generator=generator,
    )
    clean_draw = clean[samples]
    candidate_draw = candidate[samples]
    reference_draw = reference[samples]
    w1_difference = _empirical_w1(candidate_draw, clean_draw) - _empirical_w1(
        reference_draw, clean_draw
    )
    tail = {}
    for probability in quantiles:
        difference = torch.quantile(candidate_draw, probability, dim=1) - torch.quantile(
            reference_draw, probability, dim=1
        )
        tail[str(probability)] = {
            "median_angstrom": float(difference.median()),
            "lcb95_angstrom": float(torch.quantile(difference, 0.05)),
            "ucb95_angstrom": float(torch.quantile(difference, 0.95)),
        }
    endpoint_ratio = endpoint_candidate[samples].mean(dim=1) / endpoint_reference[
        samples
    ].mean(dim=1).clamp_min(1.0e-12)
    return {
        "wasserstein_difference_median_angstrom": float(w1_difference.median()),
        "wasserstein_difference_ucb95_angstrom": float(
            torch.quantile(w1_difference, float(specification["confidence"]))
        ),
        "endpoint_mean_ratio_median": float(endpoint_ratio.median()),
        "endpoint_mean_ratio_ucb95": float(torch.quantile(endpoint_ratio, 0.95)),
        "lower_tail_difference": tail,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_reverse_sde_50_noninferiority_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen SDE non-inferiority protocol")
    source = protocol["source"]
    for path, expected in (
        (args.prior_protocol, source["prior_sampler_protocol_sha256"]),
        (args.prior_result, source["prior_sampler_result_sha256"]),
        (args.checkpoint, source["checkpoint_sha256"]),
        (args.cache_root / "manifest.json", source["cache_manifest_sha256"]),
    ):
        if sha256_file(path) != str(expected):
            raise ValueError(f"frozen input hash mismatch: {path}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the formal SDE non-inferiority audit requires CUDA")
    runtime = load_tensor_free_ema_runtime(
        args.checkpoint,
        device,
        protocol_name=str(source["checkpoint_protocol"]),
        protocol_sha256=str(source["checkpoint_protocol_sha256"]),
    )
    if runtime.training_config.get("objective") != source["checkpoint_objective"]:
        raise ValueError("checkpoint objective does not match coordinate-only audit")

    panel = protocol["held_out_panel"]
    dataset = PackedAlexP1Dataset(args.cache_root, str(panel["split"]))
    indices = _held_out_indices(len(dataset), panel)
    sampling = protocol["sampling"]
    nfe_values = [int(value) for value in sampling["nfe"]]
    if nfe_values != [50, 100]:
        raise ValueError("nested SDE Gate requires exactly 50 and 100 NFE")
    vp_schedule = CosineNoiseSchedule()
    coordinate_schedule = ExponentialTorusNoiseSchedule(
        sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
    )
    maximum_time = float(runtime.training_config["maximum_time"])
    times = {
        steps: reverse_time_grid(
            vp_schedule,
            maximum_time,
            steps,
            dtype=torch.float32,
            device=device,
            spacing=str(sampling["time_grid"]),
        )
        for steps in nfe_values
    }
    if not torch.allclose(times[50], times[100][::2], atol=2.0e-6, rtol=2.0e-6):
        raise ValueError("50-step grid is not nested in the 100-step grid")
    use_bf16 = runtime.training_config["precision"] == "bf16"
    accumulators: dict[int, dict[str, Any]] = {
        steps: {
            "distances": [],
            "endpoint": [],
            "elapsed": 0.0,
            "peak_memory": 0,
            "failures": 0,
        }
        for steps in nfe_values
    }
    clean_distances: list[torch.Tensor] = []
    did_warmup = False
    batch_size = int(sampling["batch_size"])
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
            int(sampling["initialization_seed"]) + batch_index
        )
        initial = torch.rand(
            clean.shape, dtype=clean.dtype, device=device, generator=initial_generator
        )
        initial = project_translation_state(initial, packed.batch, graphs)
        noise_generator = torch.Generator(device=device).manual_seed(
            int(sampling["fine_brownian_seed"]) + batch_index
        )
        coarse_noises, fine_noises = _nested_bridge_noises(
            clean.shape,
            clean,
            times[100],
            coordinate_schedule,
            noise_generator,
        )
        noise_by_nfe = {50: coarse_noises, 100: fine_noises}
        if not did_warmup:
            _warmup_model(
                runtime.model,
                packed,
                blueprint,
                lattice_state.log_volume,
                lattice_state.log_shape,
                clean.new_full((graphs,), maximum_time),
                int(sampling["warmup_model_calls"]),
                use_bf16=use_bf16,
            )
            did_warmup = True
        order = nfe_values if batch_index % 2 == 0 else list(reversed(nfe_values))
        for steps in order:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                terminal = _rollout(
                    runtime.model,
                    initial,
                    packed,
                    blueprint,
                    lattice_state,
                    coordinate_schedule,
                    times[steps],
                    noise_by_nfe[steps],
                    use_bf16=use_bf16,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                peak_memory = torch.cuda.max_memory_allocated(device)
                if not bool(torch.isfinite(terminal).all()):
                    raise FloatingPointError("coordinate rollout produced non-finite state")
                values = accumulators[steps]
                values["elapsed"] += elapsed
                values["peak_memory"] = max(int(values["peak_memory"]), peak_memory)
                values["distances"].append(
                    _minimum_distances(terminal, packed.lattice, packed.batch).cpu()
                )
                values["endpoint"].append(
                    _translation_aligned_endpoint_rms(
                        terminal, clean, packed.lattice, packed.batch
                    ).cpu()
                )
            except (RuntimeError, ValueError, FloatingPointError):
                accumulators[steps]["failures"] += graphs
                torch.cuda.synchronize(device)

    clean_distance = torch.cat(clean_distances).double()
    rows: list[dict[str, object]] = []
    distances: dict[int, torch.Tensor] = {}
    endpoints: dict[int, torch.Tensor] = {}
    threshold = float(protocol["metrics"]["minimum_distance_angstrom"])
    for steps in nfe_values:
        values = accumulators[steps]
        if int(values["failures"]) or not values["distances"]:
            raise RuntimeError(f"reverse-SDE-{steps} produced failures")
        distances[steps] = torch.cat(values["distances"]).double()
        endpoints[steps] = torch.cat(values["endpoint"]).double()
        elapsed = float(values["elapsed"])
        rows.append(
            {
                "nfe": steps,
                "graphs": int(distances[steps].numel()),
                "sampling_failures": int(values["failures"]),
                "terminal_finite_fraction": 1.0,
                "nearest_neighbour_w1_angstrom": float(
                    _empirical_w1(distances[steps], clean_distance)
                ),
                "minimum_distance_valid_fraction": float(
                    (distances[steps] >= threshold).double().mean()
                ),
                "minimum_distance_q01_angstrom": float(
                    torch.quantile(distances[steps], 0.01)
                ),
                "minimum_distance_q05_angstrom": float(
                    torch.quantile(distances[steps], 0.05)
                ),
                "endpoint_periodic_rms_mean_angstrom": float(endpoints[steps].mean()),
                "latency_seconds": elapsed,
                "graphs_per_second": int(distances[steps].numel()) / elapsed,
                "peak_cuda_memory_mib": int(values["peak_memory"]) / (1024.0**2),
            }
        )
    row_by_nfe = {int(row["nfe"]): row for row in rows}
    bootstrap = _bootstrap(
        clean_distance,
        distances[50],
        distances[100],
        endpoints[50],
        endpoints[100],
        protocol["bootstrap"],
        [float(value) for value in protocol["metrics"]["lower_tail_quantiles"]],
    )
    candidate = row_by_nfe[50]
    reference = row_by_nfe[100]
    acceptance = protocol["acceptance"]
    checks = {
        "wasserstein_ucb": float(bootstrap["wasserstein_difference_ucb95_angstrom"])
        <= float(acceptance["wasserstein_difference_structure_bootstrap_ucb95_angstrom_max"]),
        "valid_distance": float(reference["minimum_distance_valid_fraction"])
        - float(candidate["minimum_distance_valid_fraction"])
        <= float(acceptance["valid_distance_rate_degradation_max"]),
        "endpoint_rms": float(candidate["endpoint_periodic_rms_mean_angstrom"])
        / max(float(reference["endpoint_periodic_rms_mean_angstrom"]), 1.0e-12)
        - 1.0
        <= float(acceptance["endpoint_periodic_rms_relative_degradation_max"]),
        "lower_tail_direct": all(
            float(candidate[f"minimum_distance_q{round(probability * 100):02d}_angstrom"])
            >= float(reference[f"minimum_distance_q{round(probability * 100):02d}_angstrom"])
            - float(acceptance["lower_tail_quantile_degradation_angstrom_max"])
            for probability in protocol["metrics"]["lower_tail_quantiles"]
        ),
        "lower_tail_bootstrap": all(
            float(bootstrap["lower_tail_difference"][str(float(probability))]["lcb95_angstrom"])
            >= float(acceptance["lower_tail_difference_structure_bootstrap_lcb95_angstrom_min"])
            for probability in protocol["metrics"]["lower_tail_quantiles"]
        ),
        "failures": int(candidate["sampling_failures"])
        == int(acceptance["sampling_failures"]),
        "finite": float(candidate["terminal_finite_fraction"])
        >= float(acceptance["terminal_finite_fraction"]),
        "latency": float(candidate["latency_seconds"])
        / float(reference["latency_seconds"])
        <= float(acceptance["latency_ratio_to_sde_100_max"]),
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_file_sha256": sha256_file(args.protocol),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "graphs": int(indices.numel()),
        "prior_panel_overlap": 0,
        "rows": rows,
        "bootstrap": bootstrap,
        "checks": checks,
        "latency_ratio_50_to_100": float(candidate["latency_seconds"])
        / float(reference["latency_seconds"]),
        "qualified": qualified,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
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
