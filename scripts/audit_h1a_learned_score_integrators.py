"""Compare coordinate integrators under one frozen learned EMA score."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from audit_h1a_coordinate_reverse_closure import _endpoint_rms
from audit_h1a_wrapped_reverse_kernel import (
    SCORE_ONLY_METHODS,
    _integrate_score_step,
)
from torch_geometric.data import Batch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.schedules import ExponentialTorusNoiseSchedule
from gaugeflow.production.state_projection import project_translation_state


def _quantiles(values: torch.Tensor) -> list[float]:
    return torch.quantile(
        values.double(),
        torch.tensor(
            [0.0, 0.05, 0.5, 0.95, 1.0],
            dtype=torch.float64,
            device=values.device,
        ),
    ).cpu().tolist()


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
    if protocol.get("protocol") != "h1a_learned_score_integrator_audit_v1":
        raise ValueError("unexpected learned-score integrator protocol")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["cache_manifest_sha256"]
    ):
        raise ValueError("learned-score audit cache manifest mismatch")
    checkpoint = (
        args.run_root
        / f"seed_{int(protocol['source_seed'])}"
        / f"checkpoint_step_{int(protocol['source_checkpoint_step']):08d}.pt"
    )
    if sha256_file(checkpoint) != str(protocol["source_checkpoint_sha256"]):
        raise ValueError("learned-score audit checkpoint hash mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=str(protocol["source_protocol"]),
        protocol_sha256=str(protocol["source_protocol_sha256"]),
    )
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(protocol["validation_seed"])),
    )[: int(protocol["validation_graphs"])]
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
    records: list[dict[str, Any]] = []
    for start_index, start_time in enumerate(protocol["start_times"]):
        initial_generator = torch.Generator(device=device).manual_seed(
            int(protocol["noise_seed"]) + start_index
        )
        start = packed.lattice.new_full((graphs,), float(start_time))
        noise = torch.randn(
            clean.shape,
            dtype=clean.dtype,
            device=device,
            generator=initial_generator,
        )
        noise = project_translation_state(noise, packed.batch, graphs)
        initial = clean + schedule.sigma(start)[packed.batch, None] * noise
        for steps in map(int, protocol["step_counts"]):
            times = torch.linspace(
                float(start_time), 0.0, steps + 1, dtype=clean.dtype, device=device
            )
            variances = schedule.variance(times)
            for method in protocol["methods"]:
                if method not in SCORE_ONLY_METHODS:
                    raise ValueError(f"unknown learned-score method: {method}")
                generator = torch.Generator(device=device).manual_seed(
                    int(protocol["noise_seed"]) + 10_000 * start_index + steps
                )
                coordinates = initial.clone()
                score_evaluations = 0

                def score_function(
                    value: torch.Tensor, variance: torch.Tensor
                ) -> torch.Tensor:
                    nonlocal score_evaluations
                    score_evaluations += 1
                    sigma = variance.sqrt()
                    scalar_time = (
                        torch.log(sigma / schedule.sigma_min) / schedule.log_ratio
                    ).clamp(0.0, 1.0)
                    time = scalar_time.expand(graphs)
                    prediction = runtime.model(
                        packed.atom_types,
                        value,
                        lattice_state.log_volume,
                        lattice_state.log_shape,
                        packed.batch,
                        time,
                        condition,
                        present,
                        blueprint.shape_projector,
                        blueprint.fractional_to_cartesian,
                    )
                    return (
                        prediction.coordinate_fractional_scaled_score
                        / sigma.clamp_min(1.0e-12)
                    )

                failure: str | None = None
                try:
                    for index in range(steps):
                        coordinates = _integrate_score_step(
                            str(method),
                            coordinates,
                            variances[index],
                            variances[index + 1],
                            score_function,
                            generator,
                        )
                    if not bool(torch.isfinite(coordinates).all()):
                        failure = "nonfinite_terminal_coordinate"
                except (RuntimeError, ValueError, FloatingPointError) as error:
                    failure = f"{type(error).__name__}: {error}"
                if failure is None:
                    rms = _endpoint_rms(
                        coordinates, clean, packed.lattice, packed.batch
                    )
                    mean_rms = float(rms.mean())
                    quantiles = _quantiles(rms)
                else:
                    mean_rms = math.inf
                    quantiles = []
                records.append(
                    {
                        "start_time": float(start_time),
                        "steps": steps,
                        "method": method,
                        "mean_endpoint_rms_angstrom": mean_rms,
                        "endpoint_rms_quantiles_angstrom": quantiles,
                        "score_evaluations": score_evaluations,
                        "sampling_failures": int(failure is not None),
                        "failure": failure,
                    }
                )

    by_key = {
        (record["method"], record["start_time"], record["steps"]): record
        for record in records
    }
    baseline_t01 = by_key[("ancestral_gaussian", 0.1, 100)][
        "mean_endpoint_rms_angstrom"
    ]
    baseline_t02 = by_key[("ancestral_gaussian", 0.2, 100)][
        "mean_endpoint_rms_angstrom"
    ]
    acceptance = protocol["acceptance"]
    checks: dict[str, dict[str, Any]] = {}
    for method in protocol["methods"]:
        if method == "ancestral_gaussian":
            continue
        t01 = by_key[(method, 0.1, 100)]["mean_endpoint_rms_angstrom"]
        t02 = by_key[(method, 0.2, 100)]["mean_endpoint_rms_angstrom"]
        increases = [
            by_key[(method, float(time), 200)]["mean_endpoint_rms_angstrom"]
            - by_key[(method, float(time), 100)]["mean_endpoint_rms_angstrom"]
            for time in protocol["start_times"]
        ]
        method_checks = {
            "t02_improvement": t02 / baseline_t02
            <= float(
                acceptance[
                    "candidate_t02_100step_mean_rms_ratio_to_ancestral_max"
                ]
            ),
            "t01_guardrail": t01 / baseline_t01
            <= float(
                acceptance[
                    "candidate_t01_100step_mean_rms_ratio_to_ancestral_max"
                ]
            ),
            "step_refinement": max(increases)
            <= float(
                acceptance[
                    "candidate_100_to_200_mean_rms_increase_angstrom_max"
                ]
            ),
            "sampling_failures": max(
                int(record["sampling_failures"])
                for record in records
                if record["method"] == method
            )
            == int(acceptance["sampling_failures"]),
        }
        checks[str(method)] = {
            "checks": method_checks,
            "qualified": all(method_checks.values()),
            "t01_100step_rms_ratio_to_ancestral": t01 / baseline_t01,
            "t02_100step_rms_ratio_to_ancestral": t02 / baseline_t02,
            "max_100_to_200_mean_rms_increase_angstrom": max(increases),
        }
    qualified = [method for method, value in checks.items() if value["qualified"]]
    result = {
        "protocol": protocol["protocol"],
        "validation_indices": indices.tolist(),
        "records": records,
        "candidate_checks": checks,
        "qualified_candidates": qualified,
        "decision": (
            "unique_candidate_may_enter_free_generation_diagnostic"
            if len(qualified) == 1
            else "no_unique_integrator_repair_retain_production_sampler"
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
