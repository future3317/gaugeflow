"""Attribute low-noise coordinate failure to primal/dual lattice metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from scripts.diagnose_h1a_coordinate_generator import (
    _translation_aligned_endpoint_rms,
)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    stacked = torch.stack((left.double().log(), right.double().log()))
    return float(torch.corrcoef(stacked)[0, 1])


def _metric_values(
    error: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    metric = lattice @ lattice.transpose(-1, -2)
    dual_gradient = torch.linalg.solve(
        metric[batch], error.unsqueeze(-1)
    ).squeeze(-1)
    primal_gradient = torch.einsum("ni,nij->nj", error, metric[batch])
    dual = scatter(
        (error * dual_gradient).sum(-1),
        batch,
        dim=0,
        dim_size=graph_count,
        reduce="mean",
    )
    primal = scatter(
        (error * primal_gradient).sum(-1),
        batch,
        dim=0,
        dim_size=graph_count,
        reduce="mean",
    )
    return dual, primal, dual_gradient, primal_gradient


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_primal_dual_metric_attribution_v1":
        raise ValueError("primal/dual attribution protocol mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("primal/dual attribution cache mismatch")
    if int(protocol["audit"]["optimizer_steps"]) != 0:
        raise ValueError("primal/dual attribution forbids optimizer steps")
    source_protocol = load_json_object(args.source_protocol)
    expected_source = str(protocol["prerequisites"]["source_protocol"])
    if source_protocol.get("protocol") != expected_source:
        raise ValueError("primal/dual attribution source protocol mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("primal/dual attribution requires CUDA")
    step = int(protocol["prerequisites"]["source_checkpoint_step"])
    seed = int(source_protocol["training"]["seeds"][0])
    checkpoint = args.run_root / f"seed_{seed}" / f"checkpoint_step_{step:08d}.pt"
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=expected_source,
        protocol_sha256=canonical_json_hash(source_protocol),
    )
    runtime.model.eval()
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    data = protocol["data"]
    dataset = PackedAlexP1Dataset(args.cache_root, str(data["split"]))
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(data["selection_seed"]))
    )[: int(data["graphs"])]
    use_bf16 = runtime.training_config["precision"] == "bf16"
    time_results: list[dict[str, Any]] = []
    for time_value in data["times"]:
        dual_values: list[torch.Tensor] = []
        primal_values: list[torch.Tensor] = []
        endpoint_values: list[torch.Tensor] = []
        condition_values: list[torch.Tensor] = []
        dual_gradient_values: list[torch.Tensor] = []
        primal_gradient_values: list[torch.Tensor] = []
        generator = torch.Generator(device=device).manual_seed(
            int(data["noise_seed"]) + round(float(time_value) * 1_000_000)
        )
        for start in range(0, indices.numel(), int(data["batch_size"])):
            selected = indices[start : start + int(data["batch_size"])]
            packed = Batch.from_data_list(
                [dataset[int(index)] for index in selected]
            ).to(device)
            graphs = int(packed.num_graphs)
            counts = torch.bincount(packed.batch, minlength=graphs)
            blueprint = ParentBlueprintBatch.from_node_counts(
                counts, dtype=packed.frac_coords.dtype, device=device
            )
            time = packed.lattice.new_full((graphs,), float(time_value))
            noisy = diffusion.noise_clean_batch(
                packed.atom_types,
                packed.frac_coords,
                packed.lattice,
                packed.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=time,
                generator=generator,
            )
            condition = time.new_zeros((graphs, 18))
            present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                prediction = runtime.model(
                    noisy.element_tokens,
                    noisy.fractional_coordinates,
                    noisy.log_volume,
                    noisy.log_shape,
                    packed.batch,
                    time,
                    condition,
                    present,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                )
            error = (
                prediction.coordinate_fractional_scaled_score.float()
                - noisy.coordinate_scaled_score_target.float()
            )
            noisy_lattice = LatticeVolumeShape(
                noisy.log_volume, noisy.log_shape
            ).lattice(blueprint.fractional_to_cartesian)
            dual, primal, dual_gradient, primal_gradient = _metric_values(
                error, noisy_lattice, packed.batch, graphs
            )
            sigma = diffusion.coordinate_schedule.sigma(time)[packed.batch, None]
            score = prediction.coordinate_fractional_scaled_score.float() / sigma
            variance = diffusion.coordinate_schedule.variance(time)[packed.batch]
            estimate = noisy.fractional_coordinates + variance[:, None] * score
            endpoint = _translation_aligned_endpoint_rms(
                estimate, packed.frac_coords, packed.lattice, packed.batch
            ).square()
            singular = torch.linalg.svdvals(noisy_lattice)
            relative_condition = (singular[:, 0] / singular[:, -1]).pow(4)
            dual_values.append(dual.cpu())
            primal_values.append(primal.cpu())
            endpoint_values.append(endpoint.cpu())
            condition_values.append(relative_condition.cpu())
            dual_gradient_values.append(dual_gradient.cpu().reshape(-1))
            primal_gradient_values.append(primal_gradient.cpu().reshape(-1))
        dual_all = torch.cat(dual_values).clamp_min(1.0e-30)
        primal_all = torch.cat(primal_values).clamp_min(1.0e-30)
        endpoint_all = torch.cat(endpoint_values).clamp_min(1.0e-30)
        dual_gradient_all = torch.cat(dual_gradient_values)
        primal_gradient_all = torch.cat(primal_gradient_values)
        gradient_cosine = float(
            torch.dot(dual_gradient_all, primal_gradient_all)
            / (
                torch.linalg.vector_norm(dual_gradient_all)
                * torch.linalg.vector_norm(primal_gradient_all)
            ).clamp_min(1.0e-30)
        )
        condition_all = torch.cat(condition_values).double()
        time_results.append(
            {
                "time": float(time_value),
                "dual_loss_mean": float(dual_all.mean()),
                "primal_loss_mean": float(primal_all.mean()),
                "endpoint_squared_mean_angstrom2": float(endpoint_all.mean()),
                "dual_primal_gradient_cosine": gradient_cosine,
                "dual_endpoint_log_correlation": _correlation(dual_all, endpoint_all),
                "primal_endpoint_log_correlation": _correlation(primal_all, endpoint_all),
                "median_relative_metric_condition": float(condition_all.median()),
                "maximum_relative_metric_condition": float(condition_all.max()),
            }
        )
    thresholds = protocol["decision_thresholds"]
    low = next(
        value
        for value in time_results
        if value["time"] == float(thresholds["low_noise_time"])
    )
    checks = {
        "metric_condition": low["median_relative_metric_condition"]
        >= float(thresholds["median_metric_condition_min"]),
        "gradient_opposition": low["dual_primal_gradient_cosine"]
        <= float(thresholds["dual_primal_gradient_cosine_max"]),
        "primal_endpoint_alignment": low["primal_endpoint_log_correlation"]
        >= float(thresholds["primal_endpoint_log_correlation_min"]),
        "correlation_gap": (
            low["primal_endpoint_log_correlation"]
            - low["dual_endpoint_log_correlation"]
        )
        >= float(thresholds["primal_over_dual_endpoint_correlation_gap_min"]),
    }
    metric_mismatch = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "source_protocol": expected_source,
        "source_checkpoint": str(checkpoint),
        "fixed_indices": indices.tolist(),
        "time_resolved": time_results,
        "checks": checks,
        "attribution": "primal_dual_metric_mismatch" if metric_mismatch else "learned_carrier_direction_failure",
        "optimizer_steps": 0,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
