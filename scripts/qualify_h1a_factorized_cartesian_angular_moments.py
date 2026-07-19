"""Zero-training CUDA qualification for the dynamic persistent-edge backbone."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset, collate_packed_alex
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--lattice-standardization",
        type=Path,
        default=Path("configs/statistics/h1a_p1_lattice_standardization.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    return float((left @ right / denominator.clamp_min(1.0e-30)).cpu())


def _gradient_vector(model: HybridCrystalDenoiser) -> torch.Tensor:
    values = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not values:
        raise RuntimeError("coordinate objective produced no gradients")
    return torch.cat(values)


def _precision_probe(
    diffusion: TensorFreeHybridDiffusion,
    clean: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    blueprint: ParentBlueprintBatch,
    times: torch.Tensor,
    *,
    seed: int,
    bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    diffusion.zero_grad(set_to_none=True)
    generator = torch.Generator(device=times.device).manual_seed(seed)
    with torch.autocast(
        device_type=times.device.type,
        dtype=torch.bfloat16,
        enabled=bf16,
    ):
        output = diffusion(
            *clean,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=times,
            generator=generator,
        )
    output.coordinate_loss.backward()
    candidate_count = int(output.prediction.gauge_atlas.effective_frame_count.sum())
    return (
        output.prediction.coordinate_cartesian_scaled_score.detach().float(),
        _gradient_vector(diffusion.denoiser),
        float(output.coordinate_loss.detach().cpu()),
        candidate_count,
    )


@torch.no_grad()
def _output_probe(
    diffusion: TensorFreeHybridDiffusion,
    clean: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    blueprint: ParentBlueprintBatch,
    times: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    output = diffusion(
        *clean,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=times,
        generator=torch.Generator(device=times.device).manual_seed(seed),
    )
    return output.prediction.coordinate_cartesian_scaled_score.detach().float()


def _benchmark(
    diffusion: TensorFreeHybridDiffusion,
    clean: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    blueprint: ParentBlueprintBatch,
    times: torch.Tensor,
    *,
    seed: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    with torch.no_grad():
        noisy = diffusion.noise_clean_batch(
            *clean,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=times,
            generator=torch.Generator(device=times.device).manual_seed(seed),
        )
    condition = torch.zeros((times.numel(), 18), device=times.device)
    present = torch.zeros((times.numel(), 1), dtype=torch.bool, device=times.device)

    def forward() -> None:
        with torch.no_grad(), torch.autocast(
            device_type=times.device.type, dtype=torch.bfloat16
        ):
            diffusion.denoiser(
                noisy.element_tokens,
                noisy.fractional_coordinates,
                noisy.log_volume,
                noisy.log_shape,
                clean[3],
                noisy.time,
                condition,
                present,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
            )

    for _ in range(warmup):
        forward()
    torch.cuda.synchronize(times.device)
    torch.cuda.reset_peak_memory_stats(times.device)
    started = time.perf_counter()
    for _ in range(iterations):
        forward()
    torch.cuda.synchronize(times.device)
    elapsed = time.perf_counter() - started
    graphs_per_second = times.numel() * iterations / elapsed
    peak_mib = float(torch.cuda.max_memory_allocated(times.device)) / 1024.0**2
    return graphs_per_second, peak_mib


def main() -> None:
    args = _arguments()
    protocol = load_json_object(args.protocol)
    if protocol.get("status_before_run") != "frozen_not_run":
        raise ValueError("qualification protocol was not frozen before execution")
    if protocol.get("protocol") != "h1a_dynamic_persistent_edge_v1":
        raise ValueError("unexpected dynamic persistent-edge qualification protocol")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal factorized-angular qualification requires CUDA")
    prerequisites = protocol["prerequisites"]
    if sha256_file(args.cache_root / "manifest.json") != str(
        prerequisites["cache_manifest_sha256"]
    ):
        raise ValueError("qualified cache manifest mismatch")
    causal_root = Path("reports") / str(prerequisites["causal_audit"])
    if sha256_file(causal_root / "span_result.json") != str(
        prerequisites["span_result_sha256"]
    ) or sha256_file(causal_root / "startup_result.json") != str(
        prerequisites["startup_result_sha256"]
    ):
        raise ValueError("dynamic-edge causal evidence hash mismatch")

    data = protocol["data"]
    dataset = PackedAlexP1Dataset(args.cache_root, str(data["split"]))
    graph_count = int(data["fixed_graphs"])
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(data["fixed_selection_seed"]))
    )[:graph_count]
    batch_data = collate_packed_alex([dataset[int(index)] for index in indices]).to(device)
    counts = torch.bincount(batch_data.batch, minlength=graph_count)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=batch_data.frac_coords.dtype, device=device
    )
    clean = (
        batch_data.atom_types,
        batch_data.frac_coords,
        batch_data.lattice,
        batch_data.batch,
    )
    time_values = torch.tensor(data["times"], dtype=torch.float32, device=device)
    times = time_values.repeat((graph_count + time_values.numel() - 1) // time_values.numel())[
        :graph_count
    ]

    torch.manual_seed(int(data["model_seed"]))
    torch.cuda.manual_seed_all(int(data["model_seed"]))
    model_spec = protocol["model"]
    model = HybridCrystalDenoiser(
        hidden_dim=int(model_spec["hidden_dim"]),
        vector_dim=int(model_spec["vector_dim"]),
        layers=int(model_spec["layers"]),
        radial_dim=int(model_spec["radial_dim"]),
        radial_cutoff=float(model_spec["radial_cutoff_angstrom"]),
        edge_dim=int(model_spec["edge_dim"]),
        angular_channels=int(model_spec["angular_channels"]),
        edge_refresh_rank=int(model_spec["edge_refresh_rank"]),
    ).to(device)
    standardizer = P1LatticeStandardizer.from_json(args.lattice_standardization)
    diffusion = TensorFreeHybridDiffusion(model, standardizer)
    parameters_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    mixer = model.coordinate_carrier_mixer
    probe_generator = torch.Generator(device=device).manual_seed(int(data["model_seed"]) + 1)
    probe_carrier = torch.randn(
        (23, mixer.carrier_channels, 3), device=device, generator=probe_generator
    )
    probe_state = torch.randn(
        (23, mixer.state_dim), device=device, generator=probe_generator
    )
    probe_output = mixer(probe_carrier, probe_state)
    probe_reference = torch.einsum("c,ncd->nd", mixer.base_weight, probe_carrier)
    mixer_relative_perturbation = float(
        (
            (probe_output - probe_reference).square().mean().sqrt()
            / probe_reference.square().mean().sqrt().clamp_min(1.0e-30)
        ).cpu()
    )

    residual_names = tuple(
        name
        for name in parameters_before
        if name.endswith(
            (
                "angular_scalar_residual.2.weight",
                "angular_vector_residual.2.weight",
                "coordinate_edge_residual.2.weight",
                "coordinate_carrier_mixer.carrier_projection.weight",
            )
        )
    )
    active_output = _output_probe(
        diffusion, clean, blueprint, times, seed=int(data["noise_seed"])
    )
    with torch.no_grad():
        state = model.state_dict()
        for name in residual_names:
            state[name].zero_()
    zero_residual_output = _output_probe(
        diffusion, clean, blueprint, times, seed=int(data["noise_seed"])
    )
    model.load_state_dict(parameters_before, strict=True)
    output_relative_perturbation = float(
        (
            (active_output - zero_residual_output).square().mean().sqrt()
            / zero_residual_output.square().mean().sqrt().clamp_min(1.0e-30)
        ).cpu()
    )

    fp32_output, fp32_gradient, fp32_loss, fp32_candidates = _precision_probe(
        diffusion,
        clean,
        blueprint,
        times,
        seed=int(data["noise_seed"]),
        bf16=False,
    )
    requested_gradient_fragments = (
        "angular_moments.coefficient_projection",
        "edge_source_refresh",
        "edge_target_refresh",
        "edge_context_refresh",
        "edge_vector_refresh",
        "edge_update",
        "angular_scalar_residual.0",
        "angular_vector_residual.0",
        "coordinate_edge_residual.0",
        "coordinate_carrier_mixer.state_projection",
    )
    internal_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if any(fragment in name for fragment in requested_gradient_fragments)
    ]
    if not internal_gradients or any(gradient is None for gradient in internal_gradients):
        raise RuntimeError("dynamic edge internals do not all receive first-step gradients")
    internal_gradient_norm = math.sqrt(
        sum(
            float(gradient.detach().float().square().sum().cpu())
            for gradient in internal_gradients
            if gradient is not None
        )
    )
    bf16_output, bf16_gradient, bf16_loss, bf16_candidates = _precision_probe(
        diffusion,
        clean,
        blueprint,
        times,
        seed=int(data["noise_seed"]),
        bf16=True,
    )
    output_delta = bf16_output - fp32_output
    output_relative_rmse = float(
        (output_delta.square().mean().sqrt() / fp32_output.square().mean().sqrt().clamp_min(1e-30)).cpu()
    )
    gradient_ratio = float((bf16_gradient.norm() / fp32_gradient.norm()).cpu())
    benchmark = protocol["benchmark"]
    graphs_per_second, peak_mib = _benchmark(
        diffusion,
        clean,
        blueprint,
        times,
        seed=int(data["noise_seed"]),
        warmup=int(benchmark["warmup"]),
        iterations=int(benchmark["iterations"]),
    )
    names = tuple(name for name, _ in model.named_parameters())
    parameter_count = sum(value.numel() for value in model.parameters())
    initial_nonzero = int(torch.count_nonzero(mixer.carrier_projection.weight))
    parameters_unchanged = all(
        torch.equal(value, model.state_dict()[name]) for name, value in parameters_before.items()
    )
    finite = all(
        torch.isfinite(value).all()
        for value in (fp32_output, bf16_output, fp32_gradient, bf16_gradient)
    ) and all(torch.isfinite(torch.tensor(value)) for value in (fp32_loss, bf16_loss))

    metrics: dict[str, Any] = {
        "mixer_relative_initial_perturbation": mixer_relative_perturbation,
        "coordinate_relative_initial_perturbation": output_relative_perturbation,
        "parameter_count": parameter_count,
        "added_parameter_count": parameter_count - 4_481_337,
        "adaptive_rank": mixer.rank,
        "edge_dim": model.edge_dim,
        "angular_channels": model.angular_channels,
        "edge_refresh_rank": model.edge_refresh_rank,
        "coordinate_chart": model.coordinate_chart,
        "explicit_triplet_indices": sum(
            1 for name, _ in model.named_buffers() if "triplet" in name
        ),
        "initial_angular_residual_nonzero": sum(
            int(torch.count_nonzero(value))
            for name, value in parameters_before.items()
            if name.endswith(
                (
                    "angular_scalar_residual.2.weight",
                    "angular_vector_residual.2.weight",
                    "coordinate_edge_residual.2.weight",
                )
            )
        ),
        "first_step_internal_gradient_norm": internal_gradient_norm,
        "legacy_global_head_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "coordinate_carrier_head" in name
        ),
        "carrier_projection_initial_nonzero": initial_nonzero,
        "finite_coordinate_output_and_gradient": finite,
        "fp32_coordinate_loss": fp32_loss,
        "bf16_coordinate_loss": bf16_loss,
        "bf16_fp32_output_relative_rmse": output_relative_rmse,
        "bf16_fp32_output_cosine": _cosine(fp32_output.reshape(-1), bf16_output.reshape(-1)),
        "bf16_fp32_gradient_ratio": gradient_ratio,
        "bf16_fp32_gradient_cosine": _cosine(fp32_gradient, bf16_gradient),
        "forward_graphs_per_second": graphs_per_second,
        "peak_cuda_memory_mib": peak_mib,
        "tensor_free_atlas_candidates": max(
            fp32_candidates, bf16_candidates
        ),
        "optimizer_steps": 0,
        "parameters_unchanged": parameters_unchanged,
        "forbidden_parameter_names": [
            name
            for name in names
            if any(fragment in name for fragment in ("coordinate_vector_head", "coordinate_edge_head"))
        ],
    }
    acceptance = protocol["acceptance"]
    checks = {
        "mixer_initial_perturbation": metrics["mixer_relative_initial_perturbation"]
        <= float(acceptance["mixer_relative_initial_perturbation_max"]),
        "coordinate_initial_perturbation": metrics[
            "coordinate_relative_initial_perturbation"
        ]
        <= float(acceptance["coordinate_relative_initial_perturbation_max"]),
        "parameter_count": metrics["parameter_count"] == int(acceptance["parameter_count_exact"]),
        "added_parameter_count": metrics["added_parameter_count"]
        == int(acceptance["added_parameter_count_exact"]),
        "adaptive_rank": metrics["adaptive_rank"] == int(acceptance["adaptive_rank_exact"]),
        "edge_dim": metrics["edge_dim"] == int(acceptance["edge_dim_exact"]),
        "angular_channels": metrics["angular_channels"]
        == int(acceptance["angular_channels_exact"]),
        "edge_refresh_rank": metrics["edge_refresh_rank"]
        == int(acceptance["edge_refresh_rank_exact"]),
        "coordinate_chart": metrics["coordinate_chart"]
        == str(acceptance["coordinate_chart_exact"]),
        "no_explicit_triplets": metrics["explicit_triplet_indices"]
        == int(acceptance["explicit_triplet_indices"]),
        "nonzero_residual_initialization": metrics[
            "initial_angular_residual_nonzero"
        ]
        >= int(acceptance["initial_angular_residual_nonzero_min"]),
        "first_step_internal_gradient": metrics["first_step_internal_gradient_norm"]
        >= float(acceptance["first_step_internal_gradient_norm_min"]),
        "no_legacy_head": metrics["legacy_global_head_parameters"]
        == int(acceptance["legacy_global_head_parameters"]),
        "adaptive_nonzero_initialization": metrics[
            "carrier_projection_initial_nonzero"
        ]
        >= int(acceptance["carrier_projection_initial_nonzero_min"]),
        "finite": bool(metrics["finite_coordinate_output_and_gradient"]),
        "output_relative_rmse": metrics["bf16_fp32_output_relative_rmse"]
        <= float(acceptance["bf16_fp32_output_relative_rmse_max"]),
        "output_cosine": metrics["bf16_fp32_output_cosine"]
        >= float(acceptance["bf16_fp32_output_cosine_min"]),
        "gradient_ratio": float(acceptance["bf16_fp32_gradient_ratio_min"])
        <= metrics["bf16_fp32_gradient_ratio"]
        <= float(acceptance["bf16_fp32_gradient_ratio_max"]),
        "gradient_cosine": metrics["bf16_fp32_gradient_cosine"]
        >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
        "throughput": metrics["forward_graphs_per_second"]
        >= float(acceptance["forward_graphs_per_second_min"]),
        "memory": metrics["peak_cuda_memory_mib"]
        <= float(acceptance["peak_cuda_memory_mib_max"]),
        "atlas_bypass": metrics["tensor_free_atlas_candidates"]
        == int(acceptance["tensor_free_atlas_candidates"]),
        "optimizer_steps": metrics["optimizer_steps"] == int(acceptance["optimizer_steps"]),
        "parameters_unchanged": bool(metrics["parameters_unchanged"]),
        "forbidden_names": not metrics["forbidden_parameter_names"],
    }
    result = {
        "protocol": protocol["protocol"],
        "qualified": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
