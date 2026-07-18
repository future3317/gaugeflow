"""Qualify the Cartesian-tangent coordinate objective without training."""

from __future__ import annotations

import argparse
import json
import time as wall_time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.state_projection import (
    cartesian_tangent_to_fractional,
    fractional_tangent_to_cartesian,
    project_translation_state,
)
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _fixed_indices,
    _make_batch,
)


def _gradient_vector(
    loss: torch.Tensor,
    model: torch.nn.Module,
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, bool]:
    gradients = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    values = [
        gradient.detach().float().reshape(-1)
        for gradient in gradients
        if gradient is not None
    ]
    vector = torch.cat(values) if values else loss.new_zeros(1)
    return vector, bool(torch.isfinite(loss) and torch.isfinite(vector).all())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    flat_left = left.float().reshape(-1)
    flat_right = right.float().reshape(-1)
    denominator = torch.linalg.vector_norm(flat_left) * torch.linalg.vector_norm(
        flat_right
    )
    return float(torch.dot(flat_left, flat_right) / denominator.clamp_min(1.0e-30))


def _relative_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.sqrt((left.float() - right.float()).square().mean())
    denominator = torch.sqrt(left.float().square().mean()).clamp_min(1.0e-30)
    return float(numerator / denominator)


def _maximum_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _graph_reversal(batch: torch.Tensor, graph_count: int) -> torch.Tensor:
    return torch.cat(
        [torch.where(batch == graph)[0].flip(0) for graph in range(graph_count)]
    )


def _run_precision(
    model: HybridCrystalDenoiser,
    diffusion: TensorFreeHybridDiffusion,
    batch_data: Any,
    blueprint: Any,
    time: torch.Tensor,
    noise_seed: int,
    *,
    use_bf16: bool,
) -> dict[str, Any]:
    generator = torch.Generator(device=time.device).manual_seed(noise_seed)
    with torch.autocast(
        device_type=time.device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        output = diffusion(
            batch_data.atom_types,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=time,
            generator=generator,
        )
    output_energy = (
        output.prediction.coordinate_cartesian_scaled_score.float().square().mean()
    )
    output_gradient, output_finite = _gradient_vector(
        output_energy, model, retain_graph=True
    )
    loss_gradient, loss_finite = _gradient_vector(
        output.coordinate_loss, model, retain_graph=False
    )
    noisy_lattice = LatticeVolumeShape(
        output.noisy.log_volume.float(), output.noisy.log_shape.float()
    ).lattice(blueprint.fractional_to_cartesian.float())
    reconstructed_prediction = cartesian_tangent_to_fractional(
        output.prediction.coordinate_cartesian_scaled_score.float(),
        noisy_lattice,
        batch_data.batch,
    )
    reconstructed_prediction = project_translation_state(
        reconstructed_prediction, batch_data.batch, int(batch_data.num_graphs)
    )
    target_cartesian = fractional_tangent_to_cartesian(
        output.noisy.coordinate_scaled_score_target.float(),
        noisy_lattice,
        batch_data.batch,
    )
    reconstructed_target = cartesian_tangent_to_fractional(
        target_cartesian, noisy_lattice, batch_data.batch
    )

    graphs = int(batch_data.num_graphs)
    shifts = output.noisy.fractional_coordinates.new_tensor(
        [[0.31, -0.27, 1.19], [-0.43, 0.17, 0.61]]
    )[torch.arange(graphs, device=time.device) % 2]
    condition = output.noisy.log_volume.new_zeros((graphs, 18))
    condition_present = torch.zeros(
        (graphs, 1), dtype=torch.bool, device=time.device
    )
    with torch.no_grad(), torch.autocast(
        device_type=time.device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        translated = model(
            output.noisy.element_tokens,
            output.noisy.fractional_coordinates + shifts[batch_data.batch],
            output.noisy.log_volume,
            output.noisy.log_shape,
            batch_data.batch,
            output.noisy.time,
            condition,
            condition_present,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
        )
        repeated = model(
            output.noisy.element_tokens,
            output.noisy.fractional_coordinates,
            output.noisy.log_volume,
            output.noisy.log_shape,
            batch_data.batch,
            output.noisy.time,
            condition,
            condition_present,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
        )
        permutation = _graph_reversal(batch_data.batch, graphs)
        permuted = model(
            output.noisy.element_tokens[permutation],
            output.noisy.fractional_coordinates[permutation],
            output.noisy.log_volume,
            output.noisy.log_shape,
            batch_data.batch[permutation],
            output.noisy.time,
            condition,
            condition_present,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
        )

    return {
        "_noisy": output.noisy,
        "cartesian_output": output.prediction.coordinate_cartesian_scaled_score.detach().float(),
        "output_gradient": output_gradient,
        "loss_gradient": loss_gradient,
        "output_gradient_norm": float(torch.linalg.vector_norm(output_gradient)),
        "coordinate_loss": float(output.coordinate_loss.detach()),
        "coordinate_loss_gradient_norm": float(torch.linalg.vector_norm(loss_gradient)),
        "finite": output_finite and loss_finite,
        "fractional_prediction_roundtrip_max_abs": _maximum_abs(
            reconstructed_prediction,
            output.prediction.coordinate_fractional_scaled_score,
        ),
        "fractional_target_roundtrip_max_abs": _maximum_abs(
            reconstructed_target, output.noisy.coordinate_scaled_score_target
        ),
        "translation_cartesian_relative_rmse": _relative_rmse(
            output.prediction.coordinate_cartesian_scaled_score,
            translated.coordinate_cartesian_scaled_score,
        ),
        "translation_fractional_relative_rmse": _relative_rmse(
            output.prediction.coordinate_fractional_scaled_score,
            translated.coordinate_fractional_scaled_score,
        ),
        "permutation_cartesian_relative_rmse": _relative_rmse(
            output.prediction.coordinate_cartesian_scaled_score[permutation],
            permuted.coordinate_cartesian_scaled_score,
        ),
        "permutation_fractional_relative_rmse": _relative_rmse(
            output.prediction.coordinate_fractional_scaled_score[permutation],
            permuted.coordinate_fractional_scaled_score,
        ),
        "repeat_cartesian_relative_rmse": _relative_rmse(
            output.prediction.coordinate_cartesian_scaled_score,
            repeated.coordinate_cartesian_scaled_score,
        ),
        "repeat_fractional_relative_rmse": _relative_rmse(
            output.prediction.coordinate_fractional_scaled_score,
            repeated.coordinate_fractional_scaled_score,
        ),
        "atlas_candidates": int(output.prediction.gauge_atlas.raw_candidate_count.sum()),
    }


def _chart_covariance_audit(
    lattice: torch.Tensor,
    fractional_tangent: torch.Tensor,
    batch: torch.Tensor,
) -> dict[str, float]:
    cartesian = fractional_tangent_to_cartesian(
        fractional_tangent, lattice, batch
    )
    recovered = cartesian_tangent_to_fractional(cartesian, lattice, batch)

    basis = lattice.new_tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    inverse_basis = torch.linalg.inv(basis)
    transformed_lattice = basis.unsqueeze(0) @ lattice
    transformed_fractional = fractional_tangent @ inverse_basis
    transformed_cartesian = fractional_tangent_to_cartesian(
        transformed_fractional, transformed_lattice, batch
    )
    transformed_roundtrip = cartesian_tangent_to_fractional(
        transformed_cartesian, transformed_lattice, batch
    )

    orthogonal = lattice.new_tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    rotated_lattice = lattice @ orthogonal
    rotated_cartesian = fractional_tangent_to_cartesian(
        fractional_tangent, rotated_lattice, batch
    )
    rotated_roundtrip = cartesian_tangent_to_fractional(
        rotated_cartesian, rotated_lattice, batch
    )
    return {
        "roundtrip_max_abs": _maximum_abs(recovered, fractional_tangent),
        "gl3_cartesian_invariance_max_abs": _maximum_abs(
            transformed_cartesian, cartesian
        ),
        "gl3_fractional_covariance_max_abs": _maximum_abs(
            transformed_roundtrip, transformed_fractional
        ),
        "o3_cartesian_covariance_max_abs": _maximum_abs(
            rotated_cartesian, cartesian @ orthogonal
        ),
        "o3_fractional_invariance_max_abs": _maximum_abs(
            rotated_roundtrip, fractional_tangent
        ),
    }


def _benchmark_forward(
    model: HybridCrystalDenoiser,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    graphs = int(batch_data.num_graphs)
    condition = noisy.log_volume.new_zeros((graphs, 18))
    present = torch.zeros((graphs, 1), dtype=torch.bool, device=noisy.log_volume.device)

    def forward() -> None:
        with torch.no_grad(), torch.autocast(
            device_type=noisy.log_volume.device.type,
            dtype=torch.bfloat16,
            enabled=True,
        ):
            model(
                noisy.element_tokens,
                noisy.fractional_coordinates,
                noisy.log_volume,
                noisy.log_shape,
                batch_data.batch,
                noisy.time,
                condition,
                present,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
            )

    for _ in range(warmup):
        forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = wall_time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        forward()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = float(start_event.elapsed_time(end_event)) / iterations
    return {
        "milliseconds_per_batch": elapsed_ms,
        "graphs_per_second": graphs * 1000.0 / elapsed_ms,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024.0**2),
        "wall_seconds": wall_time.perf_counter() - wall_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_geometry_precision_boundary_v2":
        raise ValueError("tangent-index correction protocol mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("tangent-index correction cache mismatch")
    standardization = load_json_object(args.lattice_standardization)
    if standardization.get("source_cache_manifest_sha256") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("lattice standardization was not fitted on the cache")
    if int(protocol["audit"]["optimizer_steps"]) != 0:
        raise ValueError("tangent-index correction forbids optimizer steps")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("tangent-index correction requires CUDA")

    model_spec = protocol["model"]
    path = protocol["path"]
    torch.manual_seed(int(path["model_seed"]))
    torch.cuda.manual_seed_all(int(path["model_seed"]))
    model = HybridCrystalDenoiser(
        hidden_dim=int(model_spec["hidden_dim"]),
        vector_dim=int(model_spec["vector_dim"]),
        layers=int(model_spec["layers"]),
        radial_dim=int(model_spec["radial_dim"]),
        radial_cutoff=float(model_spec["radial_cutoff_angstrom"]),
    ).to(device).float()
    initial = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    dataset = PackedAlexP1Dataset(args.cache_root, str(protocol["data"]["split"]))
    indices = _fixed_indices(
        len(dataset),
        int(protocol["data"]["fixed_graphs"]),
        int(protocol["data"]["fixed_selection_seed"]),
    )
    batch_data = _make_batch(dataset, indices, device)
    blueprint = _blueprint(batch_data)
    diffusion = TensorFreeHybridDiffusion(
        model,
        P1LatticeStandardizer.from_mapping(standardization),
        coordinate_sigma_min=float(path["coordinate_sigma_min"]),
        coordinate_sigma_max=float(path["coordinate_sigma_max"]),
        minimum_time=float(path["minimum_time"]),
        maximum_time=float(path["maximum_time"]),
    )
    time_grid = batch_data.lattice.new_tensor(path["time_grid"])
    time = time_grid[
        torch.arange(int(batch_data.num_graphs), device=device) % time_grid.numel()
    ]
    fp32 = _run_precision(
        model,
        diffusion,
        batch_data,
        blueprint,
        time,
        int(path["noise_seed"]),
        use_bf16=False,
    )
    bf16 = _run_precision(
        model,
        diffusion,
        batch_data,
        blueprint,
        time,
        int(path["noise_seed"]),
        use_bf16=True,
    )
    noisy_lattice = LatticeVolumeShape(
        batch_data.lattice.new_zeros(int(batch_data.num_graphs)),
        batch_data.lattice.new_zeros((int(batch_data.num_graphs), 6)),
    ).lattice(blueprint.fractional_to_cartesian)
    tangent_generator = torch.Generator(device=device).manual_seed(
        int(path["chart_seed"])
    )
    tangent = torch.randn(
        batch_data.frac_coords.shape,
        dtype=torch.float32,
        device=device,
        generator=tangent_generator,
    )
    tangent = project_translation_state(
        tangent, batch_data.batch, int(batch_data.num_graphs)
    )
    chart = _chart_covariance_audit(noisy_lattice, tangent, batch_data.batch)
    benchmark = _benchmark_forward(
        model,
        fp32["_noisy"],
        batch_data,
        blueprint,
        warmup=int(protocol["benchmark"]["warmup"]),
        iterations=int(protocol["benchmark"]["iterations"]),
    )

    output_relative_rmse = _relative_rmse(
        fp32["cartesian_output"], bf16["cartesian_output"]
    )
    output_cosine = _cosine(fp32["cartesian_output"], bf16["cartesian_output"])
    output_gradient_cosine = _cosine(
        fp32["output_gradient"], bf16["output_gradient"]
    )
    loss_gradient_cosine = _cosine(
        fp32["loss_gradient"], bf16["loss_gradient"]
    )
    output_gradient_ratio = float(bf16["output_gradient_norm"]) / max(
        float(fp32["output_gradient_norm"]), 1.0e-30
    )
    loss_gradient_ratio = float(bf16["coordinate_loss_gradient_norm"]) / max(
        float(fp32["coordinate_loss_gradient_norm"]), 1.0e-30
    )
    names = tuple(name for name, _ in model.named_parameters())
    forbidden = tuple(protocol["audit"]["forbidden_parameter_fragments"])
    legacy_count = sum(any(fragment in name for fragment in forbidden) for name in names)
    parameters_restored = all(
        torch.equal(value, model.state_dict()[name]) for name, value in initial.items()
    )
    acceptance = protocol["acceptance"]
    symmetry_limit = float(acceptance["model_symmetry_relative_rmse_max"])
    chart_limit = float(acceptance["chart_covariance_max_abs"])
    checks = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters())
        == int(acceptance["parameter_count_exact"]),
        "carrier_channels": model.coordinate_carrier.output_channels
        == int(acceptance["carrier_channels_exact"]),
        "legacy_readouts": legacy_count == int(acceptance["legacy_readout_parameters"]),
        "prediction_roundtrip": max(
            float(fp32["fractional_prediction_roundtrip_max_abs"]),
            float(bf16["fractional_prediction_roundtrip_max_abs"]),
        )
        <= float(acceptance["roundtrip_max_abs"]),
        "target_roundtrip": max(
            float(fp32["fractional_target_roundtrip_max_abs"]),
            float(bf16["fractional_target_roundtrip_max_abs"]),
        )
        <= float(acceptance["roundtrip_max_abs"]),
        "chart_covariance": max(chart.values()) <= chart_limit,
        "translation_consistency": max(
            float(fp32["translation_cartesian_relative_rmse"]),
            float(fp32["translation_fractional_relative_rmse"]),
            float(bf16["translation_cartesian_relative_rmse"]),
            float(bf16["translation_fractional_relative_rmse"]),
        )
        <= symmetry_limit,
        "repeat_determinism": max(
            float(fp32["repeat_cartesian_relative_rmse"]),
            float(fp32["repeat_fractional_relative_rmse"]),
            float(bf16["repeat_cartesian_relative_rmse"]),
            float(bf16["repeat_fractional_relative_rmse"]),
        )
        <= float(acceptance["repeat_relative_rmse_max"]),
        "permutation_consistency": max(
            float(fp32["permutation_cartesian_relative_rmse"]),
            float(fp32["permutation_fractional_relative_rmse"]),
            float(bf16["permutation_cartesian_relative_rmse"]),
            float(bf16["permutation_fractional_relative_rmse"]),
        )
        <= symmetry_limit,
        "fp32_gradients": max(
            float(fp32["output_gradient_norm"]),
            float(fp32["coordinate_loss_gradient_norm"]),
        )
        <= float(acceptance["gradient_norm_max"]),
        "bf16_gradients": max(
            float(bf16["output_gradient_norm"]),
            float(bf16["coordinate_loss_gradient_norm"]),
        )
        <= float(acceptance["gradient_norm_max"]),
        "output_gradient_ratio": float(acceptance["bf16_fp32_gradient_ratio_min"])
        <= output_gradient_ratio
        <= float(acceptance["bf16_fp32_gradient_ratio_max"]),
        "loss_gradient_ratio": float(acceptance["bf16_fp32_gradient_ratio_min"])
        <= loss_gradient_ratio
        <= float(acceptance["bf16_fp32_gradient_ratio_max"]),
        "output_gradient_cosine": output_gradient_cosine
        >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
        "loss_gradient_cosine": loss_gradient_cosine
        >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
        "output_precision": output_cosine
        >= float(acceptance["bf16_fp32_output_cosine_min"])
        and output_relative_rmse
        <= float(acceptance["bf16_fp32_output_relative_rmse_max"]),
        "finite": bool(fp32["finite"] and bf16["finite"]),
        "atlas_bypass": fp32["atlas_candidates"] == bf16["atlas_candidates"]
        == int(acceptance["tensor_free_atlas_candidates"]),
        "parameters_restored": parameters_restored,
        "optimizer_steps": int(protocol["audit"]["optimizer_steps"])
        == int(acceptance["optimizer_steps"]),
        "throughput": benchmark["graphs_per_second"]
        >= float(acceptance["forward_graphs_per_second_min"]),
        "memory": benchmark["peak_cuda_memory_mib"]
        <= float(acceptance["peak_cuda_memory_mib_max"]),
    }
    result = {
        "protocol": protocol["protocol"],
        "fixed_indices": indices.tolist(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "legacy_readout_parameters": legacy_count,
        "fp32": {
            key: value
            for key, value in fp32.items()
            if not key.startswith("_") and not isinstance(value, torch.Tensor)
        },
        "bf16": {
            key: value
            for key, value in bf16.items()
            if not key.startswith("_") and not isinstance(value, torch.Tensor)
        },
        "chart_covariance": chart,
        "benchmark": benchmark,
        "fp32_bf16_cartesian_output_relative_rmse": output_relative_rmse,
        "fp32_bf16_cartesian_output_cosine": output_cosine,
        "bf16_fp32_output_gradient_cosine": output_gradient_cosine,
        "bf16_fp32_loss_gradient_cosine": loss_gradient_cosine,
        "bf16_over_fp32_output_gradient_norm": output_gradient_ratio,
        "bf16_over_fp32_loss_gradient_norm": loss_gradient_ratio,
        "checks": checks,
        "qualified": all(checks.values()),
        "decision": protocol["decision_rule"][
            "pass" if all(checks.values()) else "fail"
        ],
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
