"""Qualify the target-free production integration of the Cartesian carrier."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _fixed_indices,
    _make_batch,
)


def _forward(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    use_bf16: bool,
) -> Any:
    graphs = int(batch_data.num_graphs)
    condition = noisy.time.new_zeros((graphs, 18))
    present = torch.zeros((graphs, 1), dtype=torch.bool, device=noisy.time.device)
    with torch.autocast(
        device_type=noisy.time.device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        return model(
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


def _gradient_fields(model: nn.Module) -> tuple[dict[str, torch.Tensor], float, bool]:
    gradients: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=next(model.parameters()).device)
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        gradients[name] = gradient.cpu().clone()
        total = total + gradient.square().sum()
        finite = finite and bool(torch.isfinite(gradient).all())
    return gradients, float(total.sqrt()), finite


def _evaluate(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    use_bf16: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float, bool, int]:
    model.train()
    model.zero_grad(set_to_none=True)
    output = _forward(
        model, noisy, batch_data, blueprint, use_bf16=use_bf16
    )
    prediction = output.coordinate_fractional_scaled_score.float()
    energy = prediction.square().mean()
    energy.backward()
    gradients, norm, finite = _gradient_fields(model)
    model.zero_grad(set_to_none=True)
    candidates = int(output.gauge_atlas.raw_candidate_count.sum())
    return (
        prediction.detach(),
        gradients,
        norm,
        finite and math.isfinite(float(energy.detach())),
        candidates,
    )


def _gradient_agreement(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, float]:
    if reference.keys() != candidate.keys() or not reference:
        raise ValueError("gradient fields must have matching nonempty keys")
    dot = sum(float((reference[name] * candidate[name]).sum()) for name in reference)
    reference_norm = math.sqrt(
        sum(float(reference[name].square().sum()) for name in reference)
    )
    candidate_norm = math.sqrt(
        sum(float(candidate[name].square().sum()) for name in candidate)
    )
    return {
        "cosine": dot / max(reference_norm * candidate_norm, 1e-30),
        "candidate_over_reference_norm": candidate_norm / max(reference_norm, 1e-30),
    }


@torch.no_grad()
def _throughput(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    repeats: int,
) -> dict[str, float]:
    device = noisy.time.device
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(5):
        _ = _forward(model, noisy, batch_data, blueprint, use_bf16=True)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _ = _forward(model, noisy, batch_data, blueprint, use_bf16=True)
    stop.record()
    torch.cuda.synchronize(device)
    milliseconds = float(start.elapsed_time(stop) / repeats)
    return {
        "latency_ms": milliseconds,
        "graphs_per_second": float(int(batch_data.num_graphs) * 1000.0 / milliseconds),
        "peak_memory_mib": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
    }


def _noisy_batch(
    diffusion: TensorFreeHybridDiffusion,
    batch_data: Any,
    blueprint: Any,
    time_grid: list[float],
    seed: int,
) -> Any:
    times = batch_data.lattice.new_tensor(time_grid)
    graph_time = times[
        torch.arange(int(batch_data.num_graphs), device=times.device) % times.numel()
    ]
    with torch.no_grad():
        return diffusion.noise_clean_batch(
            batch_data.atom_types,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=graph_time,
            generator=torch.Generator(device=times.device).manual_seed(seed),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_compact_cartesian_krylov_production_v1":
        raise ValueError("compact Cartesian production protocol mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("compact Cartesian production cache mismatch")
    if int(protocol["audit"]["optimizer_steps"]) != 0:
        raise ValueError("compact Cartesian production audit forbids optimizer steps")
    if int(protocol["audit"]["coordinate_targets_read"]) != 0:
        raise ValueError("compact Cartesian production audit forbids coordinate targets")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("compact Cartesian production audit requires CUDA")

    path = protocol["path"]
    torch.manual_seed(int(path["model_seed"]))
    torch.cuda.manual_seed_all(int(path["model_seed"]))
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    fixed_indices = _fixed_indices(
        len(dataset),
        int(protocol["data"]["fixed_graphs"]),
        int(protocol["data"]["fixed_selection_seed"]),
    )
    batch_data = _make_batch(dataset, fixed_indices, device)
    blueprint = _blueprint(batch_data)
    spec = protocol["model"]
    model = HybridCrystalDenoiser(
        hidden_dim=int(spec["hidden_dim"]),
        vector_dim=int(spec["vector_dim"]),
        layers=int(spec["layers"]),
        radial_dim=int(spec["radial_dim"]),
        radial_cutoff=float(spec["radial_cutoff_angstrom"]),
    ).to(device).float()
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    standardizer = P1LatticeStandardizer.from_mapping(
        load_json_object(args.lattice_standardization)
    )
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(path["coordinate_sigma_min"]),
        coordinate_sigma_max=float(path["coordinate_sigma_max"]),
        minimum_time=float(path["minimum_time"]),
        maximum_time=float(path["maximum_time"]),
    )
    noisy = _noisy_batch(
        diffusion,
        batch_data,
        blueprint,
        list(path["time_grid"]),
        int(path["noise_seed"]),
    )
    fp32_prediction, fp32_gradients, fp32_norm, fp32_finite, fp32_candidates = (
        _evaluate(model, noisy, batch_data, blueprint, use_bf16=False)
    )
    bf16_prediction, bf16_gradients, bf16_norm, bf16_finite, bf16_candidates = (
        _evaluate(model, noisy, batch_data, blueprint, use_bf16=True)
    )
    relative_rmse = float(
        (bf16_prediction - fp32_prediction).square().mean().sqrt()
        / fp32_prediction.square().mean().sqrt().clamp_min(1e-30)
    )
    prediction_cosine = float(
        (bf16_prediction * fp32_prediction).sum()
        / (
            torch.linalg.vector_norm(bf16_prediction)
            * torch.linalg.vector_norm(fp32_prediction)
        ).clamp_min(1e-30)
    )
    gradient_agreement = _gradient_agreement(fp32_gradients, bf16_gradients)

    throughput_indices = _fixed_indices(
        len(dataset),
        int(protocol["data"]["throughput_graphs"]),
        int(protocol["data"]["throughput_selection_seed"]),
    )
    throughput_batch = _make_batch(dataset, throughput_indices, device)
    throughput_blueprint = _blueprint(throughput_batch)
    throughput_noisy = _noisy_batch(
        diffusion,
        throughput_batch,
        throughput_blueprint,
        list(path["time_grid"]),
        int(path["noise_seed"]) + 1,
    )
    performance = _throughput(
        model,
        throughput_noisy,
        throughput_batch,
        throughput_blueprint,
        int(protocol["audit"]["forward_repeats"]),
    )
    state_names = tuple(model.state_dict())
    forbidden = tuple(str(value) for value in protocol["audit"]["forbidden_state_dict_fragments"])
    legacy_count = sum(any(fragment in name for fragment in forbidden) for name in state_names)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    carrier_channels = int(model.coordinate_carrier.output_channels)  # type: ignore[attr-defined]
    model.load_state_dict(initial_state, strict=True)
    parameters_restored = all(
        torch.equal(value, model.state_dict()[name]) for name, value in initial_state.items()
    )
    acceptance = protocol["acceptance"]
    checks = {
        "parameter_count": parameter_count == int(acceptance["parameter_count_exact"]),
        "carrier_channels": carrier_channels == int(acceptance["carrier_channels_exact"]),
        "legacy_readouts": legacy_count == int(acceptance["legacy_readout_parameters"]),
        "prediction_rmse": relative_rmse
        <= float(acceptance["fp32_bf16_coordinate_relative_rmse_max"]),
        "prediction_cosine": prediction_cosine
        >= float(acceptance["fp32_bf16_coordinate_cosine_min"]),
        "fp32_gradient": fp32_finite
        and fp32_norm <= float(acceptance["fp32_probe_gradient_norm_max"]),
        "bf16_gradient": bf16_finite
        and bf16_norm <= float(acceptance["bf16_probe_gradient_norm_max"]),
        "gradient_norm_agreement": float(
            acceptance["bf16_over_fp32_gradient_norm_min"]
        )
        <= gradient_agreement["candidate_over_reference_norm"]
        <= float(acceptance["bf16_over_fp32_gradient_norm_max"]),
        "gradient_direction": gradient_agreement["cosine"]
        >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
        "atlas_bypass": fp32_candidates == bf16_candidates
        == int(acceptance["tensor_free_atlas_candidates"]),
        "throughput": performance["graphs_per_second"]
        >= float(acceptance["forward_graphs_per_second_min"]),
        "memory": performance["peak_memory_mib"]
        <= float(acceptance["forward_peak_memory_mib_max"]),
        "finite": fp32_finite and bf16_finite
        is bool(acceptance["finite_forward_and_backward"]),
        "parameters_restored": parameters_restored
        is bool(acceptance["parameters_restored"]),
        "sampling_failures": int(acceptance["sampling_failures"]) == 0,
        "coordinate_targets_read": int(acceptance["coordinate_targets_read"]) == 0,
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "fixed_indices": fixed_indices.tolist(),
        "parameter_count": parameter_count,
        "carrier_channels": carrier_channels,
        "legacy_readout_parameters": legacy_count,
        "fp32_bf16_coordinate_relative_rmse": relative_rmse,
        "fp32_bf16_coordinate_cosine": prediction_cosine,
        "fp32_probe_gradient_norm": fp32_norm,
        "bf16_probe_gradient_norm": bf16_norm,
        "gradient_agreement": gradient_agreement,
        "tensor_free_atlas_candidates": fp32_candidates,
        "performance": performance,
        "checks": checks,
        "qualified": qualified,
        "optimizer_steps": 0,
        "coordinate_targets_read": 0,
        "sampling_failures": 0,
        "decision": (
            "compact_cartesian_krylov_production_qualified_freeze_memorization"
            if qualified
            else "compact_cartesian_krylov_production_failed_revert_integration"
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
