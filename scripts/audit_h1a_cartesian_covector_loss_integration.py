"""Qualify the Cartesian-covector loss and compact production carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.state_projection import fractional_covector_to_cartesian
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
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    values = [
        gradient.detach().float().reshape(-1)
        for gradient in gradients
        if gradient is not None
    ]
    vector = torch.cat(values) if values else loss.new_zeros(1)
    return vector, bool(torch.isfinite(loss) and torch.isfinite(vector).all())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator.clamp_min(1.0e-30))


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
    output_energy = output.prediction.coordinate_cartesian_scaled_score.float().square().mean()
    output_gradient, output_finite = _gradient_vector(
        output_energy, model, retain_graph=True
    )
    loss_gradient, loss_finite = _gradient_vector(
        output.coordinate_loss, model, retain_graph=False
    )
    noisy_lattice = LatticeVolumeShape(
        output.noisy.log_volume.float(), output.noisy.log_shape.float()
    ).lattice(blueprint.fractional_to_cartesian.float())
    reconstructed_prediction = torch.einsum(
        "ni,nij->nj",
        output.prediction.coordinate_cartesian_scaled_score.float(),
        noisy_lattice[batch_data.batch].transpose(-1, -2),
    )
    target_cartesian = fractional_covector_to_cartesian(
        output.noisy.coordinate_scaled_score_target.float(),
        noisy_lattice,
        batch_data.batch,
    )
    reconstructed_target = torch.einsum(
        "ni,nij->nj",
        target_cartesian,
        noisy_lattice[batch_data.batch].transpose(-1, -2),
    )
    return {
        "cartesian_output": output.prediction.coordinate_cartesian_scaled_score.detach().float(),
        "output_gradient": output_gradient,
        "output_gradient_norm": float(torch.linalg.vector_norm(output_gradient)),
        "coordinate_loss": float(output.coordinate_loss.detach()),
        "coordinate_loss_gradient_norm": float(torch.linalg.vector_norm(loss_gradient)),
        "finite": output_finite and loss_finite,
        "fractional_prediction_reconstruction_max_abs": float(
            (
                reconstructed_prediction
                - output.prediction.coordinate_fractional_scaled_score.float()
            )
            .abs()
            .max()
        ),
        "fractional_target_roundtrip_max_abs": float(
            (
                reconstructed_target
                - output.noisy.coordinate_scaled_score_target.float()
            )
            .abs()
            .max()
        ),
        "atlas_candidates": int(output.prediction.gauge_atlas.raw_candidate_count.sum()),
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
    if protocol.get("protocol") != "h1a_cartesian_covector_loss_integration_v1":
        raise ValueError("Cartesian-covector integration protocol mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("Cartesian-covector integration cache mismatch")
    if int(protocol["audit"]["optimizer_steps"]) != 0:
        raise ValueError("Cartesian-covector integration forbids optimizer steps")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Cartesian-covector integration requires CUDA")
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
        P1LatticeStandardizer.from_mapping(
            load_json_object(args.lattice_standardization)
        ),
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
    relative_rmse = float(
        torch.sqrt((fp32["cartesian_output"] - bf16["cartesian_output"]).square().mean())
        / torch.sqrt(fp32["cartesian_output"].square().mean()).clamp_min(1.0e-30)
    )
    prediction_cosine = _cosine(
        fp32["cartesian_output"].reshape(-1),
        bf16["cartesian_output"].reshape(-1),
    )
    gradient_cosine = _cosine(fp32["output_gradient"], bf16["output_gradient"])
    gradient_ratio = float(bf16["output_gradient_norm"]) / max(
        float(fp32["output_gradient_norm"]), 1.0e-30
    )
    names = tuple(name for name, _ in model.named_parameters())
    forbidden = tuple(protocol["audit"]["forbidden_parameter_fragments"])
    legacy_count = sum(any(fragment in name for fragment in forbidden) for name in names)
    parameters_restored = all(
        torch.equal(value, model.state_dict()[name]) for name, value in initial.items()
    )
    acceptance = protocol["acceptance"]
    checks = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters())
        == int(acceptance["parameter_count_exact"]),
        "carrier_channels": model.coordinate_carrier.output_channels
        == int(acceptance["carrier_channels_exact"]),
        "legacy_readouts": legacy_count == int(acceptance["legacy_readout_parameters"]),
        "prediction_roundtrip": max(
            fp32["fractional_prediction_reconstruction_max_abs"],
            bf16["fractional_prediction_reconstruction_max_abs"],
        )
        <= float(acceptance["fractional_prediction_reconstruction_max_abs"]),
        "target_roundtrip": max(
            fp32["fractional_target_roundtrip_max_abs"],
            bf16["fractional_target_roundtrip_max_abs"],
        )
        <= float(acceptance["fractional_target_roundtrip_max_abs"]),
        "fp32_gradient": fp32["output_gradient_norm"]
        <= float(acceptance["fp32_cartesian_output_gradient_norm_max"]),
        "bf16_gradient": bf16["output_gradient_norm"]
        <= float(acceptance["bf16_cartesian_output_gradient_norm_max"]),
        "gradient_ratio": float(acceptance["bf16_over_fp32_gradient_norm_min"])
        <= gradient_ratio
        <= float(acceptance["bf16_over_fp32_gradient_norm_max"]),
        "gradient_cosine": gradient_cosine
        >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
        "prediction_cosine": prediction_cosine
        >= float(acceptance["fp32_bf16_cartesian_output_cosine_min"]),
        "prediction_rmse": relative_rmse
        <= float(acceptance["fp32_bf16_cartesian_output_relative_rmse_max"]),
        "finite": bool(fp32["finite"] and bf16["finite"]),
        "atlas_bypass": fp32["atlas_candidates"] == bf16["atlas_candidates"]
        == int(acceptance["tensor_free_atlas_candidates"]),
        "parameters_restored": parameters_restored,
        "optimizer_steps": int(protocol["audit"]["optimizer_steps"])
        == int(acceptance["optimizer_steps"]),
    }
    result = {
        "protocol": protocol["protocol"],
        "fixed_indices": indices.tolist(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "legacy_readout_parameters": legacy_count,
        "fp32": {key: value for key, value in fp32.items() if not isinstance(value, torch.Tensor)},
        "bf16": {key: value for key, value in bf16.items() if not isinstance(value, torch.Tensor)},
        "fp32_bf16_cartesian_output_relative_rmse": relative_rmse,
        "fp32_bf16_cartesian_output_cosine": prediction_cosine,
        "bf16_fp32_gradient_cosine": gradient_cosine,
        "bf16_over_fp32_gradient_norm": gradient_ratio,
        "checks": checks,
        "qualified": all(checks.values()),
        "decision": protocol["decision_rule"]["pass"] if all(checks.values()) else protocol["decision_rule"]["fail"],
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
