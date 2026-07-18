"""Audit the quotient-output tangent rank and conditioning on one fixed state."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time as wall_time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.equivariant_denoiser import invariant_vector_rms_precondition
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _make_batch,
    _make_model,
    _predict,
)


def tangent_spectrum_metrics(
    gram: torch.Tensor,
    desired_change: torch.Tensor,
    *,
    relative_threshold: float,
) -> dict[str, Any]:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("tangent Gram matrix must be square")
    if desired_change.shape != (gram.shape[0],):
        raise ValueError("desired output change does not match tangent Gram matrix")
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
    maximum = eigenvalues[-1].clamp_min(torch.finfo(torch.float64).tiny)
    active = eigenvalues > float(relative_threshold) * maximum
    projection = eigenvectors[:, active] @ (eigenvectors[:, active].T @ desired_change.double())
    residual = torch.linalg.vector_norm(desired_change.double() - projection)
    denominator = torch.linalg.vector_norm(desired_change.double()).clamp_min(1e-30)
    positive = eigenvalues[active]
    condition = float(maximum / positive[0]) if positive.numel() else math.inf
    normalized = positive / positive.sum().clamp_min(1e-30)
    entropy_rank = float(torch.exp(-(normalized * normalized.log()).sum())) if positive.numel() else 0.0
    return {
        "output_dimension": int(gram.shape[0]),
        "tangent_rank": int(active.sum()),
        "nullity": int((~active).sum()),
        "maximum_eigenvalue": float(maximum),
        "minimum_active_eigenvalue": float(positive[0]) if positive.numel() else 0.0,
        "condition_number": condition,
        "effective_rank": entropy_rank,
        "target_projection_relative_residual": float(residual / denominator),
        "eigenvalues": eigenvalues.tolist(),
    }


def _parameter_group(name: str) -> str:
    for prefix in (
        "coordinate_vector_head",
        "coordinate_control_gate",
        "coordinate_edge_head",
        "blocks",
        "element_embedding",
        "degree_embedding",
        "time_embedding",
        "state_embedding",
    ):
        if name.startswith(prefix):
            return prefix
    return "other"


def _operator_numeric_checks(
    model: torch.nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
) -> dict[str, float | bool]:
    generator = torch.Generator(device=batch_data.lattice.device).manual_seed(5917)
    vectors = torch.randn(
        (17, 9, 3),
        dtype=batch_data.lattice.dtype,
        device=batch_data.lattice.device,
        generator=generator,
    )
    matrix = torch.randn(
        (3, 3),
        dtype=batch_data.lattice.dtype,
        device=batch_data.lattice.device,
        generator=generator,
    )
    rotation, _ = torch.linalg.qr(matrix)
    rotated = invariant_vector_rms_precondition(vectors @ rotation)
    reference = invariant_vector_rms_precondition(vectors) @ rotation
    o3_error = float((rotated - reference).abs().max())
    original = _predict(model, noisy, batch_data, blueprint, use_bf16=False)
    shifted = dataclasses.replace(
        noisy,
        fractional_coordinates=noisy.fractional_coordinates
        + noisy.fractional_coordinates.new_tensor([0.31, -0.27, 1.19]),
    )
    translated = _predict(model, shifted, batch_data, blueprint, use_bf16=False)
    translation_error = float((translated - original).abs().max())
    zero = torch.zeros_like(vectors, requires_grad=True)
    zero_output = invariant_vector_rms_precondition(zero)
    zero_output.square().sum().backward()
    finite = bool(torch.isfinite(zero_output).all()) and zero.grad is not None
    finite = finite and bool(torch.isfinite(zero.grad).all())
    return {
        "operator_o3_covariance_max_abs": o3_error,
        "full_model_translation_invariance_max_abs": translation_error,
        "zero_stratum_forward_backward_finite": finite,
    }


def _cuda_benchmark(
    protocol: dict[str, Any],
    dataset: PackedAlexP1Dataset,
    standardizer: P1LatticeStandardizer,
    *,
    device: torch.device,
) -> dict[str, float]:
    specification = protocol["cuda"]
    count = int(specification["batch_graphs"])
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(specification["selection_seed"])),
    )[:count]
    batch_data = _make_batch(dataset, indices, device)
    blueprint = _blueprint(batch_data)
    seed = int(protocol["prerequisites"]["model_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _make_model(protocol, device)
    path = protocol["path"]
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(path["coordinate_sigma_min"]),
        coordinate_sigma_max=float(path["coordinate_sigma_max"]),
        minimum_time=float(path["minimum_time"]),
        maximum_time=float(path["maximum_time"]),
    )
    trainer = ProductionTrainer(
        diffusion,
        ProductionTrainingConfig(
            learning_rate=float(specification["learning_rate"]),
            weight_decay=float(specification["weight_decay"]),
            gradient_clip_norm=float(specification["gradient_clip_norm"]),
            ema_decay=float(specification["ema_decay"]),
            coordinate_sigma_min=float(path["coordinate_sigma_min"]),
            coordinate_sigma_max=float(path["coordinate_sigma_max"]),
            minimum_time=float(path["minimum_time"]),
            maximum_time=float(path["maximum_time"]),
            precision=str(specification["precision"]),
            objective="coordinate",
        ),
    )
    generator = torch.Generator(device=device).manual_seed(seed + 11)
    for _ in range(int(specification["warmup_steps"])):
        trainer.train_step(
            batch_data.atom_types,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            blueprint,
            generator=generator,
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    tick = wall_time.perf_counter()
    finite = True
    for _ in range(int(specification["measured_steps"])):
        output, gradient = trainer.train_step(
            batch_data.atom_types,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            blueprint,
            generator=generator,
        )
        finite = finite and math.isfinite(float(output.coordinate_loss))
        finite = finite and math.isfinite(gradient)
    torch.cuda.synchronize(device)
    elapsed = wall_time.perf_counter() - tick
    return {
        "graphs_per_second": count * int(specification["measured_steps"]) / elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "finite": float(finite),
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
    if protocol.get("protocol") != "h1a_coordinate_vector_rms_preconditioner_v1":
        raise ValueError("coordinate tangent protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate tangent cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed = int(protocol["prerequisites"]["model_seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    graph_index = int(protocol["prerequisites"]["fixed_graph_index"])
    batch_data = _make_batch(dataset, torch.tensor([graph_index]), device)
    if int(batch_data.num_nodes) != int(protocol["prerequisites"]["fixed_graph_nodes"]):
        raise ValueError("coordinate tangent fixed graph changed")
    blueprint = _blueprint(batch_data)
    model = _make_model(protocol, device).float()
    standardizer = P1LatticeStandardizer.from_mapping(
        load_json_object(args.lattice_standardization)
    )
    path = protocol["path"]
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(path["coordinate_sigma_min"]),
        coordinate_sigma_max=float(path["coordinate_sigma_max"]),
        minimum_time=float(path["minimum_time"]),
        maximum_time=float(path["maximum_time"]),
    )
    time = batch_data.lattice.new_tensor([float(protocol["prerequisites"]["fixed_time"])])
    noisy = diffusion.noise_clean_batch(
        batch_data.atom_types,
        batch_data.frac_coords,
        batch_data.lattice,
        batch_data.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=time,
        generator=torch.Generator(device=device).manual_seed(
            int(protocol["prerequisites"]["noise_seed"])
        ),
    )
    prediction = _predict(model, noisy, batch_data, blueprint, use_bf16=False)
    target = noisy.coordinate_scaled_score_target
    desired_matrix = (target - prediction).double()
    desired_matrix = desired_matrix - desired_matrix.mean(dim=0, keepdim=True)
    desired = desired_matrix.reshape(-1)
    parameters = [
        (name, value)
        for name, value in model.named_parameters()
        if value.requires_grad
        and not name.startswith(("gauge_atlas.", "geometry_query_encoder."))
    ]
    rows: list[torch.Tensor] = []
    for component in range(prediction.numel()):
        gradients = torch.autograd.grad(
            prediction.reshape(-1)[component],
            [value for _, value in parameters],
            retain_graph=True,
            allow_unused=True,
        )
        rows.append(
            torch.cat(
                [
                    (torch.zeros_like(value) if gradient is None else gradient).reshape(-1)
                    for (_, value), gradient in zip(parameters, gradients, strict=True)
                ]
            ).detach()
        )
    jacobian = torch.stack(rows)
    # The output dimension is tiny, so accumulate the parameter contraction in
    # FP64.  FP32 accumulation produced spurious negative eigenvalues at the
    # same scale as the v1 rank threshold and cannot certify a null space.
    jacobian64 = jacobian.double()
    gram = jacobian64 @ jacobian64.T
    numeric = protocol["numeric"]
    spectrum = tangent_spectrum_metrics(
        gram,
        desired.detach(),
        relative_threshold=float(numeric["rank_relative_eigenvalue_threshold"]),
    )
    loss = (prediction - target).square().mean()
    loss_gradients = torch.autograd.grad(
        loss, [value for _, value in parameters], allow_unused=True
    )
    group_energy: dict[str, float] = {}
    for (name, _), gradient in zip(parameters, loss_gradients, strict=True):
        if gradient is None:
            continue
        group = _parameter_group(name)
        group_energy[group] = group_energy.get(group, 0.0) + float(
            gradient.detach().double().square().sum()
        )
    group_norm = {name: math.sqrt(value) for name, value in group_energy.items()}
    quotient_dimension = 3 * int(batch_data.num_nodes) - 3
    expected_nullity = int(numeric["expected_translation_nullity"])
    rank_full = (
        int(spectrum["tangent_rank"]) == quotient_dimension
        and int(spectrum["nullity"]) == expected_nullity
    )
    target_reachable = (
        float(spectrum["target_projection_relative_residual"])
        <= float(numeric["quotient_projected_target_residual_max"])
    )
    condition_pass = float(spectrum["condition_number"]) <= float(
        numeric["condition_number_max"]
    )
    effective_rank_pass = float(spectrum["effective_rank"]) >= float(
        numeric["effective_rank_min"]
    )
    vector_to_edge = group_norm.get("coordinate_vector_head", 0.0) / max(
        group_norm.get("coordinate_edge_head", 0.0), 1e-30
    )
    gradient_balance_pass = vector_to_edge >= float(
        numeric["vector_to_edge_gradient_norm_ratio_min"]
    )
    numeric_checks = _operator_numeric_checks(
        model, noisy, batch_data, blueprint
    )
    o3_pass = float(numeric_checks["operator_o3_covariance_max_abs"]) <= float(
        numeric["o3_covariance_max_fp32"]
    )
    translation_pass = float(
        numeric_checks["full_model_translation_invariance_max_abs"]
    ) <= float(numeric["translation_invariance_max_fp32"])
    benchmark = _cuda_benchmark(
        protocol, dataset, standardizer, device=device
    )
    cuda_specification = protocol["cuda"]
    benchmark_pass = (
        benchmark["graphs_per_second"]
        >= float(cuda_specification["graphs_per_second_min"])
        and benchmark["peak_allocated_mib"]
        <= float(cuda_specification["peak_allocated_mib_max"])
        and bool(benchmark["finite"])
    )
    checks = {
        "quotient_tangent_full_rank": rank_full,
        "quotient_target_direction_reachable": target_reachable,
        "condition_number": condition_pass,
        "effective_rank": effective_rank_pass,
        "vector_to_edge_gradient_balance": gradient_balance_pass,
        "operator_o3_covariance": o3_pass,
        "full_model_translation_invariance": translation_pass,
        "finite_zero_stratum": bool(
            numeric_checks["zero_stratum_forward_backward_finite"]
        ),
        "parameter_count": sum(value.numel() for value in model.parameters())
        == int(protocol["model"]["parameter_count"]),
        "cuda": benchmark_pass,
        "tensor_candidates": int(numeric["tensor_candidates"]) == 0,
    }
    qualified = all(checks.values())
    decision = (
        "vector_rms_preconditioner_qualified_freeze_one_state_training"
        if qualified
        else "vector_rms_preconditioner_failed_remove_before_training"
    )
    result = {
        "protocol": protocol["protocol"],
        "graph_index": graph_index,
        "nodes": int(batch_data.num_nodes),
        "quotient_dimension": quotient_dimension,
        "initial_coordinate_mse": float(loss),
        "prediction_norm": float(torch.linalg.vector_norm(prediction)),
        "target_norm": float(torch.linalg.vector_norm(target)),
        "prediction_target_cosine": float(
            (prediction * target).sum()
            / (
                torch.linalg.vector_norm(prediction)
                * torch.linalg.vector_norm(target)
            ).clamp_min(1e-30)
        ),
        "tangent": spectrum,
        "numeric_path": {
            "forward_dtype": str(prediction.dtype),
            "jacobian_dtype": str(jacobian.dtype),
            "gram_dtype": str(gram.dtype),
        },
        "module_gradient_norm": group_norm,
        "vector_to_edge_gradient_norm_ratio": vector_to_edge,
        "operator_numeric": numeric_checks,
        "cuda_benchmark": benchmark,
        "checks": checks,
        "qualified": qualified,
        "decision": decision,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
