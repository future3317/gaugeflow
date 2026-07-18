"""Audit the exact affine coordinate-readout span on one fixed H1a state."""

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
from gaugeflow.production.equivariant_denoiser import (
    invariant_graphwise_basis_unit_scale,
)
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _make_batch,
    _make_model,
    _predict,
)


def project_common_translation(value: torch.Tensor) -> torch.Tensor:
    """Project a one-graph coordinate field to the translation quotient."""
    if value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError("coordinate field must have shape [nodes, 3]")
    return value - value.mean(dim=0, keepdim=True)


def helmert_quotient_basis(
    nodes: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return an orthonormal basis of R^(3N) modulo common translation."""
    if nodes < 2:
        raise ValueError("a coordinate quotient basis needs at least two nodes")
    helmert = torch.zeros((nodes, nodes - 1), dtype=dtype, device=device)
    for column in range(nodes - 1):
        count = column + 1
        scale = (float(count * (count + 1))) ** -0.5
        helmert[:count, column] = scale
        helmert[count, column] = -float(count) * scale
    return torch.kron(
        helmert, torch.eye(3, dtype=dtype, device=device)
    )


def affine_readout_solution(
    jacobian: torch.Tensor,
    desired_change: torch.Tensor,
    *,
    relative_threshold: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the minimum-norm active-subspace solution and span metrics."""
    if jacobian.ndim != 2:
        raise ValueError("readout Jacobian must be a matrix")
    if desired_change.shape != (jacobian.shape[0],):
        raise ValueError("desired output change does not match the Jacobian")
    if relative_threshold <= 0.0:
        raise ValueError("relative rank threshold must be positive")
    jacobian64 = jacobian.double()
    gram = jacobian64 @ jacobian64.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues[-1].clamp_min(torch.finfo(torch.float64).tiny)
    active = eigenvalues > relative_threshold * maximum
    active_vectors = eigenvectors[:, active]
    active_values = eigenvalues[active]
    desired64 = desired_change.double()
    projection = active_vectors @ (active_vectors.T @ desired64)
    coefficients = active_vectors @ (
        (active_vectors.T @ desired64) / active_values
    )
    delta = jacobian64.T @ coefficients
    residual = torch.linalg.vector_norm(desired64 - projection) / torch.linalg.vector_norm(
        desired64
    ).clamp_min(1e-30)
    normalized = active_values / active_values.sum().clamp_min(1e-30)
    effective_rank = (
        float(torch.exp(-(normalized * normalized.log()).sum()))
        if active_values.numel()
        else 0.0
    )
    return delta, {
        "output_dimension": int(jacobian.shape[0]),
        "parameter_dimension": int(jacobian.shape[1]),
        "rank": int(active.sum()),
        "nullity": int((~active).sum()),
        "maximum_eigenvalue": float(maximum),
        "minimum_active_eigenvalue": (
            float(active_values[0]) if active_values.numel() else 0.0
        ),
        "condition_number": (
            float(maximum / active_values[0]) if active_values.numel() else None
        ),
        "effective_rank": effective_rank,
        "target_projection_relative_residual": float(residual),
        "eigenvalues": eigenvalues.tolist(),
    }


def _readout_parameters(
    model: torch.nn.Module, expected_names: list[str]
) -> list[tuple[str, torch.nn.Parameter]]:
    available = dict(model.named_parameters())
    missing = [name for name in expected_names if name not in available]
    if missing:
        raise ValueError(f"coordinate readout parameters are missing: {missing}")
    return [(name, available[name]) for name in expected_names]


def _jacobian(
    output: torch.Tensor, parameters: list[tuple[str, torch.nn.Parameter]]
) -> torch.Tensor:
    values = [value for _, value in parameters]
    rows: list[torch.Tensor] = []
    for component in range(output.numel()):
        gradients = torch.autograd.grad(
            output.reshape(-1)[component], values, retain_graph=True
        )
        rows.append(torch.cat([gradient.reshape(-1) for gradient in gradients]).detach())
    return torch.stack(rows)


def _column_slices(
    parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0
    for name, value in parameters:
        result[name] = slice(offset, offset + value.numel())
        offset += value.numel()
    return result


def _operator_numeric_checks(
    model: torch.nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
) -> dict[str, float | bool]:
    generator = torch.Generator(device=batch_data.lattice.device).manual_seed(5917)
    basis = torch.randn(
        (17, 9, 3),
        dtype=batch_data.lattice.dtype,
        device=batch_data.lattice.device,
        generator=generator,
    )
    batch = torch.tensor(
        [0] * 8 + [1] * 9, dtype=torch.long, device=batch_data.lattice.device
    )
    matrix = torch.randn(
        (3, 3),
        dtype=basis.dtype,
        device=basis.device,
        generator=generator,
    )
    rotation, _ = torch.linalg.qr(matrix)
    reference = invariant_graphwise_basis_unit_scale(basis, batch, 2)
    rotated = invariant_graphwise_basis_unit_scale(basis @ rotation, batch, 2)
    o3_error = float((rotated - reference @ rotation).abs().max())
    permutation = torch.randperm(
        basis.shape[0],
        device=basis.device,
        generator=torch.Generator(device=basis.device).manual_seed(5918),
    )
    permuted = invariant_graphwise_basis_unit_scale(
        basis[permutation], batch[permutation], 2
    )
    permutation_error = float((permuted - reference[permutation]).abs().max())
    original = _predict(model, noisy, batch_data, blueprint, use_bf16=False)
    shifted = dataclasses.replace(
        noisy,
        fractional_coordinates=noisy.fractional_coordinates
        + noisy.fractional_coordinates.new_tensor([0.31, -0.27, 1.19]),
    )
    translated = _predict(model, shifted, batch_data, blueprint, use_bf16=False)
    translation_error = float((translated - original).abs().max())
    zero = torch.zeros_like(basis, requires_grad=True)
    zero_output = invariant_graphwise_basis_unit_scale(zero, batch, 2)
    zero_output.square().sum().backward()
    finite = zero.grad is not None and bool(torch.isfinite(zero.grad).all())
    return {
        "operator_o3_covariance_max_abs": o3_error,
        "operator_permutation_max_abs": permutation_error,
        "full_model_translation_invariance_max_abs": translation_error,
        "zero_stratum_forward_backward_finite": finite,
    }


def _cuda_benchmark(
    protocol: dict[str, Any],
    dataset: PackedAlexP1Dataset,
    standardizer: P1LatticeStandardizer,
    *,
    device: torch.device,
) -> dict[str, float | bool]:
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
        "finite": finite,
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
    if protocol.get("protocol") != "h1a_coordinate_unit_scaled_readout_v1":
        raise ValueError("coordinate unit-scaled-readout protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate affine-readout cache mismatch")
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
        raise ValueError("coordinate affine-readout fixed graph changed")
    blueprint = _blueprint(batch_data)
    model = _make_model(protocol, device).float().eval()
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
    time = batch_data.lattice.new_tensor(
        [float(protocol["prerequisites"]["fixed_time"])]
    )
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

    def predict() -> torch.Tensor:
        return _predict(model, noisy, batch_data, blueprint, use_bf16=False)

    prediction = predict()
    target = noisy.coordinate_scaled_score_target
    quotient_basis = helmert_quotient_basis(
        int(batch_data.num_nodes), dtype=prediction.dtype, device=prediction.device
    )
    initial_quotient = quotient_basis.T @ prediction.reshape(-1)
    target_quotient = quotient_basis.T @ target.reshape(-1)
    desired = target_quotient - initial_quotient
    readout = protocol["readout"]
    parameters = _readout_parameters(model, list(readout["parameter_names"]))
    parameter_count = sum(value.numel() for _, value in parameters)
    if parameter_count != int(readout["parameter_count"]):
        raise ValueError("coordinate affine-readout parameter count mismatch")
    raw_jacobian = _jacobian(prediction, parameters)
    jacobian = quotient_basis.T @ raw_jacobian
    threshold = float(readout["rank_relative_singular_value_threshold"])
    delta, combined = affine_readout_solution(
        jacobian, desired.detach(), relative_threshold=threshold
    )
    slices = _column_slices(parameters)
    groups = {
        "vector": ["coordinate_vector_head.weight"],
        "edge": [
            "coordinate_edge_head.2.weight",
            "coordinate_edge_head.2.bias",
        ],
    }
    group_metrics: dict[str, dict[str, Any]] = {}
    for group, names in groups.items():
        columns = torch.cat(
            [
                torch.arange(
                    slices[name].start,
                    slices[name].stop,
                    device=jacobian.device,
                )
                for name in names
            ]
        )
        _, group_metrics[group] = affine_readout_solution(
            jacobian[:, columns], desired.detach(), relative_threshold=threshold
        )
    originals = [value.detach().clone() for _, value in parameters]
    offset = 0
    try:
        with torch.no_grad():
            for (_, parameter), original in zip(parameters, originals, strict=True):
                count = parameter.numel()
                parameter.copy_(
                    original + delta[offset : offset + count].reshape_as(parameter)
                )
                offset += count
        with torch.no_grad():
            actual = predict().reshape(-1).float()
    finally:
        with torch.no_grad():
            for (_, parameter), original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
    restored = all(
        torch.equal(value.detach(), original)
        for (_, value), original in zip(parameters, originals, strict=True)
    )
    initial = initial_quotient.detach()
    target_flat = target_quotient.detach()
    actual_quotient = quotient_basis.double().T @ actual.double()
    linear = initial.double() + jacobian.double() @ delta
    affine_error = torch.linalg.vector_norm(
        actual_quotient - linear
    ) / torch.linalg.vector_norm(desired.double()).clamp_min(1e-30)
    coordinate_denominator = 3 * int(batch_data.num_nodes)
    actual_mse = float(
        (actual_quotient - target_flat.double()).square().sum()
        / coordinate_denominator
    )
    acceptance = protocol["acceptance"]
    operator_numeric = _operator_numeric_checks(
        model, noisy, batch_data, blueprint
    )
    benchmark = _cuda_benchmark(
        protocol, dataset, standardizer, device=device
    )
    cuda = protocol["cuda"]
    checks = {
        "quotient_rank": int(combined["rank"]) == int(acceptance["quotient_rank"]),
        "target_projection": float(combined["target_projection_relative_residual"])
        <= float(acceptance["target_projection_relative_residual_max"]),
        "condition_number": float(combined["condition_number"])
        <= float(acceptance["condition_number_max"]),
        "effective_rank": float(combined["effective_rank"])
        >= float(acceptance["effective_rank_min"]),
        "readout_step_norm": float(torch.linalg.vector_norm(delta))
        <= float(acceptance["readout_step_norm_max"]),
        "actual_coordinate_mse": actual_mse
        <= float(acceptance["actual_coordinate_mse_max"]),
        "affine_forward": float(affine_error)
        <= float(acceptance["affine_forward_relative_error_max"]),
        "operator_o3_covariance": float(
            operator_numeric["operator_o3_covariance_max_abs"]
        )
        <= float(acceptance["operator_o3_covariance_max"]),
        "operator_permutation": float(
            operator_numeric["operator_permutation_max_abs"]
        )
        <= float(acceptance["operator_permutation_error_max"]),
        "full_model_translation": float(
            operator_numeric["full_model_translation_invariance_max_abs"]
        )
        <= float(acceptance["full_model_translation_error_max"]),
        "finite_zero_stratum": bool(
            operator_numeric["zero_stratum_forward_backward_finite"]
        )
        is bool(acceptance["finite_zero_stratum"]),
        "cuda": float(benchmark["graphs_per_second"])
        >= float(cuda["graphs_per_second_min"])
        and float(benchmark["peak_allocated_mib"])
        <= float(cuda["peak_allocated_mib_max"])
        and bool(benchmark["finite"]),
        "parameters_restored_exactly": restored
        is bool(acceptance["parameters_restored_exactly"]),
        "tensor_candidates": int(acceptance["tensor_candidates"]) == 0,
    }
    decision = (
        "unit_scaled_readout_qualified_freeze_one_state_training"
        if all(checks.values())
        else "unit_scaled_readout_failed_remove_before_training"
    )
    result = {
        "protocol": protocol["protocol"],
        "graph_index": graph_index,
        "nodes": int(batch_data.num_nodes),
        "initial_coordinate_mse": float(
            (initial.double() - target_flat.double()).square().sum()
            / coordinate_denominator
        ),
        "actual_coordinate_mse": actual_mse,
        "linear_coordinate_mse": float(
            (linear - target_flat.double()).square().sum()
            / coordinate_denominator
        ),
        "affine_forward_relative_error": float(affine_error),
        "readout_step_norm": float(torch.linalg.vector_norm(delta)),
        "readout_parameter_norm": float(
            torch.linalg.vector_norm(
                torch.cat([value.reshape(-1) for value in originals]).double()
            )
        ),
        "combined": combined,
        "groups": group_metrics,
        "operator_numeric": operator_numeric,
        "cuda_benchmark": benchmark,
        "checks": checks,
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
