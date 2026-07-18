"""Qualify the Cartesian rank-one/rank-two coordinate score readout."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _make_batch,
    _make_model,
    _predict,
)


def tangent_spectrum(
    gram: torch.Tensor, *, relative_threshold: float
) -> dict[str, float | int | list[float]]:
    eigenvalues = torch.linalg.eigvalsh(gram.double())
    maximum = eigenvalues[-1].clamp_min(torch.finfo(torch.float64).tiny)
    active = eigenvalues > float(relative_threshold) * maximum
    positive = eigenvalues[active]
    weights = positive / positive.sum().clamp_min(1e-30)
    effective_rank = (
        float(torch.exp(-(weights * weights.log()).sum()))
        if positive.numel()
        else 0.0
    )
    return {
        "output_dimension": int(gram.shape[0]),
        "tangent_rank": int(active.sum()),
        "nullity": int((~active).sum()),
        "condition_number": (
            float(maximum / positive[0]) if positive.numel() else math.inf
        ),
        "effective_rank": effective_rank,
        "minimum_active_eigenvalue": (
            float(positive[0]) if positive.numel() else 0.0
        ),
        "maximum_eigenvalue": float(maximum),
        "eigenvalues": eigenvalues.tolist(),
    }


def _cartesian_moment_reference(
    directions: torch.Tensor,
    target: torch.Tensor,
    vector_coefficient: torch.Tensor,
    tensor_coefficient: torch.Tensor,
    readout: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    channels = vector_coefficient.shape[1]
    vector = directions.new_zeros((node_count, channels, 3))
    tensor = directions.new_zeros((node_count, channels, 3, 3))
    vector.index_add_(
        0, target, vector_coefficient[..., None] * directions[:, None, :]
    )
    identity = torch.eye(3, dtype=directions.dtype, device=directions.device)
    dyadic = torch.einsum("ei,ej->eij", directions, directions) - identity / 3.0
    tensor.index_add_(
        0,
        target,
        tensor_coefficient[..., None, None] * dyadic[:, None, :, :],
    )
    degree = torch.bincount(target, minlength=node_count).clamp_min(1).sqrt()
    vector = vector / degree[:, None, None]
    tensor = tensor / degree[:, None, None, None]
    polar = torch.einsum("ncij,ncj->nci", tensor, vector)
    return torch.einsum("nci,c->ni", polar, readout)


def _numeric_operator_checks(
    model: torch.nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
) -> dict[str, float | bool]:
    device = batch_data.lattice.device
    generator = torch.Generator(device=device).manual_seed(5917)
    edges, nodes, channels = 47, 9, 8
    directions = torch.randn((edges, 3), device=device, generator=generator)
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    target = torch.randint(nodes, (edges,), device=device, generator=generator)
    vector_coefficient = torch.randn(
        (edges, channels), device=device, generator=generator
    )
    tensor_coefficient = torch.randn(
        (edges, channels), device=device, generator=generator
    )
    readout = torch.randn((channels,), device=device, generator=generator)
    reflection = torch.diag(directions.new_tensor([-1.0, 1.0, 1.0]))
    original_moment = _cartesian_moment_reference(
        directions,
        target,
        vector_coefficient,
        tensor_coefficient,
        readout,
        nodes,
    )
    reflected_moment = _cartesian_moment_reference(
        directions @ reflection,
        target,
        vector_coefficient,
        tensor_coefficient,
        readout,
        nodes,
    )
    o3_error = float((reflected_moment - original_moment @ reflection).abs().max())
    original = _predict(model, noisy, batch_data, blueprint, use_bf16=False)
    shifted = dataclasses.replace(
        noisy,
        fractional_coordinates=noisy.fractional_coordinates
        + noisy.fractional_coordinates.new_tensor([0.31, -0.27, 1.19]),
    )
    translated = _predict(model, shifted, batch_data, blueprint, use_bf16=False)
    translation_error = float((translated - original).abs().max())
    permutation = torch.randperm(
        original.shape[0], device=device, generator=generator
    )
    permuted_noisy = dataclasses.replace(
        noisy,
        element_tokens=noisy.element_tokens[permutation],
        fractional_coordinates=noisy.fractional_coordinates[permutation],
    )
    permuted = _predict(
        model, permuted_noisy, batch_data, blueprint, use_bf16=False
    )
    permutation_error = float((permuted - original[permutation]).abs().max())
    differentiable_directions = directions.detach().clone().requires_grad_(True)
    differentiable = _cartesian_moment_reference(
        differentiable_directions,
        target,
        vector_coefficient,
        tensor_coefficient,
        readout,
        nodes,
    )
    differentiable.square().sum().backward()
    finite = differentiable_directions.grad is not None and bool(
        torch.isfinite(differentiable_directions.grad).all()
    )
    return {
        "o3_covariance_max_abs": o3_error,
        "translation_invariance_max_abs": translation_error,
        "node_permutation_equivariance_max_abs": permutation_error,
        "finite_forward_backward": finite,
    }


def _benchmark(
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
    tick = time.perf_counter()
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
    elapsed = time.perf_counter() - tick
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
    if protocol.get("protocol") != "h1a_cartesian_moment_score_head_v1":
        raise ValueError("Cartesian moment score-head protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("Cartesian moment score-head cache mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen operator qualification requires CUDA")
    seed = int(protocol["prerequisites"]["model_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    graph_index = int(protocol["prerequisites"]["fixed_graph_index"])
    batch_data = _make_batch(dataset, torch.tensor([graph_index]), device)
    if int(batch_data.num_nodes) != int(protocol["prerequisites"]["fixed_graph_nodes"]):
        raise ValueError("Cartesian moment fixed graph changed")
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
    graph_time = batch_data.lattice.new_tensor(
        [float(protocol["prerequisites"]["fixed_time"])]
    )
    noisy = diffusion.noise_clean_batch(
        batch_data.atom_types,
        batch_data.frac_coords,
        batch_data.lattice,
        batch_data.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=graph_time,
        generator=torch.Generator(device=device).manual_seed(
            int(protocol["prerequisites"]["noise_seed"])
        ),
    )
    prediction = _predict(model, noisy, batch_data, blueprint, use_bf16=False)
    target_score = noisy.coordinate_scaled_score_target
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
    gram = jacobian.double() @ jacobian.double().T
    numeric = protocol["numeric"]
    spectrum = tangent_spectrum(
        gram,
        relative_threshold=float(numeric["rank_relative_eigenvalue_threshold"]),
    )
    loss = (prediction - target_score).square().mean()
    initial_loss = float(loss.detach())
    gradients = torch.autograd.grad(
        loss, [value for _, value in parameters], allow_unused=True
    )
    group_energy = {"moment": 0.0, "central": 0.0}
    for (name, _), gradient in zip(parameters, gradients, strict=True):
        if gradient is None:
            continue
        if name.startswith(("coordinate_moment_coefficients.", "coordinate_moment_readout.")):
            group_energy["moment"] += float(gradient.double().square().sum())
        if name.startswith("coordinate_edge_scalar."):
            group_energy["central"] += float(gradient.double().square().sum())
    moment_to_central = math.sqrt(group_energy["moment"]) / max(
        math.sqrt(group_energy["central"]), 1e-30
    )
    operator_checks = _numeric_operator_checks(
        model, noisy, batch_data, blueprint
    )
    expected_nullity = int(numeric["expected_translation_nullity"])
    node_count = int(batch_data.num_nodes)
    quotient_dimension = 3 * node_count - 3
    tangent_rank = spectrum["tangent_rank"]
    tangent_nullity = spectrum["nullity"]
    condition_number = spectrum["condition_number"]
    effective_rank = spectrum["effective_rank"]
    if not isinstance(tangent_rank, int) or not isinstance(tangent_nullity, int):
        raise TypeError("tangent rank metrics have invalid types")
    if not isinstance(condition_number, float) or not isinstance(
        effective_rank, float
    ):
        raise TypeError("tangent spectral metrics have invalid types")
    checks = {
        "quotient_tangent_full_rank": tangent_rank == quotient_dimension
        and tangent_nullity == expected_nullity,
        "condition_number": condition_number
        <= float(numeric["condition_number_max"]),
        "effective_rank": effective_rank >= float(numeric["effective_rank_min"]),
        "moment_gradient_balance": moment_to_central
        >= float(numeric["moment_to_central_gradient_norm_ratio_min"]),
        "translation_invariance": float(operator_checks["translation_invariance_max_abs"])
        <= float(numeric["translation_invariance_max_fp32"]),
        "node_permutation_equivariance": float(
            operator_checks["node_permutation_equivariance_max_abs"]
        )
        <= float(numeric["node_permutation_equivariance_max_fp32"]),
        "o3_covariance": float(operator_checks["o3_covariance_max_abs"])
        <= float(numeric["o3_covariance_max_fp32"]),
        "finite_forward_backward": bool(operator_checks["finite_forward_backward"]),
        "parameter_count": sum(value.numel() for value in model.parameters())
        == int(protocol["model"]["parameter_count"]),
        "tensor_candidates": int(numeric["tensor_candidates"]) == 0,
    }
    del (
        rows,
        jacobian,
        gram,
        gradients,
        loss,
        prediction,
        target_score,
        noisy,
        diffusion,
        model,
        parameters,
        batch_data,
        blueprint,
    )
    torch.cuda.empty_cache()
    benchmark = _benchmark(protocol, dataset, standardizer, device=device)
    cuda = protocol["cuda"]
    checks["cuda"] = (
        float(benchmark["graphs_per_second"]) >= float(cuda["graphs_per_second_min"])
        and float(benchmark["peak_allocated_mib"]) <= float(cuda["peak_allocated_mib_max"])
        and bool(benchmark["finite"])
    )
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "graph_index": graph_index,
        "nodes": node_count,
        "quotient_dimension": quotient_dimension,
        "initial_coordinate_mse": initial_loss,
        "tangent": spectrum,
        "moment_to_central_gradient_norm_ratio": moment_to_central,
        "operator_numeric": operator_checks,
        "cuda_benchmark": benchmark,
        "checks": checks,
        "qualified": qualified,
        "decision": (
            "cartesian_moment_head_qualified_freeze_one_state_training"
            if qualified
            else "cartesian_moment_head_failed_remove_before_training"
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
