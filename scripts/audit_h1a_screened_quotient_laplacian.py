"""Qualify the screened quotient-Laplacian coordinate preconditioner."""

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
from gaugeflow.production.equivariant_denoiser import (
    screened_quotient_laplacian_precondition,
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


def helmert_quotient_basis(
    nodes: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if nodes < 2:
        raise ValueError("quotient audit needs at least two nodes")
    helmert = torch.zeros((nodes, nodes - 1), dtype=dtype, device=device)
    for column in range(nodes - 1):
        count = column + 1
        scale = float(count * (count + 1)) ** -0.5
        helmert[:count, column] = scale
        helmert[count, column] = -float(count) * scale
    return torch.kron(helmert, torch.eye(3, dtype=dtype, device=device))


def _readout_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    names = [
        "coordinate_vector_head.weight",
        "coordinate_edge_head.2.weight",
        "coordinate_edge_head.2.bias",
    ]
    parameters = dict(model.named_parameters())
    return [(name, parameters[name]) for name in names]


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


def _spectrum_and_solution(
    jacobian: torch.Tensor,
    desired: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    jacobian64 = jacobian.double()
    gram = jacobian64 @ jacobian64.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues[-1].clamp_min(torch.finfo(torch.float64).tiny)
    active = eigenvalues > threshold * maximum
    vectors = eigenvectors[:, active]
    values = eigenvalues[active]
    desired64 = desired.double()
    projection = vectors @ (vectors.T @ desired64)
    coefficients = vectors @ ((vectors.T @ desired64) / values)
    solution = jacobian64.T @ coefficients
    normalized = values / values.sum().clamp_min(1e-30)
    return solution, {
        "rank": int(active.sum()),
        "condition_number": float(maximum / values[0]),
        "effective_rank": float(torch.exp(-(normalized * normalized.log()).sum())),
        "target_projection_relative_residual": float(
            torch.linalg.vector_norm(desired64 - projection)
            / torch.linalg.vector_norm(desired64).clamp_min(1e-30)
        ),
        "minimum_active_eigenvalue": float(values[0]),
        "maximum_eigenvalue": float(maximum),
        "eigenvalues": eigenvalues.tolist(),
    }


def _operator_checks(device: torch.device) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(5920)
    field = torch.randn((5, 3), device=device, generator=generator)
    field = field - field.mean(0, keepdim=True)
    source = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 0], device=device)
    target = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3, 0, 4], device=device)
    weight = torch.linspace(0.4, 1.0, source.numel(), device=device)
    batch = torch.zeros(5, dtype=torch.long, device=device)
    matrix = torch.randn((3, 3), device=device, generator=generator)
    rotation, _ = torch.linalg.qr(matrix)
    reference = screened_quotient_laplacian_precondition(
        field, source, target, weight, batch, 1
    )
    rotated = screened_quotient_laplacian_precondition(
        field @ rotation, source, target, weight, batch, 1
    )
    permutation = torch.randperm(5, device=device, generator=generator)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(5, device=device)
    permuted = screened_quotient_laplacian_precondition(
        field[permutation],
        inverse[source],
        inverse[target],
        weight,
        batch,
        1,
    )
    return {
        "o3_covariance_max_abs": float((rotated - reference @ rotation).abs().max()),
        "permutation_max_abs": float((permuted - reference[permutation]).abs().max()),
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
    if protocol.get("protocol") != "h1a_screened_quotient_laplacian_v1":
        raise ValueError("screened quotient-Laplacian protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("screened quotient-Laplacian cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed = int(protocol["prerequisites"]["model_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    graph_index = int(protocol["prerequisites"]["fixed_graph_index"])
    batch_data = _make_batch(dataset, torch.tensor([graph_index]), device)
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
    noisy = diffusion.noise_clean_batch(
        batch_data.atom_types,
        batch_data.frac_coords,
        batch_data.lattice,
        batch_data.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=batch_data.lattice.new_tensor(
            [float(protocol["prerequisites"]["fixed_time"])]
        ),
        generator=torch.Generator(device=device).manual_seed(
            int(protocol["prerequisites"]["noise_seed"])
        ),
    )
    prediction = _predict(model, noisy, batch_data, blueprint, use_bf16=False)
    target = noisy.coordinate_scaled_score_target
    quotient = helmert_quotient_basis(
        int(batch_data.num_nodes), dtype=prediction.dtype, device=device
    )
    initial = quotient.T @ prediction.reshape(-1)
    desired = quotient.T @ (target - prediction).reshape(-1)
    parameters = _readout_parameters(model)
    raw_jacobian = _jacobian(prediction, parameters)
    jacobian = quotient.T @ raw_jacobian
    numeric = protocol["numeric"]
    solution, spectrum = _spectrum_and_solution(
        jacobian,
        desired.detach(),
        threshold=float(numeric["rank_relative_eigenvalue_threshold"]),
    )
    originals = [value.detach().clone() for _, value in parameters]
    offset = 0
    try:
        with torch.no_grad():
            for (_, parameter), original in zip(parameters, originals, strict=True):
                count = parameter.numel()
                parameter.copy_(
                    original + solution[offset : offset + count].reshape_as(parameter)
                )
                offset += count
        with torch.no_grad():
            actual = quotient.T @ _predict(
                model, noisy, batch_data, blueprint, use_bf16=False
            ).reshape(-1)
    finally:
        with torch.no_grad():
            for (_, parameter), original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
    actual_mse = float(
        (actual.double() - (initial + desired).double()).square().sum()
        / (3 * int(batch_data.num_nodes))
    )
    shifted = dataclasses.replace(
        noisy,
        fractional_coordinates=noisy.fractional_coordinates
        + noisy.fractional_coordinates.new_tensor([0.31, -0.27, 1.19]),
    )
    with torch.no_grad():
        translation_error = float(
            (
                _predict(model, shifted, batch_data, blueprint, use_bf16=False)
                - prediction.detach()
            )
            .abs()
            .max()
        )
    operator = _operator_checks(device)
    benchmark = _benchmark(protocol, dataset, standardizer, device=device)
    cuda = protocol["cuda"]
    checks = {
        "quotient_rank": int(spectrum["rank"]) == int(numeric["quotient_rank"]),
        "target_projection": float(spectrum["target_projection_relative_residual"])
        <= float(numeric["target_projection_relative_residual_max"]),
        "condition_number": float(spectrum["condition_number"])
        <= float(numeric["condition_number_max"]),
        "effective_rank": float(spectrum["effective_rank"])
        >= float(numeric["effective_rank_min"]),
        "readout_step_norm": float(torch.linalg.vector_norm(solution))
        <= float(numeric["readout_step_norm_max"]),
        "actual_affine_fit": actual_mse <= float(numeric["actual_affine_fit_mse_max"]),
        "operator_o3": operator["o3_covariance_max_abs"]
        <= float(numeric["operator_o3_error_max"]),
        "operator_permutation": operator["permutation_max_abs"]
        <= float(numeric["operator_permutation_error_max"]),
        "full_model_translation": translation_error
        <= float(numeric["full_model_translation_error_max"]),
        "cuda": float(benchmark["graphs_per_second"])
        >= float(cuda["graphs_per_second_min"])
        and float(benchmark["peak_allocated_mib"])
        <= float(cuda["peak_allocated_mib_max"])
        and bool(benchmark["finite"]),
        "parameter_count": sum(value.numel() for value in model.parameters())
        == int(protocol["model"]["parameter_count"]),
        "tensor_candidates": int(numeric["tensor_candidates"]) == 0,
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "initial_coordinate_mse": float((prediction - target).square().mean()),
        "readout_step_norm": float(torch.linalg.vector_norm(solution)),
        "actual_affine_fit_mse": actual_mse,
        "spectrum": spectrum,
        "operator_numeric": operator,
        "full_model_translation_error": translation_error,
        "cuda_benchmark": benchmark,
        "checks": checks,
        "qualified": qualified,
        "decision": (
            "screened_quotient_laplacian_qualified_freeze_one_state_training"
            if qualified
            else "screened_quotient_laplacian_failed_remove_before_training"
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
