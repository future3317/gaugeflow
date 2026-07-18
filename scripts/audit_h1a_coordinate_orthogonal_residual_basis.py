"""Qualify a target-free block-orthogonal chart of the coordinate readout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.geometry import periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.state_projection import graph_mean
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _coordinate_loss,
    _endpoint_rms,
    _fixed_indices,
    _make_batch,
    _make_model,
    _predict,
)

READOUT_NAMES = (
    "coordinate_vector_head.weight",
    "coordinate_edge_head.2.weight",
    "coordinate_edge_head.2.bias",
)


def block_orthogonal_residual_chart(
    vector_design: torch.Tensor,
    edge_design: torch.Tensor,
    row_weights: torch.Tensor,
) -> torch.Tensor:
    """Return the exact graph-weighted block Gram--Schmidt channel chart."""
    if vector_design.ndim != 2 or edge_design.ndim != 2:
        raise ValueError("coordinate designs must be matrices")
    if vector_design.shape[0] != edge_design.shape[0]:
        raise ValueError("coordinate designs must share rows")
    if row_weights.shape != (vector_design.shape[0],):
        raise ValueError("row weights do not match the coordinate design")
    if vector_design.shape[1] < 1 or edge_design.shape[1] < 1:
        raise ValueError("both coordinate blocks must be nonempty")
    if not bool((row_weights > 0).all()):
        raise ValueError("coordinate row weights must be positive")
    vector = vector_design.double() * row_weights.double()[:, None]
    edge = edge_design.double() * row_weights.double()[:, None]
    vector_gram = vector.T @ vector
    vector_cholesky, vector_info = torch.linalg.cholesky_ex(vector_gram)
    if int(vector_info.max()) != 0:
        raise ValueError("vector coordinate block is not positive definite")
    cross = vector.T @ edge
    projection = torch.cholesky_solve(cross, vector_cholesky)
    residual = edge - vector @ projection
    residual_gram = residual.T @ residual
    residual_cholesky, residual_info = torch.linalg.cholesky_ex(residual_gram)
    if int(residual_info.max()) != 0:
        raise ValueError("edge residual coordinate block is not positive definite")
    vector_columns = vector.shape[1]
    edge_columns = edge.shape[1]
    vector_whitener = torch.linalg.solve_triangular(
        vector_cholesky.T,
        torch.eye(vector_columns, dtype=torch.float64, device=vector.device),
        upper=True,
    )
    residual_whitener = torch.linalg.solve_triangular(
        residual_cholesky.T,
        torch.eye(edge_columns, dtype=torch.float64, device=edge.device),
        upper=True,
    )
    chart = torch.zeros(
        (vector_columns + edge_columns, vector_columns + edge_columns),
        dtype=torch.float64,
        device=vector.device,
    )
    chart[:vector_columns, :vector_columns] = vector_whitener
    chart[:vector_columns, vector_columns:] = -projection @ residual_whitener
    chart[vector_columns:, vector_columns:] = residual_whitener
    return chart


def _capture_design(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    use_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    captures: dict[str, torch.Tensor] = {}

    def vector_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captures["vector"] = inputs[0]

    def edge_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captures["edge"] = inputs[0]

    vector_handle = model.coordinate_vector_head.register_forward_pre_hook(  # type: ignore[attr-defined]
        vector_hook
    )
    edge_handle = model.coordinate_edge_head[2].register_forward_pre_hook(  # type: ignore[attr-defined,index]
        edge_hook
    )
    try:
        original = _predict(
            model, noisy, batch_data, blueprint, use_bf16=use_bf16
        )
    finally:
        vector_handle.remove()
        edge_handle.remove()
    graphs = int(batch_data.num_graphs)
    with torch.autocast(device_type=noisy.time.device.type, enabled=False):
        lattice = LatticeVolumeShape(noisy.log_volume, noisy.log_shape).lattice(
            blueprint.fractional_to_cartesian
        ).float()
        edges = periodic_radius_multigraph(
            noisy.fractional_coordinates.float(),
            lattice,
            batch_data.batch,
            cutoff=float(model.radial.cutoff),  # type: ignore[attr-defined]
        )
        edge_hidden = captures["edge"].float()
        if edge_hidden.shape[0] != edges.source.numel():
            raise RuntimeError("captured edge basis does not match production edge order")
        envelope = model.radial.envelope(edges.distance).float()  # type: ignore[attr-defined]
        degree = torch.bincount(
            edges.target, minlength=int(batch_data.num_nodes)
        ).to(lattice)
        vector_basis = captures["vector"].float().transpose(-1, -2)
        edge_messages = (
            edge_hidden[:, :, None]
            * envelope[:, :, None]
            * edges.displacement[:, None, :]
        )
        edge_basis = vector_basis.new_zeros(
            (int(batch_data.num_nodes), edge_hidden.shape[1], 3)
        )
        edge_basis.index_add_(0, edges.target, edge_messages)
        edge_basis = edge_basis / degree.clamp_min(1).sqrt()[:, None, None]
        bias_messages = envelope * edges.displacement
        bias_basis = vector_basis.new_zeros((int(batch_data.num_nodes), 1, 3))
        bias_basis.index_add_(0, edges.target, bias_messages[:, None, :])
        bias_basis = bias_basis / degree.clamp_min(1).sqrt()[:, None, None]
        cartesian_basis = torch.cat((vector_basis, edge_basis, bias_basis), dim=1)
        cartesian_basis = cartesian_basis - graph_mean(
            cartesian_basis, batch_data.batch, graphs
        )[batch_data.batch]
        fractional_basis = torch.einsum(
            "nci,nji->ncj", cartesian_basis, lattice[batch_data.batch]
        )
        fractional_basis = fractional_basis - graph_mean(
            fractional_basis, batch_data.batch, graphs
        )[batch_data.batch]
        design = fractional_basis.transpose(1, 2).reshape(
            3 * int(batch_data.num_nodes), -1
        )
        parameters = dict(model.named_parameters())
        weights = torch.cat(
            tuple(parameters[name].reshape(-1).float() for name in READOUT_NAMES)
        )
        reconstruction = (design @ weights).reshape_as(original)
    return design, original.float(), float((reconstruction - original.float()).abs().max())


def _gradient_fields(model: nn.Module) -> tuple[dict[str, torch.Tensor], float, bool]:
    gradients: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=next(model.parameters()).device)
    finite = True
    for name, parameter in model.named_parameters():
        if name in READOUT_NAMES or parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        gradients[name] = gradient.cpu().clone()
        total = total + gradient.square().sum()
        finite = finite and bool(torch.isfinite(gradient).all())
    return gradients, float(total.sqrt()), finite


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


def _evaluate(
    model: nn.Module,
    diffusion: TensorFreeHybridDiffusion,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    chart: torch.Tensor,
    solution: torch.Tensor,
    *,
    use_bf16: bool,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    model.train()
    model.zero_grad(set_to_none=True)
    design, _, _ = _capture_design(
        model, noisy, batch_data, blueprint, use_bf16=use_bf16
    )
    transformed = design.float() @ chart.to(design.device, torch.float32)
    prediction = (transformed @ solution.to(design.device, torch.float32)).reshape_as(
        noisy.coordinate_scaled_score_target
    )
    loss = _coordinate_loss(
        prediction,
        noisy.coordinate_scaled_score_target,
        batch_data.batch,
        int(batch_data.num_graphs),
    )
    endpoint = _endpoint_rms(
        prediction,
        noisy,
        batch_data.frac_coords,
        batch_data.lattice,
        batch_data.batch,
        diffusion,
    )
    low = noisy.time <= 0.02
    loss.backward()
    gradients, gradient_norm, finite = _gradient_fields(model)
    model.zero_grad(set_to_none=True)
    return prediction.detach(), {
        "coordinate_mse": float(loss.detach()),
        "low_time_endpoint_rms_angstrom": float(endpoint[low].square().mean().sqrt()),
        "backbone_gradient_norm": gradient_norm,
        "finite": finite and math.isfinite(float(loss.detach())),
    }, gradients


def _operator_benchmark(
    design: torch.Tensor,
    chart: torch.Tensor,
    solution: torch.Tensor,
    repeats: int,
) -> dict[str, float]:
    if design.device.type != "cuda":
        raise ValueError("operator benchmark requires CUDA")
    chart32 = chart.to(design.device, torch.float32)
    solution32 = solution.to(design.device, torch.float32)
    torch.cuda.reset_peak_memory_stats(design.device)
    baseline_memory = torch.cuda.memory_allocated(design.device)
    for _ in range(10):
        _ = (design.float() @ chart32) @ solution32
    torch.cuda.synchronize(design.device)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _ = (design.float() @ chart32) @ solution32
    stop.record()
    torch.cuda.synchronize(design.device)
    return {
        "latency_ms": float(start.elapsed_time(stop) / repeats),
        "peak_memory_mib": float(
            (torch.cuda.max_memory_allocated(design.device) - baseline_memory)
            / (1024**2)
        ),
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
    if protocol.get("protocol") != "h1a_coordinate_orthogonal_residual_basis_v1":
        raise ValueError("orthogonal-residual coordinate protocol mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("orthogonal-residual coordinate cache mismatch")
    if int(protocol["audit"]["optimizer_steps"]) != 0:
        raise ValueError("orthogonal-residual audit forbids optimizer steps")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen orthogonal-residual audit requires CUDA")
    path = protocol["path"]
    torch.manual_seed(int(path["model_seed"]))
    torch.cuda.manual_seed_all(int(path["model_seed"]))
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    data = protocol["data"]
    indices = _fixed_indices(
        len(dataset), int(data["fixed_graphs"]), int(data["fixed_selection_seed"])
    )
    batch_data = _make_batch(dataset, indices, device)
    blueprint = _blueprint(batch_data)
    model = _make_model(protocol, device).float()
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
    times = batch_data.lattice.new_tensor(path["time_grid"])
    graph_time = times[
        torch.arange(int(batch_data.num_graphs), device=device) % times.numel()
    ]
    with torch.no_grad():
        noisy = diffusion.noise_clean_batch(
            batch_data.atom_types,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=graph_time,
            generator=torch.Generator(device=device).manual_seed(int(path["noise_seed"])),
        )
        design, _, reconstruction = _capture_design(
            model, noisy, batch_data, blueprint, use_bf16=False
        )
    vector_columns = int(protocol["audit"]["vector_columns"])
    row_graph = batch_data.batch[:, None].expand(-1, 3).reshape(-1)
    graph_rows = torch.bincount(row_graph, minlength=int(batch_data.num_graphs)).double()
    row_weights = graph_rows[row_graph].rsqrt()
    chart = block_orthogonal_residual_chart(
        design[:, :vector_columns].double(),
        design[:, vector_columns:].double(),
        row_weights,
    )
    weighted_basis = row_weights[:, None] * (design.double() @ chart)
    gram = weighted_basis.T @ weighted_basis
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    singular = torch.linalg.svdvals(gram)
    target = noisy.coordinate_scaled_score_target.reshape(-1).double()
    weighted_target = row_weights * target
    solution = weighted_basis.T @ weighted_target
    prediction = (design.double() @ chart @ solution)
    raw_solution = torch.linalg.lstsq(
        (design.double() * row_weights[:, None]).cpu(),
        weighted_target.cpu(),
        rcond=1e-10,
        driver="gelsd",
    ).solution.to(device)
    raw_prediction = design.double() @ raw_solution
    span_error = float(
        torch.linalg.vector_norm(prediction - raw_prediction)
        / torch.linalg.vector_norm(raw_prediction).clamp_min(1e-30)
    )
    vector_prediction = (design.double() @ chart[:, :vector_columns]) @ solution[:vector_columns]
    edge_prediction = (design.double() @ chart[:, vector_columns:]) @ solution[vector_columns:]
    weighted_vector_norm = torch.linalg.vector_norm(row_weights * vector_prediction)
    weighted_edge_norm = torch.linalg.vector_norm(row_weights * edge_prediction)
    weighted_total_norm = torch.linalg.vector_norm(row_weights * prediction)
    cancellation = float(
        (weighted_vector_norm + weighted_edge_norm) / weighted_total_norm.clamp_min(1e-30)
    )
    fp32_prediction, fp32, fp32_gradients = _evaluate(
        model, diffusion, noisy, batch_data, blueprint, chart, solution, use_bf16=False
    )
    bf16_prediction, bf16, bf16_gradients = _evaluate(
        model, diffusion, noisy, batch_data, blueprint, chart, solution, use_bf16=True
    )
    agreement = _gradient_agreement(fp32_gradients, bf16_gradients)
    mse_ratio = bf16["coordinate_mse"] / max(fp32["coordinate_mse"], 1e-30)
    prediction_relative_rmse = float(
        (bf16_prediction - fp32_prediction).square().mean().sqrt()
        / fp32_prediction.square().mean().sqrt().clamp_min(1e-30)
    )
    benchmark = _operator_benchmark(
        design.detach(), chart, solution, int(protocol["audit"]["latency_repeats"])
    )
    model.load_state_dict(initial_state, strict=True)
    parameters_restored = all(
        torch.equal(value, model.state_dict()[name]) for name, value in initial_state.items()
    )
    acceptance = protocol["acceptance"]
    checks = {
        "weighted_gram_condition": float(singular[0] / singular[-1])
        <= float(acceptance["weighted_gram_condition_number_max"]),
        "weighted_gram_error": float((gram - identity).abs().max())
        <= float(acceptance["weighted_gram_max_abs_error_max"]),
        "span_prediction": span_error
        <= float(acceptance["span_prediction_relative_error_max"]),
        "orthogonal_solution_norm": float(torch.linalg.vector_norm(solution))
        <= float(acceptance["orthogonal_solution_norm_max"]),
        "orthogonal_block_cancellation": cancellation
        <= float(acceptance["orthogonal_block_cancellation_ratio_max"]),
        "fp32_mse": fp32["coordinate_mse"]
        <= float(acceptance["fp32_coordinate_mse_max"]),
        "fp32_endpoint": fp32["low_time_endpoint_rms_angstrom"]
        <= float(acceptance["fp32_low_time_endpoint_rms_angstrom_max"]),
        "bf16_mse": bf16["coordinate_mse"]
        <= float(acceptance["bf16_coordinate_mse_max"]),
        "bf16_endpoint": bf16["low_time_endpoint_rms_angstrom"]
        <= float(acceptance["bf16_low_time_endpoint_rms_angstrom_max"]),
        "bf16_mse_ratio": mse_ratio
        <= float(acceptance["bf16_over_fp32_mse_ratio_max"]),
        "bf16_prediction": prediction_relative_rmse
        <= float(acceptance["bf16_prediction_relative_rmse_max"]),
        "fp32_gradient": bool(fp32["finite"])
        and fp32["backbone_gradient_norm"]
        <= float(acceptance["fp32_backbone_gradient_norm_max"]),
        "bf16_gradient": bool(bf16["finite"])
        and bf16["backbone_gradient_norm"]
        <= float(acceptance["bf16_backbone_gradient_norm_max"]),
        "gradient_norm_agreement": float(
            acceptance["bf16_over_fp32_gradient_norm_min"]
        )
        <= agreement["candidate_over_reference_norm"]
        <= float(acceptance["bf16_over_fp32_gradient_norm_max"]),
        "gradient_direction": agreement["cosine"]
        >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
        "operator_latency": benchmark["latency_ms"]
        <= float(acceptance["chart_operator_latency_ms_max"]),
        "operator_memory": benchmark["peak_memory_mib"]
        <= float(acceptance["chart_operator_peak_memory_mib_max"]),
        "design_reconstruction": reconstruction
        <= float(acceptance["design_reconstruction_max_abs"]),
        "parameters_restored": parameters_restored
        is bool(acceptance["parameters_restored"]),
        "sampling_failures": int(acceptance["sampling_failures"]) == 0,
        "tensor_candidates": int(acceptance["tensor_candidates"]) == 0,
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "fixed_indices": indices.tolist(),
        "design_shape": list(design.shape),
        "chart_shape": list(chart.shape),
        "weighted_gram_condition_number": float(singular[0] / singular[-1]),
        "weighted_gram_max_abs_error": float((gram - identity).abs().max()),
        "span_prediction_relative_error": span_error,
        "orthogonal_solution_norm": float(torch.linalg.vector_norm(solution)),
        "effective_raw_solution_norm": float(torch.linalg.vector_norm(chart @ solution)),
        "orthogonal_block_cancellation_ratio": cancellation,
        "design_reconstruction_max_abs": reconstruction,
        "fp32": fp32,
        "bf16": bf16,
        "bf16_over_fp32_mse_ratio": mse_ratio,
        "bf16_prediction_relative_rmse": prediction_relative_rmse,
        "gradient_agreement": agreement,
        "operator_benchmark": benchmark,
        "checks": checks,
        "qualified": qualified,
        "optimizer_steps": 0,
        "sampling_failures": 0,
        "tensor_candidates": 0,
        "decision": (
            "orthogonal_residual_basis_qualified_freeze_production_integration"
            if qualified
            else "orthogonal_residual_basis_failed_retain_combined_head"
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
