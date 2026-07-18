"""Audit whether one existing coordinate branch is sufficient and BF16-stable."""

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


def branch_slices(vector_columns: int, total_columns: int) -> dict[str, slice]:
    """Return the unique column interval owned by each candidate branch."""
    if vector_columns < 1 or total_columns <= vector_columns:
        raise ValueError("invalid coordinate branch dimensions")
    return {
        "vector_only": slice(0, vector_columns),
        "edge_only": slice(vector_columns, total_columns),
        "combined": slice(0, total_columns),
    }


def weighted_fit(
    design: torch.Tensor,
    target: torch.Tensor,
    row_graph: torch.Tensor,
    graph_count: int,
    *,
    rcond: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Solve graph-equal least squares and report the active singular spectrum."""
    if design.ndim != 2 or target.shape != (design.shape[0],):
        raise ValueError("affine design and target shapes do not match")
    if row_graph.shape != target.shape or graph_count < 1 or rcond <= 0.0:
        raise ValueError("invalid graph-equal affine fit inputs")
    counts = torch.bincount(row_graph, minlength=graph_count).double()
    weights = counts[row_graph].rsqrt()
    weighted_design = design.double() * weights[:, None]
    weighted_target = target.double() * weights
    cpu_design = weighted_design.cpu()
    solution = torch.linalg.lstsq(
        cpu_design,
        weighted_target.cpu(),
        rcond=rcond,
        driver="gelsd",
    ).solution.to(design.device)
    singular_values = torch.linalg.svdvals(cpu_design)
    active = singular_values > rcond * singular_values[0].clamp_min(1e-30)
    return solution, {
        "rows": int(design.shape[0]),
        "columns": int(design.shape[1]),
        "rank": int(active.sum()),
        "maximum_singular_value": float(singular_values[0]),
        "minimum_active_singular_value": (
            float(singular_values[active][-1]) if active.any() else 0.0
        ),
        "condition_number": (
            float(singular_values[0] / singular_values[active][-1])
            if active.any()
            else math.inf
        ),
    }


def assign_branch(model: nn.Module, branch: str, solution: torch.Tensor) -> None:
    """Zero both readouts, then assign exactly one declared branch or both."""
    parameters = dict(model.named_parameters())
    vector_count = parameters[READOUT_NAMES[0]].numel()
    edge_count = sum(parameters[name].numel() for name in READOUT_NAMES[1:])
    expected = {
        "vector_only": vector_count,
        "edge_only": edge_count,
        "combined": vector_count + edge_count,
    }
    if branch not in expected or solution.numel() != expected[branch]:
        raise ValueError("coordinate branch solution has the wrong shape")
    with torch.no_grad():
        for name in READOUT_NAMES:
            parameters[name].zero_()
        offset = 0
        selected = (
            READOUT_NAMES[:1]
            if branch == "vector_only"
            else READOUT_NAMES[1:]
            if branch == "edge_only"
            else READOUT_NAMES
        )
        for name in selected:
            parameter = parameters[name]
            count = parameter.numel()
            parameter.copy_(solution[offset : offset + count].reshape_as(parameter))
            offset += count


def _capture_design(
    model: nn.Module, noisy: Any, batch_data: Any, blueprint: Any
) -> tuple[torch.Tensor, torch.Tensor, float]:
    captures: dict[str, torch.Tensor] = {}

    def vector_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captures["vector"] = inputs[0].detach()

    def edge_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captures["edge"] = inputs[0].detach()

    vector_handle = model.coordinate_vector_head.register_forward_pre_hook(  # type: ignore[attr-defined]
        vector_hook
    )
    edge_handle = model.coordinate_edge_head[2].register_forward_pre_hook(  # type: ignore[attr-defined,index]
        edge_hook
    )
    try:
        with torch.no_grad():
            prediction = _predict(
                model, noisy, batch_data, blueprint, use_bf16=False
            ).detach()
    finally:
        vector_handle.remove()
        edge_handle.remove()
    graphs = int(batch_data.num_graphs)
    with torch.no_grad():
        lattice = LatticeVolumeShape(noisy.log_volume, noisy.log_shape).lattice(
            blueprint.fractional_to_cartesian
        )
        edges = periodic_radius_multigraph(
            noisy.fractional_coordinates,
            lattice,
            batch_data.batch,
            cutoff=float(model.radial.cutoff),  # type: ignore[attr-defined]
        )
        edge_hidden = captures["edge"]
        if edge_hidden.shape[0] != edges.source.numel():
            raise RuntimeError("captured edge basis does not match production edge order")
        envelope = model.radial.envelope(edges.distance)  # type: ignore[attr-defined]
        degree = torch.bincount(
            edges.target, minlength=int(batch_data.num_nodes)
        ).to(lattice)
        vector_basis = captures["vector"].transpose(-1, -2)
        edge_messages = (
            edge_hidden[:, :, None]
            * envelope[:, :, None]
            * edges.displacement[:, None, :]
        )
        edge_basis = vector_basis.new_zeros(
            (int(batch_data.num_nodes), edge_hidden.shape[1], 3)
        )
        edge_basis.index_add_(0, edges.target, edge_messages.to(edge_basis.dtype))
        edge_basis = edge_basis / degree.clamp_min(1).sqrt()[:, None, None]
        bias_messages = envelope * edges.displacement
        bias_basis = vector_basis.new_zeros((int(batch_data.num_nodes), 1, 3))
        bias_basis.index_add_(
            0, edges.target, bias_messages[:, None, :].to(bias_basis.dtype)
        )
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
        weights = torch.cat(tuple(parameters[name].reshape(-1) for name in READOUT_NAMES))
        reconstruction = (design @ weights).reshape_as(prediction)
    return design, prediction, float((reconstruction - prediction).abs().max())


def _gradients(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    use_bf16: bool,
) -> tuple[dict[str, torch.Tensor], float, bool]:
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = _predict(model, noisy, batch_data, blueprint, use_bf16=use_bf16)
    loss = _coordinate_loss(
        prediction,
        noisy.coordinate_scaled_score_target,
        batch_data.batch,
        int(batch_data.num_graphs),
    )
    loss.backward()
    gradients: dict[str, torch.Tensor] = {}
    total = prediction.new_zeros(())
    finite = math.isfinite(float(loss.detach()))
    for name, parameter in model.named_parameters():
        if name in READOUT_NAMES or parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        gradients[name] = gradient.cpu().clone()
        total = total + gradient.square().sum()
        finite = finite and bool(torch.isfinite(gradient).all())
    model.zero_grad(set_to_none=True)
    return gradients, float(total.sqrt()), finite


def _gradient_agreement(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, float]:
    if reference.keys() != candidate.keys() or not reference:
        raise ValueError("gradient fields must have the same nonempty keys")
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
def _prediction_metrics(
    model: nn.Module,
    diffusion: TensorFreeHybridDiffusion,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    use_bf16: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = _predict(
        model, noisy, batch_data, blueprint, use_bf16=use_bf16
    ).float()
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
    return prediction, {
        "coordinate_mse": float(loss),
        "low_time_endpoint_rms_angstrom": float(
            endpoint[low].square().mean().sqrt()
        ),
    }


def _evaluate_branch(
    branch: str,
    model: nn.Module,
    diffusion: TensorFreeHybridDiffusion,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    design: torch.Tensor,
    row_graph: torch.Tensor,
    column_slice: slice,
    *,
    rcond: float,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    branch_design = design[:, column_slice]
    target = noisy.coordinate_scaled_score_target.reshape(-1)
    solution, spectrum = weighted_fit(
        branch_design,
        target,
        row_graph,
        int(batch_data.num_graphs),
        rcond=rcond,
    )
    first_rows = row_graph == 0
    first_solution, first_spectrum = weighted_fit(
        branch_design[first_rows],
        target[first_rows],
        torch.zeros(int(first_rows.sum()), dtype=torch.long, device=row_graph.device),
        1,
        rcond=rcond,
    )
    first_prediction = branch_design[first_rows].double() @ first_solution
    first_residual = float(
        torch.linalg.vector_norm(first_prediction - target[first_rows].double())
        / torch.linalg.vector_norm(target[first_rows].double()).clamp_min(1e-30)
    )
    assign_branch(model, branch, solution)
    fp32_prediction, fp32 = _prediction_metrics(
        model,
        diffusion,
        noisy,
        batch_data,
        blueprint,
        use_bf16=False,
    )
    bf16_prediction, bf16 = _prediction_metrics(
        model,
        diffusion,
        noisy,
        batch_data,
        blueprint,
        use_bf16=True,
    )
    fp32_gradients, fp32_gradient_norm, fp32_finite = _gradients(
        model, noisy, batch_data, blueprint, use_bf16=False
    )
    bf16_gradients, bf16_gradient_norm, bf16_finite = _gradients(
        model, noisy, batch_data, blueprint, use_bf16=True
    )
    agreement = _gradient_agreement(fp32_gradients, bf16_gradients)
    mse_ratio = bf16["coordinate_mse"] / max(fp32["coordinate_mse"], 1e-30)
    prediction_relative_rmse = float(
        (bf16_prediction - fp32_prediction).square().mean().sqrt()
        / fp32_prediction.square().mean().sqrt().clamp_min(1e-30)
    )
    checks = {
        "one_state_rank": int(first_spectrum["rank"])
        == int(acceptance["one_state_quotient_rank"]),
        "one_state_projection": first_residual
        <= float(acceptance["one_state_projection_relative_residual_max"]),
        "panel_fp32_mse": fp32["coordinate_mse"]
        <= float(acceptance["panel_fp32_coordinate_mse_max"]),
        "panel_fp32_endpoint": fp32["low_time_endpoint_rms_angstrom"]
        <= float(acceptance["panel_fp32_low_time_endpoint_rms_angstrom_max"]),
        "solution_norm": float(torch.linalg.vector_norm(solution))
        <= float(acceptance["solution_norm_max"]),
        "bf16_mse": bf16["coordinate_mse"]
        <= float(acceptance["bf16_coordinate_mse_max"]),
        "bf16_endpoint": bf16["low_time_endpoint_rms_angstrom"]
        <= float(acceptance["bf16_low_time_endpoint_rms_angstrom_max"]),
        "bf16_mse_ratio": mse_ratio
        <= float(acceptance["bf16_over_fp32_mse_ratio_max"]),
        "bf16_prediction": prediction_relative_rmse
        <= float(acceptance["bf16_prediction_relative_rmse_max"]),
        "fp32_gradient": fp32_finite
        and fp32_gradient_norm
        <= float(acceptance["fp32_backbone_gradient_norm_max"]),
        "bf16_gradient": bf16_finite
        and bf16_gradient_norm
        <= float(acceptance["bf16_backbone_gradient_norm_max"]),
        "gradient_norm_agreement": float(
            acceptance["bf16_over_fp32_gradient_norm_min"]
        )
        <= agreement["candidate_over_reference_norm"]
        <= float(acceptance["bf16_over_fp32_gradient_norm_max"]),
        "gradient_direction": agreement["cosine"]
        >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
    }
    return {
        "solution_norm": float(torch.linalg.vector_norm(solution)),
        "spectrum": spectrum,
        "one_state": {
            "rank": int(first_spectrum["rank"]),
            "projection_relative_residual": first_residual,
        },
        "fp32": {**fp32, "backbone_gradient_norm": fp32_gradient_norm},
        "bf16": {**bf16, "backbone_gradient_norm": bf16_gradient_norm},
        "bf16_over_fp32_mse_ratio": mse_ratio,
        "bf16_prediction_relative_rmse": prediction_relative_rmse,
        "gradient_agreement": agreement,
        "checks": checks,
        "qualified": all(checks.values()),
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
    if protocol.get("protocol") != "h1a_coordinate_branch_minimality_v1":
        raise ValueError("coordinate branch-minimality protocol mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate branch-minimality cache mismatch")
    if int(protocol["audit"]["optimizer_steps"]) != 0:
        raise ValueError("branch-minimality audit forbids optimizer steps")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen branch-minimality audit requires CUDA")
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
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
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
            generator=torch.Generator(device=device).manual_seed(
                int(path["noise_seed"])
            ),
        )
    design, _, reconstruction = _capture_design(model, noisy, batch_data, blueprint)
    slices = branch_slices(int(protocol["model"]["vector_dim"]), design.shape[1])
    row_graph = batch_data.batch[:, None].expand(-1, 3).reshape(-1)
    acceptance = protocol["acceptance"]
    results = {
        branch: _evaluate_branch(
            branch,
            model,
            diffusion,
            noisy,
            batch_data,
            blueprint,
            design,
            row_graph,
            slices[branch],
            rcond=float(protocol["audit"]["lstsq_rcond"]),
            acceptance=acceptance,
        )
        for branch in protocol["audit"]["branches"]
    }
    model.load_state_dict(initial_state, strict=True)
    parameters_restored = all(
        torch.equal(value, model.state_dict()[name]) for name, value in initial_state.items()
    )
    passing = [
        branch
        for branch in ("vector_only", "edge_only")
        if bool(results[branch]["qualified"])
    ]
    selected = (
        "vector_only"
        if "vector_only" in passing
        else "edge_only"
        if "edge_only" in passing
        else None
    )
    common_checks = {
        "design_reconstruction": reconstruction
        <= float(acceptance["design_reconstruction_max_abs"]),
        "parameters_restored": parameters_restored
        is bool(acceptance["parameters_restored"]),
        "sampling_failures": int(acceptance["sampling_failures"]) == 0,
        "tensor_candidates": int(acceptance["tensor_candidates"]) == 0,
    }
    qualified = selected is not None and all(common_checks.values())
    result = {
        "protocol": protocol["protocol"],
        "fixed_indices": indices.tolist(),
        "design_shape": list(design.shape),
        "design_reconstruction_max_abs": reconstruction,
        "branches": results,
        "common_checks": common_checks,
        "selected_branch": selected,
        "qualified": qualified,
        "optimizer_steps": 0,
        "sampling_failures": 0,
        "tensor_candidates": 0,
        "decision": (
            f"{selected}_qualified_freeze_production_qualification"
            if qualified
            else "single_branch_minimality_failed_retain_combined_head"
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
