"""Audit FP32/BF16 stability of a scaled exact coordinate readout without training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import TracebackType
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
    _fixed_indices,
    _make_batch,
    _make_model,
    _predict,
)

READOUT_PARAMETER_NAMES = (
    "coordinate_vector_head.weight",
    "coordinate_edge_head.2.weight",
    "coordinate_edge_head.2.bias",
)


def is_power_of_two(value: float) -> bool:
    """Return whether ``value`` is an exact positive binary power."""
    if not math.isfinite(value) or value <= 0.0:
        return False
    _, exponent = math.frexp(value)
    return value == math.ldexp(0.5, exponent)


class ScaledCoordinateReadout:
    """Temporarily apply an exactly reversible power-of-two readout chart."""

    def __init__(self, model: nn.Module, scale: float) -> None:
        if not is_power_of_two(scale):
            raise ValueError("coordinate readout scale must be a positive power of two")
        self.model = model
        self.scale = float(scale)
        self._saved: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _scale_output(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return output * self.scale

    def __enter__(self) -> ScaledCoordinateReadout:
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name in READOUT_PARAMETER_NAMES:
                parameter = parameters[name]
                self._saved[name] = parameter.detach().clone()
                parameter.div_(self.scale)
        self._handles = [
            self.model.coordinate_vector_head.register_forward_hook(  # type: ignore[attr-defined]
                self._scale_output
            ),
            self.model.coordinate_edge_head[2].register_forward_hook(  # type: ignore[attr-defined,index]
                self._scale_output
            ),
        ]
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in self._saved.items():
                parameters[name].copy_(value)


def weighted_affine_fit(
    design: torch.Tensor,
    target: torch.Tensor,
    row_graph: torch.Tensor,
    graph_count: int,
    *,
    rcond: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Solve graph-equal least squares and return its active singular spectrum."""
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


def assign_affine_readout(
    model: nn.Module,
    solution: torch.Tensor,
    names: tuple[str, ...] = READOUT_PARAMETER_NAMES,
) -> None:
    """Copy a flat affine solution into only the declared coordinate readout."""
    parameters = dict(model.named_parameters())
    expected = sum(parameters[name].numel() for name in names)
    if solution.numel() != expected:
        raise ValueError("affine readout solution has the wrong length")
    offset = 0
    with torch.no_grad():
        for name in names:
            parameter = parameters[name]
            count = parameter.numel()
            parameter.copy_(solution[offset : offset + count].reshape_as(parameter))
            offset += count


def _capture_scaled_affine_design(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    scale: float,
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
        cartesian_basis = scale * torch.cat(
            (vector_basis, edge_basis, bias_basis), dim=1
        )
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
            tuple(parameters[name].reshape(-1) for name in READOUT_PARAMETER_NAMES)
        )
        reconstruction = (design @ weights).reshape_as(prediction)
    return design, prediction, float((reconstruction - prediction).abs().max())


def _backbone_gradients(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    use_bf16: bool,
) -> tuple[dict[str, torch.Tensor], float, float, bool]:
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
        if name in READOUT_PARAMETER_NAMES or parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        gradients[name] = gradient.cpu().clone()
        total = total + gradient.square().sum()
        finite = finite and bool(torch.isfinite(gradient).all())
    model.zero_grad(set_to_none=True)
    return gradients, float(total.sqrt()), float(loss.detach()), finite


def gradient_agreement(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Compare two named gradient fields without changing parameter order."""
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


def _component_cancellation(
    design: torch.Tensor, solution: torch.Tensor, vector_columns: int
) -> dict[str, float]:
    vector = design[:, :vector_columns].double() @ solution[:vector_columns]
    edge = design[:, vector_columns:].double() @ solution[vector_columns:]
    total = vector + edge
    return {
        "vector_prediction_norm": float(torch.linalg.vector_norm(vector)),
        "edge_prediction_norm": float(torch.linalg.vector_norm(edge)),
        "total_prediction_norm": float(torch.linalg.vector_norm(total)),
        "component_to_total_norm_ratio": float(
            (torch.linalg.vector_norm(vector) + torch.linalg.vector_norm(edge))
            / torch.linalg.vector_norm(total).clamp_min(1e-30)
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
    if protocol.get("protocol") != "h1a_scaled_variable_projection_stability_v1":
        raise ValueError("scaled variable-projection stability protocol mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("scaled variable-projection cache mismatch")
    audit = protocol["audit"]
    if tuple(audit["readout_parameters"]) != READOUT_PARAMETER_NAMES:
        raise ValueError("scaled variable-projection readout declaration mismatch")
    if int(audit["optimizer_steps"]) != 0:
        raise ValueError("stability audit forbids optimizer steps")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen stability audit requires CUDA")

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
        baseline_prediction = _predict(
            model, noisy, batch_data, blueprint, use_bf16=False
        ).detach()
    _, baseline_gradient_norm, baseline_loss, baseline_gradient_finite = (
        _backbone_gradients(
            model, noisy, batch_data, blueprint, use_bf16=False
        )
    )

    scale = float(audit["readout_scale"])
    with ScaledCoordinateReadout(model, scale):
        with torch.no_grad():
            scaled_initial_prediction = _predict(
                model, noisy, batch_data, blueprint, use_bf16=False
            ).detach()
        function_preservation = float(
            (scaled_initial_prediction - baseline_prediction).abs().max()
        )
        design, _, design_reconstruction = _capture_scaled_affine_design(
            model, noisy, batch_data, blueprint, scale=scale
        )
        row_graph = batch_data.batch[:, None].expand(-1, 3).reshape(-1)
        solution, spectrum = weighted_affine_fit(
            design,
            noisy.coordinate_scaled_score_target.reshape(-1),
            row_graph,
            int(batch_data.num_graphs),
            rcond=float(audit["lstsq_rcond"]),
        )
        assign_affine_readout(model, solution)
        fp32_gradients, fp32_gradient_norm, fp32_loss, fp32_gradient_finite = (
            _backbone_gradients(
                model, noisy, batch_data, blueprint, use_bf16=False
            )
        )
        bf16_gradients, bf16_gradient_norm, bf16_loss, bf16_gradient_finite = (
            _backbone_gradients(
                model, noisy, batch_data, blueprint, use_bf16=True
            )
        )
        with torch.no_grad():
            fp32_prediction = _predict(
                model, noisy, batch_data, blueprint, use_bf16=False
            ).float()
            bf16_prediction = _predict(
                model, noisy, batch_data, blueprint, use_bf16=True
            ).float()
        prediction_relative_rmse = float(
            (bf16_prediction - fp32_prediction).square().mean().sqrt()
            / fp32_prediction.square().mean().sqrt().clamp_min(1e-30)
        )
        agreement = gradient_agreement(fp32_gradients, bf16_gradients)
        cancellation = _component_cancellation(
            design, solution, int(protocol["model"]["vector_dim"])
        )

    parameters_restored = all(
        torch.equal(value, model.state_dict()[name]) for name, value in initial_state.items()
    )
    acceptance = protocol["acceptance"]
    mse_ratio = bf16_loss / max(fp32_loss, 1e-30)
    exact_over_baseline_gradient = fp32_gradient_norm / max(
        baseline_gradient_norm, 1e-30
    )
    checks = {
        "function_preservation": function_preservation
        <= float(acceptance["function_preservation_max_abs"]),
        "design_reconstruction": design_reconstruction
        <= float(acceptance["design_reconstruction_max_abs"]),
        "design_rank": int(spectrum["rank"]) == int(acceptance["design_rank"]),
        "scaled_solution_norm": float(torch.linalg.vector_norm(solution))
        <= float(acceptance["scaled_solution_norm_max"]),
        "fp32_coordinate_mse": fp32_loss
        <= float(acceptance["fp32_coordinate_mse_max"]),
        "bf16_coordinate_mse": bf16_loss
        <= float(acceptance["bf16_coordinate_mse_max"]),
        "bf16_over_fp32_mse": mse_ratio
        <= float(acceptance["bf16_over_fp32_mse_ratio_max"]),
        "bf16_prediction": prediction_relative_rmse
        <= float(acceptance["bf16_prediction_relative_rmse_max"]),
        "fp32_backbone_gradient": fp32_gradient_finite
        and fp32_gradient_norm
        <= float(acceptance["fp32_backbone_gradient_norm_max"]),
        "bf16_backbone_gradient": bf16_gradient_finite
        and bf16_gradient_norm
        <= float(acceptance["bf16_backbone_gradient_norm_max"]),
        "gradient_norm_agreement": float(
            acceptance["bf16_over_fp32_gradient_norm_min"]
        )
        <= agreement["candidate_over_reference_norm"]
        <= float(acceptance["bf16_over_fp32_gradient_norm_max"]),
        "gradient_direction": agreement["cosine"]
        >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
        "parameters_restored": parameters_restored
        is bool(acceptance["parameters_restored"]),
        "sampling_failures": int(acceptance["sampling_failures"]) == 0,
        "tensor_candidates": int(acceptance["tensor_candidates"]) == 0,
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "fixed_indices": indices.tolist(),
        "checks": checks,
        "qualified": qualified,
        "function_preservation_max_abs": function_preservation,
        "design_reconstruction_max_abs": design_reconstruction,
        "spectrum": spectrum,
        "scaled_solution_norm": float(torch.linalg.vector_norm(solution)),
        "effective_unscaled_solution_norm": float(
            scale * torch.linalg.vector_norm(solution)
        ),
        "baseline": {
            "coordinate_mse": baseline_loss,
            "backbone_gradient_norm": baseline_gradient_norm,
            "gradient_finite": baseline_gradient_finite,
        },
        "scaled_exact": {
            "fp32_coordinate_mse": fp32_loss,
            "bf16_coordinate_mse": bf16_loss,
            "bf16_over_fp32_mse_ratio": mse_ratio,
            "bf16_prediction_relative_rmse": prediction_relative_rmse,
            "fp32_backbone_gradient_norm": fp32_gradient_norm,
            "bf16_backbone_gradient_norm": bf16_gradient_norm,
            "exact_over_baseline_fp32_gradient_norm": exact_over_baseline_gradient,
            "gradient_agreement": agreement,
            "fp32_gradient_finite": fp32_gradient_finite,
            "bf16_gradient_finite": bf16_gradient_finite,
            "component_cancellation": cancellation,
        },
        "optimizer_steps": 0,
        "sampling_failures": 0,
        "tensor_candidates": 0,
        "parameters_restored": parameters_restored,
        "decision": (
            "scaled_variable_projection_stability_qualified_freeze_training"
            if qualified
            else "scaled_variable_projection_stability_failed_reject_before_training"
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
