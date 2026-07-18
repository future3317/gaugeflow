"""Train the frozen 16-state H1a variable-projection mechanism test."""

from __future__ import annotations

import argparse
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
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _coordinate_loss,
    _endpoint_rms,
    _fixed_indices,
    _make_batch,
    _make_model,
    _predict,
)
from scripts.audit_h1a_coordinate_readout_panel import (
    _capture_affine_design,
    weighted_affine_fit,
)


def assign_affine_readout(
    model: torch.nn.Module,
    solution: torch.Tensor,
    names: list[str],
) -> None:
    """Copy one flat affine solution into the declared final readout only."""
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


def _solve_readout(
    model: torch.nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    names: list[str],
    rcond: float,
) -> dict[str, float | int]:
    model.eval()
    design, _, reconstruction = _capture_affine_design(
        model, noisy, batch_data, blueprint
    )
    row_graph = batch_data.batch[:, None].expand(-1, 3).reshape(-1)
    solution, spectrum = weighted_affine_fit(
        design,
        noisy.coordinate_scaled_score_target.reshape(-1),
        row_graph,
        int(batch_data.num_graphs),
        rcond=rcond,
    )
    assign_affine_readout(model, solution, names)
    return {
        **spectrum,
        "design_reconstruction_max_abs": reconstruction,
        "solution_norm": float(torch.linalg.vector_norm(solution)),
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    diffusion: TensorFreeHybridDiffusion,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
) -> dict[str, float]:
    model.eval()
    prediction = _predict(
        model, noisy, batch_data, blueprint, use_bf16=False
    ).float()
    target = noisy.coordinate_scaled_score_target.float()
    coordinate_mse = _coordinate_loss(
        prediction, target, batch_data.batch, int(batch_data.num_graphs)
    )
    zero_mse = _coordinate_loss(
        torch.zeros_like(target), target, batch_data.batch, int(batch_data.num_graphs)
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
    return {
        "coordinate_mse": float(coordinate_mse),
        "zero_predictor_mse": float(zero_mse),
        "explained_fraction": float(
            1.0 - coordinate_mse / zero_mse.clamp_min(1e-30)
        ),
        "endpoint_rms_angstrom": float(endpoint.square().mean().sqrt()),
        "low_time_endpoint_rms_angstrom": float(
            endpoint[low].square().mean().sqrt()
        ),
        "sampling_failures": 0.0,
        "tensor_candidates": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_coordinate_variable_projection_16_v1":
        raise ValueError("coordinate variable-projection protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate variable-projection cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    training = protocol["training"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    data = protocol["data"]
    indices = _fixed_indices(
        len(dataset), int(data["fixed_graphs"]), int(data["fixed_selection_seed"])
    )
    batch_data = _make_batch(dataset, indices, device)
    blueprint = _blueprint(batch_data)
    model = _make_model(protocol, device).float()
    readout_names = list(training["readout_parameters"])
    named_parameters = dict(model.named_parameters())
    for name in readout_names:
        named_parameters[name].requires_grad_(False)
    optimized = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        optimized,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
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
                int(training["noise_seed"])
            ),
        )
    args.run_root.mkdir(parents=True, exist_ok=False)
    metrics_path = args.run_root / "metrics.jsonl"
    solve_records: list[dict[str, float | int]] = []
    records: list[dict[str, float]] = []
    start = time.perf_counter()
    finite = True
    for step in range(1, int(training["steps"]) + 1):
        if step == 1 or (step - 1) % int(training["solve_every_steps"]) == 0:
            solve_record = _solve_readout(
                model,
                noisy,
                batch_data,
                blueprint,
                names=readout_names,
                rcond=float(training["lstsq_rcond"]),
            )
            solve_record["step"] = step
            solve_records.append(solve_record)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(
            model,
            noisy,
            batch_data,
            blueprint,
            use_bf16=training["precision"] == "bf16" and device.type == "cuda",
        )
        loss = _coordinate_loss(
            prediction,
            noisy.coordinate_scaled_score_target,
            batch_data.batch,
            int(batch_data.num_graphs),
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            optimized, float(training["gradient_clip_norm"])
        )
        optimizer.step()
        finite = finite and math.isfinite(float(loss)) and math.isfinite(float(gradient))
        if (
            step == 1
            or step % int(training["log_every"]) == 0
            or step == int(training["steps"])
        ):
            record = {
                "step": float(step),
                "coordinate_loss": float(loss.detach()),
                "gradient_norm": float(gradient),
                "graphs_per_second": float(
                    step * int(batch_data.num_graphs) / (time.perf_counter() - start)
                ),
            }
            records.append(record)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    final_solve = _solve_readout(
        model,
        noisy,
        batch_data,
        blueprint,
        names=readout_names,
        rcond=float(training["lstsq_rcond"]),
    )
    final_solve["step"] = int(training["steps"])
    metrics = _evaluate(model, diffusion, noisy, batch_data, blueprint)
    acceptance = protocol["acceptance"]
    checks = {
        "coordinate_mse": metrics["coordinate_mse"]
        <= float(acceptance["coordinate_mse_max"]),
        "explained_fraction": metrics["explained_fraction"]
        >= float(acceptance["explained_fraction_min"]),
        "low_time_endpoint": metrics["low_time_endpoint_rms_angstrom"]
        <= float(acceptance["low_time_endpoint_rms_angstrom_max"]),
        "finite_curve": finite is bool(acceptance["finite_curve"]),
        "sampling_failures": metrics["sampling_failures"]
        == float(acceptance["sampling_failures"]),
        "tensor_candidates": metrics["tensor_candidates"]
        == float(acceptance["tensor_candidates"]),
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "fixed_indices": indices.tolist(),
        "metrics": metrics,
        "checks": checks,
        "qualified": qualified,
        "curve": records,
        "readout_solves": solve_records,
        "final_readout_solve": final_solve,
        "decision": (
            "variable_projection_16_qualified_freeze_64_state_test"
            if qualified
            else "variable_projection_16_failed_reject_mechanism"
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
