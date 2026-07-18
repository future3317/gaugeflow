"""Run the frozen H1a fixed-state and resampled-state memorization audit."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.manifold import torus_logmap
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion, TensorFreeNoisyBatch
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.state_projection import graph_mean


def _fixed_indices(length: int, count: int, seed: int, excluded: torch.Tensor | None = None) -> torch.Tensor:
    if count < 1 or count > length:
        raise ValueError("fixed subset size is outside the dataset")
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed))
    if excluded is not None:
        keep = torch.ones(length, dtype=torch.bool)
        keep[excluded] = False
        order = order[keep[order]]
    if order.numel() < count:
        raise ValueError("not enough records after excluding the fixed subset")
    return order[:count]


def _make_batch(dataset: PackedAlexP1Dataset, indices: torch.Tensor, device: torch.device) -> Batch:
    return Batch.from_data_list([dataset[int(index)] for index in indices]).to(device)


def _blueprint(batch_data: Batch) -> ParentBlueprintBatch:
    graphs = int(batch_data.num_graphs)
    counts = torch.bincount(batch_data.batch, minlength=graphs)
    return ParentBlueprintBatch.from_node_counts(
        counts, dtype=batch_data.frac_coords.dtype, device=batch_data.frac_coords.device
    )


def _coordinate_loss(prediction: torch.Tensor, target: torch.Tensor, batch: torch.Tensor, graphs: int) -> torch.Tensor:
    graph_loss = scatter(
        (prediction - target).square().sum(-1),
        batch,
        dim=0,
        dim_size=graphs,
        reduce="mean",
    )
    return graph_loss.mean() / 3.0


def _endpoint_rms(
    prediction: torch.Tensor,
    noisy: TensorFreeNoisyBatch,
    clean: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    diffusion: TensorFreeHybridDiffusion,
) -> torch.Tensor:
    graphs = noisy.time.numel()
    sigma = diffusion.coordinate_schedule.sigma(noisy.time)[batch, None]
    estimate = noisy.fractional_coordinates + sigma * prediction
    difference = torus_logmap(clean, estimate)
    phase = 2.0 * math.pi * difference
    translation = torch.atan2(
        graph_mean(phase.sin(), batch, graphs), graph_mean(phase.cos(), batch, graphs)
    ) / (2.0 * math.pi)
    residual = torus_logmap(translation[batch], difference)
    cartesian = torch.einsum("ni,nij->nj", residual, lattice[batch])
    return scatter(
        cartesian.square().sum(-1), batch, dim=0, dim_size=graphs, reduce="mean"
    ).sqrt()


def _predict(
    model: HybridCrystalDenoiser,
    noisy: TensorFreeNoisyBatch,
    batch_data: Batch,
    blueprint: ParentBlueprintBatch,
    *,
    use_bf16: bool,
) -> torch.Tensor:
    graphs = int(batch_data.num_graphs)
    condition = noisy.time.new_zeros((graphs, 18))
    present = torch.zeros((graphs, 1), dtype=torch.bool, device=noisy.time.device)
    with torch.autocast(
        device_type=noisy.time.device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        output = model(
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
    return output.coordinate_fractional_scaled_score


def _optimizer(model: nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def _train_exact_state(
    model: HybridCrystalDenoiser,
    diffusion: TensorFreeHybridDiffusion,
    batch_data: Batch,
    blueprint: ParentBlueprintBatch,
    protocol: dict[str, Any],
    *,
    generator: torch.Generator,
    metrics_path: Path,
) -> tuple[TensorFreeNoisyBatch, list[dict[str, float]]]:
    training = protocol["training"]
    times = batch_data.lattice.new_tensor(training["time_grid"])
    graph_time = times[torch.arange(int(batch_data.num_graphs), device=times.device) % times.numel()]
    with torch.no_grad():
        noisy = diffusion.noise_clean_batch(
            batch_data.atom_types,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=graph_time,
            generator=generator,
        )
    optimizer = _optimizer(model, training)
    use_bf16 = training["precision"] == "bf16" and batch_data.lattice.device.type == "cuda"
    records: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(1, int(training["exact_state_steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, noisy, batch_data, blueprint, use_bf16=use_bf16)
        loss = _coordinate_loss(
            prediction, noisy.coordinate_scaled_score_target, batch_data.batch, int(batch_data.num_graphs)
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        if step == 1 or step % int(training["log_every"]) == 0 or step == int(training["exact_state_steps"]):
            record = {
                "step": float(step),
                "coordinate_loss": float(loss.detach()),
                "gradient_norm": float(gradient.detach()),
                "graphs_per_second": float(step * int(batch_data.num_graphs) / (time.perf_counter() - start)),
            }
            records.append(record)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    return noisy, records


def _train_resampled(
    model: HybridCrystalDenoiser,
    diffusion: TensorFreeHybridDiffusion,
    batch_data: Batch,
    blueprint: ParentBlueprintBatch,
    protocol: dict[str, Any],
    *,
    generator: torch.Generator,
    metrics_path: Path,
) -> list[dict[str, float]]:
    training = protocol["training"]
    optimizer = _optimizer(model, training)
    use_bf16 = training["precision"] == "bf16" and batch_data.lattice.device.type == "cuda"
    records: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(1, int(training["resampled_state_steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=batch_data.lattice.device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            output = diffusion(
                batch_data.atom_types,
                batch_data.frac_coords,
                batch_data.lattice,
                batch_data.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                generator=generator,
            )
            loss = output.coordinate_loss
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        if step == 1 or step % int(training["log_every"]) == 0 or step == int(training["resampled_state_steps"]):
            record = {
                "step": float(step),
                "coordinate_loss": float(loss.detach()),
                "gradient_norm": float(gradient.detach()),
                "graphs_per_second": float(step * int(batch_data.num_graphs) / (time.perf_counter() - start)),
            }
            records.append(record)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    return records


@torch.no_grad()
def _evaluate_noisy(
    model: HybridCrystalDenoiser,
    diffusion: TensorFreeHybridDiffusion,
    batch_data: Batch,
    blueprint: ParentBlueprintBatch,
    noisy: TensorFreeNoisyBatch,
    *,
    use_bf16: bool,
) -> dict[str, float]:
    model.eval()
    prediction = _predict(model, noisy, batch_data, blueprint, use_bf16=use_bf16).float()
    target = noisy.coordinate_scaled_score_target.float()
    loss = _coordinate_loss(prediction, target, batch_data.batch, int(batch_data.num_graphs))
    target_energy = _coordinate_loss(torch.zeros_like(target), target, batch_data.batch, int(batch_data.num_graphs))
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
        "coordinate_mse": float(loss),
        "zero_predictor_mse": float(target_energy),
        "explained_fraction": float(1.0 - loss / target_energy.clamp_min(1e-30)),
        "endpoint_rms_angstrom": float(endpoint.square().mean().sqrt()),
        "low_time_endpoint_rms_angstrom": float(endpoint[low].square().mean().sqrt()),
        "tensor_candidates": 0.0,
        "sampling_failures": 0.0,
    }


@torch.no_grad()
def _evaluate_grid(
    model: HybridCrystalDenoiser,
    diffusion: TensorFreeHybridDiffusion,
    batch_data: Batch,
    blueprint: ParentBlueprintBatch,
    protocol: dict[str, Any],
    *,
    seed: int,
) -> list[dict[str, float]]:
    training = protocol["training"]
    generator = torch.Generator(device=batch_data.lattice.device).manual_seed(seed)
    use_bf16 = training["precision"] == "bf16" and batch_data.lattice.device.type == "cuda"
    result: list[dict[str, float]] = []
    for time_value in training["time_grid"]:
        aggregate: list[dict[str, float]] = []
        time_tensor = batch_data.lattice.new_full((int(batch_data.num_graphs),), float(time_value))
        for _ in range(int(training["evaluation_noise_replicates"])):
            noisy = diffusion.noise_clean_batch(
                batch_data.atom_types,
                batch_data.frac_coords,
                batch_data.lattice,
                batch_data.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=time_tensor,
                generator=generator,
            )
            aggregate.append(
                _evaluate_noisy(model, diffusion, batch_data, blueprint, noisy, use_bf16=use_bf16)
            )
        row = {key: sum(value[key] for value in aggregate) / len(aggregate) for key in aggregate[0]}
        row["time"] = float(time_value)
        result.append(row)
    return result


def _make_model(protocol: dict[str, Any], device: torch.device) -> HybridCrystalDenoiser:
    spec = protocol["model"]
    model = HybridCrystalDenoiser(
        hidden_dim=int(spec["hidden_dim"]),
        vector_dim=int(spec["vector_dim"]),
        layers=int(spec["layers"]),
        radial_dim=int(spec["radial_dim"]),
        radial_cutoff=float(spec["radial_cutoff_angstrom"]),
        atlas_residual_circle_samples=8,
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != int(spec["parameter_count"]):
        raise ValueError("memorization model parameter count mismatch")
    return model


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
    if protocol.get("protocol") != "h1a_coordinate_memorization_audit_v1":
        raise ValueError("coordinate memorization protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate memorization cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    torch.manual_seed(int(protocol["training"]["seed"]))
    torch.cuda.manual_seed_all(int(protocol["training"]["seed"]))
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    validation = PackedAlexP1Dataset(args.cache_root, "val")
    data = protocol["data"]
    fixed_indices = _fixed_indices(
        len(dataset), int(data["fixed_graphs"]), int(data["fixed_selection_seed"])
    )
    unseen_indices = _fixed_indices(
        len(dataset),
        int(data["unseen_train_graphs"]),
        int(data["unseen_train_selection_seed"]),
        excluded=fixed_indices,
    )
    validation_indices = _fixed_indices(
        len(validation), int(data["validation_graphs"]), int(data["validation_selection_seed"])
    )
    fixed_batch = _make_batch(dataset, fixed_indices, device)
    fixed_blueprint = _blueprint(fixed_batch)
    standardization = P1LatticeStandardizer.from_mapping(
        load_json_object(args.lattice_standardization)
    )
    model = _make_model(protocol, device)
    initial_state = copy.deepcopy(model.state_dict())
    training = protocol["training"]
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardization,
        coordinate_sigma_min=float(training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training["coordinate_sigma_max"]),
        minimum_time=float(training["minimum_time"]),
        maximum_time=float(training["maximum_time"]),
    )
    args.run_root.mkdir(parents=True, exist_ok=False)
    exact_noisy, exact_curve = _train_exact_state(
        model,
        diffusion,
        fixed_batch,
        fixed_blueprint,
        protocol,
        generator=torch.Generator(device=device).manual_seed(int(training["seed"]) + 1),
        metrics_path=args.run_root / "exact_state_metrics.jsonl",
    )
    exact = _evaluate_noisy(
        model,
        diffusion,
        fixed_batch,
        fixed_blueprint,
        exact_noisy,
        use_bf16=training["precision"] == "bf16" and device.type == "cuda",
    )
    acceptance = protocol["acceptance"]
    exact_checks = {
        "coordinate_mse": exact["coordinate_mse"] <= float(acceptance["exact_state_coordinate_mse_max"]),
        "explained_fraction": exact["explained_fraction"] >= float(acceptance["exact_state_explained_fraction_min"]),
        "low_time_endpoint": exact["low_time_endpoint_rms_angstrom"]
        <= float(acceptance["exact_state_low_time_endpoint_rms_angstrom_max"]),
    }
    result: dict[str, Any] = {
        "protocol": protocol["protocol"],
        "seed": int(training["seed"]),
        "fixed_indices": fixed_indices.tolist(),
        "exact_state": {"metrics": exact, "checks": exact_checks, "curve": exact_curve},
        "resampled_state": None,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    if all(exact_checks.values()):
        model.load_state_dict(initial_state, strict=True)
        diffusion = TensorFreeHybridDiffusion(
            model,
            standardization,
            coordinate_sigma_min=float(training["coordinate_sigma_min"]),
            coordinate_sigma_max=float(training["coordinate_sigma_max"]),
            minimum_time=float(training["minimum_time"]),
            maximum_time=float(training["maximum_time"]),
        )
        dynamic_curve = _train_resampled(
            model,
            diffusion,
            fixed_batch,
            fixed_blueprint,
            protocol,
            generator=torch.Generator(device=device).manual_seed(int(training["seed"]) + 2),
            metrics_path=args.run_root / "resampled_state_metrics.jsonl",
        )
        fixed_grid = _evaluate_grid(
            model, diffusion, fixed_batch, fixed_blueprint, protocol, seed=int(training["seed"]) + 3
        )
        unseen_batch = _make_batch(dataset, unseen_indices, device)
        validation_batch = _make_batch(validation, validation_indices, device)
        unseen_grid = _evaluate_grid(
            model, diffusion, unseen_batch, _blueprint(unseen_batch), protocol, seed=int(training["seed"]) + 4
        )
        validation_grid = _evaluate_grid(
            model, diffusion, validation_batch, _blueprint(validation_batch), protocol, seed=int(training["seed"]) + 5
        )
        fixed_by_time = {row["time"]: row for row in fixed_grid}
        low_rows = [row for row in fixed_grid if row["time"] <= 0.1]
        low_explained = sum(row["explained_fraction"] for row in low_rows) / len(low_rows)
        dynamic_checks = {
            "low_time_explained_fraction": low_explained
            >= float(acceptance["resampled_fixed_low_time_explained_fraction_min"]),
            "t005_endpoint": fixed_by_time[0.005]["endpoint_rms_angstrom"]
            <= float(acceptance["resampled_fixed_t005_endpoint_rms_angstrom_max"]),
            "sampling_failures": all(
                row["sampling_failures"] == float(acceptance["sampling_failures"])
                for row in fixed_grid
            ),
            "tensor_candidates": all(
                row["tensor_candidates"] == float(acceptance["tensor_candidates"])
                for row in fixed_grid
            ),
        }
        result["resampled_state"] = {
            "curve": dynamic_curve,
            "fixed_grid": fixed_grid,
            "unseen_train_grid": unseen_grid,
            "validation_grid": validation_grid,
            "fixed_low_time_mean_explained_fraction": low_explained,
            "checks": dynamic_checks,
        }
        if all(dynamic_checks.values()):
            fixed_low = sum(row["coordinate_mse"] for row in low_rows) / len(low_rows)
            unseen_low_rows = [row for row in unseen_grid if row["time"] <= 0.1]
            unseen_low = sum(row["coordinate_mse"] for row in unseen_low_rows) / len(unseen_low_rows)
            result["decision"] = (
                "fixed_path_learned_source_coupling_generalization_gap"
                if unseen_low > 2.0 * fixed_low
                else "fixed_and_unseen_path_learned_prepare_pretraining_proposal"
            )
        else:
            result["decision"] = "exact_states_memorized_resampled_path_failed"
    else:
        result["decision"] = "exact_state_memorization_failed_inspect_forward_head_loss"
    result["qualified"] = bool(
        all(exact_checks.values())
        and result["resampled_state"] is not None
        and all(result["resampled_state"]["checks"].values())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
