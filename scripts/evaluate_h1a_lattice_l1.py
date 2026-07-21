"""Evaluate the frozen coordinate-free H1a lattice L1 protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Element
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape, project_lattice_state
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.runtime import load_tensor_free_ema_runtime


def _wasserstein(left: torch.Tensor, right: torch.Tensor, points: int) -> float:
    probabilities = torch.linspace(0.0, 1.0, points, dtype=torch.float64)
    return float(
        (torch.quantile(left.double(), probabilities) - torch.quantile(right.double(), probabilities)).abs().mean()
    )


def _normalized_wasserstein(left: torch.Tensor, right: torch.Tensor, points: int) -> float:
    scale = torch.quantile(right.double(), 0.75) - torch.quantile(right.double(), 0.25)
    if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
        raise ValueError("reference IQR must be finite and positive")
    return _wasserstein(left, right, points) / float(scale)


def _atomic_mass_table(reference: torch.Tensor) -> torch.Tensor:
    values = [float(Element.from_Z(number).atomic_mass) for number in range(1, 119)]
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def _lattice_statistics(
    element_tokens: torch.Tensor,
    batch: torch.Tensor,
    lattice: torch.Tensor,
    shape_latent: torch.Tensor,
) -> dict[str, torch.Tensor]:
    graphs = lattice.shape[0]
    counts = torch.bincount(batch, minlength=graphs)
    volume = torch.linalg.det(lattice)
    mass = torch.zeros(graphs, dtype=lattice.dtype, device=lattice.device)
    mass.index_add_(0, batch, _atomic_mass_table(lattice)[element_tokens])
    lengths = torch.linalg.vector_norm(lattice, dim=-1)
    normalized = lattice / lengths.unsqueeze(-1)
    cosines = torch.stack(
        (
            (normalized[:, 1] * normalized[:, 2]).sum(dim=-1),
            (normalized[:, 0] * normalized[:, 2]).sum(dim=-1),
            (normalized[:, 0] * normalized[:, 1]).sum(dim=-1),
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosines))
    return {
        "volume_per_atom": volume / counts,
        "density_g_cm3": 1.66053906660 * mass / volume,
        "shape_latent": shape_latent,
        "condition_number": torch.linalg.cond(lattice),
        "angles_degree": angles,
        "lengths_angstrom": lengths,
    }


def _append_statistics(
    destination: dict[str, list[torch.Tensor]],
    source: dict[str, torch.Tensor],
) -> None:
    for name, value in source.items():
        destination.setdefault(name, []).append(value.detach().cpu())


def _concatenate_statistics(values: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.cat(items, dim=0) for name, items in values.items()}


def _reference_envelope_fraction(
    generated: torch.Tensor,
    reference: torch.Tensor,
    probabilities: torch.Tensor,
) -> float:
    bounds = torch.quantile(reference.double(), probabilities)
    return float(((generated >= bounds[0]) & (generated <= bounds[1])).double().mean())


def _load_panel(
    dataset: PackedAlexP1Dataset,
    count: int,
    seed: int,
) -> torch.Tensor:
    if count > len(dataset):
        raise ValueError("evaluation panel exceeds the selected dataset split")
    return torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count]


@torch.no_grad()
def _teacher_forced_metrics(
    runtime: Any,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    protocol: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
        categorical_path=str(runtime.training_config["categorical_path"]),
    )
    times = [float(value) for value in evaluation["teacher_times"]]
    totals = {
        value: {"volume_error": 0.0, "shape_error": 0.0, "volume_base": 0.0, "shape_base": 0.0, "graphs": 0}
        for value in times
    }
    generators = {
        value: torch.Generator(device=device).manual_seed(int(evaluation["teacher_noise_seed"]) + index)
        for index, value in enumerate(times)
    }
    for start in range(0, indices.numel(), int(evaluation["sampling_batch_size"])):
        selected = indices[start : start + int(evaluation["sampling_batch_size"])]
        packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
        graphs = int(packed.num_graphs)
        counts = torch.bincount(packed.batch, minlength=graphs)
        blueprint = ParentBlueprintBatch.from_node_counts(
            counts,
            dtype=packed.lattice.dtype,
            device=device,
        )
        for value in times:
            output = diffusion.forward_lattice(
                packed.atom_types,
                packed.lattice,
                packed.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                lattice_time=torch.full((graphs,), value, dtype=packed.lattice.dtype, device=device),
                generator=generators[value],
            )
            record = totals[value]
            record["volume_error"] += float(
                (output.prediction.clean_volume_latent - output.noisy.clean_volume_latent_target).square().sum()
            )
            record["shape_error"] += float(
                (output.prediction.clean_shape_latent - output.noisy.clean_shape_latent_target).square().sum()
            )
            record["volume_base"] += float(output.noisy.clean_volume_latent_target.square().sum())
            record["shape_base"] += float(output.noisy.clean_shape_latent_target.square().sum())
            record["graphs"] += graphs
    by_time: dict[str, dict[str, float]] = {}
    volume_error = shape_error = volume_base = shape_base = 0.0
    for value, record in totals.items():
        volume_error += float(record["volume_error"])
        shape_error += float(record["shape_error"])
        volume_base += float(record["volume_base"])
        shape_base += float(record["shape_base"])
        by_time[f"{value:.1f}"] = {
            "volume_mse": float(record["volume_error"]) / int(record["graphs"]),
            "shape_mse": float(record["shape_error"]) / (5 * int(record["graphs"])),
            "volume_mse_ratio": float(record["volume_error"]) / float(record["volume_base"]),
            "shape_mse_ratio": float(record["shape_error"]) / float(record["shape_base"]),
        }
    return {
        "by_time": by_time,
        "aggregate_volume_mse_ratio": volume_error / volume_base,
        "aggregate_shape_mse_ratio": shape_error / shape_base,
    }


@torch.no_grad()
def _sampling_metrics(
    runtime: Any,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    protocol: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    evaluation = protocol["evaluation"]
    batch_size = int(evaluation["sampling_batch_size"])
    sampler = TensorFreeReverseSampler(
        runtime.model,
        runtime.lattice_standardizer,
        maximum_time=float(runtime.training_config["maximum_time"]),
        categorical_path=str(runtime.training_config["categorical_path"]),
    )
    reference_lists: dict[str, list[torch.Tensor]] = {}
    generated_lists: dict[str, list[torch.Tensor]] = {}
    seed = int(evaluation["sampling_seed"])
    initialization_generator = torch.Generator(device=device).manual_seed(seed)
    continuous_generator = torch.Generator(device=device).manual_seed(seed + 1)
    failures = 0
    finite_positive = 0
    forward_calls = 0
    atlas_calls = 0

    def count_forward(_module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        nonlocal forward_calls
        forward_calls += 1

    def count_atlas(_module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        nonlocal atlas_calls
        atlas_calls += 1

    forward_handle = runtime.model.register_forward_hook(count_forward)
    atlas_handle = runtime.model.gauge_atlas.register_forward_hook(count_atlas)
    start_time = time.perf_counter()
    try:
        for start in range(0, indices.numel(), batch_size):
            selected = indices[start : start + batch_size]
            packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
            graphs = int(packed.num_graphs)
            counts = torch.bincount(packed.batch, minlength=graphs)
            blueprint = ParentBlueprintBatch.from_node_counts(
                counts,
                dtype=packed.lattice.dtype,
                device=device,
            )
            reference_state = LatticeVolumeShape.from_lattice(
                packed.lattice,
                blueprint.fractional_to_cartesian,
            )
            reference_shape = project_lattice_state(
                reference_state.log_shape,
                blueprint.shape_projector,
            )
            _append_statistics(
                reference_lists,
                _lattice_statistics(
                    packed.atom_types,
                    packed.batch,
                    packed.lattice,
                    runtime.lattice_standardizer.encode_shape(reference_shape),
                ),
            )
            try:
                generated = sampler.sample_lattice(
                    packed.atom_types,
                    blueprint,
                    steps=int(evaluation["sampling_steps"]),
                    initialization_generator=initialization_generator,
                    continuous_generator=continuous_generator,
                    continuous_mode=str(evaluation["continuous_mode"]),
                    time_grid=str(evaluation["time_grid"]),
                )
            except SamplingFailure:
                failures += graphs
                continue
            determinant = torch.linalg.det(generated.lattice)
            finite = torch.isfinite(generated.lattice).all(dim=(-2, -1)) & (determinant > 0.0)
            finite_positive += int(finite.sum())
            _append_statistics(
                generated_lists,
                _lattice_statistics(
                    packed.atom_types,
                    packed.batch,
                    generated.lattice,
                    runtime.lattice_standardizer.encode_shape(generated.log_shape),
                ),
            )
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        forward_handle.remove()
        atlas_handle.remove()
    reference = _concatenate_statistics(reference_lists)
    generated = _concatenate_statistics(generated_lists)
    points = int(evaluation["wasserstein_quantile_points"])
    envelope_probabilities = torch.tensor(
        evaluation["reference_envelope_quantiles"],
        dtype=torch.float64,
    )
    shape_w1 = [
        _normalized_wasserstein(generated["shape_latent"][:, index], reference["shape_latent"][:, index], points)
        for index in range(5)
    ]
    envelope_fractions = {
        name: _reference_envelope_fraction(generated[name], reference[name], envelope_probabilities)
        for name in ("volume_per_atom", "density_g_cm3", "condition_number")
    }
    metrics = {
        "graphs": int(indices.numel()),
        "sampling_failures": failures,
        "finite_positive_lattice_fraction": finite_positive / int(indices.numel()),
        "normalized_volume_per_atom_w1": _normalized_wasserstein(
            generated["volume_per_atom"], reference["volume_per_atom"], points
        ),
        "normalized_density_w1": _normalized_wasserstein(
            generated["density_g_cm3"], reference["density_g_cm3"], points
        ),
        "normalized_shape_latent_w1": shape_w1,
        "mean_normalized_shape_latent_w1": sum(shape_w1) / len(shape_w1),
        "reference_envelope_fractions": envelope_fractions,
        "reference_envelope_fraction": min(envelope_fractions.values()),
        "graphs_per_second": int(indices.numel()) / elapsed,
        "full_geometry_forward_calls": forward_calls,
        "tensor_atlas_calls": atlas_calls,
        "generated_quantiles": {
            name: torch.quantile(
                value.double().reshape(-1),
                torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999], dtype=torch.float64),
            ).tolist()
            for name, value in generated.items()
            if name != "shape_latent"
        },
        "reference_quantiles": {
            name: torch.quantile(
                value.double().reshape(-1),
                torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999], dtype=torch.float64),
            ).tolist()
            for name, value in reference.items()
            if name != "shape_latent"
        },
    }
    return metrics, reference, generated


@torch.no_grad()
def _deterministic_replay(
    runtime: Any,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    protocol: dict[str, Any],
    device: torch.device,
) -> float:
    evaluation = protocol["evaluation"]
    selected = indices[: min(64, indices.numel())]
    packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
    graphs = int(packed.num_graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        torch.bincount(packed.batch, minlength=graphs),
        dtype=packed.lattice.dtype,
        device=device,
    )
    sampler = TensorFreeReverseSampler(
        runtime.model,
        runtime.lattice_standardizer,
        maximum_time=float(runtime.training_config["maximum_time"]),
        categorical_path=str(runtime.training_config["categorical_path"]),
    )
    outputs = []
    for _ in range(2):
        seed = int(evaluation["sampling_seed"]) + 100
        outputs.append(
            sampler.sample_lattice(
                packed.atom_types,
                blueprint,
                steps=int(evaluation["sampling_steps"]),
                initialization_generator=torch.Generator(device=device).manual_seed(seed),
                continuous_generator=torch.Generator(device=device).manual_seed(seed + 1),
                continuous_mode=str(evaluation["continuous_mode"]),
                time_grid=str(evaluation["time_grid"]),
            ).lattice
        )
    return float((outputs[0] - outputs[1]).abs().max())


def _training_execution(run_root: Path) -> dict[str, float]:
    records = [
        json.loads(line)
        for line in (run_root / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("training metrics are empty")
    steady = records[1:] if len(records) > 1 else records
    throughput = torch.tensor([float(record["graphs_per_second"]) for record in steady])
    return {
        "median_graphs_per_second": float(throughput.median()),
        "peak_cuda_memory_mib": max(float(record["peak_cuda_memory_mib"]) for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    protocol = load_json_object(arguments.protocol)
    if protocol.get("protocol") != "h1a_lattice_l1_v1" or protocol.get("status_before_run") != "frozen_not_run":
        raise ValueError("unexpected or unfrozen L1 protocol")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    runtime = load_tensor_free_ema_runtime(
        arguments.checkpoint,
        device,
        protocol_name=str(protocol["protocol"]),
        protocol_sha256=canonical_json_hash(protocol),
    )
    declared_split = str(protocol["evaluation"]["split"])
    cache_split = {"validation": "val"}.get(declared_split)
    if cache_split is None:
        raise ValueError("L1 protocol must use the independent validation split")
    dataset = PackedAlexP1Dataset(arguments.cache_root, cache_split)
    indices = _load_panel(
        dataset,
        int(protocol["evaluation"]["graphs"]),
        int(protocol["evaluation"]["panel_seed"]),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    teacher = _teacher_forced_metrics(runtime, dataset, indices, protocol, device)
    sampling, _, _ = _sampling_metrics(runtime, dataset, indices, protocol, device)
    replay = _deterministic_replay(runtime, dataset, indices, protocol, device)
    execution = _training_execution(arguments.run_root)
    execution["evaluation_peak_cuda_memory_mib"] = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0**2) if device.type == "cuda" else 0.0
    )
    acceptance = protocol["acceptance"]
    checks = {
        "teacher_volume": teacher["aggregate_volume_mse_ratio"] <= float(acceptance["teacher_volume_mse_ratio_max"]),
        "teacher_shape": teacher["aggregate_shape_mse_ratio"] <= float(acceptance["teacher_shape_mse_ratio_max"]),
        "teacher_t05_volume": teacher["by_time"]["0.5"]["volume_mse_ratio"]
        <= float(acceptance["teacher_t05_volume_mse_ratio_max"]),
        "teacher_t05_shape": teacher["by_time"]["0.5"]["shape_mse_ratio"]
        <= float(acceptance["teacher_t05_shape_mse_ratio_max"]),
        "volume_distribution": sampling["normalized_volume_per_atom_w1"]
        <= float(acceptance["normalized_volume_per_atom_w1_max"]),
        "density_distribution": sampling["normalized_density_w1"] <= float(acceptance["normalized_density_w1_max"]),
        "shape_distribution": sampling["mean_normalized_shape_latent_w1"]
        <= float(acceptance["mean_normalized_shape_latent_w1_max"]),
        "reference_envelope": sampling["reference_envelope_fraction"]
        >= float(acceptance["reference_envelope_fraction_min"]),
        "finite_positive": sampling["finite_positive_lattice_fraction"]
        >= float(acceptance["finite_positive_lattice_fraction"]),
        "sampling_failures": sampling["sampling_failures"] == int(acceptance["sampling_failures"]),
        "deterministic_replay": replay <= float(acceptance["deterministic_replay_max_abs"]),
        "training_throughput": execution["median_graphs_per_second"]
        >= float(acceptance["training_graphs_per_second_min"]),
        "training_memory": execution["peak_cuda_memory_mib"] <= float(acceptance["peak_cuda_memory_mib_max"]),
        "no_coordinate_forward": sampling["full_geometry_forward_calls"]
        == int(acceptance["target_coordinate_forward_calls"]),
        "no_tensor_atlas": sampling["tensor_atlas_calls"] == int(acceptance["tensor_candidates"]),
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "code_commit": protocol["code_commit"],
        "teacher_forced": teacher,
        "sampling": sampling,
        "deterministic_replay_max_abs": replay,
        "execution": execution,
        "checks": checks,
        "qualified": qualified,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not qualified:
        raise RuntimeError("L1 failed its frozen acceptance checks")


if __name__ == "__main__":
    main()
