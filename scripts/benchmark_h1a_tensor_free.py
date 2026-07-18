"""Run the corrected train-distribution and held-out H1a benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from gaugeflow.file_utils import load_json_object
from gaugeflow.geometry import periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.runtime import load_tensor_free_ema_runtime


def _minimum_distances(
    fractional_coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    edges = periodic_radius_multigraph(
        fractional_coordinates, lattice, batch, cutoff=8.0
    )
    if edges.target.numel() == 0:
        return lattice.new_full((lattice.shape[0],), math.inf)
    return scatter(
        edges.distance,
        batch[edges.target],
        dim=0,
        dim_size=lattice.shape[0],
        reduce="min",
    )


def _formula_keys(tokens: torch.Tensor, batch: torch.Tensor, graphs: int) -> list[str]:
    counts = torch.zeros((graphs, 118), dtype=torch.int32, device=tokens.device)
    counts.index_put_((batch, tokens), torch.ones_like(tokens, dtype=torch.int32), accumulate=True)
    rows = counts.cpu().tolist()
    return [
        ";".join(f"{index + 1}:{count}" for index, count in enumerate(row) if count)
        for row in rows
    ]


def _histogram(values: torch.Tensor, classes: int) -> torch.Tensor:
    return torch.bincount(values.long(), minlength=classes).double()


def _jsd(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)
    left_term = torch.where(left > 0.0, left * (left / middle).log(), 0.0)
    right_term = torch.where(right > 0.0, right * (right / middle).log(), 0.0)
    return float(0.5 * (left_term.sum() + right_term.sum()))


def _wasserstein(left: torch.Tensor, right: torch.Tensor, points: int) -> float:
    probabilities = torch.linspace(0.0, 1.0, points, dtype=torch.float64)
    return float(
        (torch.quantile(left.double(), probabilities) - torch.quantile(right.double(), probabilities))
        .abs()
        .mean()
    )


def _reference_statistics(
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    element_histogram = torch.zeros(118, dtype=torch.float64)
    node_counts: list[torch.Tensor] = []
    volumes: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    for start in range(0, indices.numel(), 64):
        selected = indices[start : start + 64]
        packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
        graphs = int(packed.num_graphs)
        counts = torch.bincount(packed.batch, minlength=graphs)
        element_histogram += _histogram(packed.atom_types.cpu(), 118)
        node_counts.append(counts.cpu())
        determinant = torch.linalg.det(packed.lattice)
        volumes.append((determinant / counts).cpu())
        distances.append(
            _minimum_distances(packed.frac_coords, packed.lattice, packed.batch).cpu()
        )
    return {
        "element_histogram": element_histogram,
        "node_counts": torch.cat(node_counts),
        "volume_per_atom": torch.cat(volumes),
        "minimum_distance": torch.cat(distances),
    }


def _reference_formula_set(
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> set[str]:
    formulas: set[str] = set()
    for start in range(0, indices.numel(), 64):
        selected = indices[start : start + 64]
        packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
        formulas.update(
            _formula_keys(packed.atom_types, packed.batch, int(packed.num_graphs))
        )
    return formulas


@torch.no_grad()
def _sample_seed(
    checkpoint: Path,
    protocol: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=str(protocol["checkpoint_protocol"]),
        protocol_sha256=str(protocol["checkpoint_protocol_sha256"]),
    )
    specification = protocol["sampling"]
    samples = int(specification["samples_per_seed"])
    count_generator = torch.Generator().manual_seed(int(specification["seed"]) + seed)
    counts = runtime.node_count_prior.sample(samples, generator=count_generator)
    sampler = TensorFreeReverseSampler(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_fractional_sigma_max=float(
            runtime.training_config["coordinate_fractional_sigma_max"]
        ),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    generator = torch.Generator(device=device).manual_seed(
        int(specification["seed"]) + seed + 1
    )
    element_histogram = torch.zeros(118, dtype=torch.float64)
    volumes: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    formulas: list[str] = []
    masks = 0
    failures = 0
    finite_positive = 0
    for start in range(0, samples, int(specification["batch_size"])):
        selected_counts = counts[start : start + int(specification["batch_size"])].to(device)
        blueprint = ParentBlueprintBatch.from_node_counts(
            selected_counts, dtype=torch.float32, device=device
        )
        try:
            generated = sampler.sample(
                blueprint,
                steps=int(specification["steps"]),
                generator=generator,
                stochastic=bool(specification["stochastic"]),
                time_grid=str(specification["time_grid"]),
            )
        except SamplingFailure:
            failures += int(selected_counts.numel())
            continue
        masks += int(generated.diagnostics.masked_count[-1])
        determinant = torch.linalg.det(generated.lattice)
        finite = torch.isfinite(generated.lattice).all(dim=(-2, -1)) & (determinant > 0.0)
        finite_positive += int(finite.sum())
        element_histogram += _histogram(generated.element_tokens.cpu(), 118)
        volumes.append((determinant / selected_counts).cpu())
        distances.append(
            _minimum_distances(
                generated.fractional_coordinates,
                generated.lattice,
                generated.batch,
            ).cpu()
        )
        formulas.extend(
            _formula_keys(generated.element_tokens, generated.batch, int(selected_counts.numel()))
        )
    return {
        "samples": samples,
        "sampling_failures": failures,
        "terminal_masks": masks,
        "finite_positive_lattices": finite_positive,
        "element_histogram": element_histogram,
        "node_counts": counts,
        "volume_per_atom": torch.cat(volumes),
        "minimum_distance": torch.cat(distances),
        "formulas": formulas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    protocol = load_json_object(arguments.protocol)
    if protocol.get("protocol") != "h1a_tensor_free_benchmark_v2":
        raise ValueError("unexpected tensor-free benchmark protocol")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    distribution_spec = protocol["distribution_reference"]
    distribution_dataset = PackedAlexP1Dataset(
        arguments.cache_root, str(distribution_spec["split"])
    )
    distribution_indices = torch.randperm(
        len(distribution_dataset),
        generator=torch.Generator().manual_seed(int(distribution_spec["seed"])),
    )[: int(distribution_spec["graphs"])]
    reference = _reference_statistics(
        distribution_dataset, distribution_indices, device=device
    )
    novelty_spec = protocol["novelty_reference"]
    novelty_dataset = PackedAlexP1Dataset(
        arguments.cache_root, str(novelty_spec["split"])
    )
    novelty_indices = torch.randperm(
        len(novelty_dataset),
        generator=torch.Generator().manual_seed(int(novelty_spec["seed"])),
    )[: int(novelty_spec["graphs"])]
    novelty_formula_set = _reference_formula_set(
        novelty_dataset, novelty_indices, device=device
    )
    generated = {
        str(seed): _sample_seed(
            arguments.run_root
            / f"seed_{int(seed)}"
            / f"checkpoint_step_{int(protocol['checkpoint_step']):08d}.pt",
            protocol,
            seed=int(seed),
            device=device,
        )
        for seed in protocol["seeds"]
    }
    metrics = protocol["metrics"]
    threshold = float(metrics["minimum_distance_angstrom"])
    quantile_points = int(metrics["wasserstein_quantile_points"])
    all_elements = sum((value["element_histogram"] for value in generated.values()))
    all_counts = torch.cat([value["node_counts"] for value in generated.values()])
    all_volumes = torch.cat([value["volume_per_atom"] for value in generated.values()])
    all_distances = torch.cat([value["minimum_distance"] for value in generated.values()])
    all_formulas = sum((value["formulas"] for value in generated.values()), [])
    reference_count_classes = max(int(reference["node_counts"].max()), int(all_counts.max())) + 1
    volume_iqr = torch.quantile(reference["volume_per_atom"].double(), 0.75) - torch.quantile(
        reference["volume_per_atom"].double(), 0.25
    )
    distance_iqr = torch.quantile(reference["minimum_distance"].double(), 0.75) - torch.quantile(
        reference["minimum_distance"].double(), 0.25
    )
    envelope_probabilities = torch.tensor(
        metrics["reference_volume_envelope_quantiles"], dtype=torch.float64
    )
    volume_envelope = torch.quantile(
        reference["volume_per_atom"].double(), envelope_probabilities
    )
    aggregate = {
        "minimum_distance_fraction": float((all_distances >= threshold).double().mean()),
        "minimum_distance_each_seed_fraction": {
            seed: float((value["minimum_distance"] >= threshold).double().mean())
            for seed, value in generated.items()
        },
        "finite_positive_lattices_fraction": sum(
            int(value["finite_positive_lattices"]) for value in generated.values()
        )
        / sum(int(value["samples"]) for value in generated.values()),
        "sampling_failures": sum(int(value["sampling_failures"]) for value in generated.values()),
        "terminal_masks": sum(int(value["terminal_masks"]) for value in generated.values()),
        "volume_reference_envelope_fraction": float(
            ((all_volumes >= volume_envelope[0]) & (all_volumes <= volume_envelope[1]))
            .double()
            .mean()
        ),
        "element_marginal_jsd": _jsd(all_elements, reference["element_histogram"]),
        "node_count_jsd": _jsd(
            _histogram(all_counts, reference_count_classes),
            _histogram(reference["node_counts"], reference_count_classes),
        ),
        "normalized_volume_wasserstein": _wasserstein(
            all_volumes, reference["volume_per_atom"], quantile_points
        )
        / float(volume_iqr),
        "normalized_minimum_distance_wasserstein": _wasserstein(
            all_distances, reference["minimum_distance"], quantile_points
        )
        / float(distance_iqr),
        "formula_uniqueness_fraction": len(set(all_formulas)) / len(all_formulas),
        "formula_novelty_vs_held_out_subset_fraction": sum(
            formula not in novelty_formula_set for formula in all_formulas
        )
        / len(all_formulas),
        "generated_minimum_distance_quantiles_angstrom": torch.quantile(
            all_distances.double(),
            torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64),
        ).tolist(),
        "train_reference_minimum_distance_quantiles_angstrom": torch.quantile(
            reference["minimum_distance"].double(),
            torch.tensor(
                [0.0, 0.01, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64
            ),
        ).tolist(),
    }
    acceptance = protocol["acceptance"]
    checks = {
        "sampling_failures": aggregate["sampling_failures"]
        == int(acceptance["sampling_failures"]),
        "terminal_masks": aggregate["terminal_masks"] == int(acceptance["terminal_masks"]),
        "finite_positive_lattices": aggregate["finite_positive_lattices_fraction"]
        >= float(acceptance["finite_positive_lattices_fraction"]),
        "minimum_distance_each_seed": min(
            aggregate["minimum_distance_each_seed_fraction"].values()
        )
        >= float(acceptance["minimum_distance_each_seed_fraction_min"]),
        "minimum_distance_aggregate": aggregate["minimum_distance_fraction"]
        >= float(acceptance["minimum_distance_aggregate_fraction_min"]),
        "volume_envelope": aggregate["volume_reference_envelope_fraction"]
        >= float(acceptance["volume_reference_envelope_fraction_min"]),
        "element_marginal": aggregate["element_marginal_jsd"]
        <= float(acceptance["element_marginal_jsd_max"]),
        "node_count": aggregate["node_count_jsd"] <= float(acceptance["node_count_jsd_max"]),
        "volume_wasserstein": aggregate["normalized_volume_wasserstein"]
        <= float(acceptance["normalized_volume_wasserstein_max"]),
        "minimum_distance_wasserstein": aggregate["normalized_minimum_distance_wasserstein"]
        <= float(acceptance["normalized_minimum_distance_wasserstein_max"]),
        "formula_uniqueness": aggregate["formula_uniqueness_fraction"]
        >= float(acceptance["formula_uniqueness_fraction_min"]),
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "distribution_reference_graphs": int(distribution_indices.numel()),
        "novelty_reference_graphs": int(novelty_indices.numel()),
        "generated_graphs": int(all_counts.numel()),
        "aggregate": aggregate,
        "checks": checks,
        "qualified": qualified,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not qualified:
        raise RuntimeError("H1a tensor-free benchmark failed its frozen acceptance checks")


if __name__ == "__main__":
    main()
