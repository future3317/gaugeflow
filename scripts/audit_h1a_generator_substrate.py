"""Audit the H1a graph representation, lattice statistics and torus prior."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from gaugeflow.geometry import periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import (
    PACKED_ALEX_P1_PROTOCOL,
    PackedAlexP1Dataset,
)
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape

SPLITS = ("train", "val", "test")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _integer_grid(shell: int, device: torch.device) -> torch.Tensor:
    axis = torch.arange(-shell, shell + 1, device=device, dtype=torch.float32)
    grid = torch.cartesian_prod(axis, axis, axis)
    return grid[(grid != 0).any(dim=-1)]


@torch.no_grad()
def terminal_mixing_census(
    datasets: dict[str, PackedAlexP1Dataset],
    *,
    sigma_max: float,
    threshold: float,
    maximum_shell: int,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[dict[str, Any], dict[str, Any]]:
    residual_parts: list[torch.Tensor] = []
    log_volume_sum = torch.zeros((), dtype=torch.float64)
    residual_volume_sum = torch.zeros((), dtype=torch.float64)
    residual_volume_square_sum = torch.zeros((), dtype=torch.float64)
    shape_sum = torch.zeros(6, dtype=torch.float64)
    shape_outer = torch.zeros((6, 6), dtype=torch.float64)
    train_rows = 0
    largest_shell = 0
    within_threshold = 0
    for split in SPLITS:
        dataset = datasets[split]
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            lattice = dataset.lattice[start:stop].to(device)
            metric_inverse = torch.linalg.inv(lattice @ lattice.transpose(-1, -2))
            eigenvalue_min = torch.linalg.eigvalsh(metric_inverse)[:, 0]
            minimum = None
            for shell in range(1, maximum_shell + 1):
                grid = _integer_grid(shell, device)
                quadratic = torch.einsum(
                    "ki,bij,kj->bk", grid, metric_inverse, grid
                )
                minimum = quadratic.amin(dim=-1)
                certified = eigenvalue_min * float((shell + 1) ** 2) > minimum * (
                    1.0 + 2.0e-6
                )
                if bool(certified.all()):
                    largest_shell = max(largest_shell, shell)
                    break
            else:
                raise RuntimeError("reciprocal shortest-vector census exceeded its frozen shell")
            assert minimum is not None
            residual = torch.exp(-2.0 * math.pi**2 * sigma_max**2 * minimum)
            within_threshold += int((residual <= threshold).sum())
            residual_parts.append(residual.cpu())
            if split == "train":
                chart = torch.eye(3, dtype=lattice.dtype, device=device).expand(
                    lattice.shape[0], -1, -1
                )
                state = LatticeVolumeShape.from_lattice(lattice, chart)
                counts = dataset.node_counts[start:stop].to(
                    device=device, dtype=state.log_volume.dtype
                )
                volume_residual = state.log_volume - counts.log()
                log_volume_sum += state.log_volume.double().sum().cpu()
                residual_volume_sum += volume_residual.double().sum().cpu()
                residual_volume_square_sum += volume_residual.double().square().sum().cpu()
                shape = state.log_shape.double().cpu()
                shape_sum += shape.sum(dim=0)
                shape_outer += shape.T @ shape
                train_rows += stop - start
    all_residual = torch.cat(residual_parts).double()
    quantiles = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], dtype=torch.float64)
    mixing = {
        "rows": int(all_residual.numel()),
        "largest_certifying_reciprocal_shell": largest_shell,
        "fraction_within_threshold": within_threshold / all_residual.numel(),
        "epsilon_mix_quantiles": {
            label: float(value)
            for label, value in zip(
                ("p50", "p90", "p99", "p999", "max"),
                torch.quantile(all_residual, quantiles).tolist(),
                strict=True,
            )
        },
    }
    shape_mean = shape_sum / train_rows
    shape_covariance = (
        shape_outer - train_rows * torch.outer(shape_mean, shape_mean)
    ) / (train_rows - 1)
    volume_residual_mean = residual_volume_sum / train_rows
    volume_residual_variance = (
        residual_volume_square_sum
        - train_rows * volume_residual_mean.square()
    ) / (train_rows - 1)
    shape_eigenvalues, shape_eigenvectors = torch.linalg.eigh(shape_covariance)
    active = shape_eigenvalues > 1.0e-10
    shape_basis = shape_eigenvectors[:, active]
    shape_scales = shape_eigenvalues[active].sqrt()
    statistics = {
        "train_rows": train_rows,
        "log_volume_mean": float(log_volume_sum / train_rows),
        "log_volume_minus_log_n_mean": float(volume_residual_mean),
        "log_volume_minus_log_n_std": float(volume_residual_variance.sqrt()),
        "log_shape_mean": shape_mean.tolist(),
        "log_shape_covariance": shape_covariance.tolist(),
        "log_shape_covariance_eigenvalues": shape_eigenvalues.tolist(),
        "log_shape_whitening_basis_columns": shape_basis.tolist(),
        "log_shape_whitening_scales": shape_scales.tolist(),
    }
    return mixing, statistics


def _sample_indices(
    datasets: dict[str, PackedAlexP1Dataset], sample_rows: int, seed: int
) -> list[tuple[str, int]]:
    generator = torch.Generator().manual_seed(seed)
    candidates: list[tuple[str, int]] = []
    per_stratum = min(256, sample_rows // 8)
    for count in (1, 2, 3, 20):
        stratum: list[tuple[str, int]] = []
        for split in SPLITS:
            indices = torch.nonzero(
                datasets[split].node_counts == count, as_tuple=False
            ).flatten()
            stratum.extend((split, int(index)) for index in indices.tolist())
        if not stratum:
            raise RuntimeError(f"frozen edge stratum N={count} is empty")
        order = torch.randperm(len(stratum), generator=generator)[:per_stratum]
        candidates.extend(stratum[int(index)] for index in order)
    selected = set(candidates)
    split_starts: list[tuple[str, int, int]] = []
    cursor = 0
    for split in SPLITS:
        split_starts.append((split, cursor, cursor + len(datasets[split])))
        cursor += len(datasets[split])
    for global_index in torch.randperm(cursor, generator=generator).tolist():
        chosen = next(
            (split, global_index - start)
            for split, start, stop in split_starts
            if start <= global_index < stop
        )
        if chosen in selected:
            continue
        candidates.append(chosen)
        selected.add(chosen)
        if len(candidates) == sample_rows:
            break
    if len(candidates) != sample_rows:
        raise RuntimeError("unable to construct the frozen unique edge sample")
    return candidates


@torch.no_grad()
def edge_census(
    datasets: dict[str, PackedAlexP1Dataset],
    *,
    sample_rows: int,
    cutoff: float,
    seed: int,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    selected = _sample_indices(datasets, sample_rows, seed)
    old_edges = 0
    radius_edges = 0
    self_images = 0
    outside_old_edges = 0
    repeated_pair_edges = 0
    n1_rows = 0
    n1_nonempty = 0
    peak_memory = 0
    elapsed = 0.0
    for start in range(0, len(selected), batch_size):
        records = [
            datasets[split][index]
            for split, index in selected[start : start + batch_size]
        ]
        packed = Batch.from_data_list(records).to(device)
        counts = torch.bincount(packed.batch, minlength=packed.num_graphs)
        old_edges += int((counts * (counts - 1)).sum())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        edges = periodic_radius_multigraph(
            packed.frac_coords,
            packed.lattice,
            packed.batch,
            cutoff=cutoff,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
        elapsed += time.perf_counter() - tick
        radius_edges += edges.source.numel()
        self_images += int((edges.source == edges.target).sum())
        for graph in range(packed.num_graphs):
            node_mask = packed.batch == graph
            node_ids = torch.nonzero(node_mask, as_tuple=False).flatten()
            first = int(node_ids[0])
            stop = int(node_ids[-1]) + 1
            graph_edge = (edges.target >= first) & (edges.target < stop)
            pairs = torch.stack(
                (edges.source[graph_edge] - first, edges.target[graph_edge] - first),
                dim=-1,
            )
            nonself = pairs[:, 0] != pairs[:, 1]
            unique_nonself = torch.unique(pairs[nonself], dim=0).shape[0]
            nodes = stop - first
            outside_old_edges += nodes * (nodes - 1) - unique_nonself
            if pairs.numel():
                _, multiplicity = torch.unique(pairs, dim=0, return_counts=True)
                repeated_pair_edges += int((multiplicity - 1).clamp_min(0).sum())
            if nodes == 1:
                n1_rows += 1
                n1_nonempty += int(pairs.shape[0] > 0)
    return {
        "sample_rows": len(selected),
        "cutoff_angstrom": cutoff,
        "old_closest_pair_edges": old_edges,
        "periodic_radius_edges": radius_edges,
        "self_image_edges": self_images,
        "additional_repeated_pair_edges": repeated_pair_edges,
        "old_edges_outside_cutoff": outside_old_edges,
        "n1_rows": n1_rows,
        "n1_nonempty_fraction": n1_nonempty / max(n1_rows, 1),
        "wall_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "configs/gates/h1a_generator_substrate_audit_v1.json",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("E:/DATA/T2C-Flow/processed/gaugeflow_h1a_v1/p1_structure_cache_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "reports/h1a_generator_substrate_audit_v1/pre_repair_census.json",
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    protocol = _load_json(arguments.protocol)
    if protocol.get("protocol") != "h1a_generator_substrate_audit_v1":
        raise ValueError("unexpected H1a substrate protocol")
    cache_manifest = _load_json(arguments.cache_root / "manifest.json")
    if (
        cache_manifest.get("protocol") != PACKED_ALEX_P1_PROTOCOL
        or not bool(cache_manifest.get("qualified"))
    ):
        raise ValueError("H1a substrate audit requires the qualified P1 cache")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    datasets = {
        split: PackedAlexP1Dataset(arguments.cache_root, split)
        for split in SPLITS
    }
    mixing_protocol = protocol["terminal_coordinate_prior"]
    mixing, statistics = terminal_mixing_census(
        datasets,
        sigma_max=float(mixing_protocol["sigma_max_angstrom"]),
        threshold=float(mixing_protocol["mixing_residual_threshold"]),
        maximum_shell=int(mixing_protocol["maximum_reciprocal_search_shell"]),
        device=device,
    )
    graph_protocol = protocol["periodic_graph"]
    edges = edge_census(
        datasets,
        sample_rows=int(graph_protocol["sample_rows"]),
        cutoff=float(graph_protocol["cutoff_angstrom"]),
        seed=int(protocol["seed"]),
        device=device,
    )
    result = {
        "protocol": protocol["protocol"],
        "cache_protocol": cache_manifest["protocol"],
        "edge_census": edges,
        "terminal_mixing": mixing,
        "training_lattice_statistics": statistics,
        "qualified": False,
        "decision": "diagnostics_only_pending_path_and_gradient_audits",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
