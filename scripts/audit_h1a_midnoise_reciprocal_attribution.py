"""Frozen H1a middle-noise identifiability and reciprocal-deficiency audit.

This script does not train or modify the denoiser.  It loads the active
dynamic-persistent-edge EMA checkpoint and asks three separate questions:

1. can a revealed-composition structural oracle retrieve the clean endpoint
   from a middle-noise state among same-composition alternatives;
2. is the held-out score residual disproportionately concentrated at low
   physical reciprocal frequency; and
3. does a frozen, target-independent low-k Cartesian carrier linearly correct
   held-out residuals better than a matched high-k control.

Only agreement of all three preregistered checks can authorize a new global
reciprocal carrier.  The audit never changes H1a or any later Gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.runtime import TensorFreeEmaRuntime, load_tensor_free_ema_runtime
from gaugeflow.production.state_projection import (
    fractional_tangent_to_cartesian,
    graph_mean,
    graph_sum,
)


@lru_cache(maxsize=32)
def _projective_integer_grid_cpu(max_index: int) -> torch.Tensor:
    if max_index < 1:
        raise ValueError("reciprocal index bound must be positive")
    axis = torch.arange(-max_index, max_index + 1, dtype=torch.long)
    grid = torch.cartesian_prod(axis, axis, axis)
    nonzero = (grid != 0).any(dim=-1)
    grid = grid[nonzero]
    first_nonzero = (grid != 0).to(torch.long).argmax(dim=-1)
    first_value = grid.gather(1, first_nonzero.unsqueeze(-1)).squeeze(-1)
    return grid[first_value > 0].contiguous()


def _complete_projective_modes(
    lattice: torch.Tensor,
    q_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return a complete projective physical reciprocal ball for each graph."""
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
        raise ValueError("lattice must have shape [graphs,3,3]")
    if q_max <= 0.0:
        raise ValueError("physical reciprocal cutoff must be positive")
    work_lattice = (
        lattice if lattice.dtype in {torch.float32, torch.float64} else lattice.float()
    )
    operator_norm = torch.linalg.matrix_norm(work_lattice, ord=2)
    max_index = int(torch.ceil(q_max * operator_norm.max() / (2.0 * math.pi))) + 1
    integer_modes = _projective_integer_grid_cpu(max_index).to(lattice.device)
    rhs = integer_modes.to(work_lattice).transpose(0, 1).unsqueeze(0).expand(
        lattice.shape[0], -1, -1
    )
    reciprocal = 2.0 * math.pi * torch.linalg.solve(
        work_lattice, rhs
    ).transpose(1, 2)
    norms = torch.linalg.vector_norm(reciprocal, dim=-1)
    valid = (norms > 1.0e-8) & (norms <= q_max + 1.0e-6)
    boundary = integer_modes.abs().amax(dim=-1) == max_index
    if bool((valid & boundary.unsqueeze(0)).any()):
        raise RuntimeError("reciprocal enumeration bound is not complete")
    return integer_modes, reciprocal, norms, valid, max_index


def _reciprocal_basis(
    fractional_coordinates: torch.Tensor,
    atom_types: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    *,
    q_max: float,
) -> dict[str, torch.Tensor | int]:
    graphs = lattice.shape[0]
    modes, reciprocal, norms, valid, max_index = _complete_projective_modes(
        lattice, q_max
    )
    coordinates = fractional_coordinates.to(dtype=reciprocal.dtype)
    phase = 2.0 * math.pi * (
        coordinates @ modes.to(coordinates).transpose(0, 1)
    )
    cosine = phase.cos()
    sine = phase.sin()
    counts = torch.bincount(batch, minlength=graphs).clamp_min(1).to(cosine)
    density_weights = torch.stack(
        (
            torch.ones_like(atom_types, dtype=cosine.dtype),
            2.0 * (atom_types.to(cosine) + 1.0) / 118.0 - 1.0,
        ),
        dim=-1,
    )
    rho_cosine = graph_sum(
        density_weights[:, :, None] * cosine[:, None, :], batch, graphs
    ) / counts.sqrt()[:, None, None]
    rho_sine = graph_sum(
        density_weights[:, :, None] * sine[:, None, :], batch, graphs
    ) / counts.sqrt()[:, None, None]
    return {
        "modes": modes,
        "reciprocal": reciprocal,
        "norms": norms,
        "valid": valid,
        "cosine": cosine,
        "sine": sine,
        "rho_cosine": rho_cosine,
        "rho_sine": rho_sine,
        "max_index": max_index,
    }


def _reciprocal_carrier(
    basis: dict[str, torch.Tensor | int],
    batch: torch.Tensor,
    *,
    band: tuple[float, float],
    radial_channels: int,
) -> torch.Tensor:
    """Construct fixed O(NQ) Cartesian vector carriers for a physical band."""
    if radial_channels < 2 or not 0.0 <= band[0] < band[1]:
        raise ValueError("invalid reciprocal probe band or radial channel count")
    reciprocal = basis["reciprocal"]
    norms = basis["norms"]
    valid = basis["valid"]
    cosine = basis["cosine"]
    sine = basis["sine"]
    rho_cosine = basis["rho_cosine"]
    rho_sine = basis["rho_sine"]
    assert isinstance(reciprocal, torch.Tensor)
    assert isinstance(norms, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    assert isinstance(cosine, torch.Tensor)
    assert isinstance(sine, torch.Tensor)
    assert isinstance(rho_cosine, torch.Tensor)
    assert isinstance(rho_sine, torch.Tensor)
    band_valid = valid & (norms >= band[0]) & (norms <= band[1])
    centers = torch.linspace(
        band[0], band[1], radial_channels, device=norms.device, dtype=norms.dtype
    )
    width = (band[1] - band[0]) / float(radial_channels - 1)
    radial = torch.exp(
        -0.5 * ((norms[:, :, None] - centers[None, None, :]) / width) ** 2
    ) * band_valid[:, :, None]
    radial = radial / radial.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    unit_reciprocal = reciprocal / norms.clamp_min(1.0e-8).unsqueeze(-1)
    graph_rho_cosine = rho_cosine[batch]
    graph_rho_sine = rho_sine[batch]
    phase_response = (
        sine[:, None, :] * graph_rho_cosine
        - cosine[:, None, :] * graph_rho_sine
    )
    node_reciprocal = unit_reciprocal[batch]
    node_radial = radial[batch]
    carrier = math.sqrt(2.0) * torch.einsum(
        "nam,nmd,nmc->nacd", phase_response, node_reciprocal, node_radial
    )
    carrier = carrier.flatten(1, 2)
    graphs = int(valid.shape[0])
    return carrier - graph_mean(carrier, batch, graphs)[batch]


def _spectral_power_by_bin(
    basis: dict[str, torch.Tensor | int],
    vector: torch.Tensor,
    batch: torch.Tensor,
    bin_edges: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = basis["cosine"]
    sine = basis["sine"]
    norms = basis["norms"]
    valid = basis["valid"]
    assert isinstance(cosine, torch.Tensor)
    assert isinstance(sine, torch.Tensor)
    assert isinstance(norms, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    graphs = int(valid.shape[0])
    counts = torch.bincount(batch, minlength=graphs).clamp_min(1).to(vector)
    cosine_amplitude = graph_sum(
        cosine[:, :, None] * vector[:, None, :], batch, graphs
    ) / counts.sqrt()[:, None, None]
    sine_amplitude = graph_sum(
        sine[:, :, None] * vector[:, None, :], batch, graphs
    ) / counts.sqrt()[:, None, None]
    projective_power = 2.0 * (
        cosine_amplitude.square() + sine_amplitude.square()
    ).sum(dim=-1)
    powers: list[torch.Tensor] = []
    mode_counts: list[torch.Tensor] = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = valid & (norms > lower) & (norms <= upper)
        powers.append((projective_power * mask).sum(dim=-1))
        mode_counts.append(mask.sum(dim=-1))
    return torch.stack(powers, dim=-1), torch.stack(mode_counts, dim=-1)


def _probe_sufficient_statistics(
    carrier: torch.Tensor,
    residual: torch.Tensor,
    batch: torch.Tensor,
    graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(batch, minlength=graphs).clamp_min(1).to(carrier)
    node_weight = counts.reciprocal()[batch]
    xtx = torch.einsum("ncd,ned,n->ce", carrier, carrier, node_weight)
    xty = torch.einsum("ncd,nd,n->c", carrier, residual, node_weight)
    return xtx, xty


def _fit_ridge_probe(
    xtx: torch.Tensor,
    xty: torch.Tensor,
    ridge_relative: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    matrix = xtx.double().cpu()
    target = xty.double().cpu()
    channels = matrix.shape[0]
    ridge = ridge_relative * float(torch.trace(matrix)) / max(channels, 1)
    regularized = matrix + ridge * torch.eye(channels, dtype=torch.float64)
    coefficients = torch.linalg.solve(regularized, target)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    threshold = max(float(eigenvalues.max()) * 1.0e-10, 1.0e-14)
    positive = eigenvalues[eigenvalues > threshold]
    condition = float(positive.max() / positive.min()) if positive.numel() else math.inf
    return coefficients, {
        "channels": channels,
        "ridge": ridge,
        "rank": int(positive.numel()),
        "condition_number": condition,
    }


def _graph_coordinate_mse(
    residual: torch.Tensor,
    batch: torch.Tensor,
    graphs: int,
) -> torch.Tensor:
    counts = torch.bincount(batch, minlength=graphs).clamp_min(1).to(residual)
    energy = graph_sum(residual.square().sum(dim=-1), batch, graphs)
    return energy / (3.0 * counts)


def _bootstrap_ratio_improvement(
    baseline: torch.Tensor,
    corrected: torch.Tensor,
    *,
    seed: int,
    samples: int,
) -> list[float]:
    baseline = baseline.double().cpu()
    corrected = corrected.double().cpu()
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        baseline.numel(), (samples, baseline.numel()), generator=generator
    )
    ratio = 1.0 - corrected[draws].mean(dim=-1) / baseline[draws].mean(
        dim=-1
    ).clamp_min(1.0e-15)
    return torch.quantile(
        ratio, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64)
    ).tolist()


def _pair_lattice_descriptor(
    fractional_coordinates: torch.Tensor,
    lattice: torch.Tensor,
    atom_types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutation/O(3)/translation-invariant complete pair descriptor."""
    sites = int(atom_types.numel())
    if sites < 2:
        raise ValueError("retrieval oracle requires at least two sites")
    pairs = torch.triu_indices(sites, sites, offset=1, device=lattice.device)
    delta = fractional_coordinates[pairs[1]] - fractional_coordinates[pairs[0]]
    delta = delta - delta.round()
    shifts = torch.cartesian_prod(
        *[torch.arange(-1, 2, device=lattice.device, dtype=delta.dtype)] * 3
    )
    cartesian = torch.einsum(
        "psa,ab->psb", delta[:, None, :] + shifts[None, :, :], lattice
    )
    distance = torch.linalg.vector_norm(cartesian, dim=-1).amin(dim=-1)
    left = torch.minimum(atom_types[pairs[0]], atom_types[pairs[1]])
    right = torch.maximum(atom_types[pairs[0]], atom_types[pairs[1]])
    code = left * 119 + right
    ordered: list[torch.Tensor] = []
    for value in torch.unique(code, sorted=True):
        ordered.append(distance[code == value].sort().values)
    pair_descriptor = torch.cat(ordered)
    metric = lattice @ lattice.transpose(-1, -2)
    log_volume = 0.5 * torch.logdet(metric)
    unit_metric = metric / torch.det(metric).pow(1.0 / 3.0)
    log_shape_eigenvalues = torch.linalg.eigvalsh(unit_metric).clamp_min(1.0e-12).log()
    lattice_descriptor = torch.cat((log_volume.reshape(1), log_shape_eigenvalues))
    return pair_descriptor, lattice_descriptor


def _composition_groups(
    dataset: PackedAlexP1Dataset,
    *,
    minimum_size: int,
    maximum_size: int,
    minimum_sites: int,
    panel_graphs: int,
    seed: int,
) -> tuple[list[int], dict[int, list[int]]]:
    groups: dict[tuple[int, tuple[tuple[int, int], ...]], list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        start = int(dataset.offsets[index])
        stop = int(dataset.offsets[index + 1])
        if stop - start < minimum_sites:
            continue
        tokens = dataset.atom_tokens[start:stop].to(torch.long)
        values, counts = torch.unique(tokens, return_counts=True)
        key = (
            stop - start,
            tuple(zip(values.tolist(), counts.tolist(), strict=True)),
        )
        groups[key].append(index)
    eligible = [
        value
        for _, value in sorted(groups.items(), key=lambda item: item[0])
        if minimum_size <= len(value) <= maximum_size
    ]
    order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(seed))
    selected: list[int] = []
    selected_groups: dict[int, list[int]] = {}
    for position in order.tolist():
        group = eligible[position]
        if len(selected) + len(group) > panel_graphs:
            continue
        selected.extend(group)
        for index in group:
            selected_groups[index] = group
        if len(selected) == panel_graphs:
            break
    return selected, selected_groups


def _load_noisy_batch(
    runtime: TensorFreeEmaRuntime,
    diffusion: TensorFreeHybridDiffusion,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    time_value: float,
    generator: torch.Generator,
    *,
    device: torch.device,
    run_model: bool,
) -> tuple[Batch, Any, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    packed = Batch.from_data_list([dataset[int(index)] for index in indices]).to(device)
    graphs = int(packed.num_graphs)
    counts = torch.bincount(packed.batch, minlength=graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=packed.frac_coords.dtype, device=device
    )
    time = packed.lattice.new_full((graphs,), time_value)
    noisy = diffusion.noise_clean_batch(
        packed.atom_types,
        packed.frac_coords,
        packed.lattice,
        packed.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=time,
        generator=generator,
    )
    noisy_lattice = LatticeVolumeShape(
        noisy.log_volume.float(), noisy.log_shape.float()
    ).lattice(blueprint.fractional_to_cartesian.float())
    if not run_model:
        return packed, noisy, noisy_lattice, None, None
    condition = time.new_zeros((graphs, 18))
    condition_present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
    use_bf16 = runtime.training_config["precision"] == "bf16" and device.type == "cuda"
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        prediction = runtime.model(
            noisy.element_tokens,
            noisy.fractional_coordinates,
            noisy.log_volume,
            noisy.log_shape,
            packed.batch,
            time,
            condition,
            condition_present,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
        )
    target_cartesian = fractional_tangent_to_cartesian(
        noisy.coordinate_scaled_score_target.float(), noisy_lattice, packed.batch
    )
    cell_scale = torch.exp(noisy.log_volume.float() / 3.0)[packed.batch, None]
    target = target_cartesian / cell_scale
    predicted = prediction.coordinate_cartesian_scaled_score.float() / cell_scale
    return packed, noisy, noisy_lattice, target, predicted


@torch.no_grad()
def _train_frozen_probes(
    runtime: TensorFreeEmaRuntime,
    diffusion: TensorFreeHybridDiffusion,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    specification: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    batch_size = int(specification["batch_size"])
    bands = {name: tuple(value) for name, value in specification["probe_bands"].items()}
    for time_index, time_value in enumerate(specification["times"]):
        accumulators = {
            name: {
                "xtx": torch.zeros(
                    (2 * int(specification["radial_channels"]),) * 2,
                    dtype=torch.float64,
                ),
                "xty": torch.zeros(
                    2 * int(specification["radial_channels"]), dtype=torch.float64
                ),
            }
            for name in bands
        }
        generator = torch.Generator(device=device).manual_seed(
            int(specification["train_noise_seed"]) + time_index
        )
        for start in range(0, indices.numel(), batch_size):
            selected = indices[start : start + batch_size]
            packed, noisy, noisy_lattice, target, predicted = _load_noisy_batch(
                runtime,
                diffusion,
                dataset,
                selected,
                float(time_value),
                generator,
                device=device,
                run_model=True,
            )
            assert target is not None and predicted is not None
            residual = target - predicted
            basis = _reciprocal_basis(
                noisy.fractional_coordinates,
                packed.atom_types,
                noisy_lattice,
                packed.batch,
                q_max=float(specification["q_max_inverse_angstrom"]),
            )
            for name, band in bands.items():
                carrier = _reciprocal_carrier(
                    basis,
                    packed.batch,
                    band=band,
                    radial_channels=int(specification["radial_channels"]),
                )
                xtx, xty = _probe_sufficient_statistics(
                    carrier, residual, packed.batch, int(packed.num_graphs)
                )
                accumulators[name]["xtx"] += xtx.double().cpu()
                accumulators[name]["xty"] += xty.double().cpu()
        output[str(time_value)] = {}
        for name, values in accumulators.items():
            coefficients, audit = _fit_ridge_probe(
                values["xtx"], values["xty"], float(specification["ridge_relative"])
            )
            output[str(time_value)][name] = {
                "coefficients": coefficients,
                "fit_audit": audit,
            }
    return output


@torch.no_grad()
def _evaluate_reciprocal_diagnostics(
    runtime: TensorFreeEmaRuntime,
    diffusion: TensorFreeHybridDiffusion,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    fitted: dict[str, dict[str, Any]],
    specification: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    probe_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    batch_size = int(specification["batch_size"])
    bands = {name: tuple(value) for name, value in specification["probe_bands"].items()}
    bin_edges = list(map(float, specification["spectrum_bin_edges_inverse_angstrom"]))
    maximum_index = 0
    for time_index, time_value in enumerate(specification["times"]):
        generator = torch.Generator(device=device).manual_seed(
            int(specification["validation_noise_seed"]) + time_index
        )
        baseline_parts: list[torch.Tensor] = []
        corrected_parts: dict[str, list[torch.Tensor]] = {name: [] for name in bands}
        coverage_parts: dict[str, list[torch.Tensor]] = {name: [] for name in bands}
        residual_power_parts: list[torch.Tensor] = []
        target_power_parts: list[torch.Tensor] = []
        mode_count_parts: list[torch.Tensor] = []
        for start in range(0, indices.numel(), batch_size):
            selected = indices[start : start + batch_size]
            packed, noisy, noisy_lattice, target, predicted = _load_noisy_batch(
                runtime,
                diffusion,
                dataset,
                selected,
                float(time_value),
                generator,
                device=device,
                run_model=True,
            )
            assert target is not None and predicted is not None
            residual = target - predicted
            basis = _reciprocal_basis(
                noisy.fractional_coordinates,
                packed.atom_types,
                noisy_lattice,
                packed.batch,
                q_max=float(specification["q_max_inverse_angstrom"]),
            )
            maximum_index = max(maximum_index, int(basis["max_index"]))
            basis_norms = basis["norms"]
            basis_valid = basis["valid"]
            assert isinstance(basis_norms, torch.Tensor)
            assert isinstance(basis_valid, torch.Tensor)
            baseline_parts.append(
                _graph_coordinate_mse(
                    residual, packed.batch, int(packed.num_graphs)
                ).cpu()
            )
            for name, band in bands.items():
                coverage_parts[name].append(
                    (
                        basis_valid
                        & (basis_norms >= band[0])
                        & (basis_norms <= band[1])
                    )
                    .any(dim=-1)
                    .cpu()
                )
                carrier = _reciprocal_carrier(
                    basis,
                    packed.batch,
                    band=band,
                    radial_channels=int(specification["radial_channels"]),
                )
                coefficient = fitted[str(time_value)][name]["coefficients"].to(
                    device=device, dtype=carrier.dtype
                )
                correction = torch.einsum("ncd,c->nd", carrier, coefficient)
                corrected_parts[name].append(
                    _graph_coordinate_mse(
                        residual - correction, packed.batch, int(packed.num_graphs)
                    ).cpu()
                )
            residual_power, mode_counts = _spectral_power_by_bin(
                basis, residual, packed.batch, bin_edges
            )
            target_power, target_mode_counts = _spectral_power_by_bin(
                basis, target, packed.batch, bin_edges
            )
            if not torch.equal(mode_counts, target_mode_counts):
                raise RuntimeError("residual and target spectrum used different modes")
            residual_power_parts.append(residual_power.cpu())
            target_power_parts.append(target_power.cpu())
            mode_count_parts.append(mode_counts.cpu())
        baseline = torch.cat(baseline_parts).double()
        for band_index, name in enumerate(bands):
            corrected = torch.cat(corrected_parts[name]).double()
            improvement = 1.0 - float(corrected.mean() / baseline.mean())
            interval = _bootstrap_ratio_improvement(
                baseline,
                corrected,
                seed=int(specification["bootstrap_seed"]) + 10 * time_index + band_index,
                samples=int(specification["bootstrap_samples"]),
            )
            probe_rows.append(
                {
                    "time": float(time_value),
                    "band": name,
                    "baseline_mse": float(baseline.mean()),
                    "corrected_mse": float(corrected.mean()),
                    "relative_improvement": improvement,
                    "bootstrap_95_low": interval[0],
                    "bootstrap_median": interval[1],
                    "bootstrap_95_high": interval[2],
                    "graphs_with_modes_fraction": float(
                        torch.cat(coverage_parts[name]).double().mean()
                    ),
                    **fitted[str(time_value)][name]["fit_audit"],
                }
            )
        residual_power = torch.cat(residual_power_parts).double()
        target_power = torch.cat(target_power_parts).double()
        mode_counts = torch.cat(mode_count_parts).double()
        for bin_index, (lower, upper) in enumerate(
            zip(bin_edges[:-1], bin_edges[1:], strict=True)
        ):
            valid_graph = mode_counts[:, bin_index] > 0
            residual_per_mode = residual_power[valid_graph, bin_index] / mode_counts[
                valid_graph, bin_index
            ]
            target_per_mode = target_power[valid_graph, bin_index] / mode_counts[
                valid_graph, bin_index
            ]
            spectrum_rows.append(
                {
                    "time": float(time_value),
                    "q_lower": lower,
                    "q_upper": upper,
                    "graphs_with_modes": int(valid_graph.sum()),
                    "mean_modes": float(mode_counts[valid_graph, bin_index].mean()),
                    "residual_power_per_mode": float(residual_per_mode.mean()),
                    "target_power_per_mode": float(target_per_mode.mean()),
                    "normalized_residual_ratio": float(
                        residual_per_mode.mean() / target_per_mode.mean().clamp_min(1.0e-15)
                    ),
                }
            )
    return probe_rows, spectrum_rows, {"maximum_complete_integer_index": maximum_index}


@torch.no_grad()
def _middle_noise_retrieval(
    runtime: TensorFreeEmaRuntime,
    diffusion: TensorFreeHybridDiffusion,
    dataset: PackedAlexP1Dataset,
    specification: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, group_lookup = _composition_groups(
        dataset,
        minimum_size=int(specification["minimum_group_size"]),
        maximum_size=int(specification["maximum_group_size"]),
        minimum_sites=int(specification["minimum_sites"]),
        panel_graphs=int(specification["panel_graphs"]),
        seed=int(specification["selection_seed"]),
    )
    if len(selected) < int(specification["minimum_realized_panel_graphs"]):
        raise RuntimeError("not enough same-composition endpoints for retrieval panel")
    clean_descriptors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for index in selected:
        record = dataset[index]
        clean_descriptors[index] = tuple(
            value.double().cpu()
            for value in _pair_lattice_descriptor(
                record.frac_coords.double(),
                record.lattice.squeeze(0).double(),
                record.atom_types,
            )
        )
    rows: list[dict[str, Any]] = []
    index_tensor = torch.tensor(selected, dtype=torch.long)
    for time_index, time_value in enumerate(specification["times"]):
        generator = torch.Generator(device=device).manual_seed(
            int(specification["noise_seed"]) + time_index
        )
        correct = 0
        margins: list[float] = []
        batch_size = int(specification["batch_size"])
        for start in range(0, index_tensor.numel(), batch_size):
            batch_indices = index_tensor[start : start + batch_size]
            packed, noisy, noisy_lattice, _, _ = _load_noisy_batch(
                runtime,
                diffusion,
                dataset,
                batch_indices,
                float(time_value),
                generator,
                device=device,
                run_model=False,
            )
            for graph, index in enumerate(batch_indices.tolist()):
                begin = int(packed.ptr[graph])
                end = int(packed.ptr[graph + 1])
                noisy_pair, noisy_lattice_descriptor = _pair_lattice_descriptor(
                    noisy.fractional_coordinates[begin:end].double(),
                    noisy_lattice[graph].double(),
                    packed.atom_types[begin:end],
                )
                costs: list[tuple[float, int]] = []
                for candidate in group_lookup[index]:
                    clean_pair, clean_lattice_descriptor = clean_descriptors[candidate]
                    pair_cost = (noisy_pair.cpu() - clean_pair).square().mean()
                    lattice_cost = (
                        noisy_lattice_descriptor.cpu() - clean_lattice_descriptor
                    ).square().mean()
                    costs.append((float(pair_cost + lattice_cost), candidate))
                costs.sort(key=lambda item: (item[0], item[1]))
                correct += int(costs[0][1] == index)
                own_cost = next(cost for cost, candidate in costs if candidate == index)
                best_other = min(cost for cost, candidate in costs if candidate != index)
                margins.append(best_other - own_cost)
        margin_tensor = torch.tensor(margins, dtype=torch.float64)
        rows.append(
            {
                "time": float(time_value),
                "graphs": len(selected),
                "top1_accuracy": correct / len(selected),
                "chance_accuracy": sum(
                    1.0 / len(group_lookup[index]) for index in selected
                )
                / len(selected),
                "mean_own_vs_best_other_margin": float(margin_tensor.mean()),
                "median_own_vs_best_other_margin": float(margin_tensor.median()),
            }
        )
    audit = {
        "realized_panel_graphs": len(selected),
        "composition_groups": len({tuple(value) for value in group_lookup.values()}),
        "minimum_group_size": min(len(group_lookup[index]) for index in selected),
        "maximum_group_size": max(len(group_lookup[index]) for index in selected),
        "indices_sha256": canonical_json_hash(selected),
        "revealed_clean_composition": True,
        "endpoint_identity_used": False,
    }
    return rows, audit


def _aggregate_decision(
    retrieval: list[dict[str, Any]],
    probe: list[dict[str, Any]],
    spectrum: list[dict[str, Any]],
    acceptance: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    pooled_retrieval = sum(row["top1_accuracy"] for row in retrieval) / len(retrieval)
    retrieval_check = (
        pooled_retrieval >= float(acceptance["retrieval_mean_accuracy_min"])
        and min(row["top1_accuracy"] for row in retrieval)
        >= float(acceptance["retrieval_each_time_accuracy_min"])
        and sum(row["mean_own_vs_best_other_margin"] for row in retrieval) / len(retrieval)
        > float(acceptance["retrieval_mean_margin_min"])
    )
    low_probe = [row for row in probe if row["band"] == "low"]
    high_probe = [row for row in probe if row["band"] == "high_control"]
    low_mean = sum(row["relative_improvement"] for row in low_probe) / len(low_probe)
    high_mean = sum(row["relative_improvement"] for row in high_probe) / len(high_probe)
    probe_check = (
        low_mean >= float(acceptance["low_probe_mean_improvement_min"])
        and min(row["relative_improvement"] for row in low_probe)
        >= float(acceptance["low_probe_each_time_improvement_min"])
        and min(row["bootstrap_95_low"] for row in low_probe)
        >= float(acceptance["low_probe_each_time_ci_low_min"])
        and low_mean - high_mean
        >= float(acceptance["low_minus_high_probe_improvement_min"])
        and min(row["graphs_with_modes_fraction"] for row in low_probe)
        >= float(acceptance["low_probe_graph_coverage_min"])
    )
    by_time: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in spectrum:
        by_time[float(row["time"])].append(row)
    ratios: list[float] = []
    for rows in by_time.values():
        low_rows = [row for row in rows if float(row["q_upper"]) <= 1.5]
        high_rows = [row for row in rows if float(row["q_lower"]) >= 2.5]
        low_ratio = sum(row["normalized_residual_ratio"] for row in low_rows) / len(
            low_rows
        )
        high_ratio = sum(row["normalized_residual_ratio"] for row in high_rows) / len(
            high_rows
        )
        ratios.append(low_ratio / max(high_ratio, 1.0e-15))
    supporting_times = sum(
        ratio >= float(acceptance["low_to_high_spectral_ratio_min"])
        for ratio in ratios
    )
    spectrum_check = supporting_times >= int(
        acceptance["spectral_supporting_times_min"]
    ) and sum(ratios) / len(ratios) >= float(
        acceptance["pooled_low_to_high_spectral_ratio_min"]
    )
    checks = {
        "middle_noise_endpoint_recoverable": retrieval_check,
        "low_k_residual_spectral_excess": spectrum_check,
        "frozen_low_k_probe_generalizes": probe_check,
    }
    metrics = {
        "retrieval_mean_accuracy": pooled_retrieval,
        "low_probe_mean_improvement": low_mean,
        "high_control_probe_mean_improvement": high_mean,
        "low_minus_high_probe_improvement": low_mean - high_mean,
        "spectral_low_to_high_ratios": ratios,
        "spectral_supporting_times": supporting_times,
        "spectral_mean_low_to_high_ratio": sum(ratios) / len(ratios),
    }
    return checks, metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty diagnostic table {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    checks = result["checks"]
    metrics = result["decision_metrics"]
    lines = [
        "# H1a middle-noise reciprocal attribution v1",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        "This is a frozen-checkpoint diagnostic. It does not retrain the generator, "
        "change H1a, or authorize any later Gate.",
        "",
        "## Three independent checks",
        "",
        f"- middle-noise endpoint recoverability: `{checks['middle_noise_endpoint_recoverable']}` "
        f"(mean top-1 `{metrics['retrieval_mean_accuracy']:.6f}`);",
        f"- low-k residual spectral excess: `{checks['low_k_residual_spectral_excess']}` "
        f"(mean low/high ratio `{metrics['spectral_mean_low_to_high_ratio']:.6f}`);",
        f"- frozen low-k probe generalization: `{checks['frozen_low_k_probe_generalizes']}` "
        f"(low `{metrics['low_probe_mean_improvement']:.6f}`, high control "
        f"`{metrics['high_control_probe_mean_improvement']:.6f}`).",
        "",
        "A reciprocal production carrier is permitted only when all three checks pass. "
        "The detailed curves are stored in the adjacent CSV files.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_midnoise_reciprocal_attribution_v1":
        raise ValueError("unexpected middle-noise reciprocal audit protocol")
    prerequisites = protocol["prerequisites"]
    checkpoint_protocol = load_json_object(args.checkpoint_protocol)
    if sha256_file(args.checkpoint_protocol) != str(
        prerequisites["checkpoint_protocol_sha256"]
    ):
        raise ValueError("checkpoint protocol hash mismatch")
    if sha256_file(args.checkpoint) != str(prerequisites["checkpoint_sha256"]):
        raise ValueError("checkpoint hash mismatch")
    if sha256_file(args.checkpoint_result) != str(
        prerequisites["checkpoint_result_sha256"]
    ):
        raise ValueError("checkpoint result hash mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        prerequisites["cache_manifest_sha256"]
    ):
        raise ValueError("cache manifest hash mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_protocol_hash = canonical_json_hash(checkpoint_protocol)
    runtime = load_tensor_free_ema_runtime(
        args.checkpoint,
        device,
        protocol_name=str(checkpoint_protocol["protocol"]),
        protocol_sha256=checkpoint_protocol_hash,
    )
    runtime.model.eval()
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    diagnostic = protocol["diagnostic"]
    train_dataset = PackedAlexP1Dataset(args.cache_root, "train")
    validation_dataset = PackedAlexP1Dataset(args.cache_root, "val")
    train_indices = torch.randperm(
        len(train_dataset),
        generator=torch.Generator().manual_seed(int(diagnostic["train_selection_seed"])),
    )[: int(diagnostic["train_graphs"])]
    validation_indices = torch.randperm(
        len(validation_dataset),
        generator=torch.Generator().manual_seed(
            int(diagnostic["validation_selection_seed"])
        ),
    )[: int(diagnostic["validation_graphs"])]
    fitted = _train_frozen_probes(
        runtime,
        diffusion,
        train_dataset,
        train_indices,
        diagnostic,
        device=device,
    )
    probe_rows, spectrum_rows, reciprocal_audit = _evaluate_reciprocal_diagnostics(
        runtime,
        diffusion,
        validation_dataset,
        validation_indices,
        fitted,
        diagnostic,
        device=device,
    )
    retrieval_rows, retrieval_audit = _middle_noise_retrieval(
        runtime,
        diffusion,
        validation_dataset,
        {**diagnostic, **protocol["retrieval_oracle"]},
        device=device,
    )
    checks, decision_metrics = _aggregate_decision(
        retrieval_rows, probe_rows, spectrum_rows, protocol["acceptance"]
    )
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "train_indices_sha256": canonical_json_hash(train_indices.tolist()),
        "validation_indices_sha256": canonical_json_hash(validation_indices.tolist()),
        "checks": checks,
        "decision_metrics": decision_metrics,
        "retrieval_audit": retrieval_audit,
        "reciprocal_audit": reciprocal_audit,
        "qualified": qualified,
        "decision": (
            "authorize_separate_reciprocal_carrier_qualification"
            if qualified
            else "do_not_implement_reciprocal_carrier"
        ),
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "middle_noise_retrieval.csv", retrieval_rows)
    _write_csv(args.output / "reciprocal_spectrum.csv", spectrum_rows)
    _write_csv(args.output / "frozen_low_k_probe.csv", probe_rows)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(args.output / "summary.md", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
