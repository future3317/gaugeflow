"""Shared structure-paired statistics for modality-time diagnostics."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Batch

from .alex_p1_data import PackedAlexP1Dataset
from .blueprint import ParentBlueprintBatch
from .hybrid_diffusion import TensorFreeHybridDiffusion
from .runtime import load_tensor_free_ema_runtime

CORNER_NAMES = (
    "clean_clean",
    "noisy_element",
    "noisy_lattice",
    "diagonal",
    "interior",
)


@torch.no_grad()
def corner_graph_losses(
    checkpoint: Path,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    coordinate_time: torch.Tensor,
    interior_element_time: torch.Tensor,
    interior_lattice_time: torch.Tensor,
    *,
    device: torch.device,
    noise_seed: int,
    protocol_name: str,
    protocol_sha256: str,
    batch_size: int = 16,
) -> tuple[dict[str, torch.Tensor], int]:
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=protocol_name,
        protocol_sha256=protocol_sha256,
    )
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    use_bf16 = runtime.training_config["precision"] == "bf16" and device.type == "cuda"
    losses: dict[str, list[torch.Tensor]] = {name: [] for name in CORNER_NAMES}
    candidate_count = 0
    for corner in CORNER_NAMES:
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        element_time, lattice_time = corner_side_times(
            corner,
            coordinate_time,
            interior_element_time,
            interior_lattice_time,
        )
        for start in range(0, indices.numel(), batch_size):
            stop = min(start + batch_size, indices.numel())
            selected = indices[start:stop]
            packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
            graphs = int(packed.num_graphs)
            counts = torch.bincount(packed.batch, minlength=graphs)
            blueprint = ParentBlueprintBatch.from_node_counts(
                counts,
                dtype=packed.frac_coords.dtype,
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = diffusion(
                    packed.atom_types,
                    packed.frac_coords,
                    packed.lattice,
                    packed.batch,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                    time=coordinate_time[start:stop],
                    element_time=element_time[start:stop],
                    lattice_time=lattice_time[start:stop],
                    generator=generator,
                )
            losses[corner].append((output.graph_coordinate_loss / 3.0).float().cpu())
            candidate_count += int(output.prediction.gauge_atlas.effective_frame_count.sum())
    return {name: torch.cat(values) for name, values in losses.items()}, candidate_count


def corner_side_times(
    name: str,
    coordinate_time: torch.Tensor,
    interior_element_time: torch.Tensor,
    interior_lattice_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros_like(coordinate_time)
    if name == "clean_clean":
        return zeros, zeros
    if name == "noisy_element":
        return coordinate_time, zeros
    if name == "noisy_lattice":
        return zeros, coordinate_time
    if name == "diagonal":
        return coordinate_time, coordinate_time
    if name == "interior":
        return interior_element_time, interior_lattice_time
    raise ValueError(f"unknown modality corner: {name}")


def paired_bootstrap_ratio(
    initial: torch.Tensor,
    final: torch.Tensor,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    if initial.shape != final.shape or initial.ndim != 1 or initial.numel() < 2:
        raise ValueError("paired bootstrap requires matching structure vectors")
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        initial.numel(),
        (replicates, initial.numel()),
        generator=generator,
    )
    ratios = final[draws].mean(-1) / initial[draws].mean(-1).clamp_min(1.0e-12)
    quantiles = torch.quantile(
        ratios.double(), torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64)
    )
    return {
        "q025": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q975": float(quantiles[2]),
    }


def paired_bootstrap_mean_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    """Bootstrap the structure-paired mean of ``left - right``."""
    if left.shape != right.shape or left.ndim != 1 or left.numel() < 2:
        raise ValueError("paired difference bootstrap requires matching structure vectors")
    difference = left.double() - right.double()
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        difference.numel(),
        (replicates, difference.numel()),
        generator=generator,
    )
    means = difference[draws].mean(-1)
    quantiles = torch.quantile(
        means, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64)
    )
    return {
        "mean": float(difference.mean()),
        "q025": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q975": float(quantiles[2]),
    }
