"""Audit pre-clipping J1 gradient geometry without taking optimizer steps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.checkpointing import (
    load_production_checkpoint,
    read_production_checkpoint_metadata,
)
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from scripts.evaluate_h1a_j1_independent_modality_times import (
    CORNER_NAMES,
    _corner_side_times,
)

MODULE_GROUPS = (
    "input_time_embeddings",
    "base_message_blocks",
    "dynamic_edge_angular",
    "time_fusion",
    "coordinate_readout",
    "inactive_other",
)


def _module_group(parameter_name: str) -> str:
    if parameter_name.startswith("modality_time_fusion."):
        return "time_fusion"
    if parameter_name.startswith(
        (
            "element_embedding.",
            "degree_embedding.",
            "time_embedding.",
            "element_time_embedding.",
            "lattice_time_embedding.",
            "state_embedding.",
        )
    ):
        return "input_time_embeddings"
    if parameter_name.startswith("blocks."):
        dynamic_fragments = (
            ".angular_moments.",
            ".edge_source_refresh.",
            ".edge_target_refresh.",
            ".edge_context_refresh.",
            ".edge_vector_refresh.",
            ".edge_update.",
            ".edge_norm.",
            ".angular_scalar_residual.",
            ".angular_vector_residual.",
        )
        return (
            "dynamic_edge_angular"
            if any(fragment in parameter_name for fragment in dynamic_fragments)
            else "base_message_blocks"
        )
    if parameter_name.startswith(
        (
            "coordinate_control_gate.",
            "coordinate_edge_encoder.",
            "edge_state_initializer.",
            "coordinate_edge_residual.",
            "coordinate_carrier.",
            "coordinate_carrier_mixer.",
        )
    ):
        return "coordinate_readout"
    if parameter_name.startswith(
        (
            "element_head.",
            "volume_head.",
            "shape_head.",
            "gauge_atlas.",
            "geometry_query_encoder.",
        )
    ):
        return "inactive_other"
    raise ValueError(f"unassigned gradient-audit parameter: {parameter_name}")


def _flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    parts = []
    for parameter in model.parameters():
        parts.append(
            torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
            if parameter.grad is None
            else parameter.grad.detach().float().reshape(-1)
        )
    return torch.cat(parts)


def _module_gradient_energy(model: torch.nn.Module) -> dict[str, float]:
    reference = next(model.parameters())
    energy = {
        group: torch.zeros((), dtype=torch.float64, device=reference.device)
        for group in MODULE_GROUPS
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            energy[_module_group(name)] += parameter.grad.detach().double().square().sum()
    total = sum(energy.values()).clamp_min(1.0e-30)
    return {group: float((energy[group] / total).cpu()) for group in MODULE_GROUPS}


def _clock_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    prefixes = {
        "coordinate": "time_embedding.",
        "element": "element_time_embedding.",
        "lattice": "lattice_time_embedding.",
        "fusion": "modality_time_fusion.",
    }
    result = {}
    for label, prefix in prefixes.items():
        squared = sum(
            (
                parameter.grad.detach().float().square().sum()
                if parameter.grad is not None
                else parameter.new_zeros((), dtype=torch.float32)
            )
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        )
        result[label] = float(squared.sqrt().cpu())
    return result


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    )
    return {
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "max": float(quantiles[4]),
        "mean": float(tensor.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_j1_gradient_geometry_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen gradient-geometry protocol")
    prerequisites = protocol["prerequisites"]
    expected_hashes = {
        Path("configs/gates/h1a_j1_independent_modality_times_v1.json"): prerequisites[
            "source_protocol_file_sha256"
        ],
        Path("reports/h1a_j1_independent_modality_times_v1/result.json"): prerequisites[
            "source_result_file_sha256"
        ],
        args.checkpoint: prerequisites["source_checkpoint_sha256"],
        args.cache_root / "manifest.json": prerequisites["cache_manifest_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"gradient-audit prerequisite hash mismatch: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    metadata = read_production_checkpoint_metadata(args.checkpoint)
    source_protocol = load_json_object(
        Path("configs/gates/h1a_j1_independent_modality_times_v1.json")
    )
    if (
        metadata.get("protocol") != prerequisites["source_protocol"]
        or metadata.get("protocol_sha256") != canonical_json_hash(source_protocol)
    ):
        raise ValueError("gradient source checkpoint protocol mismatch")
    model_config = metadata["model_config"]
    model = HybridCrystalDenoiser(**model_config).to(device)
    load_production_checkpoint(args.checkpoint, model=model, map_location=device)
    if model.modality_time_conditioning != "separate":
        raise ValueError("gradient audit requires the trained C2 model")
    model.train()
    standardizer = P1LatticeStandardizer.from_mapping(metadata["lattice_standardization"])
    training_config = metadata["training_config"]
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training_config["coordinate_sigma_max"]),
        minimum_time=float(training_config["minimum_time"]),
        maximum_time=float(training_config["maximum_time"]),
    )

    evaluation = protocol["evaluation"]
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["panel_seed"])),
    )[: int(evaluation["graphs"])]
    reference = torch.zeros(1, device=device)
    time_generator = torch.Generator(device=device).manual_seed(
        int(evaluation["noise_seed"]) - 1
    )
    coordinate_time = diffusion.sample_time(indices.numel(), reference, generator=time_generator)
    interior_element_time = diffusion.sample_time(
        indices.numel(), reference, generator=time_generator
    )
    interior_lattice_time = diffusion.sample_time(
        indices.numel(), reference, generator=time_generator
    )

    batch_size = int(evaluation["batch_size"])
    clip_norm = float(evaluation["clip_norm_reference"])
    rows: list[dict[str, Any]] = []
    pair_cosines: dict[str, list[float]] = {
        f"{left}__{right}": []
        for left_index, left in enumerate(CORNER_NAMES)
        for right in CORNER_NAMES[left_index + 1 :]
    }
    alpha_values: list[float] = []
    for microbatch, start in enumerate(range(0, indices.numel(), batch_size)):
        stop = min(start + batch_size, indices.numel())
        packed = Batch.from_data_list(
            [dataset[int(index)] for index in indices[start:stop]]
        ).to(device)
        graphs = int(packed.num_graphs)
        counts = torch.bincount(packed.batch, minlength=graphs)
        blueprint = ParentBlueprintBatch.from_node_counts(
            counts, dtype=packed.frac_coords.dtype, device=device
        )
        gradients: dict[str, torch.Tensor] = {}
        for regime_index, regime in enumerate(CORNER_NAMES):
            model.zero_grad(set_to_none=True)
            element_time, lattice_time = _corner_side_times(
                regime,
                coordinate_time[start:stop],
                interior_element_time[start:stop],
                interior_lattice_time[start:stop],
            )
            generator = torch.Generator(device=device).manual_seed(
                int(evaluation["noise_seed"]) + microbatch
            )
            use_bf16 = device.type == "cuda"
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
                    element_time=element_time,
                    lattice_time=lattice_time,
                    generator=generator,
                )
            output.coordinate_loss.backward()
            gradient = _flatten_gradients(model)
            norm = float(torch.linalg.vector_norm(gradient).cpu())
            alpha = min(1.0, clip_norm / max(norm, 1.0e-30))
            alpha_values.append(alpha)
            gradients[regime] = gradient
            rows.append(
                {
                    "microbatch": microbatch,
                    "regime": regime,
                    "coordinate_loss": float(output.coordinate_loss.detach().cpu()),
                    "gradient_norm": norm,
                    "clip_scale_alpha": alpha,
                    "module_gradient_energy_fraction": _module_gradient_energy(model),
                    "clock_gradient_norms": _clock_gradient_norms(model),
                }
            )
        for left_index, left in enumerate(CORNER_NAMES):
            for right in CORNER_NAMES[left_index + 1 :]:
                cosine = torch.nn.functional.cosine_similarity(
                    gradients[left], gradients[right], dim=0, eps=1.0e-12
                )
                pair_cosines[f"{left}__{right}"].append(float(cosine.cpu()))
        del gradients

    cosine_summary = {}
    thresholds = protocol["diagnostic_thresholds"]
    persistent_pairs = []
    for pair, values in pair_cosines.items():
        summary = _summary(values)
        negative_fraction = sum(value < 0.0 for value in values) / len(values)
        persistent = (
            negative_fraction
            >= float(thresholds["persistent_conflict_negative_fraction_min"])
            and summary["median"]
            < float(thresholds["persistent_conflict_median_cosine_max"])
        )
        cosine_summary[pair] = {
            **summary,
            "negative_fraction": negative_fraction,
            "persistent_conflict": persistent,
        }
        if persistent:
            persistent_pairs.append(pair)
    alpha_summary = _summary(alpha_values)
    severe_clipping = alpha_summary["median"] < float(
        thresholds["severe_clipping_median_alpha_max"]
    )
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for value in (
            row["coordinate_loss"],
            row["gradient_norm"],
            row["clip_scale_alpha"],
            *row["module_gradient_energy_fraction"].values(),
            *row["clock_gradient_norms"].values(),
        )
    ) and all(math.isfinite(value) for values in pair_cosines.values() for value in values)
    classification = (
        "persistent_conflict_and_severe_clipping"
        if persistent_pairs and severe_clipping
        else "conflict_without_severe_clipping"
        if persistent_pairs
        else "norm_scale_without_persistent_conflict"
        if severe_clipping
        else "no_conflict"
    )
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "validation_indices_sha256": canonical_json_hash(indices.tolist()),
        "optimizer_steps": 0,
        "rows": rows,
        "alpha_summary": alpha_summary,
        "gradient_cosines": cosine_summary,
        "persistent_conflict_pairs": persistent_pairs,
        "severe_clipping": severe_clipping,
        "finite": finite,
        "classification": classification,
        "optimizer_change_authorized": persistent_pairs != [] and severe_clipping,
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
