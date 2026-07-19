"""Audit early gradient flow in the active H1a persistent-edge backbone."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset, collate_packed_alex
from gaugeflow.production.blueprint import EmpiricalNodeCountPrior, ParentBlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig


def _parameter_group(name: str) -> str | None:
    if ".angular_moments.coefficient_projection." in name:
        return "angular_coefficient_projection"
    if ".edge_update." in name:
        return "edge_update"
    if ".angular_scalar_residual.2." in name:
        return "angular_scalar_output"
    if ".angular_scalar_residual." in name:
        return "angular_scalar_internal"
    if ".angular_vector_residual.2." in name:
        return "angular_vector_output"
    if ".angular_vector_residual." in name:
        return "angular_vector_internal"
    if "coordinate_edge_residual.2." in name:
        return "coordinate_edge_output"
    if "coordinate_edge_residual." in name:
        return "coordinate_edge_internal"
    if "coordinate_carrier_mixer.state_projection." in name:
        return "adaptive_mixer_v"
    if "coordinate_carrier_mixer.carrier_projection." in name:
        return "adaptive_mixer_u"
    return None


class _RawGradientRecorder:
    """Collect pre-clipping parameter gradients through backward hooks."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.square_norms: dict[str, float] = defaultdict(float)
        self.element_counts: dict[str, int] = defaultdict(int)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.parameters: dict[str, list[torch.nn.Parameter]] = defaultdict(list)
        for name, parameter in model.named_parameters():
            group = _parameter_group(name)
            if group is None:
                continue
            self.parameters[group].append(parameter)
            self.handles.append(parameter.register_hook(self._hook(group)))

    def _hook(self, group: str) -> Callable[[torch.Tensor], torch.Tensor]:
        def record(gradient: torch.Tensor) -> torch.Tensor:
            value = gradient.detach().float()
            self.square_norms[group] += float(value.square().sum().cpu())
            self.element_counts[group] += value.numel()
            return gradient

        return record

    def reset(self) -> None:
        self.square_norms.clear()
        self.element_counts.clear()

    def norms(self) -> dict[str, float]:
        return {
            group: math.sqrt(self.square_norms.get(group, 0.0))
            for group in sorted(self.parameters)
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _load_standardizer(path: Path, manifest_hash: str) -> P1LatticeStandardizer:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("source_cache_manifest_sha256") != manifest_hash:
        raise ValueError("lattice standardization does not match the cache")
    return P1LatticeStandardizer.from_mapping(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--lattice-standardization",
        type=Path,
        default=Path("configs/statistics/h1a_p1_lattice_standardization.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_persistent_edge_causal_attribution_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen persistent-edge audit protocol")
    source = protocol["source"]
    specification = protocol["gradient_startup"]
    model_spec = protocol["model"]
    manifest_hash = sha256_file(args.cache_root / "manifest.json")
    if manifest_hash != str(source["cache_manifest_sha256"]):
        raise ValueError("persistent-edge audit cache mismatch")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed = int(specification["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    batch_size = int(specification["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(specification["num_workers"]),
        collate_fn=collate_packed_alex,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=int(specification["num_workers"]) > 0,
    )
    model_config = {
        "hidden_dim": int(model_spec["hidden_dim"]),
        "vector_dim": int(model_spec["vector_dim"]),
        "layers": int(model_spec["layers"]),
        "radial_dim": int(model_spec["radial_dim"]),
        "radial_cutoff": float(model_spec["radial_cutoff_angstrom"]),
        "atlas_residual_circle_samples": 8,
        "edge_dim": int(model_spec["edge_dim"]),
        "angular_channels": int(model_spec["angular_channels"]),
    }
    model = HybridCrystalDenoiser(**model_config).to(device)
    if sum(value.numel() for value in model.parameters()) != int(
        model_spec["parameter_count"]
    ):
        raise ValueError("persistent-edge audit model count mismatch")
    if model.coordinate_chart != str(model_spec["coordinate_chart"]):
        raise ValueError("persistent-edge audit coordinate chart mismatch")
    training_config = ProductionTrainingConfig(
        learning_rate=float(specification["learning_rate"]),
        weight_decay=float(specification["weight_decay"]),
        gradient_clip_norm=float(specification["gradient_clip_norm"]),
        ema_decay=float(specification["ema_decay"]),
        coordinate_sigma_min=float(specification["coordinate_sigma_min"]),
        coordinate_sigma_max=float(specification["coordinate_sigma_max"]),
        minimum_time=float(specification["minimum_time"]),
        maximum_time=float(specification["maximum_time"]),
        precision=str(specification["precision"]),
        objective=str(specification["objective"]),
    )
    diffusion = TensorFreeHybridDiffusion(
        model,
        _load_standardizer(args.lattice_standardization, manifest_hash),
        coordinate_sigma_min=training_config.coordinate_sigma_min,
        coordinate_sigma_max=training_config.coordinate_sigma_max,
        minimum_time=training_config.minimum_time,
        maximum_time=training_config.maximum_time,
    )
    trainer = ProductionTrainer(diffusion, training_config)
    recorder = _RawGradientRecorder(model)
    expected_groups = {
        "angular_coefficient_projection",
        "edge_update",
        "angular_scalar_output",
        "angular_scalar_internal",
        "angular_vector_output",
        "angular_vector_internal",
        "coordinate_edge_output",
        "coordinate_edge_internal",
        "adaptive_mixer_u",
        "adaptive_mixer_v",
    }
    if set(recorder.parameters) != expected_groups:
        raise ValueError("gradient audit did not bind every preregistered module")
    initial_parameters = {
        group: [parameter.detach().cpu().clone() for parameter in parameters]
        for group, parameters in recorder.parameters.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    curve_path = args.output.with_suffix(".jsonl")
    if curve_path.exists() or args.output.exists():
        raise FileExistsError("persistent-edge startup audit output already exists")

    generator = torch.Generator(device=device).manual_seed(seed + 1)
    data_iterator = iter(loader)
    records: list[dict[str, object]] = []
    node_prior = EmpiricalNodeCountPrior.fit(dataset.node_counts)
    del node_prior  # the audit reproduces training only; no checkpoint is emitted
    try:
        while trainer.step < int(specification["steps"]):
            batch_data = next(data_iterator).to(device, non_blocking=True)
            graphs = int(batch_data.num_graphs)
            counts = torch.bincount(batch_data.batch, minlength=graphs)
            blueprint = ParentBlueprintBatch.from_node_counts(
                counts, dtype=batch_data.frac_coords.dtype, device=device
            )
            recorder.reset()
            output, total_gradient_norm = trainer.train_step(
                batch_data.atom_types,
                batch_data.frac_coords,
                batch_data.lattice,
                batch_data.batch,
                blueprint,
                generator=generator,
            )
            record: dict[str, object] = {
                "step": trainer.step,
                "coordinate_loss": float(output.coordinate_loss.detach().cpu()),
                "total_gradient_norm_before_clip": float(total_gradient_norm.cpu()),
                "raw_group_gradient_norm": recorder.norms(),
            }
            records.append(record)
            with curve_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if trainer.step in set(map(int, specification["summary_steps"])):
                print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        recorder.close()

    threshold = float(specification["activation_norm_threshold"])
    first_active: dict[str, int | None] = {}
    cumulative_energy: dict[str, float] = {}
    for group in sorted(expected_groups):
        values = [float(record["raw_group_gradient_norm"][group]) for record in records]  # type: ignore[index]
        first_active[group] = next(
            (index + 1 for index, value in enumerate(values) if value > threshold), None
        )
        cumulative_energy[group] = sum(value * value for value in values)
    parameter_delta = {}
    for group, parameters in recorder.parameters.items():
        square = 0.0
        for initial, parameter in zip(initial_parameters[group], parameters, strict=True):
            difference = parameter.detach().cpu().float() - initial.float()
            square += float(difference.square().sum())
        parameter_delta[group] = math.sqrt(square)
    selected_steps = set(map(int, specification["summary_steps"]))
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "steps": int(specification["steps"]),
        "graphs_presented": int(specification["steps"]) * batch_size,
        "seed": seed,
        "first_active_step": first_active,
        "cumulative_raw_gradient_energy": cumulative_energy,
        "parameter_delta_norm": parameter_delta,
        "selected_records": [
            record for record in records if int(record["step"]) in selected_steps
        ],
        "all_finite": all(
            math.isfinite(float(value))
            for record in records
            for value in (
                record["coordinate_loss"],
                record["total_gradient_norm_before_clip"],
                *record["raw_group_gradient_norm"].values(),  # type: ignore[union-attr]
            )
        ),
        "decision_boundary": protocol["decision_rule"]["boundary"],
        "training_config": dataclasses.asdict(training_config),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
