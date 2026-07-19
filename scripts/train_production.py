"""Train the production tensor-free hybrid diffusion (S1a substrate only)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset, collate_packed_alex
from gaugeflow.production.blueprint import EmpiricalNodeCountPrior, ParentBlueprintBatch
from gaugeflow.production.checkpointing import (
    load_production_checkpoint,
    read_production_checkpoint_metadata,
    save_production_checkpoint,
)
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig

_GRADIENT_GROUPS = (
    "input_state_embeddings",
    "base_message_blocks",
    "dynamic_edge_angular",
    "coordinate_readout",
    "element_readout",
    "lattice_readout",
    "tensor_atlas",
)


def _gradient_group(parameter_name: str) -> str:
    if parameter_name.startswith(("element_embedding.", "degree_embedding.", "time_embedding.", "state_embedding.")):
        return "input_state_embeddings"
    if parameter_name.startswith(("gauge_atlas.", "geometry_query_encoder.")):
        return "tensor_atlas"
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
    if parameter_name.startswith("element_head."):
        return "element_readout"
    if parameter_name.startswith(("volume_head.", "shape_head.")):
        return "lattice_readout"
    raise ValueError(f"unassigned production parameter: {parameter_name}")


def _clipped_module_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    """Return post-global-clip L2 norms for a complete parameter partition."""
    reference = next(model.parameters())
    squared = {name: torch.zeros((), dtype=torch.float32, device=reference.device) for name in _GRADIENT_GROUPS}
    for parameter_name, parameter in model.named_parameters():
        group = _gradient_group(parameter_name)
        if parameter.grad is not None:
            squared[group] = squared[group] + parameter.grad.detach().float().square().sum()
    values = torch.stack(tuple(squared[name] for name in _GRADIENT_GROUPS)).sqrt().cpu()
    return {name: float(value) for name, value in zip(_GRADIENT_GROUPS, values, strict=True)}


def _validate_coordinate_exposure(
    training_spec: dict[str, object], *, dataset_size: int, steps: int, batch_size: int
) -> None:
    """Validate either complete passes or one preregistered prefix screen."""
    passes = float(training_spec["data_passes"])
    exposure_mode = str(training_spec.get("exposure_mode", "complete_passes"))
    if exposure_mode == "prefix_screen":
        if not 0.0 < passes < 1.0 or steps >= math.ceil(dataset_size / batch_size):
            raise ValueError("coordinate prefix screen must stop within its first shuffled pass")
        expected_presentations = steps * batch_size
        if int(training_spec["graph_presentations"]) != expected_presentations:
            raise ValueError("coordinate prefix-screen presentation count is inconsistent")
        if not math.isclose(passes, expected_presentations / dataset_size, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("coordinate prefix-screen exposure fraction is inconsistent")
        return
    if exposure_mode != "complete_passes" or passes < 1.0 or not passes.is_integer():
        raise ValueError("coordinate training requires complete passes unless prefix_screen is explicit")
    complete_passes = int(passes)
    steps_per_pass = math.ceil(dataset_size / batch_size)
    expected_presentations = complete_passes * dataset_size
    if steps != complete_passes * steps_per_pass or int(training_spec["graph_presentations"]) != expected_presentations:
        raise ValueError("coordinate training counts do not match the declared complete data passes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--lattice-standardization",
        type=Path,
        default=Path("configs/statistics/h1a_p1_lattice_standardization.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("status_before_run") != "frozen_not_run":
        raise ValueError("training protocol was not frozen before execution")
    model_spec = protocol.get("model")
    training_spec = protocol.get("training")
    prerequisites = protocol.get("prerequisites")
    if not isinstance(model_spec, dict) or not isinstance(training_spec, dict) or not isinstance(prerequisites, dict):
        raise ValueError("production protocol requires model, training and prerequisites objects")
    seeds = [int(value) for value in training_spec["seeds"]]
    if args.seed not in seeds:
        raise ValueError("seed is not preregistered by the production protocol")
    steps = int(training_spec["steps"])
    batch_size = int(training_spec["batch_size"])
    objective = str(training_spec["objective"])
    if objective == "coordinate" and "coordinate_clean_side_information" not in training_spec:
        raise ValueError("coordinate training must explicitly declare its observed-side-information contract")
    log_every = int(training_spec["log_every"])
    num_workers = int(training_spec["num_workers"])
    checkpoint_steps = {int(value) for value in training_spec["checkpoint_steps"]}
    if (
        steps < 1
        or batch_size < 1
        or log_every < 1
        or num_workers < 0
        or 0 not in checkpoint_steps
        or steps not in checkpoint_steps
        or any(value < 0 or value > steps for value in checkpoint_steps)
    ):
        raise ValueError("production protocol has invalid training counts")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    observed_manifest_hash = sha256_file(args.cache_root / "manifest.json")
    if observed_manifest_hash != str(prerequisites["cache_manifest_sha256"]):
        raise ValueError("production protocol cache manifest mismatch")
    qualification_name = str(prerequisites["qualified_mechanism"])
    qualification_result = Path("reports") / qualification_name / "result.json"
    if sha256_file(qualification_result) != str(prerequisites["qualification_result_sha256"]):
        raise ValueError("production protocol mechanism qualification mismatch")
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    if objective == "coordinate":
        _validate_coordinate_exposure(
            training_spec,
            dataset_size=len(dataset),
            steps=steps,
            batch_size=batch_size,
        )
    if bool(training_spec.get("from_scratch_required", False)) and args.resume is not None:
        raise ValueError("this frozen protocol requires a from-scratch run")
    node_prior = EmpiricalNodeCountPrior.fit(dataset.node_counts)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_packed_alex,
        generator=loader_generator,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    resume_metadata = read_production_checkpoint_metadata(args.resume) if args.resume is not None else None
    protocol_sha256 = canonical_json_hash(protocol)
    if resume_metadata is not None and (
        resume_metadata.get("protocol") != protocol["protocol"]
        or resume_metadata.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError("resume checkpoint does not match the frozen protocol")
    if resume_metadata is None:
        standardization_value = json.loads(args.lattice_standardization.read_text(encoding="utf-8"))
        if standardization_value.get("source_cache_manifest_sha256") != observed_manifest_hash:
            raise ValueError("lattice standardization was not fitted on this cache manifest")
    else:
        standardization_value = resume_metadata.get("lattice_standardization")
    if not isinstance(standardization_value, dict):
        raise ValueError("training requires lattice-standardization metadata")
    lattice_standardizer = P1LatticeStandardizer.from_mapping(standardization_value)
    model_config = (
        resume_metadata["model_config"]
        if resume_metadata is not None
        else {
            "hidden_dim": int(model_spec["hidden_dim"]),
            "vector_dim": int(model_spec["vector_dim"]),
            "layers": int(model_spec["layers"]),
            "radial_dim": int(model_spec["radial_dim"]),
            "radial_cutoff": float(model_spec["radial_cutoff_angstrom"]),
            "atlas_residual_circle_samples": 8,
            "edge_dim": int(model_spec["edge_dim"]),
            "angular_channels": int(model_spec["angular_channels"]),
            "edge_refresh_rank": int(model_spec["edge_refresh_rank"]),
        }
    )
    model = HybridCrystalDenoiser(**model_config)
    model = model.to(device)
    observed_parameter_count = sum(value.numel() for value in model.parameters())
    if observed_parameter_count != int(model_spec["parameter_count"]):
        raise ValueError("production model parameter count does not match the frozen protocol")
    training_config = (
        ProductionTrainingConfig(**resume_metadata["training_config"])
        if resume_metadata is not None
        else ProductionTrainingConfig(
            learning_rate=float(training_spec["learning_rate"]),
            weight_decay=float(training_spec["weight_decay"]),
            gradient_clip_norm=float(training_spec["gradient_clip_norm"]),
            ema_decay=float(training_spec["ema_decay"]),
            coordinate_sigma_min=float(training_spec["coordinate_sigma_min"]),
            coordinate_sigma_max=float(training_spec["coordinate_sigma_max"]),
            minimum_time=float(training_spec["minimum_time"]),
            maximum_time=float(training_spec["maximum_time"]),
            precision=str(training_spec["precision"]),
            objective=objective,
            coordinate_clean_side_information=bool(training_spec.get("coordinate_clean_side_information", False)),
        )
    )
    diffusion = TensorFreeHybridDiffusion(
        model,
        lattice_standardizer,
        coordinate_sigma_min=training_config.coordinate_sigma_min,
        coordinate_sigma_max=training_config.coordinate_sigma_max,
        minimum_time=training_config.minimum_time,
        maximum_time=training_config.maximum_time,
    )
    trainer = ProductionTrainer(diffusion, training_config)
    if args.resume is not None:
        step, node_prior, _ = load_production_checkpoint(
            args.resume,
            model=model,
            ema=trainer.ema,
            optimizer=trainer.optimizer,
            map_location=device,
            restore_rng=True,
        )
        trainer.step = step
    if args.resume is None and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to append a from-scratch run to a nonempty output")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha256,
        "model_config": model_config,
        "training_config": dataclasses.asdict(training_config),
        "lattice_standardization": standardization_value,
        "seed": args.seed,
        "tensor_condition_enabled": False,
        "coordinate_chart": model.coordinate_chart,
        "blueprint": "P1_empirical_node_count",
        "training_stage": training_config.objective,
    }
    if args.resume is None:
        save_production_checkpoint(
            args.output / "checkpoint_step_00000000.pt",
            model=model,
            ema=trainer.ema,
            optimizer=trainer.optimizer,
            training_step=0,
            node_count_prior=node_prior,
            metadata=checkpoint_metadata,
        )
    log_path = args.output / "training_metrics.jsonl"
    data_iterator = iter(loader)
    device_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    throughput_start = time.perf_counter()
    throughput_graphs = 0
    graphs_seen_this_invocation = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while trainer.step < steps:
        try:
            batch_data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch_data = next(data_iterator)
        batch_data = batch_data.to(device, non_blocking=True)
        graph_count = int(batch_data.num_graphs)
        counts = torch.bincount(batch_data.batch, minlength=graph_count)
        blueprint = ParentBlueprintBatch.from_node_counts(counts, dtype=batch_data.frac_coords.dtype, device=device)
        output, gradient_norm = trainer.train_step(
            batch_data.atom_types,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            blueprint,
            generator=device_generator,
        )
        throughput_graphs += graph_count
        graphs_seen_this_invocation += graph_count
        if trainer.step % log_every == 0 or trainer.step in checkpoint_steps or trainer.step in {1, steps}:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - throughput_start
            record = {
                "step": trainer.step,
                "loss": float(
                    (output.loss if training_config.objective == "joint" else output.coordinate_loss).detach().cpu()
                ),
                "element_loss": float(output.element_loss.detach().cpu()),
                "coordinate_loss": float(output.coordinate_loss.detach().cpu()),
                "volume_loss": float(output.volume_loss.detach().cpu()),
                "shape_loss": float(output.shape_loss.detach().cpu()),
                "masked_fraction": float(output.masked_fraction.detach().cpu()),
                "gradient_norm": float(gradient_norm.cpu()),
                "clipped_module_gradient_norms": _clipped_module_gradient_norms(model),
                "graphs_seen_this_invocation": graphs_seen_this_invocation,
                "graphs_per_second": throughput_graphs / elapsed,
                "peak_cuda_memory_mib": (
                    float(torch.cuda.max_memory_allocated(device)) / (1024.0**2) if device.type == "cuda" else 0.0
                ),
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            throughput_start = time.perf_counter()
            throughput_graphs = 0
        if trainer.step in checkpoint_steps:
            save_production_checkpoint(
                args.output / f"checkpoint_step_{trainer.step:08d}.pt",
                model=model,
                ema=trainer.ema,
                optimizer=trainer.optimizer,
                training_step=trainer.step,
                node_count_prior=node_prior,
                metadata=checkpoint_metadata,
            )


if __name__ == "__main__":
    main()
