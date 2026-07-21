"""Run a bounded CUDA overfit/gradient/exact-resume smoke for Stage-B transfer."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.checkpointing import (
    load_production_checkpoint,
    read_production_checkpoint_metadata,
)
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.matpes_data import collate_matpes_records
from gaugeflow.production.matpes_index import IndexedMatPESDataset
from gaugeflow.production.physical_pretraining import (
    PhysicalRepresentationModel,
    load_functional_physical_normalizer,
)
from gaugeflow.production.physical_training import (
    PhysicalTransferTrainer,
    PhysicalTransferTrainingConfig,
)
from gaugeflow.production.training import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graphs", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    if arguments.graphs < 1 or arguments.warmup_steps < 1:
        raise ValueError("physical smoke requires positive graph and step counts")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA physical smoke requested without CUDA")
    torch.manual_seed(5705)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(5705)
        torch.use_deterministic_algorithms(True)

    metadata = read_production_checkpoint_metadata(arguments.a1_checkpoint)
    model_config = metadata.get("model_config")
    training_config = metadata.get("training_config")
    standardization = metadata.get("lattice_standardization")
    if not all(isinstance(value, dict) for value in (model_config, training_config, standardization)):
        raise ValueError("A1 checkpoint metadata is incomplete")
    assert isinstance(model_config, dict)
    assert isinstance(training_config, dict)
    assert isinstance(standardization, dict)
    backbone = HybridCrystalDenoiser(**model_config).to(device)
    a1_ema = ExponentialMovingAverage(backbone, float(training_config["ema_decay"]))
    load_production_checkpoint(
        arguments.a1_checkpoint,
        model=backbone,
        ema=a1_ema,
        map_location=device,
    )
    a1_ema.copy_to(backbone)
    del a1_ema
    normalizer, vocabulary = load_functional_physical_normalizer(arguments.normalizer)
    dataset = IndexedMatPESDataset(arguments.index, "train", verify_hashes=True, require_qualified=True)
    records = [dataset[index] for index in range(arguments.graphs)]
    physical_batch = collate_matpes_records(
        records,
        functional_vocabulary=vocabulary,
        teacher_dim=1,
    ).to(device)
    physical_model = PhysicalRepresentationModel(
        backbone,
        teacher_dim=1,
        functional_count=len(vocabulary),
    ).to(device)
    diffusion = TensorFreeHybridDiffusion(
        backbone,
        P1LatticeStandardizer.from_mapping(standardization),
        coordinate_sigma_min=float(training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training_config["coordinate_sigma_max"]),
        minimum_time=float(training_config["minimum_time"]),
        maximum_time=float(training_config["maximum_time"]),
        categorical_path=str(training_config["categorical_path"]),
        composition_conditioning=bool(training_config["composition_conditioning"]),
    )
    trainer = PhysicalTransferTrainer(
        physical_model,
        diffusion,
        PhysicalTransferTrainingConfig(
            precision="bf16" if device.type == "cuda" else "fp32",
            feature_weight=0.0,
        ),
    )
    counts = torch.bincount(physical_batch.batch, minlength=arguments.graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(counts, device=device)

    def one_step(seed: int) -> tuple[float, float, float]:
        generator = torch.Generator(device=device).manual_seed(seed)
        trainer.begin_optimization_step()
        physical = trainer.accumulate_physical_step(
            physical_batch,
            normalizer,
            loss_weight=0.75,
        )
        replay = trainer.accumulate_alex_replay_step(
            physical_batch.element_tokens,
            physical_batch.fractional_coordinates,
            physical_batch.lattice,
            physical_batch.batch,
            blueprint,
            loss_weight=0.25,
            generator=generator,
        )
        gradient = trainer.finish_optimization_step()
        return float(physical.loss.detach()), float(replay), float(gradient)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    curve = [one_step(6705) for _ in range(arguments.warmup_steps)]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=arguments.output.parent) as temporary:
        checkpoint = Path(temporary) / "resume.pt"
        torch.save(
            {
                "model": physical_model.state_dict(),
                "trainer": trainer.state_dict(),
            },
            checkpoint,
        )
        reference_metrics = one_step(7705)
        reference_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in physical_model.named_parameters()
        }
        state: dict[str, Any] = torch.load(checkpoint, map_location=device, weights_only=True)
        physical_model.load_state_dict(state["model"], strict=True)
        trainer.load_state_dict(state["trainer"])
        repeated_metrics = one_step(7705)
        resume_parameter_error = max(
            float((parameter.detach().cpu() - reference_parameters[name]).abs().max())
            for name, parameter in physical_model.named_parameters()
        )
    resume_metric_error = max(
        abs(first - second)
        for first, second in zip(reference_metrics, repeated_metrics, strict=True)
    )
    finite = all(
        torch.isfinite(torch.tensor(value))
        for row in (*curve, reference_metrics, repeated_metrics)
        for value in row
    )
    result = {
        "schema": "gaugeflow.physical_transfer_smoke.v1",
        "qualified": False,
        "scope": "bounded software/CUDA smoke; not a Stage-B scientific qualification",
        "a1_checkpoint": str(arguments.a1_checkpoint),
        "a1_checkpoint_sha256": sha256_file(arguments.a1_checkpoint),
        "index_manifest_sha256": sha256_file(arguments.index / "manifest.json"),
        "normalizer_sha256": sha256_file(arguments.normalizer),
        "graphs": arguments.graphs,
        "warmup_curve": [
            {"physical_loss": row[0], "alex_replay_loss": row[1], "gradient_norm": row[2]}
            for row in curve
        ],
        "resume_reference": reference_metrics,
        "resume_repeated": repeated_metrics,
        "resume_metric_max_abs_error": resume_metric_error,
        "resume_parameter_max_abs_error": resume_parameter_error,
        "finite": bool(finite),
        "exact_resume": resume_metric_error == 0.0 and resume_parameter_error == 0.0,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
