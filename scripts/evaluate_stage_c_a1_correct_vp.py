"""Re-evaluate only Stage-C generated NN/volume metrics with the corrected VP kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from evaluate_gaugeflow_base_a1 import reference_statistics
from evaluate_physical_representation import evaluate_generation_retention

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.checkpointing import load_production_checkpoint, read_production_checkpoint_metadata
from gaugeflow.production.composition_runtime import load_qualified_composition_model
from gaugeflow.production.continued_checkpointing import build_continued_pretraining_objects
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.physical_checkpointing import (
    load_physical_ema_for_evaluation,
    read_physical_checkpoint_metadata,
)
from gaugeflow.production.training import ExponentialMovingAverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--a1-protocol", type=Path, required=True)
    parser.add_argument("--alex-cache", type=Path, required=True)
    parser.add_argument("--composition-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    a1_protocol = load_json_object(args.a1_protocol)
    evaluation = a1_protocol["evaluation"]
    alex = PackedAlexP1Dataset(args.alex_cache, "val")
    indices = torch.randperm(
        len(alex), generator=torch.Generator().manual_seed(int(evaluation["validation_seed"]))
    )[: int(evaluation["validation_graphs"])]
    reference = reference_statistics(
        alex, indices, batch_size=int(evaluation["batch_size"]), device=device
    )
    a1_metadata = read_production_checkpoint_metadata(args.a1_checkpoint)
    model_config = a1_metadata["model_config"]
    a1_training = a1_metadata["training_config"]
    standardization = a1_metadata["lattice_standardization"]
    node_model = HybridCrystalDenoiser(**model_config).to(device)
    _, node_prior, _ = load_production_checkpoint(args.a1_checkpoint, model=node_model, map_location=device)
    del node_model
    composition = load_qualified_composition_model(
        args.composition_checkpoint,
        args.composition_protocol,
        device=device,
        expected_checkpoint_sha256=str(a1_protocol["composition_checkpoint_sha256"]),
    )
    stage_c_metadata = read_physical_checkpoint_metadata(args.checkpoints[0])
    stage_b_metadata = stage_c_metadata["stage_b_metadata"]
    physical_config = stage_b_metadata["physical_training_config"]
    objects = build_continued_pretraining_objects(stage_b_metadata, device=device, optimizer_owner=False)
    ema = ExponentialMovingAverage(objects.model, float(physical_config["ema_decay"]))
    results: dict[str, object] = {}
    for checkpoint in args.checkpoints:
        observed_step, loaded_metadata = load_physical_ema_for_evaluation(
            checkpoint, model=objects.model, ema=ema, map_location=device
        )
        if loaded_metadata != read_physical_checkpoint_metadata(checkpoint):
            raise AssertionError("Stage-C metadata changed while loading")
        objects.model.eval()
        generation = evaluate_generation_retention(
            objects.model.backbone,
            node_prior,
            standardization,
            a1_training,
            evaluation,
            reference,
            composition,
            device=device,
        )
        results[str(observed_step)] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "stage_c_global_step": observed_step,
            "generation_retention": generation,
        }
    output = {
        "schema": "gaugeflow.stage_c_correct_vp_generation_requalification.v1",
        "sampler_kernel": "vp_reverse_step_correct_v1",
        "a1_protocol": str(args.a1_protocol),
        "a1_protocol_sha256": sha256_file(args.a1_protocol),
        "a1_validation_indices": indices.tolist(),
        "checkpoints": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
