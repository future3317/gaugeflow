"""Generate P1 crystals with the production tensor-free reverse sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pymatgen.core import Lattice, Structure

from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.checkpointing import (
    load_production_checkpoint,
    read_production_checkpoint_metadata,
)
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.training import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sampler-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.sampler_steps < 1:
        raise ValueError("sample and step counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    metadata = read_production_checkpoint_metadata(args.checkpoint)
    model_config = metadata.get("model_config")
    training_config = metadata.get("training_config")
    if not isinstance(model_config, dict) or not isinstance(training_config, dict):
        raise ValueError("checkpoint does not contain production model/training configuration")
    model = HybridCrystalDenoiser(**model_config).to(device)
    ema = ExponentialMovingAverage(model, float(training_config["ema_decay"]))
    _, node_prior, _ = load_production_checkpoint(
        args.checkpoint, model=model, ema=ema, map_location=device
    )
    ema.copy_to(model)
    sampler = TensorFreeReverseSampler(
        model,
        coordinate_sigma_max=float(training_config["coordinate_sigma_max"]),
        maximum_time=float(training_config["maximum_time"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    count_generator = torch.Generator().manual_seed(args.seed)
    sample_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    records: list[dict[str, object]] = []
    failures = 0
    for index in range(args.num_samples):
        node_count = node_prior.sample(1, generator=count_generator, device=device)
        blueprint = ParentBlueprintBatch.from_p1_counts(node_count, device=device)
        try:
            generated = sampler.sample(
                blueprint,
                steps=args.sampler_steps,
                generator=sample_generator,
                stochastic=not args.deterministic,
                time_grid="uniform_log_alpha",
            )
            atomic_numbers = generated.atomic_numbers.cpu().tolist()
            coordinates = generated.fractional_coordinates.cpu().tolist()
            lattice = generated.lattice[0].cpu().tolist()
            structure = Structure(
                Lattice(lattice), atomic_numbers, coordinates, coords_are_cartesian=False
            )
            cif_name = f"sample_{index:05d}.cif"
            structure.to(filename=str(args.output / cif_name))
            records.append(
                {
                    "sample": index,
                    "status": "success",
                    "cif": cif_name,
                    "node_count": int(node_count.item()),
                    "atomic_numbers": atomic_numbers,
                    "fractional_coordinates": coordinates,
                    "lattice": lattice,
                    "terminal_mask_count": int(generated.diagnostics.masked_count[-1]),
                }
            )
        except SamplingFailure as error:
            failures += 1
            records.append({"sample": index, "status": "failure", "reason": str(error)})
    summary = {
        "protocol": "s1a_tensor_free_production_v1",
        "checkpoint": str(args.checkpoint),
        "samples": args.num_samples,
        "failures": failures,
        "sampler_steps": args.sampler_steps,
        "records": records,
    }
    (args.output / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("samples", "failures", "sampler_steps")}, sort_keys=True))


if __name__ == "__main__":
    main()
