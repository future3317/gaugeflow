"""Train the production tensor-free hybrid diffusion (S1a substrate only)."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gaugeflow.file_utils import sha256_file
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--lattice-standardization",
        type=Path,
        default=Path("configs/statistics/h1a_p1_lattice_standardization.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--vector-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--radial-dim", type=int, default=16)
    parser.add_argument("--radial-cutoff", type=float, default=8.0)
    parser.add_argument("--coordinate-fractional-sigma-max", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.checkpoint_every < 1 or args.log_every < 1:
        raise ValueError("training counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    node_prior = EmpiricalNodeCountPrior.fit(dataset.node_counts)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_packed_alex,
        generator=loader_generator,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    resume_metadata = (
        read_production_checkpoint_metadata(args.resume) if args.resume is not None else None
    )
    if resume_metadata is None:
        standardization_value = json.loads(
            args.lattice_standardization.read_text(encoding="utf-8")
        )
        observed_manifest_hash = sha256_file(args.cache_root / "manifest.json")
        if standardization_value.get("source_cache_manifest_sha256") != observed_manifest_hash:
            raise ValueError("lattice standardization was not fitted on this cache manifest")
    else:
        standardization_value = resume_metadata.get("lattice_standardization")
    if not isinstance(standardization_value, dict):
        raise ValueError("training requires lattice-standardization metadata")
    lattice_standardizer = P1LatticeStandardizer.from_mapping(
        standardization_value
    )
    model_config = (
        resume_metadata["model_config"]
        if resume_metadata is not None
        else {
            "hidden_dim": args.hidden_dim,
            "vector_dim": args.vector_dim,
            "layers": args.layers,
            "radial_dim": args.radial_dim,
            "radial_cutoff": args.radial_cutoff,
            "atlas_residual_circle_samples": 8,
        }
    )
    model = HybridCrystalDenoiser(**model_config).to(device)
    training_config = (
        ProductionTrainingConfig(**resume_metadata["training_config"])
        if resume_metadata is not None
        else ProductionTrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            ema_decay=args.ema_decay,
            coordinate_fractional_sigma_max=args.coordinate_fractional_sigma_max,
            precision=args.precision,
        )
    )
    diffusion = TensorFreeHybridDiffusion(
        model,
        lattice_standardizer,
        coordinate_fractional_sigma_max=training_config.coordinate_fractional_sigma_max,
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
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = {
        "protocol": "h1a_p1_generator_pilot_v1",
        "model_config": model_config,
        "training_config": dataclasses.asdict(training_config),
        "lattice_standardization": standardization_value,
        "seed": args.seed,
        "tensor_condition_enabled": False,
        "blueprint": "P1_empirical_node_count",
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
    while trainer.step < args.steps:
        try:
            batch_data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch_data = next(data_iterator)
        batch_data = batch_data.to(device, non_blocking=True)
        graph_count = int(batch_data.num_graphs)
        counts = torch.bincount(batch_data.batch, minlength=graph_count)
        blueprint = ParentBlueprintBatch.from_node_counts(
            counts, dtype=batch_data.frac_coords.dtype, device=device
        )
        output, gradient_norm = trainer.train_step(
            batch_data.atom_types,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            blueprint,
            generator=device_generator,
        )
        if trainer.step % args.log_every == 0 or trainer.step == 1:
            record = {
                "step": trainer.step,
                "loss": float(output.loss.detach().cpu()),
                "element_loss": float(output.element_loss.detach().cpu()),
                "coordinate_loss": float(output.coordinate_loss.detach().cpu()),
                "volume_loss": float(output.volume_loss.detach().cpu()),
                "shape_loss": float(output.shape_loss.detach().cpu()),
                "masked_fraction": float(output.masked_fraction.detach().cpu()),
                "gradient_norm": gradient_norm,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if trainer.step % args.checkpoint_every == 0 or trainer.step == args.steps:
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
