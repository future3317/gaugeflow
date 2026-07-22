"""Qualify one-cost group-equivalent Stage-D response augmentation on CUDA."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.continued_checkpointing import build_continued_pretraining_objects
from gaugeflow.production.physical_checkpointing import (
    load_physical_ema_for_evaluation,
    read_physical_checkpoint_metadata,
)
from gaugeflow.production.response_data import (
    StageDResponseDataset,
    augment_equivalent_response_batch,
    collate_response_records,
)
from gaugeflow.production.response_multitask import (
    ResponseMultiTaskModel,
    ResponseTargets,
    ResponseTaskWeights,
    response_multitask_loss,
)
from gaugeflow.production.training import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=5731)
    parser.add_argument("--probe-weight", type=float, default=0.1)
    parser.add_argument(
        "--transforms",
        default="rotation,origin,permutation",
        help="Comma-separated subset of rotation,improper,basis,origin,permutation",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _target_squared_norm(target: ResponseTargets) -> float:
    names = (
        "piezoelectric",
        "dielectric",
        "elastic",
        "born_effective_charge",
        "gamma_soft",
        "gamma_log_magnitude",
        "internal_strain",
    )
    return float(sum(getattr(target, name).double().square().sum() for name in names))


def _relative_residual(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def _gradient_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"non-finite Stage-D gradient in {name}")
            snapshot[name] = parameter.grad.detach().cpu().clone()
    if not snapshot:
        raise RuntimeError("Stage-D audit produced no gradients")
    return snapshot


def _gradient_cosine(
    reference: dict[str, torch.Tensor], model: torch.nn.Module
) -> tuple[float, int]:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    compared = 0
    for name, parameter in model.named_parameters():
        if name not in reference:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Stage-D transformed view lost gradient {name}")
        left = reference[name].double()
        right = parameter.grad.detach().cpu().double()
        dot += float(torch.sum(left * right))
        left_norm += float(torch.sum(left.square()))
        right_norm += float(torch.sum(right.square()))
        compared += parameter.numel()
    denominator = math.sqrt(left_norm * right_norm)
    if denominator == 0.0:
        raise RuntimeError("Stage-D paired gradients have zero norm")
    return dot / denominator, compared


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage-D audit {args.output}")
    if args.batch_size < 1 or args.probe_weight < 0.0:
        raise ValueError("Stage-D audit arguments are invalid")
    transforms = {item.strip() for item in args.transforms.split(",") if item.strip()}
    allowed_transforms = {"rotation", "improper", "basis", "origin", "permutation"}
    if not transforms <= allowed_transforms or "improper" in transforms and "rotation" not in transforms:
        raise ValueError("Stage-D equivalent-view transform set is invalid")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-D equivalent-view qualification requires CUDA")

    dataset = StageDResponseDataset(args.cache, "train")
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.batch_size]
    host = collate_response_records([dataset[int(index)] for index in indices])
    augmentation_generator = torch.Generator().manual_seed(args.seed + 1)
    start = time.perf_counter()
    transformed_host = augment_equivalent_response_batch(
        host,
        generator=augmentation_generator,
        include_rotation="rotation" in transforms,
        include_improper="improper" in transforms,
        include_basis="basis" in transforms,
        include_origin="origin" in transforms,
        include_permutation="permutation" in transforms,
    )
    augmentation_ms = 1e3 * (time.perf_counter() - start)

    metadata = read_physical_checkpoint_metadata(args.checkpoint)
    stage_b_metadata = metadata.get("stage_b_metadata")
    if not isinstance(stage_b_metadata, dict):
        raise ValueError("Stage-C checkpoint lacks its Stage-B model contract")
    objects = build_continued_pretraining_objects(
        stage_b_metadata,
        device=device,
        optimizer_owner=False,
    )
    physical_config = stage_b_metadata.get("physical_training_config")
    if not isinstance(physical_config, dict):
        raise ValueError("Stage-C checkpoint lacks its physical training contract")
    ema = ExponentialMovingAverage(objects.model, float(physical_config["ema_decay"]))
    step, loaded_metadata = load_physical_ema_for_evaluation(
        args.checkpoint,
        model=objects.model,
        ema=ema,
        map_location=device,
    )
    if loaded_metadata != metadata:
        raise AssertionError("Stage-C metadata changed while loading")
    torch.manual_seed(args.seed + 2)
    source_count = int(dataset.payload["source_index"].max()) + 1
    model = ResponseMultiTaskModel(
        objects.model.backbone,
        source_count=source_count,
    ).to(device)
    model.train()
    weights = ResponseTaskWeights(piezoelectric_probe=args.probe_weight)
    original = host.to(device, non_blocking=True)
    transformed = transformed_host.to(device, non_blocking=True)

    model.zero_grad(set_to_none=True)
    original_prediction = model(
        original.element_tokens,
        original.fractional_coordinates,
        original.lattice,
        original.batch,
        original.source_index,
    )
    original_loss = response_multitask_loss(
        original_prediction,
        original.targets,
        original.batch,
        original.graph_count,
        weights=weights,
    )
    original_loss.loss.backward()
    original_gradient = _gradient_snapshot(model)

    model.zero_grad(set_to_none=True)
    transformed_prediction = model(
        transformed.element_tokens,
        transformed.fractional_coordinates,
        transformed.lattice,
        transformed.batch,
        transformed.source_index,
    )
    transformed_loss = response_multitask_loss(
        transformed_prediction,
        transformed.targets,
        transformed.batch,
        transformed.graph_count,
        weights=weights,
    )
    transformed_loss.loss.backward()
    gradient_cosine, compared_parameters = _gradient_cosine(original_gradient, model)

    loss_residual = _relative_residual(
        float(original_loss.loss), float(transformed_loss.loss)
    )
    gamma_scale = max(
        float(original_prediction.gamma_log_magnitude.abs().amax()),
        float(original_prediction.gamma_soft_logits.abs().amax()),
        1.0,
    )
    gamma_residual = max(
        float(
            (
                original_prediction.gamma_log_magnitude
                - transformed_prediction.gamma_log_magnitude
            )
            .abs()
            .amax()
        ),
        float(
            (
                original_prediction.gamma_soft_logits
                - transformed_prediction.gamma_soft_logits
            )
            .abs()
            .amax()
        ),
    ) / gamma_scale
    volume_residual = float(
        (
            torch.linalg.det(original.lattice).abs()
            - torch.linalg.det(transformed.lattice).abs()
        )
        .abs()
        .div(torch.linalg.det(original.lattice).abs().clamp_min(1.0))
        .amax()
    )
    target_norm_residual = _relative_residual(
        _target_squared_norm(host.targets),
        _target_squared_norm(transformed_host.targets),
    )
    thresholds = {
        "loss_relative_residual": 2e-3,
        "gradient_cosine_minimum": 0.995,
        "gamma_relative_residual": 2e-3,
        "volume_relative_residual": 5e-5,
        "target_norm_relative_residual": 5e-5,
    }
    qualified = (
        loss_residual <= thresholds["loss_relative_residual"]
        and gradient_cosine >= thresholds["gradient_cosine_minimum"]
        and gamma_residual <= thresholds["gamma_relative_residual"]
        and volume_residual <= thresholds["volume_relative_residual"]
        and target_norm_residual <= thresholds["target_norm_relative_residual"]
    )
    result = {
        "schema": "gaugeflow.stage_d_equivalent_views.v1",
        "status": "qualified" if qualified else "failed",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": step,
        "cache": str(args.cache),
        "cache_sha256": dataset.manifest["cache_sha256"],
        "batch_size": host.graph_count,
        "atoms": int(host.element_tokens.numel()),
        "seed": args.seed,
        "probe_weight": args.probe_weight,
        "transforms": sorted(transforms),
        "augmentation_ms": augmentation_ms,
        "loss_original": float(original_loss.loss),
        "loss_transformed": float(transformed_loss.loss),
        "loss_relative_residual": loss_residual,
        "gradient_cosine": gradient_cosine,
        "gradient_parameters_compared": compared_parameters,
        "gamma_relative_residual": gamma_residual,
        "volume_relative_residual": volume_residual,
        "target_norm_relative_residual": target_norm_residual,
        "thresholds": thresholds,
        "boundary": (
            "Qualifies one-cost exact-object augmentation and the invariant D0 objective; "
            "it is not response predictive performance or an E/F authorization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not qualified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
