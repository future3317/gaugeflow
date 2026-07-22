"""Train one paired Stage-D D0 response arm from the selected Stage-C base."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.continued_checkpointing import build_continued_pretraining_objects
from gaugeflow.production.physical_checkpointing import (
    load_physical_ema_for_evaluation,
    read_physical_checkpoint_metadata,
)
from gaugeflow.production.response_data import (
    ResponseBatch,
    StageDResponseDataset,
    augment_equivalent_response_batch,
    collate_response_records,
)
from gaugeflow.production.response_multitask import (
    ResponseLossOutput,
    ResponseMultiTaskModel,
    ResponseTaskWeights,
    response_multitask_loss,
)
from gaugeflow.production.response_normalization import load_response_normalizer
from gaugeflow.production.training import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "probe"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_model(
    checkpoint: Path,
    *,
    source_count: int,
    seed: int,
    device: torch.device,
) -> tuple[ResponseMultiTaskModel, dict[str, Any], int]:
    metadata = read_physical_checkpoint_metadata(checkpoint)
    stage_b_metadata = metadata.get("stage_b_metadata")
    if not isinstance(stage_b_metadata, dict):
        raise ValueError("selected Stage-C checkpoint lacks its Stage-B model contract")
    objects = build_continued_pretraining_objects(
        stage_b_metadata,
        device=device,
        optimizer_owner=False,
    )
    physical_config = stage_b_metadata.get("physical_training_config")
    if not isinstance(physical_config, dict):
        raise ValueError("selected Stage-C checkpoint lacks its physical contract")
    source_ema = ExponentialMovingAverage(objects.model, float(physical_config["ema_decay"]))
    step, loaded_metadata = load_physical_ema_for_evaluation(
        checkpoint,
        model=objects.model,
        ema=source_ema,
        map_location=device,
    )
    if loaded_metadata != metadata:
        raise AssertionError("selected Stage-C metadata changed while loading")
    torch.manual_seed(seed)
    model = ResponseMultiTaskModel(
        objects.model.backbone,
        source_count=source_count,
    ).to(device)
    return model, metadata, step


def _next_indices(
    *,
    count: int,
    batch_size: int,
    permutation: torch.Tensor,
    cursor: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    chunks: list[torch.Tensor] = []
    needed = batch_size
    epochs = 0
    while needed:
        available = count - cursor
        take = min(available, needed)
        if take:
            chunks.append(permutation[cursor : cursor + take])
            cursor += take
            needed -= take
        if cursor == count:
            permutation = torch.randperm(count, generator=generator)
            cursor = 0
            epochs += 1
    return torch.cat(chunks), permutation, cursor, epochs


def _move_and_normalize(
    host: ResponseBatch,
    *,
    device: torch.device,
    normalizer: Any,
) -> tuple[ResponseBatch, Any]:
    moved = host.pin_memory().to(device, non_blocking=True)
    target = normalizer.normalize(moved.targets, moved.source_index, moved.batch)
    return moved, target


def _forward_loss(
    model: ResponseMultiTaskModel,
    batch: ResponseBatch,
    target: Any,
    weights: ResponseTaskWeights,
    *,
    use_bf16: bool,
) -> ResponseLossOutput:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        prediction = model(
            batch.element_tokens,
            batch.fractional_coordinates,
            batch.lattice,
            batch.batch,
            batch.source_index,
        )
        return response_multitask_loss(
            prediction,
            target,
            batch.batch,
            batch.graph_count,
            weights=weights,
        )


@torch.inference_mode()
def _evaluate(
    model: ResponseMultiTaskModel,
    dataset: StageDResponseDataset,
    *,
    normalizer: Any,
    weights: ResponseTaskWeights,
    batch_size: int,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float]:
    model.eval()
    names = (
        "loss",
        "piezoelectric_loss",
        "piezoelectric_probe_loss",
        "dielectric_loss",
        "elastic_loss",
        "born_loss",
        "gamma_loss",
        "internal_strain_loss",
    )
    totals = {name: 0.0 for name in names}
    graphs = 0
    for start in range(0, len(dataset), batch_size):
        host = collate_response_records(
            [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        )
        batch, target = _move_and_normalize(host, device=device, normalizer=normalizer)
        output = _forward_loss(model, batch, target, weights, use_bf16=use_bf16)
        for name in names:
            totals[name] += float(getattr(output, name)) * batch.graph_count
        graphs += batch.graph_count
    return {name: value / graphs for name, value in totals.items()}


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _evaluate_ema(
    model: ResponseMultiTaskModel,
    ema: ExponentialMovingAverage,
    dataset: StageDResponseDataset,
    **kwargs: Any,
) -> dict[str, float]:
    """Evaluate EMA weights without changing the live optimizer-owned model."""

    live_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    ema.copy_to(model)
    try:
        return _evaluate(model, dataset, **kwargs)
    finally:
        model.load_state_dict(live_state, strict=True)


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    protocol_name = protocol.get("protocol")
    if protocol_name not in (
        "stage_d_d0_response_probe_v1",
        "stage_d_response_training_v1",
    ):
        raise ValueError("unexpected Stage-D response protocol")
    formal_training = protocol_name == "stage_d_response_training_v1"
    if args.arm not in protocol.get("arms", {}):
        raise ValueError("Stage-D response arm is absent from the protocol")
    arm_config = protocol["arms"][args.arm]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-D response training requires CUDA")
    torch.set_float32_matmul_precision("high")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    checkpoint_path = args.output / "checkpoint.pt"
    result_path = args.output / "result.json"
    metrics_path = args.output / "training_metrics.jsonl"
    validation_metrics_path = args.output / "validation_metrics.jsonl"
    best_path = args.output / "best_checkpoint.pt"
    if not args.resume and any(
        path.exists()
        for path in (
            checkpoint_path,
            result_path,
            metrics_path,
            validation_metrics_path,
            best_path,
        )
    ):
        raise FileExistsError("Stage-D output already contains training artifacts")

    train = StageDResponseDataset(args.cache, "train")
    validation = StageDResponseDataset(args.cache, "val")
    test = StageDResponseDataset(args.cache, "test") if formal_training else None
    source_count = int(train.payload["source_index"].max()) + 1
    seed = int(protocol["seed"])
    model, source_metadata, source_step = _load_model(
        args.checkpoint,
        source_count=source_count,
        seed=seed,
        device=device,
    )
    normalizer = load_response_normalizer(
        args.normalizer,
        expected_cache_sha256=str(train.manifest["cache_sha256"]),
    ).to(device)
    backbone_parameters = list(model.backbone.parameters())
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    if not backbone_parameters or not head_parameters:
        raise RuntimeError("Stage-D optimizer parameter partition is empty")
    optimizer = torch.optim.AdamW(
        (
            {
                "params": backbone_parameters,
                "lr": float(protocol["backbone_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(protocol["head_learning_rate"]),
            },
        ),
        weight_decay=float(protocol["weight_decay"]),
        fused=True,
    )
    steps = int(protocol["steps"])
    warmup = int(protocol["warmup_steps"])

    def learning_rate_factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    ema = ExponentialMovingAverage(model, float(protocol["ema_decay"]))
    data_generator = torch.Generator().manual_seed(seed + 10)
    augmentation_generator = torch.Generator().manual_seed(seed + 11)
    permutation = torch.randperm(len(train), generator=data_generator)
    cursor = 0
    epoch = 0
    completed_step = 0
    best_validation_loss = float("inf")
    best_step = 0
    stale_validations = 0
    stopped_early = False
    if args.resume:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected = {
            "protocol_sha256": sha256_file(args.protocol),
            "cache_sha256": train.manifest["cache_sha256"],
            "normalizer_sha256": sha256_file(args.normalizer),
            "source_checkpoint_sha256": sha256_file(args.checkpoint),
            "arm": args.arm,
        }
        if any(state.get(name) != value for name, value in expected.items()):
            raise ValueError("Stage-D resume provenance does not match")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.load_state_dict(state["ema"])
        data_generator.set_state(state["data_generator_state"])
        augmentation_generator.set_state(state["augmentation_generator_state"])
        permutation = state["permutation"]
        cursor = int(state["cursor"])
        epoch = int(state["epoch"])
        completed_step = int(state["step"])
        best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        best_step = int(state.get("best_step", 0))
        stale_validations = int(state.get("stale_validations", 0))
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)

    weights = ResponseTaskWeights(
        piezoelectric_probe=float(arm_config["piezoelectric_probe_weight"])
    )
    use_bf16 = protocol["precision"] == "bf16"
    batch_size = int(protocol["batch_size"])
    log_every = int(protocol["log_every"])
    checkpoint_every = int(protocol["checkpoint_every"])
    validation_every = int(protocol.get("validation_every", steps))
    early_stopping_patience = int(protocol.get("early_stopping_patience", steps + 1))
    minimum_validation_improvement = float(
        protocol.get("minimum_validation_improvement", 0.0)
    )
    if (
        validation_every < 1
        or early_stopping_patience < 1
        or minimum_validation_improvement < 0.0
    ):
        raise ValueError("Stage-D validation schedule is invalid")
    started = time.perf_counter()
    last_step = completed_step
    for step in range(completed_step + 1, steps + 1):
        last_step = step
        indices, permutation, cursor, crossed = _next_indices(
            count=len(train),
            batch_size=batch_size,
            permutation=permutation,
            cursor=cursor,
            generator=data_generator,
        )
        epoch += crossed
        host = collate_response_records([train[int(index)] for index in indices])
        host = augment_equivalent_response_batch(host, generator=augmentation_generator)
        batch, target = _move_and_normalize(host, device=device, normalizer=normalizer)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = _forward_loss(model, batch, target, weights, use_bf16=use_bf16)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Stage-D training loss is non-finite")
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(protocol["gradient_clip"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Stage-D gradient norm is non-finite")
        optimizer.step()
        scheduler.step()
        ema.update(model)

        if step % log_every == 0 or step == steps:
            metric = {
                "step": step,
                "epoch": epoch + cursor / len(train),
                "loss": float(output.loss),
                "piezoelectric_loss": float(output.piezoelectric_loss),
                "piezoelectric_probe_loss": float(output.piezoelectric_probe_loss),
                "dielectric_loss": float(output.dielectric_loss),
                "elastic_loss": float(output.elastic_loss),
                "born_loss": float(output.born_loss),
                "gamma_loss": float(output.gamma_loss),
                "internal_strain_loss": float(output.internal_strain_loss),
                "gradient_norm": float(gradient_norm),
                "graphs_per_second": step * batch_size / (time.perf_counter() - started),
            }
            with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(metric, sort_keys=True) + "\n")
        if formal_training and (step % validation_every == 0 or step == steps):
            validation_metrics = _evaluate_ema(
                model,
                ema,
                validation,
                normalizer=normalizer,
                weights=weights,
                batch_size=batch_size,
                device=device,
                use_bf16=use_bf16,
            )
            validation_record = {"step": step, **validation_metrics}
            with validation_metrics_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(json.dumps(validation_record, sort_keys=True) + "\n")
            if validation_metrics["loss"] < (
                best_validation_loss - minimum_validation_improvement
            ):
                best_validation_loss = validation_metrics["loss"]
                best_step = step
                stale_validations = 0
                _atomic_checkpoint(
                    best_path,
                    {
                        "schema": "gaugeflow.stage_d_response_best.v1",
                        "step": step,
                        "validation": validation_metrics,
                        "model": {
                            name: value.detach().cpu().clone()
                            for name, value in ema.shadow.items()
                        },
                        "cache_sha256": train.manifest["cache_sha256"],
                        "normalizer_sha256": sha256_file(args.normalizer),
                        "source_checkpoint_sha256": sha256_file(args.checkpoint),
                        "protocol_sha256": sha256_file(args.protocol),
                    },
                )
            else:
                stale_validations += 1

        if step % checkpoint_every == 0 or step == steps or (
            formal_training and step % validation_every == 0
        ):
            _atomic_checkpoint(
                checkpoint_path,
                {
                    "schema": "gaugeflow.stage_d_response_checkpoint.v1",
                    "protocol_sha256": sha256_file(args.protocol),
                    "cache_sha256": train.manifest["cache_sha256"],
                    "normalizer_sha256": sha256_file(args.normalizer),
                    "source_checkpoint_sha256": sha256_file(args.checkpoint),
                    "source_checkpoint_step": source_step,
                    "arm": args.arm,
                    "step": step,
                    "epoch": epoch,
                    "cursor": cursor,
                    "permutation": permutation,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema": ema.state_dict(),
                    "data_generator_state": data_generator.get_state(),
                    "augmentation_generator_state": augmentation_generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(device),
                    "source_metadata": source_metadata,
                    "best_validation_loss": best_validation_loss,
                    "best_step": best_step,
                    "stale_validations": stale_validations,
                },
            )
        if formal_training and stale_validations >= early_stopping_patience:
            stopped_early = True
            break

    if formal_training:
        if not best_path.is_file() or test is None:
            raise RuntimeError("formal Stage-D training did not produce a best checkpoint")
        best = torch.load(best_path, map_location=device, weights_only=False)
        if (
            not isinstance(best, dict)
            or best.get("schema") != "gaugeflow.stage_d_response_best.v1"
            or int(best.get("step", -1)) != best_step
        ):
            raise ValueError("formal Stage-D best checkpoint is invalid")
        model.load_state_dict(best["model"], strict=True)
        validation_metrics = _evaluate(
            model,
            validation,
            normalizer=normalizer,
            weights=weights,
            batch_size=batch_size,
            device=device,
            use_bf16=use_bf16,
        )
        test_metrics = _evaluate(
            model,
            test,
            normalizer=normalizer,
            weights=weights,
            batch_size=batch_size,
            device=device,
            use_bf16=use_bf16,
        )
    else:
        ema.copy_to(model)
        validation_metrics = _evaluate(
            model,
            validation,
            normalizer=normalizer,
            weights=weights,
            batch_size=batch_size,
            device=device,
            use_bf16=use_bf16,
        )
        test_metrics = None
    result = {
        "schema": (
            "gaugeflow.stage_d_response_training.v1"
            if formal_training
            else "gaugeflow.stage_d_d0_response_arm.v1"
        ),
        "status": "complete",
        "arm": args.arm,
        "steps": last_step,
        "maximum_steps": steps,
        "seed": seed,
        "source_checkpoint_step": source_step,
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "cache_sha256": train.manifest["cache_sha256"],
        "normalizer_sha256": sha256_file(args.normalizer),
        "protocol_sha256": sha256_file(args.protocol),
        "probe_weight": weights.piezoelectric_probe,
        "validation": validation_metrics,
        "test": test_metrics,
        "best_step": best_step if formal_training else last_step,
        "best_checkpoint_sha256": sha256_file(best_path) if formal_training else None,
        "stopped_early": stopped_early,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
