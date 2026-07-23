"""Run one Stage-E common-noise tensor-orbit conditioning arm."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.continued_checkpointing import (
    build_continued_pretraining_objects,
)
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
from gaugeflow.production.response_normalization import load_response_normalizer
from gaugeflow.production.tensor_conditioning import (
    predict_common_noisy_state,
    tensor_conditioning_training_loss,
)
from gaugeflow.production.training import ExponentialMovingAverage
from gaugeflow.tensor import piezo_to_irreps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "baseline",
            "orbit_mimic",
            "orbit_mimic_retention",
            "orbit_mimic_exact_null",
            "clean_side",
            "mixed_side",
        ),
        required=True,
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_backbones(
    checkpoint: Path,
    device: torch.device,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    metadata = read_physical_checkpoint_metadata(checkpoint)
    stage_b_metadata = metadata.get("stage_b_metadata")
    if not isinstance(stage_b_metadata, dict):
        raise ValueError("selected Stage-C checkpoint lacks its Stage-B contract")
    objects = build_continued_pretraining_objects(
        stage_b_metadata,
        device=device,
        optimizer_owner=False,
    )
    physical_config = stage_b_metadata.get("physical_training_config")
    if not isinstance(physical_config, dict):
        raise ValueError("selected Stage-C checkpoint lacks its EMA contract")
    source_ema = ExponentialMovingAverage(
        objects.model, float(physical_config["ema_decay"])
    )
    _, loaded_metadata = load_physical_ema_for_evaluation(
        checkpoint,
        model=objects.model,
        ema=source_ema,
        map_location=device,
    )
    if loaded_metadata != metadata:
        raise AssertionError("selected Stage-C metadata changed while loading")
    student = objects.model.backbone
    teacher = copy.deepcopy(student).eval()
    teacher.requires_grad_(False)
    return student, teacher, objects.diffusion, metadata


def _select_trainable_parameters(
    model: Any,
    last_blocks: int,
    *,
    exact_null: bool,
) -> list[torch.nn.Parameter]:
    if not 1 <= last_blocks <= len(model.blocks):
        raise ValueError("Stage-E last-block count is outside the backbone")
    first_trainable_block = len(model.blocks) - last_blocks
    terminal_prefixes = (
        "composition_head.",
        "element_head.",
        "coordinate_control_gate.",
        "coordinate_edge_encoder.",
        "coordinate_edge_residual.",
        "coordinate_carrier.",
        "coordinate_carrier_mixer.",
        "volume_head.",
        "shape_head.",
    )
    selected: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        block_selected = any(
            name.startswith(f"blocks.{index}.")
            for index in range(first_trainable_block, len(model.blocks))
        )
        condition_specific = (
            name.startswith("geometry_query_encoder.")
            or name.startswith("tensor_condition_adapters.")
            or name.startswith("tensor_condition_lattice_adapter.")
            or (
                name.startswith("gauge_atlas.")
                and name != "gauge_atlas.null_condition"
            )
        )
        trainable = condition_specific or (
            not exact_null
            and (name.startswith(terminal_prefixes) or block_selected)
        )
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("Stage-E parameter selection is empty")
    return selected


def _next_indices(
    count: int,
    batch_size: int,
    permutation: torch.Tensor,
    cursor: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    chunks: list[torch.Tensor] = []
    remaining = batch_size
    while remaining:
        take = min(count - cursor, remaining)
        chunks.append(permutation[cursor : cursor + take])
        cursor += take
        remaining -= take
        if cursor == count:
            permutation = torch.randperm(count, generator=generator)
            cursor = 0
    return torch.cat(chunks), permutation, cursor


def _condition(
    batch: ResponseBatch,
    normalizer: Any,
) -> torch.Tensor:
    normalized = normalizer.normalize_piezoelectric(
        batch.targets.piezoelectric,
        batch.source_index,
    )
    return piezo_to_irreps(normalized)


def _charts(batch: ResponseBatch) -> tuple[torch.Tensor, torch.Tensor]:
    shape = torch.eye(6, dtype=batch.lattice.dtype, device=batch.lattice.device)
    chart = torch.eye(3, dtype=batch.lattice.dtype, device=batch.lattice.device)
    return (
        shape.expand(batch.graph_count, -1, -1),
        chart.expand(batch.graph_count, -1, -1),
    )


def _posterior_information(output: Any) -> torch.Tensor:
    atlas = output.gauge_atlas
    tiny = torch.finfo(atlas.posterior.dtype).tiny
    term = atlas.posterior * (
        atlas.posterior.clamp_min(tiny).log()
        - atlas.candidate_prior.clamp_min(tiny).log()
    )
    return (term * atlas.candidate_mask).sum(dim=-1).mean()


@torch.inference_mode()
def _evaluate(
    diffusion: Any,
    teacher: Any,
    dataset: StageDResponseDataset,
    normalizer: Any,
    *,
    count: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float]:
    diffusion.denoiser.eval()
    totals = {
        name: 0.0
        for name in (
            "fine",
            "rotated_fine",
            "orbit_mimic",
            "representative_response",
            "null_retention",
            "posterior_information",
            "target_swap_separation",
            "condition_norm",
            "atlas_gate",
            "aligned_tensor_norm",
            "unique_candidates",
        )
    }
    graphs = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    selected = min(count, len(dataset))
    evaluation_batch_size = max(2, batch_size)
    for start in range(0, selected, evaluation_batch_size):
        host = collate_response_records(
            [
                dataset[index]
                for index in range(
                    start, min(start + evaluation_batch_size, selected)
                )
            ]
        )
        moved = host.pin_memory().to(device, non_blocking=True)
        condition = _condition(moved, normalizer)
        present = moved.targets.piezoelectric_mask[:, None]
        shape_projector, chart = _charts(moved)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
        ):
            output = tensor_conditioning_training_loss(
                diffusion,
                teacher,
                moved.element_tokens,
                moved.fractional_coordinates,
                moved.lattice,
                moved.batch,
                shape_projector,
                chart,
                condition,
                present,
                orbit_weight=1.0,
                retention_weight=1.0,
                generator=generator,
            )
            swapped_condition = condition.roll(1, dims=0)
            swapped = predict_common_noisy_state(
                diffusion.denoiser,
                output.first_fine.noisy,
                moved.batch,
                swapped_condition,
                present,
                shape_projector,
                chart,
            )
            swapped_fine = diffusion.loss_from_prediction(
                moved.element_tokens,
                moved.lattice,
                moved.batch,
                chart,
                output.first_fine.noisy,
                swapped,
            )
        weight = moved.graph_count
        values = {
            "fine": output.fine,
            "rotated_fine": output.rotated_fine.loss,
            "orbit_mimic": output.orbit_mimic.loss,
            "representative_response": output.orbit_mimic.response,
            "null_retention": output.null_retention.loss,
            "posterior_information": _posterior_information(
                output.first_fine.prediction
            ),
            "target_swap_separation": swapped_fine.loss - output.first_fine.loss,
            "condition_norm": torch.linalg.vector_norm(condition, dim=-1).mean(),
            "atlas_gate": output.first_fine.prediction.gauge_atlas.gate.mean(),
            "aligned_tensor_norm": torch.linalg.vector_norm(
                output.first_fine.prediction.gauge_atlas.aligned_tensor.flatten(1),
                dim=-1,
            ).mean(),
            "unique_candidates": output.first_fine.prediction.gauge_atlas.effective_frame_count.float().mean(),
        }
        for name, value in values.items():
            totals[name] += float(value) * weight
        graphs += weight
    return {name: value / graphs for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") not in {
        "stage_e_e0_orbit_mimic_v1",
        "stage_e_e0_orbit_mimic_smoke_v1",
        "stage_e_e0_exact_null_adapter_v1",
        "stage_e_e1_clean_side_v1",
        "stage_e_e1_mixed_side_v1",
    }:
        raise ValueError("unexpected Stage-E protocol")
    arm = protocol["arms"].get(args.arm)
    if not isinstance(arm, dict):
        raise ValueError("Stage-E arm is absent from the protocol")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-E training requires CUDA")
    torch.set_float32_matmul_precision("high")
    args.output.mkdir(parents=True, exist_ok=False)
    train = StageDResponseDataset(args.cache, "train")
    validation = StageDResponseDataset(args.cache, "val")
    normalizer = load_response_normalizer(
        args.normalizer,
        expected_cache_sha256=str(train.manifest["cache_sha256"]),
    ).to(device)
    student, teacher, diffusion, source_metadata = _load_backbones(
        args.checkpoint, device
    )
    parameters = _select_trainable_parameters(
        student,
        int(protocol["trainable_last_blocks"]),
        exact_null=bool(arm.get("exact_null", False)),
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
        fused=True,
    )
    ema = ExponentialMovingAverage(student, float(protocol["ema_decay"]))
    seed = int(protocol["seed"])
    torch.manual_seed(seed)
    data_generator = torch.Generator().manual_seed(seed + 1)
    augmentation_generator = torch.Generator().manual_seed(seed + 2)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 3)
    permutation = torch.randperm(len(train), generator=data_generator)
    cursor = 0
    batch_size = int(protocol["batch_size"])
    use_bf16 = protocol["precision"] == "bf16"
    orbit_weight = float(arm["orbit_weight"])
    retention_weight = float(arm["retention_weight"])
    started = time.perf_counter()
    metrics_path = args.output / "training_metrics.jsonl"
    for step in range(1, int(protocol["steps"]) + 1):
        indices, permutation, cursor = _next_indices(
            len(train), batch_size, permutation, cursor, data_generator
        )
        host = collate_response_records([train[int(index)] for index in indices])
        host = augment_equivalent_response_batch(
            host, generator=augmentation_generator
        )
        moved = host.pin_memory().to(device, non_blocking=True)
        condition = _condition(moved, normalizer)
        present = moved.targets.piezoelectric_mask[:, None]
        shape_projector, chart = _charts(moved)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        clean_side_probability = float(arm.get("clean_side_probability", 0.0))
        if not 0.0 <= clean_side_probability <= 1.0:
            raise ValueError("clean_side_probability must lie in [0,1]")
        use_clean_side = bool(
            arm.get("clean_side_information", False)
            or (
                clean_side_probability > 0.0
                and torch.rand((), generator=data_generator).item() < clean_side_probability
            )
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
        ):
            output = tensor_conditioning_training_loss(
                diffusion,
                teacher,
                moved.element_tokens,
                moved.fractional_coordinates,
                moved.lattice,
                moved.batch,
                shape_projector,
                chart,
                condition,
                present,
                orbit_weight=orbit_weight,
                retention_weight=retention_weight,
                generator=noise_generator,
                clean_side_information=use_clean_side,
            )
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Stage-E loss is non-finite")
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(protocol["gradient_clip"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Stage-E gradient norm is non-finite")
        optimizer.step()
        ema.update(student)
        if step % int(protocol["log_every"]) == 0 or step == int(protocol["steps"]):
            record = {
                "step": step,
                "loss": float(output.loss),
                "fine": float(output.fine),
                "orbit_mimic": float(output.orbit_mimic.loss),
                "representative_response": float(output.orbit_mimic.response),
                "null_retention": float(output.null_retention.loss),
                "gradient_norm": float(gradient_norm),
                "graphs_per_second": step * batch_size / (time.perf_counter() - started),
                "clean_side_information": use_clean_side,
            }
            with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    ema.copy_to(student)
    validation_metrics = _evaluate(
        diffusion,
        teacher,
        validation,
        normalizer,
        count=int(protocol["validation_graphs"]),
        batch_size=batch_size,
        seed=seed + 100,
        device=device,
        use_bf16=use_bf16,
    )
    checkpoint_path = args.output / "checkpoint.pt"
    torch.save(
        {
            "schema": (
                "gaugeflow.stage_e_e1.v1"
                if protocol.get("protocol") in {"stage_e_e1_clean_side_v1", "stage_e_e1_mixed_side_v1"}
                else "gaugeflow.stage_e_e0.v1"
            ),
            "arm": args.arm,
            "model": {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            },
            "source_metadata": source_metadata,
            "protocol_sha256": sha256_file(args.protocol),
            "cache_sha256": train.manifest["cache_sha256"],
            "normalizer_sha256": sha256_file(args.normalizer),
            "source_checkpoint_sha256": sha256_file(args.checkpoint),
        },
        checkpoint_path,
    )
    result = {
        "schema": (
            "gaugeflow.stage_e_e1_result.v1"
            if protocol.get("protocol") in {"stage_e_e1_clean_side_v1", "stage_e_e1_mixed_side_v1"}
            else "gaugeflow.stage_e_e0_result.v1"
        ),
        "arm": args.arm,
        "steps": int(protocol["steps"]),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "validation": validation_metrics,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": torch.cuda.get_device_name(device),
        "boundary": protocol["boundary"],
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
