"""Paired Stage-C/E rollout with an independent frozen Stage-D evaluator."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, cast

import torch
from train_stage_d_response import _load_model as load_stage_d_model
from train_stage_e_orbit_mimic import _load_backbones

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.composition_runtime import load_qualified_composition_model
from gaugeflow.production.generation_metrics import (
    minimum_periodic_distances,
    quantile_wasserstein,
    robust_scale,
)
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.response_data import (
    StageDResponseDataset,
    collate_response_records,
)
from gaugeflow.production.response_normalization import load_response_normalizer
from gaugeflow.production.reverse_sampler import (
    ContinuousReverseMode,
    SamplingFailure,
    TensorFreeReverseSampler,
)
from gaugeflow.tensor import fixed_so3_frames, piezo_to_irreps, rotate_rank3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--stage-c-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-d-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-e-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _paired_bootstrap_interval(
    difference: torch.Tensor,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    count = difference.numel()
    indices = torch.randint(count, (resamples, count), generator=generator)
    means = difference.cpu()[indices].mean(dim=1)
    quantiles = torch.quantile(means, torch.tensor([0.025, 0.975]))
    return float(quantiles[0]), float(quantiles[1])


def _orbit_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    rotated = rotate_rank3(
        target[:, None], rotations[None].to(target)
    )
    squared = (prediction[:, None] - rotated).flatten(2).square().mean(dim=-1)
    # Both tensors already live in the train-only normalized Stage-D chart.
    # An absolute chart RMSE remains well defined for physical-zero targets;
    # normalizing by each target norm would make those cases diverge.
    return squared.min(dim=1).values.sqrt()


def _sampler(
    model: Any,
    metadata: dict[str, Any],
    composition_model: Any,
) -> TensorFreeReverseSampler:
    stage_b = metadata["stage_b_metadata"]
    training = stage_b["a1_training_config"]
    return TensorFreeReverseSampler(
        model,
        P1LatticeStandardizer.from_mapping(stage_b["lattice_standardization"]),
        coordinate_sigma_min=float(training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training["coordinate_sigma_max"]),
        maximum_time=float(training["maximum_time"]),
        categorical_path="orderless_reveal",
        composition_model=composition_model,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "stage_e_e0_paired_rollout_v1":
        raise ValueError("unexpected Stage-E rollout protocol")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Stage-E rollout requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.set_float32_matmul_precision("high")

    validation = StageDResponseDataset(args.cache, "val")
    train = StageDResponseDataset(args.cache, "train")
    sample_count = int(protocol["samples"])
    selected = torch.randperm(
        len(validation), generator=torch.Generator().manual_seed(int(protocol["seed"]))
    )[:sample_count]
    records = [validation[int(index)] for index in selected]
    response_normalizer = load_response_normalizer(
        args.normalizer,
        expected_cache_sha256=str(validation.manifest["cache_sha256"]),
    ).to(device)

    base_model, unused_teacher, _, source_metadata = _load_backbones(
        args.stage_c_checkpoint, device
    )
    del unused_teacher
    conditional_model = copy.deepcopy(base_model)
    stage_e = torch.load(args.stage_e_checkpoint, map_location=device, weights_only=False)
    if (
        stage_e.get("schema") != "gaugeflow.stage_e_e0.v1"
        or stage_e.get("arm") != "orbit_mimic"
        or stage_e.get("source_checkpoint_sha256") != sha256_file(args.stage_c_checkpoint)
    ):
        raise ValueError("Stage-E checkpoint is not the selected condition-required arm")
    conditional_model.load_state_dict(stage_e["model"], strict=True)
    base_model.eval()
    conditional_model.eval()

    composition_model = load_qualified_composition_model(
        args.composition_checkpoint,
        args.composition_protocol,
        device=device,
    )
    base_sampler = _sampler(base_model, source_metadata, composition_model)
    conditional_sampler = _sampler(conditional_model, source_metadata, composition_model)

    source_count = int(train.payload["source_index"].max()) + 1
    evaluator, _, _ = load_stage_d_model(
        args.stage_c_checkpoint,
        source_count=source_count,
        seed=int(protocol["seed"]),
        device=device,
    )
    stage_d = torch.load(args.stage_d_checkpoint, map_location=device, weights_only=False)
    if stage_d.get("schema") != "gaugeflow.stage_d_response_best.v1":
        raise ValueError("Stage-D evaluator checkpoint schema is invalid")
    evaluator.load_state_dict(stage_d["model"], strict=True)
    evaluator.eval()

    rotations = fixed_so3_frames(
        int(protocol["orbit_frames"]), seed=int(protocol["seed"]) + 1
    ).to(device)
    target_batch = collate_response_records(records).to(device)
    target_normalized = response_normalizer.normalize_piezoelectric(
        target_batch.targets.piezoelectric, target_batch.source_index
    )
    conditions = piezo_to_irreps(target_normalized)
    target_volume = (
        torch.linalg.det(target_batch.lattice) / target_batch.node_counts
    ).cpu()
    target_distance = minimum_periodic_distances(
        target_batch.fractional_coordinates,
        target_batch.lattice,
        target_batch.batch,
    ).cpu()

    results: dict[str, dict[str, Any]] = {}
    orbit_errors: dict[str, list[torch.Tensor]] = {"base": [], "conditioned": []}
    for role, sampler in (("base", base_sampler), ("conditioned", conditional_sampler)):
        volumes: list[torch.Tensor] = []
        distances: list[torch.Tensor] = []
        finite_positive = valid_distance = failures = 0
        for start in range(0, sample_count, int(protocol["batch_size"])):
            stop = min(start + int(protocol["batch_size"]), sample_count)
            counts = target_batch.node_counts[start:stop]
            blueprint = ParentBlueprintBatch.from_node_counts(counts, device=device)
            seed = int(protocol["seed"]) + 1000 + start
            try:
                generated = sampler.sample(
                    blueprint,
                    tensor_condition=(conditions[start:stop] if role == "conditioned" else None),
                    steps=int(protocol["reverse_steps"]),
                    initialization_generator=torch.Generator(device=device).manual_seed(seed),
                    categorical_generator=torch.Generator(device=device).manual_seed(seed + 1),
                    continuous_generator=torch.Generator(device=device).manual_seed(seed + 2),
                    continuous_mode=cast(
                        ContinuousReverseMode, str(protocol["continuous_mode"])
                    ),
                    time_grid=str(protocol["time_grid"]),
                )
            except (SamplingFailure, RuntimeError, ValueError, FloatingPointError):
                failures += stop - start
                continue
            determinant = torch.linalg.det(generated.lattice)
            finite_positive += int(
                (torch.isfinite(generated.lattice).all(dim=(-2, -1)) & (determinant > 0)).sum()
            )
            distance = minimum_periodic_distances(
                generated.fractional_coordinates,
                generated.lattice,
                generated.batch,
            )
            valid_distance += int((distance >= 0.5).sum())
            volumes.append((determinant / counts).cpu())
            distances.append(distance.cpu())
            prediction = evaluator(
                generated.element_tokens,
                generated.fractional_coordinates,
                generated.lattice,
                generated.batch,
                target_batch.source_index[start:stop],
            ).piezoelectric
            orbit_errors[role].append(
                _orbit_error(prediction, target_normalized[start:stop], rotations).cpu()
            )
        if failures or not volumes:
            results[role] = {"sampling_failures": failures}
            continue
        volume = torch.cat(volumes)
        distance = torch.cat(distances)
        points = int(protocol["wasserstein_quantile_points"])
        results[role] = {
            "sampling_failures": failures,
            "finite_positive_lattice_fraction": finite_positive / sample_count,
            "minimum_distance_fraction_at_0_5_angstrom": valid_distance / distance.numel(),
            "normalized_nearest_neighbor_wasserstein": quantile_wasserstein(
                distance, target_distance, points=points
            ) / robust_scale(target_distance),
            "normalized_volume_wasserstein": quantile_wasserstein(
                volume, target_volume, points=points
            ) / robust_scale(target_volume),
            "mean_tensor_orbit_error": float(torch.cat(orbit_errors[role]).mean()),
        }

    if any(results[role].get("sampling_failures", 1) for role in results):
        checks = {"sampling_failures": False}
        interval = (float("nan"), float("nan"))
    else:
        difference = torch.cat(orbit_errors["conditioned"]) - torch.cat(orbit_errors["base"])
        interval = _paired_bootstrap_interval(
            difference,
            resamples=int(protocol["bootstrap_resamples"]),
            seed=int(protocol["seed"]) + 2,
        )
        limits = protocol["checks"]
        checks = {
            "sampling_failures": all(
                results[role]["sampling_failures"] == limits["sampling_failures"]
                for role in results
            ),
            "finite_positive_lattice": all(
                results[role]["finite_positive_lattice_fraction"]
                == limits["finite_positive_lattice_fraction"]
                for role in results
            ),
            "minimum_distance": all(
                results[role]["minimum_distance_fraction_at_0_5_angstrom"]
                >= limits["minimum_distance_fraction_at_0_5_angstrom_min"]
                for role in results
            ),
            "nn_retention": (
                results["conditioned"]["normalized_nearest_neighbor_wasserstein"]
                - results["base"]["normalized_nearest_neighbor_wasserstein"]
                <= limits["conditioned_minus_base_nn_w1_max"]
            ),
            "volume_retention": (
                results["conditioned"]["normalized_volume_wasserstein"]
                - results["base"]["normalized_volume_wasserstein"]
                <= limits["conditioned_minus_base_volume_w1_max"]
            ),
            "tensor_orbit_improvement": interval[1] < 0.0,
        }
    payload = {
        "schema": "gaugeflow.stage_e_e0_paired_rollout.v1",
        "protocol_sha256": sha256_file(args.protocol),
        "stage_c_checkpoint_sha256": sha256_file(args.stage_c_checkpoint),
        "stage_d_checkpoint_sha256": sha256_file(args.stage_d_checkpoint),
        "stage_e_checkpoint_sha256": sha256_file(args.stage_e_checkpoint),
        "selected_validation_indices": selected.tolist(),
        "metrics": results,
        "paired_conditioned_minus_base_orbit_error_interval_95": interval,
        "checks": checks,
        "qualified": all(checks.values()),
        "boundary": protocol["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
