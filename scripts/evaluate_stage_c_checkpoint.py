"""Evaluate one Stage-C EMA checkpoint on all three declared held-out panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from evaluate_gaugeflow_base_a1 import reference_statistics
from evaluate_physical_representation import (
    evaluate_generation_retention,
    evaluate_physical_checkpoint,
)

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.checkpointing import load_production_checkpoint
from gaugeflow.production.composition_runtime import load_qualified_composition_model
from gaugeflow.production.continued_checkpointing import build_continued_pretraining_objects
from gaugeflow.production.continued_pretraining import collate_structure_records
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lemat_index import IndexedLeMatDataset
from gaugeflow.production.matpes_index import IndexedMatPESDataset
from gaugeflow.production.physical_checkpointing import (
    load_physical_ema_for_evaluation,
    read_physical_checkpoint_metadata,
)
from gaugeflow.production.physical_pretraining import load_functional_physical_normalizer
from gaugeflow.production.stage_c_evaluation import (
    balanced_functional_panel,
    graphwise_structure_replay_loss,
)
from gaugeflow.production.teacher_feature_cache import MatPESTeacherFeatureCache
from gaugeflow.production.training import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--a1-evaluation-protocol", type=Path, required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--matpes-index", type=Path, required=True)
    parser.add_argument("--lemat-index", type=Path, required=True)
    parser.add_argument("--lemat-graphs-per-functional", type=int, default=500)
    parser.add_argument("--lemat-selection-seed", type=int, default=5721)
    parser.add_argument("--lemat-noise-seed", type=int, default=5722)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--alex-cache", type=Path, required=True)
    parser.add_argument("--composition-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-protocol", type=Path, required=True)
    parser.add_argument("--stage-b-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _ratio(current: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return float(current[key]) / float(baseline[key])


@torch.inference_mode()
def evaluate_lemat_structure_checkpoint(
    diffusion: TensorFreeHybridDiffusion,
    dataset: IndexedLeMatDataset,
    *,
    graphs_per_functional: int,
    selection_seed: int,
    noise_seed: int,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    """Evaluate one paired, functional-balanced LeMat calibration panel."""

    panels = balanced_functional_panel(
        dataset.functional_group_index,
        functional_count=len(dataset.functional_names),
        graphs_per_functional=graphs_per_functional,
        seed=selection_seed,
    )
    per_functional: dict[str, dict[str, float | int]] = {}
    all_losses: list[torch.Tensor] = []
    for functional_index, (functional, indices) in enumerate(
        zip(dataset.functional_names, panels, strict=True)
    ):
        generator = torch.Generator(device=device).manual_seed(noise_seed + functional_index)
        losses: list[torch.Tensor] = []
        for start in range(0, indices.numel(), batch_size):
            records = dataset.select(indices[start : start + batch_size])
            clean = collate_structure_records(records).to(device)
            blueprint = ParentBlueprintBatch.from_node_counts(
                clean.node_counts,
                dtype=clean.fractional_coordinates.dtype,
                device=device,
            )
            use_bf16 = precision == "bf16"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = diffusion(
                    clean.element_tokens,
                    clean.fractional_coordinates,
                    clean.lattice,
                    clean.batch,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                    generator=generator,
                )
                graph_loss = graphwise_structure_replay_loss(output)
            losses.append(graph_loss.detach().cpu().double())
        values = torch.cat(losses)
        all_losses.append(values)
        per_functional[functional] = {
            "graphs": values.numel(),
            "mean_product_space_loss": float(values.mean()),
            "standard_error": float(values.std(unbiased=True) / values.numel() ** 0.5),
        }
    means = torch.tensor(
        [value["mean_product_space_loss"] for value in per_functional.values()],
        dtype=torch.float64,
    )
    combined = torch.cat(all_losses)
    return {
        "sampling_law": "equal graph count per functional; target-independent split-local selection",
        "functional_names": list(dataset.functional_names),
        "graphs_per_functional": graphs_per_functional,
        "graphs": combined.numel(),
        "selection_seed": selection_seed,
        "noise_seed": noise_seed,
        "selection_indices_sha256": canonical_json_hash(
            [panel.tolist() for panel in panels]
        ),
        "precision": precision,
        "macro_mean_product_space_loss": float(means.mean()),
        "micro_mean_product_space_loss": float(combined.mean()),
        "per_functional": per_functional,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage-C evaluation: {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-C checkpoint evaluation requires CUDA")

    metadata = read_physical_checkpoint_metadata(args.checkpoint)
    if metadata.get("protocol") != "stage_c_lemat_continued_pretraining_v2":
        raise ValueError("checkpoint is not a Stage-C-v1 continuation")
    stage_b_metadata = metadata.get("stage_b_metadata")
    if not isinstance(stage_b_metadata, dict):
        raise ValueError("Stage-C checkpoint lacks its Stage-B model contract")
    baseline = load_json_object(args.stage_b_result)
    baseline_checkpoints = baseline.get("physical_checkpoints")
    baseline_generation = baseline.get("generation_retention")
    if not isinstance(baseline_checkpoints, dict) or not isinstance(baseline_generation, dict):
        raise ValueError("Stage-B result lacks physical or generation evidence")
    stage_b_step = max(int(step) for step in baseline_checkpoints)
    objects = build_continued_pretraining_objects(
        stage_b_metadata,
        device=device,
        optimizer_owner=False,
    )
    physical_config = stage_b_metadata.get("physical_training_config")
    if not isinstance(physical_config, dict):
        raise ValueError("Stage-B metadata lacks the physical training config")
    ema = ExponentialMovingAverage(objects.model, float(physical_config["ema_decay"]))
    observed_step, loaded_metadata = load_physical_ema_for_evaluation(
        args.checkpoint,
        model=objects.model,
        ema=ema,
        map_location=device,
    )
    if loaded_metadata != metadata:
        raise AssertionError("Stage-C metadata changed while loading")
    if observed_step <= stage_b_step:
        raise ValueError("mid-training evaluation requires a post-Stage-B checkpoint")
    objects.model.eval()

    precision = str(physical_config["precision"])
    if precision not in {"fp32", "bf16"}:
        raise ValueError("Stage-C checkpoint declares an unsupported evaluation precision")
    lemat_manifest_path = args.lemat_index / "manifest.json"
    lemat_manifest = load_json_object(lemat_manifest_path)
    lemat_index_path = args.lemat_index / str(lemat_manifest["index_file"])
    if not bool(lemat_manifest.get("qualified")):
        raise ValueError("LeMat evaluation index is not qualified")
    if sha256_file(lemat_index_path) != str(lemat_manifest["index_sha256"]):
        raise ValueError("LeMat evaluation index hash mismatch")
    # Raw parquet hashes were already verified by the qualified index build.
    # Rehashing 5.4M-source artifacts for every checkpoint changes no
    # acceptance set and would dominate evaluation latency.
    lemat = IndexedLeMatDataset(
        args.lemat_index,
        "calibration",
        verify_hashes=False,
    )
    lemat_structure = evaluate_lemat_structure_checkpoint(
        objects.diffusion,
        lemat,
        graphs_per_functional=args.lemat_graphs_per_functional,
        selection_seed=args.lemat_selection_seed,
        noise_seed=args.lemat_noise_seed,
        batch_size=64,
        device=device,
        precision=precision,
    )

    normalizer, vocabulary = load_functional_physical_normalizer(args.normalizer)
    if vocabulary != objects.functional_vocabulary:
        raise ValueError("normalizer vocabulary disagrees with the checkpoint")
    calibration = IndexedMatPESDataset(
        args.matpes_index,
        "calibration",
        teacher_feature_cache=args.teacher_cache,
    )
    feature_cache = calibration.teacher_feature_cache
    if not isinstance(feature_cache, MatPESTeacherFeatureCache):
        raise ValueError("MatPES calibration data lacks teacher features")
    physical = evaluate_physical_checkpoint(
        objects.model,
        calibration,
        normalizer,
        vocabulary,
        feature_cache.feature_dim,
        batch_size=64,
        device=device,
    )

    a1_protocol = load_json_object(args.a1_evaluation_protocol)
    evaluation = a1_protocol.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("A1 evaluation protocol is incomplete")
    a1_training = stage_b_metadata.get("a1_training_config")
    standardization = stage_b_metadata.get("lattice_standardization")
    model_config = stage_b_metadata.get("model_config")
    if not all(isinstance(value, dict) for value in (a1_training, standardization, model_config)):
        raise ValueError("Stage-B metadata lacks the A1 generation contract")
    assert isinstance(a1_training, dict)
    assert isinstance(standardization, dict)
    assert isinstance(model_config, dict)
    a1_backbone = HybridCrystalDenoiser(**model_config).to(device)
    _, node_prior, _ = load_production_checkpoint(
        args.a1_checkpoint,
        model=a1_backbone,
        map_location=device,
    )
    del a1_backbone
    torch.cuda.empty_cache()

    alex_validation = PackedAlexP1Dataset(args.alex_cache, "val")
    validation_indices = torch.randperm(
        len(alex_validation),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    reference = reference_statistics(
        alex_validation,
        validation_indices,
        batch_size=int(evaluation["batch_size"]),
        device=device,
    )
    composition_model = load_qualified_composition_model(
        args.composition_checkpoint,
        args.composition_protocol,
        device=device,
        expected_checkpoint_sha256=str(a1_protocol["composition_checkpoint_sha256"]),
    )
    generation = evaluate_generation_retention(
        objects.model.backbone,
        node_prior,
        standardization,
        a1_training,
        evaluation,
        reference,
        composition_model,
        device=device,
    )

    baseline_physical = baseline_checkpoints[str(stage_b_step)]["aggregate"]
    current_physical = physical["aggregate"]
    comparison = {
        "physical_relative_to_stage_b": {
            "equal_head_composite_loss_ratio": _ratio(
                current_physical, baseline_physical, "equal_head_composite_loss"
            ),
            "normalized_energy_rmse_ratio": _ratio(
                current_physical, baseline_physical, "normalized_energy_rmse"
            ),
            "normalized_force_rmse_ratio": _ratio(
                current_physical, baseline_physical, "normalized_force_rmse"
            ),
            "normalized_kelvin_stress_rmse_ratio": _ratio(
                current_physical, baseline_physical, "normalized_kelvin_stress_rmse"
            ),
            "teacher_feature_cosine_delta": float(current_physical["teacher_feature_cosine"])
            - float(baseline_physical["teacher_feature_cosine"]),
        },
        "generation_delta_from_stage_b": {
            key: float(generation[key]) - float(baseline_generation[key])
            for key in (
                "normalized_nearest_neighbor_wasserstein",
                "normalized_volume_wasserstein",
                "element_marginal_jsd",
                "node_count_jsd",
                "minimum_distance_fraction_at_0_5_angstrom",
            )
        },
    }
    result = {
        "schema": "gaugeflow.stage_c_checkpoint_evaluation.v2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "global_step": observed_step,
        "stage_c_step": observed_step - stage_b_step,
        "lemat_index_binding": {
            "manifest_sha256": sha256_file(lemat_manifest_path),
            "index_sha256": sha256_file(lemat_index_path),
            "source_hashes_reused_from_qualified_manifest": True,
        },
        "lemat_structure_calibration": lemat_structure,
        "physical_calibration": physical,
        "generation_retention": generation,
        "generation_validation_indices": validation_indices.tolist(),
        "stage_b_comparison": comparison,
        "status": "diagnostic_only_training_continues",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
