"""Localize E1 failure between high-mask estimation and irreversible exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset, collate_packed_alex
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.categorical_mask import AbsorbingMaskDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.reverse_sampler import reverse_time_grid
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.state_projection import project_translation_state


def _top1_correct(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
    return int(((logits.argmax(dim=-1) == target) & mask).sum()), int(mask.sum())


def _composition_matches(predicted: torch.Tensor, target: torch.Tensor, batch: torch.Tensor) -> int:
    return sum(
        int(
            torch.equal(
                predicted[batch == graph].sort().values,
                target[batch == graph].sort().values,
            )
        )
        for graph in range(int(batch.max()) + 1)
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_e1_categorical_exposure_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen E1 exposure protocol")
    source = protocol["source"]
    if sha256_file(Path("reports/h1a_e1_element_reverse_v1/result.json")) != source["result_sha256"]:
        raise ValueError("source E1 result hash mismatch")
    if sha256_file(args.checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("source E1 checkpoint hash mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_protocol = load_json_object(Path("configs/gates/h1a_e1_element_reverse_v1.json"))
    source_protocol_sha256 = canonical_json_hash(source_protocol)
    runtime = load_tensor_free_ema_runtime(
        args.checkpoint,
        device,
        protocol_name=str(source_protocol["protocol"]),
        protocol_sha256=source_protocol_sha256,
    )
    model = runtime.model
    categorical = AbsorbingMaskDiffusion()
    evaluation = protocol["evaluation"]
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    snapshot_targets = [float(value) for value in evaluation["snapshot_times"]]
    totals = {
        value: {
            "on_policy_correct": 0,
            "oracle_correct": 0,
            "masked": 0,
            "all_mask_correct": 0,
            "all_mask_nodes": 0,
            "wrong_revealed": 0,
            "revealed": 0,
            "observed_time_sum": 0.0,
            "batches": 0,
        }
        for value in snapshot_targets
    }
    multistep_correct = 0
    multistep_composition = 0
    one_shot_correct = 0
    one_shot_composition = 0
    total_nodes = 0
    total_graphs = 0
    generator = torch.Generator(device=device).manual_seed(int(evaluation["categorical_seed"]))
    one_shot_generator = torch.Generator(device=device).manual_seed(int(evaluation["one_shot_seed"]))
    batch_size = int(evaluation["batch_size"])
    for chunk in indices.split(batch_size):
        data = collate_packed_alex([dataset[int(index)] for index in chunk]).to(device)
        graphs = int(data.num_graphs)
        counts = torch.bincount(data.batch, minlength=graphs)
        blueprint = ParentBlueprintBatch.from_node_counts(
            counts,
            dtype=data.frac_coords.dtype,
            device=device,
        )
        coordinates = project_translation_state(data.frac_coords, data.batch, graphs)
        lattice_state = LatticeVolumeShape.from_lattice(
            data.lattice,
            blueprint.fractional_to_cartesian,
        )
        log_volume = lattice_state.log_volume
        log_shape = torch.einsum(
            "bij,bj->bi",
            blueprint.shape_projector,
            lattice_state.log_shape,
        )
        condition = torch.zeros((graphs, 18), dtype=coordinates.dtype, device=device)
        condition_present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
        clean_time = torch.zeros((graphs,), dtype=coordinates.dtype, device=device)
        times = reverse_time_grid(
            categorical.schedule,
            float(runtime.training_config["maximum_time"]),
            int(evaluation["reverse_steps"]),
            dtype=coordinates.dtype,
            device=device,
            spacing="uniform_log_alpha",
        )
        snapshot_indices: dict[int, list[float]] = {}
        for target in snapshot_targets:
            index = int(torch.argmin((times[:-1] - target).abs()))
            snapshot_indices.setdefault(index, []).append(target)
        tokens = torch.full_like(data.atom_types, categorical.mask_index)
        initial_time = times[0].expand(graphs)
        initial_prediction = model(
            tokens,
            coordinates,
            log_volume,
            log_shape,
            data.batch,
            clean_time,
            condition,
            condition_present,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            element_time=initial_time,
            lattice_time=clean_time,
        )
        one_shot_probability = categorical.reverse_probabilities(
            tokens,
            initial_prediction.clean_element_logits,
            initial_time,
            clean_time,
            data.batch,
        )
        one_shot = torch.multinomial(
            one_shot_probability,
            1,
            replacement=True,
            generator=one_shot_generator,
        ).squeeze(-1)
        one_shot_correct += int((one_shot == data.atom_types).sum())
        one_shot_composition += _composition_matches(one_shot, data.atom_types, data.batch)

        for step in range(int(evaluation["reverse_steps"])):
            time_from = times[step].expand(graphs)
            time_to = times[step + 1].expand(graphs)
            prediction = model(
                tokens,
                coordinates,
                log_volume,
                log_shape,
                data.batch,
                clean_time,
                condition,
                condition_present,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                element_time=time_from,
                lattice_time=clean_time,
            )
            if step in snapshot_indices:
                masked = tokens == categorical.mask_index
                correct, masked_nodes = _top1_correct(
                    prediction.clean_element_logits,
                    data.atom_types,
                    masked,
                )
                oracle_tokens = torch.where(masked, tokens, data.atom_types)
                oracle_prediction = model(
                    oracle_tokens,
                    coordinates,
                    log_volume,
                    log_shape,
                    data.batch,
                    clean_time,
                    condition,
                    condition_present,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                    element_time=time_from,
                    lattice_time=clean_time,
                )
                oracle_correct, _ = _top1_correct(
                    oracle_prediction.clean_element_logits,
                    data.atom_types,
                    masked,
                )
                all_mask_tokens = torch.full_like(tokens, categorical.mask_index)
                all_mask_prediction = model(
                    all_mask_tokens,
                    coordinates,
                    log_volume,
                    log_shape,
                    data.batch,
                    clean_time,
                    condition,
                    condition_present,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                    element_time=time_from,
                    lattice_time=clean_time,
                )
                revealed = ~masked
                for target_time in snapshot_indices[step]:
                    row = totals[target_time]
                    row["on_policy_correct"] += correct
                    row["oracle_correct"] += oracle_correct
                    row["masked"] += masked_nodes
                    row["all_mask_correct"] += int(
                        (
                            all_mask_prediction.clean_element_logits.argmax(-1)
                            == data.atom_types
                        ).sum()
                    )
                    row["all_mask_nodes"] += int(data.atom_types.numel())
                    row["wrong_revealed"] += int(
                        ((tokens != data.atom_types) & revealed).sum()
                    )
                    row["revealed"] += int(revealed.sum())
                    row["observed_time_sum"] += float(times[step])
                    row["batches"] += 1
            probability = categorical.reverse_probabilities(
                tokens,
                prediction.clean_element_logits,
                time_from,
                time_to,
                data.batch,
            )
            tokens = torch.multinomial(
                probability,
                1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
        multistep_correct += int((tokens == data.atom_types).sum())
        multistep_composition += _composition_matches(tokens, data.atom_types, data.batch)
        total_nodes += int(data.atom_types.numel())
        total_graphs += graphs

    snapshots: list[dict[str, float]] = []
    for target_time in snapshot_targets:
        row = totals[target_time]
        snapshots.append(
            {
                "target_time": target_time,
                "observed_time": row["observed_time_sum"] / row["batches"],
                "masked_fraction": row["masked"] / total_nodes,
                "on_policy_masked_top1": row["on_policy_correct"] / max(row["masked"], 1),
                "oracle_revealed_masked_top1": row["oracle_correct"] / max(row["masked"], 1),
                "oracle_minus_on_policy": (row["oracle_correct"] - row["on_policy_correct"])
                / max(row["masked"], 1),
                "all_mask_top1": row["all_mask_correct"] / row["all_mask_nodes"],
                "wrong_revealed_fraction": row["wrong_revealed"] / max(row["revealed"], 1),
            }
        )
    one_shot_site = one_shot_correct / total_nodes
    multistep_site = multistep_correct / total_nodes
    thresholds = protocol["classification_thresholds"]
    high_mask_limited = snapshots[0]["all_mask_top1"] <= float(
        thresholds["high_mask_all_mask_top1_max"]
    )
    compounding = one_shot_site - multistep_site >= float(
        thresholds["one_shot_minus_multistep_site_accuracy_min"]
    )
    exposure_rows = [row for row in snapshots if row["target_time"] <= 0.5]
    exposure_poisoning = any(
        row["oracle_minus_on_policy"]
        >= float(thresholds["oracle_minus_on_policy_masked_top1_min"])
        and row["wrong_revealed_fraction"]
        >= float(thresholds["wrong_revealed_fraction_min"])
        for row in exposure_rows
    )
    if high_mask_limited and (compounding or exposure_poisoning):
        classification = "mixed"
    elif compounding or exposure_poisoning:
        classification = "irreversible_exposure"
    elif high_mask_limited:
        classification = "representation_limited"
    else:
        classification = "neither"
    result: dict[str, Any] = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "validation_indices_sha256": canonical_json_hash(indices.tolist()),
        "snapshots": snapshots,
        "one_shot": {
            "site_accuracy": one_shot_site,
            "exact_composition_accuracy": one_shot_composition / total_graphs,
        },
        "multistep": {
            "site_accuracy": multistep_site,
            "exact_composition_accuracy": multistep_composition / total_graphs,
        },
        "checks": {
            "high_mask_representation_limited": high_mask_limited,
            "one_shot_advantage": compounding,
            "oracle_carrier_advantage": exposure_poisoning,
            "finite": all(
                torch.isfinite(torch.tensor(list(row.values()))).all() for row in snapshots
            ),
        },
        "classification": classification,
        "decision_text": protocol["decision_rule"][classification],
        "optimizer_steps": 0,
        "l1_authorized": False,
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
