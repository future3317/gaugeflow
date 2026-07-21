"""Evaluate and select one frozen GaugeFlow-base capacity candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from diagnose_h1a_coordinate_generator import _score_calibration
from evaluate_h1a_generated_side_coordinate_exposure import _nearest_neighbours, _wasserstein
from evaluate_h1a_p1_protocol import _validation_losses

from gaugeflow.file_utils import canonical_json_hash, load_json_object, numeric_tree_is_finite, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.checkpointing import read_production_checkpoint_metadata
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.runtime import load_tensor_free_ema_runtime


def select_capacity_candidate(
    rows: list[dict[str, Any]],
    specification: dict[str, Any],
) -> str | None:
    """Apply the preregistered Pareto-margin rule without capacity preference leakage."""

    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None
    best_ratio = min(float(row["validation_coordinate_ratio"]) for row in eligible)
    best_w1 = min(
        float(row["clean_side_conditional_rollout"]["node_nearest_w1_normalized"])
        for row in eligible
    )
    ratio_margin = float(specification["validation_ratio_best_absolute_margin"])
    w1_margin = float(specification["full_prior_w1_best_absolute_margin"])
    sufficient = [
        row
        for row in eligible
        if float(row["validation_coordinate_ratio"]) <= best_ratio + ratio_margin
        and float(row["clean_side_conditional_rollout"]["node_nearest_w1_normalized"])
        <= best_w1 + w1_margin
    ]
    return str(
        min(
            sufficient,
            key=lambda row: (
                int(row["parameter_count"]),
                float(row["validation_coordinate_ratio"]),
                float(row["clean_side_conditional_rollout"]["node_nearest_w1_normalized"]),
            ),
        )["candidate"]
    )


@torch.no_grad()
def _clean_side_conditional_rollout_metrics(
    checkpoint: Path,
    candidate_protocol: dict[str, Any],
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    evaluation: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    protocol_name = str(candidate_protocol["protocol"])
    protocol_hash = canonical_json_hash(candidate_protocol)
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=protocol_name,
        protocol_sha256=protocol_hash,
    )
    sampler = TensorFreeReverseSampler(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
        categorical_path=str(runtime.training_config["categorical_path"]),
    )
    initial_generator = torch.Generator(device=device).manual_seed(
        int(evaluation["coordinate_initial_seed"])
    )
    brownian_generator = torch.Generator(device=device).manual_seed(
        int(evaluation["coordinate_brownian_seed"])
    )
    reference_nodes: list[torch.Tensor] = []
    generated_nodes: list[torch.Tensor] = []
    generated_graph_minima: list[torch.Tensor] = []
    failures: list[dict[str, Any]] = []
    batch_size = int(evaluation["full_prior_batch_size"])
    for start in range(0, indices.numel(), batch_size):
        selected = indices[start : start + batch_size]
        source = dataset.select_model_batch(selected, device=device)
        graphs = int(source.lattice.shape[0])
        counts = torch.bincount(source.batch, minlength=graphs)
        blueprint = ParentBlueprintBatch.from_node_counts(
            counts,
            dtype=source.lattice.dtype,
            device=device,
        )
        initial = sampler.initialize_coordinate_state(blueprint, generator=initial_generator)
        try:
            generated = sampler.sample_coordinates(
                source.atom_types,
                source.lattice,
                blueprint,
                steps=int(evaluation["coordinate_steps"]),
                initial_state=initial,
                continuous_generator=brownian_generator,
                continuous_mode="reverse_sde",
                time_grid="uniform_log_alpha",
            )
        except (SamplingFailure, RuntimeError, ValueError) as error:
            failures.append(
                {
                    "start": start,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        reference_node, _ = _nearest_neighbours(
            source.fractional_coordinates,
            source.lattice,
            source.batch,
        )
        generated_node, generated_graph = _nearest_neighbours(
            generated.fractional_coordinates,
            source.lattice,
            source.batch,
        )
        reference_nodes.append(reference_node)
        generated_nodes.append(generated_node)
        generated_graph_minima.append(generated_graph)
    if not generated_nodes:
        return {
            "finite": False,
            "sampling_failures": failures,
            "node_nearest_w1_normalized": math.inf,
            "valid_minimum_distance_fraction": 0.0,
        }
    reference = torch.cat(reference_nodes).double()
    generated = torch.cat(generated_nodes).double()
    graph_minimum = torch.cat(generated_graph_minima).double()
    reference_iqr = torch.quantile(reference, 0.75) - torch.quantile(reference, 0.25)
    if float(reference_iqr) <= 0.0:
        raise ValueError("conditional-rollout reference nearest-neighbour IQR must be positive")
    probabilities = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float64)
    return {
        "graphs": int(indices.numel()),
        "panel_indices_sha256": canonical_json_hash(indices.tolist()),
        "node_nearest_w1_angstrom": _wasserstein(
            generated,
            reference,
            int(evaluation["wasserstein_points"]),
        ),
        "node_nearest_w1_normalized": _wasserstein(
            generated,
            reference,
            int(evaluation["wasserstein_points"]),
        )
        / float(reference_iqr),
        "generated_node_nearest_quantiles_angstrom": torch.quantile(
            generated, probabilities
        ).tolist(),
        "reference_node_nearest_quantiles_angstrom": torch.quantile(
            reference, probabilities
        ).tolist(),
        "valid_minimum_distance_fraction": float(
            (
                graph_minimum
                >= float(evaluation["minimum_distance_angstrom"])
            )
            .double()
            .mean()
        ),
        "finite": bool(torch.isfinite(generated).all()),
        "sampling_failures": failures,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        help="evaluate one preregistered candidate without making the final capacity selection",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "gaugeflow_base_capacity_screen_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen capacity-selection protocol")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("capacity selection requires CUDA")
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    evaluation = protocol["evaluation"]
    validation_indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    score_indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["score_seed"])),
    )[: int(evaluation["score_graphs"])]
    full_prior_indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["full_prior_seed"])),
    )[: int(evaluation["full_prior_graphs"])]
    registered_candidates = list(protocol["candidates"])
    if args.candidate is not None:
        registered_candidates = [
            candidate
            for candidate in registered_candidates
            if str(candidate["name"]) == args.candidate
        ]
        if len(registered_candidates) != 1:
            raise ValueError("requested capacity candidate is not preregistered")
    rows: list[dict[str, Any]] = []
    for candidate in registered_candidates:
        name = str(candidate["name"])
        candidate_path = Path(candidate["protocol"])
        smoke_path = Path("reports/gaugeflow_base_capacity_execution_smoke_v1") / f"{name}.json"
        if sha256_file(candidate_path) != str(candidate["protocol_sha256"]):
            raise ValueError(f"candidate protocol changed: {name}")
        if sha256_file(smoke_path) != str(candidate["smoke_result_sha256"]):
            raise ValueError(f"candidate smoke result changed: {name}")
        candidate_protocol = load_json_object(candidate_path)
        training = candidate_protocol["training"]
        final_step = int(training["steps"])
        run = args.runs_root / name / "seed_5705"
        checkpoint = run / f"checkpoint_step_{final_step:08d}.pt"
        records = [
            json.loads(line)
            for line in (run / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if not records or int(records[-1]["step"]) != final_step or not numeric_tree_is_finite(records):
            raise ValueError(f"incomplete or non-finite training log: {name}")
        metadata = read_production_checkpoint_metadata(checkpoint)
        if (
            metadata.get("protocol") != candidate_protocol["protocol"]
            or metadata.get("protocol_sha256") != canonical_json_hash(candidate_protocol)
            or int(metadata.get("effective_batch_size", -1)) != 64
        ):
            raise ValueError(f"checkpoint provenance mismatch: {name}")
        validation = {
            str(step): _validation_losses(
                run / f"checkpoint_step_{int(step):08d}.pt",
                dataset,
                validation_indices,
                device=device,
                seed=int(evaluation["validation_noise_seed"]),
                protocol_name=str(candidate_protocol["protocol"]),
                protocol_sha256=canonical_json_hash(candidate_protocol),
            )
            for step in training["checkpoint_steps"]
        }
        ratio = validation[str(final_step)]["coordinate"] / validation["0"]["coordinate"]
        runtime = load_tensor_free_ema_runtime(
            checkpoint,
            device,
            protocol_name=str(candidate_protocol["protocol"]),
            protocol_sha256=canonical_json_hash(candidate_protocol),
        )
        score = _score_calibration(
            runtime,
            dataset,
            score_indices,
            {
                "batch_size": 16,
                "noise_seed": int(evaluation["score_noise_seed"]),
                "times": evaluation["score_times"],
            },
            device=device,
        )
        del runtime
        conditional_rollout = _clean_side_conditional_rollout_metrics(
            checkpoint,
            candidate_protocol,
            dataset,
            full_prior_indices,
            evaluation,
            device=device,
        )
        t06 = {float(row["time"]): row for row in score}[0.6]
        acceptance = protocol["eligibility"]
        checks = {
            "finite": numeric_tree_is_finite(
                {
                    "training": records,
                    "validation": validation,
                    "score": score,
                    "conditional_rollout": conditional_rollout,
                }
            ),
            "validation_coordinate_ratio": ratio
            <= float(acceptance["validation_coordinate_ratio_max"]),
            "t06_score_explained_fraction": float(t06["score_explained_fraction"])
            >= float(acceptance["t06_score_explained_fraction_min"]),
            "clean_side_conditional_rollout_node_nearest_w1": float(
                conditional_rollout["node_nearest_w1_normalized"]
            )
            <= float(acceptance["full_prior_node_nearest_w1_normalized_max"]),
            "clean_side_conditional_rollout_valid_distance": float(
                conditional_rollout["valid_minimum_distance_fraction"]
            )
            >= float(acceptance["full_prior_valid_minimum_distance_fraction_min"]),
            "zero_sampling_failures": len(conditional_rollout["sampling_failures"])
            == int(acceptance["sampling_failures"]),
        }
        rows.append(
            {
                "candidate": name,
                "parameter_count": int(candidate["parameter_count"]),
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation_coordinate_ratio": ratio,
                "validation": validation,
                "score_calibration": score,
                "clean_side_conditional_rollout": conditional_rollout,
                "training_throughput_graphs_per_second": float(records[-1]["graphs_per_second"]),
                "peak_cuda_memory_mib": float(records[-1]["peak_cuda_memory_mib"]),
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )
        torch.cuda.empty_cache()
    complete_screen = args.candidate is None
    selected = (
        select_capacity_candidate(rows, protocol["selection"])
        if complete_screen
        else None
    )
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "candidates": rows,
        "complete_screen": complete_screen,
        "metric_interpretation": {
            "clean_side_conditional_rollout": (
                "coordinates start from the coordinate prior, but atom types, lattice and node count "
                "are ground-truth side information; this is not free joint generation"
            ),
            "protocol_legacy_name": (
                "the preregistered JSON uses full_prior in internal field names; thresholds and "
                "candidate sets were not changed after execution"
            ),
        },
        "selected_candidate": selected,
        "qualified": selected is not None if complete_screen else None,
        "decision": (
            (
                f"select {selected} as the minimum sufficient GaugeFlow-base capacity"
                if selected is not None
                else "no capacity is eligible; stop before joint GaugeFlow-base pretraining"
            )
            if complete_screen
            else "interim candidate evaluation only; wait for every preregistered capacity"
        ),
        "boundary": protocol["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not complete_screen or selected is not None else 2)


if __name__ == "__main__":
    main()
