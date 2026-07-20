"""Audit whether the qualified coordinate field uses element and lattice side information."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from scripts.diagnose_h1a_coordinate_generator import _translation_aligned_endpoint_rms

VARIANTS = (
    "clean_elements_clean_lattice",
    "corrupted_elements_clean_lattice",
    "clean_elements_corrupted_lattice",
    "corrupted_elements_corrupted_lattice",
    "permuted_elements_clean_lattice",
    "clean_elements_shuffled_lattice_shape",
)


def _cyclically_permute_node_tokens(
    tokens: torch.Tensor, batch: torch.Tensor, graph_count: int
) -> torch.Tensor:
    """Rotate node tokens once inside every graph without changing composition."""
    if tokens.ndim != 1 or batch.shape != tokens.shape or batch.dtype != torch.long:
        raise ValueError("node-token permutation needs rank-one tokens and int64 batch")
    counts = torch.bincount(batch, minlength=graph_count)
    if bool((counts < 1).any()):
        raise ValueError("node-token permutation requires nonempty graphs")
    starts = counts.cumsum(0) - counts
    node = torch.arange(tokens.numel(), dtype=torch.long, device=tokens.device)
    local = node - starts[batch]
    donor = starts[batch] + torch.remainder(local + 1, counts[batch])
    return tokens[donor]


def _lattice_shape_donor_positions(
    node_counts: torch.Tensor,
    log_volumes: torch.Tensor,
    *,
    bin_width: float,
) -> tuple[torch.Tensor, float]:
    """Return a deterministic within-(N, volume-bin) shape derangement."""
    if (
        node_counts.ndim != 1
        or log_volumes.shape != node_counts.shape
        or node_counts.dtype != torch.long
        or bin_width <= 0.0
    ):
        raise ValueError("invalid lattice donor metadata")
    bins = torch.floor(log_volumes.double() / bin_width).to(torch.long)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for position, (count, volume_bin) in enumerate(zip(node_counts.tolist(), bins.tolist(), strict=True)):
        groups[(int(count), int(volume_bin))].append(position)
    donors = torch.arange(node_counts.numel(), dtype=torch.long)
    changed = torch.zeros_like(donors, dtype=torch.bool)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        source = torch.tensor(positions, dtype=torch.long)
        donors[source] = source.roll(-1)
        changed[source] = True
    return donors, float(changed.double().mean())


def _paired_relative_bootstrap(
    reference: torch.Tensor,
    variant: torch.Tensor,
    *,
    generator: torch.Generator,
    replicates: int,
) -> dict[str, float]:
    """Bootstrap the paired relative mean change over structures."""
    if reference.ndim != 1 or variant.shape != reference.shape or reference.numel() < 2:
        raise ValueError("paired bootstrap requires equal nontrivial vectors")
    if replicates < 100 or not torch.isfinite(reference).all() or not torch.isfinite(variant).all():
        raise ValueError("paired bootstrap inputs are invalid")
    draws = torch.randint(
        reference.numel(),
        (replicates, reference.numel()),
        generator=generator,
    )
    sampled_reference = reference.double()[draws].mean(dim=1).clamp_min(1.0e-30)
    sampled_variant = variant.double()[draws].mean(dim=1)
    distribution = sampled_variant / sampled_reference - 1.0
    quantiles = torch.quantile(distribution, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64))
    return {
        "relative_mean_change": float(variant.double().mean() / reference.double().mean() - 1.0),
        "bootstrap_q025": float(quantiles[0]),
        "bootstrap_median": float(quantiles[1]),
        "bootstrap_q975": float(quantiles[2]),
    }


def _panel_donors(
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    *,
    bin_width: float,
) -> tuple[torch.Tensor, float]:
    node_counts: list[int] = []
    log_volumes: list[float] = []
    for index in indices.tolist():
        item = dataset[int(index)]
        node_counts.append(int(item.atom_types.numel()))
        lattice = item.lattice.reshape(-1, 3, 3)
        log_volumes.append(float(torch.linalg.det(lattice).log().reshape(-1)[0]))
    return _lattice_shape_donor_positions(
        torch.tensor(node_counts, dtype=torch.long),
        torch.tensor(log_volumes, dtype=torch.float64),
        bin_width=bin_width,
    )


def _metric_record(
    graph_error: torch.Tensor,
    graph_target: torch.Tensor,
    graph_prediction: torch.Tensor,
    graph_dot: torch.Tensor,
    endpoint_squared: torch.Tensor,
) -> dict[str, float]:
    error = graph_error.double().mean()
    target = graph_target.double().mean()
    prediction = graph_prediction.double().mean()
    dot = graph_dot.double().mean()
    return {
        "graph_mean_score_mse_per_component": float(error),
        "graph_mean_zero_score_mse_per_component": float(target),
        "graph_mean_score_explained_fraction": float(1.0 - error / target),
        "graph_mean_prediction_to_target_norm": float(torch.sqrt(prediction / target)),
        "graph_mean_prediction_target_cosine": float(dot / torch.sqrt(prediction * target)),
        "endpoint_rms_angstrom": float(torch.sqrt(endpoint_squared.double().mean())),
    }


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
        protocol.get("protocol") != "h1a_j0_side_information_sensitivity_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen J0 protocol")
    evaluation = protocol["evaluation"]
    acceptance = protocol["acceptance"]
    prerequisites = protocol["prerequisites"]
    if tuple(evaluation["variants"]) != VARIANTS or int(evaluation["optimizer_steps"]) != 0:
        raise ValueError("J0 variants or zero-training boundary changed")
    hash_contract = {
        Path("configs/gates/h1a_coordinate_clean_side_information_one_pass_v1.json"): prerequisites[
            "source_protocol_file_sha256"
        ],
        Path("reports/h1a_coordinate_clean_side_information_one_pass_v1/result.json"): prerequisites[
            "source_result_sha256"
        ],
        args.cache_root / "manifest.json": prerequisites["cache_manifest_sha256"],
        args.checkpoint: prerequisites["source_checkpoint_sha256"],
    }
    for path, expected in hash_contract.items():
        if sha256_file(path) != str(expected):
            raise ValueError(f"J0 prerequisite hash mismatch: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_result = load_json_object(
        Path("reports/h1a_coordinate_clean_side_information_one_pass_v1/result.json")
    )
    runtime = load_tensor_free_ema_runtime(
        args.checkpoint,
        device,
        protocol_name=str(prerequisites["source_protocol"]),
        protocol_sha256=str(source_result["protocol_sha256"]),
    )
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    dataset = PackedAlexP1Dataset(args.cache_root, str(evaluation["split"]))
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(evaluation["panel_seed"]))
    )[: int(evaluation["graphs"])]
    donors, donor_coverage = _panel_donors(
        dataset,
        indices,
        bin_width=float(evaluation["lattice_log_volume_bin_width"]),
    )

    times = [float(value) for value in evaluation["times"]]
    batch_size = int(evaluation["batch_size"])
    use_bf16 = runtime.training_config["precision"] == "bf16" and device.type == "cuda"
    per_graph: dict[float, dict[str, dict[str, list[torch.Tensor]]]] = {
        time: {
            variant: {name: [] for name in ("error", "target", "prediction", "dot", "endpoint")}
            for variant in VARIANTS
        }
        for time in times
    }
    standalone_clean_endpoint: dict[float, list[torch.Tensor]] = {time: [] for time in times}
    intervention: dict[float, dict[str, float]] = {}
    tensor_candidate_count = 0

    for time_value in times:
        coordinate_generator = torch.Generator(device=device).manual_seed(
            int(evaluation["coordinate_noise_seed"]) + round(time_value * 1_000_000)
        )
        side_generator = torch.Generator(device=device).manual_seed(
            int(evaluation["side_noise_seed"]) + round(time_value * 1_000_000)
        )
        masked = 0
        nodes_seen = 0
        permuted_changed = 0
        lattice_delta_energy = 0.0
        lattice_values = 0
        shuffled_delta_energy = 0.0
        shuffled_values = 0
        for start in range(0, indices.numel(), batch_size):
            stop = min(start + batch_size, indices.numel())
            selected = indices[start:stop]
            packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
            donor_indices = indices[donors[start:stop]]
            donor_packed = Batch.from_data_list([dataset[int(index)] for index in donor_indices]).to(device)
            graphs = int(packed.num_graphs)
            counts = torch.bincount(packed.batch, minlength=graphs)
            blueprint = ParentBlueprintBatch.from_node_counts(
                counts, dtype=packed.frac_coords.dtype, device=device
            )
            time = packed.lattice.new_full((graphs,), time_value)
            clean = diffusion.noise_clean_batch(
                packed.atom_types,
                packed.frac_coords,
                packed.lattice,
                packed.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=time,
                generator=coordinate_generator,
                clean_side_information=True,
            )
            corrupted = diffusion.noise_clean_batch(
                packed.atom_types,
                packed.frac_coords,
                packed.lattice,
                packed.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=time,
                generator=side_generator,
                clean_side_information=False,
            )
            permuted_tokens = _cyclically_permute_node_tokens(
                packed.atom_types, packed.batch, graphs
            )
            donor_shape = LatticeVolumeShape.from_lattice(
                donor_packed.lattice, blueprint.fractional_to_cartesian
            ).log_shape
            donor_shape = torch.einsum("bij,bj->bi", blueprint.shape_projector, donor_shape)

            variant_tokens = (
                clean.element_tokens,
                corrupted.element_tokens,
                clean.element_tokens,
                corrupted.element_tokens,
                permuted_tokens,
                clean.element_tokens,
            )
            variant_volume = (
                clean.log_volume,
                clean.log_volume,
                corrupted.log_volume,
                corrupted.log_volume,
                clean.log_volume,
                clean.log_volume,
            )
            variant_shape = (
                clean.log_shape,
                clean.log_shape,
                corrupted.log_shape,
                corrupted.log_shape,
                clean.log_shape,
                donor_shape,
            )
            variant_count = len(VARIANTS)
            stacked_batch = torch.cat(
                [packed.batch + variant * graphs for variant in range(variant_count)]
            )
            stacked_time = time.repeat(variant_count)
            graph_count = graphs * variant_count
            condition = clean.log_volume.new_zeros((graph_count, 18))
            condition_present = torch.zeros((graph_count, 1), dtype=torch.bool, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                prediction = runtime.model(
                    torch.cat(variant_tokens),
                    clean.fractional_coordinates.repeat(variant_count, 1),
                    torch.cat(variant_volume),
                    torch.cat(variant_shape),
                    stacked_batch,
                    stacked_time,
                    condition,
                    condition_present,
                    blueprint.shape_projector.repeat(variant_count, 1, 1),
                    blueprint.fractional_to_cartesian.repeat(variant_count, 1, 1),
                )
                # The source qualification used 16-graph forward batches.  A
                # separate same-shape clean call verifies bitwise-equivalent
                # scientific inputs without conflating the BF16 GEMM shape
                # change introduced by the six-variant vectorized audit call.
                standalone_clean_prediction = runtime.model(
                    clean.element_tokens,
                    clean.fractional_coordinates,
                    clean.log_volume,
                    clean.log_shape,
                    packed.batch,
                    time,
                    clean.log_volume.new_zeros((graphs, 18)),
                    torch.zeros((graphs, 1), dtype=torch.bool, device=device),
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                )
            tensor_candidate_count += int(prediction.gauge_atlas.effective_frame_count.sum())
            tensor_candidate_count += int(
                standalone_clean_prediction.gauge_atlas.effective_frame_count.sum()
            )

            sigma = diffusion.coordinate_schedule.sigma(time)[packed.batch].unsqueeze(-1)
            variance = diffusion.coordinate_schedule.variance(time)[packed.batch]
            target = clean.coordinate_scaled_score_target.float() / sigma
            standalone_score = (
                standalone_clean_prediction.coordinate_fractional_scaled_score.float() / sigma
            )
            standalone_estimate = (
                clean.fractional_coordinates + variance.unsqueeze(-1) * standalone_score
            )
            standalone_clean_endpoint[time_value].append(
                _translation_aligned_endpoint_rms(
                    standalone_estimate,
                    packed.frac_coords,
                    packed.lattice,
                    packed.batch,
                )
                .square()
                .cpu()
            )
            graph_target = scatter(
                target.square().sum(dim=-1) / 3.0,
                packed.batch,
                dim=0,
                dim_size=graphs,
                reduce="mean",
            )
            for variant_index, variant in enumerate(VARIANTS):
                begin = variant_index * packed.num_nodes
                end = begin + packed.num_nodes
                predicted = prediction.coordinate_fractional_scaled_score[begin:end].float() / sigma
                graph_error = scatter(
                    (predicted - target).square().sum(dim=-1) / 3.0,
                    packed.batch,
                    dim=0,
                    dim_size=graphs,
                    reduce="mean",
                )
                graph_prediction = scatter(
                    predicted.square().sum(dim=-1) / 3.0,
                    packed.batch,
                    dim=0,
                    dim_size=graphs,
                    reduce="mean",
                )
                graph_dot = scatter(
                    (predicted * target).sum(dim=-1) / 3.0,
                    packed.batch,
                    dim=0,
                    dim_size=graphs,
                    reduce="mean",
                )
                estimate = clean.fractional_coordinates + variance.unsqueeze(-1) * predicted
                endpoint = _translation_aligned_endpoint_rms(
                    estimate, packed.frac_coords, packed.lattice, packed.batch
                ).square()
                values = per_graph[time_value][variant]
                values["error"].append(graph_error.cpu())
                values["target"].append(graph_target.cpu())
                values["prediction"].append(graph_prediction.cpu())
                values["dot"].append(graph_dot.cpu())
                values["endpoint"].append(endpoint.cpu())

            masked += int((corrupted.element_tokens == diffusion.categorical.mask_index).sum())
            nodes_seen += int(packed.num_nodes)
            permuted_changed += int((permuted_tokens != packed.atom_types).sum())
            lattice_delta_energy += float(
                (corrupted.log_volume - clean.log_volume).square().sum()
                + (corrupted.log_shape - clean.log_shape).square().sum()
            )
            lattice_values += int(corrupted.log_volume.numel() + corrupted.log_shape.numel())
            shuffled_delta_energy += float((donor_shape - clean.log_shape).square().sum())
            shuffled_values += int(donor_shape.numel())
        intervention[time_value] = {
            "element_masked_fraction": masked / nodes_seen,
            "permuted_element_changed_fraction": permuted_changed / nodes_seen,
            "lattice_latent_rms": math.sqrt(lattice_delta_energy / lattice_values),
            "shuffled_lattice_shape_rms": math.sqrt(shuffled_delta_energy / shuffled_values),
        }

    source_by_time = {float(row["time"]): row for row in source_result["score_calibration"]}
    rows: list[dict[str, Any]] = []
    clean_reference_max_abs = 0.0
    comparison_by_time: dict[float, dict[str, dict[str, float]]] = {}
    bootstrap_generator = torch.Generator().manual_seed(int(evaluation["bootstrap_seed"]))
    for time_value in times:
        concatenated = {
            variant: {name: torch.cat(parts) for name, parts in metrics.items()}
            for variant, metrics in per_graph[time_value].items()
        }
        clean = concatenated[VARIANTS[0]]
        comparison_by_time[time_value] = {}
        for variant in VARIANTS:
            metrics = concatenated[variant]
            record: dict[str, Any] = {
                "time": time_value,
                "variant": variant,
                **_metric_record(
                    metrics["error"],
                    metrics["target"],
                    metrics["prediction"],
                    metrics["dot"],
                    metrics["endpoint"],
                ),
            }
            if variant != VARIANTS[0]:
                score_change = _paired_relative_bootstrap(
                    clean["error"],
                    metrics["error"],
                    generator=bootstrap_generator,
                    replicates=int(evaluation["bootstrap_replicates"]),
                )
                endpoint_change = _paired_relative_bootstrap(
                    clean["endpoint"],
                    metrics["endpoint"],
                    generator=bootstrap_generator,
                    replicates=int(evaluation["bootstrap_replicates"]),
                )
                record["relative_score_mse_change"] = score_change
                record["relative_endpoint_mse_change"] = endpoint_change
                comparison_by_time[time_value][variant] = score_change
            rows.append(record)

        # The clean branch uses exactly the qualified panel, batching, noise
        # seed and runtime.  Compare the node-weighted source metrics directly.
        clean_prediction = torch.cat(per_graph[time_value][VARIANTS[0]]["prediction"])
        clean_target = torch.cat(per_graph[time_value][VARIANTS[0]]["target"])
        clean_dot = torch.cat(per_graph[time_value][VARIANTS[0]]["dot"])
        clean_error = torch.cat(per_graph[time_value][VARIANTS[0]]["error"])
        clean_endpoint = torch.cat(standalone_clean_endpoint[time_value])
        reproduced = {
            "score_mse_per_component": float(clean_error.mean()),
            "zero_score_mse_per_component": float(clean_target.mean()),
            "score_explained_fraction": float(1.0 - clean_error.mean() / clean_target.mean()),
            "prediction_to_target_norm": float(torch.sqrt(clean_prediction.mean() / clean_target.mean())),
            "prediction_target_cosine": float(
                clean_dot.mean() / torch.sqrt(clean_prediction.mean() * clean_target.mean())
            ),
            "endpoint_rms_angstrom": float(torch.sqrt(clean_endpoint.mean())),
        }
        # Existing source aggregation is node-weighted while the sensitivity
        # bootstrap is intentionally structure-weighted.  Exact reproduction
        # is checked for endpoint RMS; score comparisons are reported but not
        # mixed across weighting conventions.
        clean_reference_max_abs = max(
            clean_reference_max_abs,
            abs(reproduced["endpoint_rms_angstrom"] - float(source_by_time[time_value]["endpoint_rms_angstrom"])),
        )

    focal = float(evaluation["focal_time"])
    focal_comparison = comparison_by_time[focal]
    element_effect = focal_comparison["corrupted_elements_clean_lattice"]
    lattice_effect = focal_comparison["clean_elements_corrupted_lattice"]
    relative_min = float(acceptance["focal_relative_score_mse_degradation_min"])
    lower_min = float(acceptance["focal_paired_bootstrap_lower_bound_min"])
    chemistry_used = (
        element_effect["relative_mean_change"] >= relative_min
        and element_effect["bootstrap_q025"] > lower_min
    )
    lattice_used = (
        lattice_effect["relative_mean_change"] >= relative_min
        and lattice_effect["bootstrap_q025"] > lower_min
    )
    focal_intervention = intervention[focal]
    checks = {
        "finite": all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if key not in {"variant", "relative_score_mse_change", "relative_endpoint_mse_change"}
        ),
        "clean_reference_reproduced": clean_reference_max_abs
        <= float(acceptance["clean_reference_metric_max_abs"]),
        "element_corruption_valid": float(acceptance["focal_element_masked_fraction_min"])
        <= focal_intervention["element_masked_fraction"]
        <= float(acceptance["focal_element_masked_fraction_max"]),
        "lattice_corruption_valid": focal_intervention["lattice_latent_rms"]
        >= float(acceptance["focal_lattice_latent_rms_min"]),
        "element_permutation_valid": focal_intervention["permuted_element_changed_fraction"]
        >= float(acceptance["permuted_element_changed_fraction_min"]),
        "lattice_shuffle_coverage": donor_coverage
        >= float(acceptance["shuffled_lattice_graph_coverage_min"]),
        "lattice_shuffle_nontrivial": focal_intervention["shuffled_lattice_shape_rms"]
        >= float(acceptance["shuffled_lattice_shape_rms_min"]),
        "chemistry_side_information_used": chemistry_used,
        "lattice_side_information_used": lattice_used,
        "tensor_bypass": tensor_candidate_count == int(acceptance["tensor_candidates"]),
    }
    authorize_j1 = all(checks.values())
    decision_key = "authorize_j1" if authorize_j1 else "do_not_authorize_j1"
    both = focal_comparison["corrupted_elements_corrupted_lattice"]["relative_mean_change"]
    interaction = both - element_effect["relative_mean_change"] - lattice_effect["relative_mean_change"]
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "panel_indices_sha256": canonical_json_hash(indices.tolist()),
        "semantics": prerequisites["source_semantics"],
        "rows": rows,
        "intervention_diagnostics": {str(key): value for key, value in intervention.items()},
        "shuffled_lattice_graph_coverage": donor_coverage,
        "clean_reference_endpoint_max_abs": clean_reference_max_abs,
        "focal_time": focal,
        "focal_element_score_effect": element_effect,
        "focal_lattice_score_effect": lattice_effect,
        "focal_both_relative_score_effect": both,
        "focal_interaction_relative_score_effect": interaction,
        "checks": checks,
        "authorize_j1": authorize_j1,
        "decision": decision_key,
        "decision_text": protocol["decision_rule"][decision_key],
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
