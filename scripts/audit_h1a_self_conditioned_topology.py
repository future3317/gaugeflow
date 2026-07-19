"""Audit whether a frozen quotient Tweedie estimate supplies clean topology."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from audit_h1a_latent_clean_topology import (
    CarrierAccumulator,
    FinalEdgeCapture,
    ScalarRidgeAccumulator,
    _binary_auc,
    _bootstrap_improvement,
    _edge_graph_sum,
    _forward_batch,
    _graph_coordinate_mse,
    exact_all_pair_geometry,
    fit_carrier,
    fit_standardized_ridge,
    topology_carrier,
    topology_fields,
)
from diagnose_h1a_coordinate_generator import _translation_aligned_endpoint_rms
from torch import nn

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.quotient_score import quotient_tweedie_endpoint
from gaugeflow.production.runtime import load_tensor_free_ema_runtime

VARIANTS = (
    "noisy_current",
    "frozen_linear_probe",
    "tweedie_self_conditioned",
    "clean_oracle",
)


def _parameter_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty self-conditioning audit table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tweedie_field(
    batch_data: dict[str, Any],
    diffusion: TensorFreeHybridDiffusion,
    specification: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = batch_data["packed"]
    noisy = batch_data["noisy"]
    prediction = batch_data["prediction"]
    graphs = int(batch_data["graphs"])
    sigma = diffusion.coordinate_schedule.sigma(noisy.time.float())
    estimate = quotient_tweedie_endpoint(
        noisy.fractional_coordinates.float(),
        prediction.coordinate_fractional_scaled_score.float(),
        sigma,
        packed.batch,
        graphs,
    )
    source, target, distance, _ = exact_all_pair_geometry(estimate, batch_data["noisy_lattice"], packed.batch)
    pairs = batch_data["edges"]
    if not (torch.equal(source, pairs.source) and torch.equal(target, pairs.target)):
        raise RuntimeError("Tweedie and v2 all-pair supports differ")
    topology = specification["topology"]
    _, field, _ = topology_fields(
        batch_data["clean_distance"],
        distance,
        target,
        packed.batch,
        cutoff=float(topology["production_cutoff_angstrom"]),
        multiplier=float(topology["first_shell_multiplier"]),
        relative_width=float(topology["transition_relative_width"]),
    )
    endpoint_rms = _translation_aligned_endpoint_rms(
        estimate,
        packed.frac_coords.float(),
        packed.lattice.float(),
        packed.batch,
    )
    return field, endpoint_rms


def _select_indices(
    dataset: PackedAlexP1Dataset,
    *,
    minimum_sites: int,
    count: int,
    seed: int,
) -> torch.Tensor:
    eligible = torch.nonzero(dataset.node_counts >= minimum_sites, as_tuple=False).flatten()
    order = torch.randperm(eligible.numel(), generator=torch.Generator().manual_seed(seed))
    selected = eligible[order[:count]]
    if selected.numel() != count:
        raise RuntimeError("not enough multi-site structures for self-conditioning audit")
    return selected


def _field_metrics(
    field: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    edge_graph: torch.Tensor,
    graphs: int,
    target_mean: float,
    threshold: float,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    counts = torch.bincount(edge_graph, minlength=graphs).clamp_min(1).to(field)
    mse = _edge_graph_sum((field - clean).square(), edge_graph, graphs) / counts
    noisy_mse = _edge_graph_sum((noisy - clean).square(), edge_graph, graphs) / counts
    mean_mse = _edge_graph_sum((target_mean - clean).square(), edge_graph, graphs) / counts
    soft_jaccard = _edge_graph_sum(torch.minimum(field, clean), edge_graph, graphs) / (
        _edge_graph_sum(torch.maximum(field, clean), edge_graph, graphs).clamp_min(1.0e-12)
    )
    clean_hard = clean >= threshold
    auc: list[float] = []
    for graph in range(graphs):
        mask = edge_graph == graph
        value = _binary_auc(field[mask].cpu(), clean_hard[mask].cpu())
        if math.isfinite(value):
            auc.append(value)
    return {
        "mse": mse.cpu(),
        "noisy_mse": noisy_mse.cpu(),
        "mean_mse": mean_mse.cpu(),
        "soft_jaccard": soft_jaccard.cpu(),
    }, auc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--topology-protocol", type=Path, required=True)
    parser.add_argument("--persistence-protocol", type=Path, required=True)
    parser.add_argument("--persistence-result", type=Path, required=True)
    parser.add_argument("--checkpoint-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    specification = load_json_object(args.protocol)
    if specification.get("protocol") != "h1a_self_conditioned_topology_attribution_v1":
        raise ValueError("unexpected self-conditioning protocol")
    prerequisites = specification["prerequisites"]
    frozen = (
        (args.topology_protocol, "topology_protocol_sha256"),
        (args.persistence_protocol, "persistence_protocol_sha256"),
        (args.persistence_result, "persistence_result_sha256"),
        (args.checkpoint_protocol, "checkpoint_protocol_sha256"),
        (args.checkpoint_result, "checkpoint_result_sha256"),
        (args.checkpoint, "checkpoint_sha256"),
        (args.cache_root / "manifest.json", "cache_manifest_sha256"),
    )
    for path, key in frozen:
        if sha256_file(path) != str(prerequisites[key]):
            raise ValueError(f"frozen input hash mismatch: {path}")
    topology_specification = load_json_object(args.topology_protocol)
    for section in ("topology", "probe"):
        source = topology_specification[section]
        active = specification[section]
        for key, value in active.items():
            if key in source and source[key] != value:
                raise ValueError(f"v2 {section} contract changed at {key}")
    source_times = list(map(float, topology_specification["diagnostic"]["times"]))
    for time_value in specification["diagnostic"]["times"]:
        source_index = int(
            specification["diagnostic"]["source_v2_time_indices"][str(time_value)]
        )
        if source_index >= len(source_times) or source_times[source_index] != float(
            time_value
        ):
            raise ValueError("self-conditioning noise seed is not aligned to v2 time")
    if load_json_object(args.persistence_result).get("decision") != prerequisites["persistence_decision"]:
        raise ValueError("persistence decision mismatch")
    checkpoint_protocol = load_json_object(args.checkpoint_protocol)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    runtime = load_tensor_free_ema_runtime(
        args.checkpoint,
        device,
        protocol_name=str(checkpoint_protocol["protocol"]),
        protocol_sha256=canonical_json_hash(checkpoint_protocol),
    )
    runtime.model.eval()
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    diagnostic = specification["diagnostic"]
    train_dataset = PackedAlexP1Dataset(args.cache_root, "train")
    validation_dataset = PackedAlexP1Dataset(args.cache_root, "val")
    train_indices = _select_indices(
        train_dataset,
        minimum_sites=int(diagnostic["minimum_sites"]),
        count=int(diagnostic["train_graphs"]),
        seed=int(diagnostic["train_selection_seed"]),
    )
    validation_indices = _select_indices(
        validation_dataset,
        minimum_sites=int(diagnostic["minimum_sites"]),
        count=int(diagnostic["validation_graphs"]),
        seed=int(diagnostic["validation_selection_seed"]),
    )
    fingerprint_before = _parameter_fingerprint(runtime.model)
    capture = FinalEdgeCapture(runtime.model)
    topology_rows: list[dict[str, Any]] = []
    carrier_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for time_index, time_value in enumerate(diagnostic["times"]):
            source_time_index = int(
                diagnostic["source_v2_time_indices"][str(time_value)]
            )
            probe_accumulator: ScalarRidgeAccumulator | None = None
            carrier_accumulator: CarrierAccumulator | None = None
            train_generator = torch.Generator(device=device).manual_seed(
                int(diagnostic["train_noise_seed"]) + source_time_index
            )
            for start in range(0, train_indices.numel(), int(diagnostic["batch_size"])):
                batch_data = _forward_batch(
                    runtime,
                    diffusion,
                    train_dataset,
                    train_indices[start : start + int(diagnostic["batch_size"])],
                    float(time_value),
                    train_generator,
                    capture,
                    specification,
                    device=device,
                )
                packed = batch_data["packed"]
                edges = batch_data["edges"]
                edge_graph = packed.batch[edges.target]
                if probe_accumulator is None:
                    probe_accumulator = ScalarRidgeAccumulator.create(batch_data["features"].shape[-1])
                    carrier_accumulator = CarrierAccumulator.create(batch_data["radial"].shape[-1])
                probe_accumulator.update(
                    batch_data["features"],
                    batch_data["clean_field"],
                    edge_graph,
                    batch_data["graphs"],
                )
                clean_carrier = topology_carrier(
                    batch_data["clean_field"],
                    batch_data["vector_radial"],
                    edges.target,
                    packed.batch,
                    packed.num_nodes,
                    batch_data["graphs"],
                )
                if carrier_accumulator is None:
                    raise RuntimeError("carrier accumulator was not initialized")
                carrier_accumulator.update(
                    clean_carrier,
                    batch_data["residual"],
                    packed.batch,
                    batch_data["graphs"],
                )
            if probe_accumulator is None or carrier_accumulator is None:
                raise RuntimeError("empty self-conditioning train panel")
            probe = fit_standardized_ridge(probe_accumulator, float(specification["probe"]["ridge_relative"]))
            coefficient = fit_carrier(carrier_accumulator, float(specification["carrier"]["ridge_relative"]))
            metric_parts = {
                name: {key: [] for key in ("mse", "noisy_mse", "mean_mse", "soft_jaccard")} for name in VARIANTS
            }
            auc_parts: dict[str, list[float]] = {name: [] for name in VARIANTS}
            baseline_parts: list[torch.Tensor] = []
            corrected_parts: dict[str, list[torch.Tensor]] = {name: [] for name in VARIANTS}
            endpoint_parts: list[torch.Tensor] = []
            validation_generator = torch.Generator(device=device).manual_seed(
                int(diagnostic["validation_noise_seed"]) + source_time_index
            )
            for start in range(0, validation_indices.numel(), int(diagnostic["batch_size"])):
                batch_data = _forward_batch(
                    runtime,
                    diffusion,
                    validation_dataset,
                    validation_indices[start : start + int(diagnostic["batch_size"])],
                    float(time_value),
                    validation_generator,
                    capture,
                    specification,
                    device=device,
                )
                packed = batch_data["packed"]
                edges = batch_data["edges"]
                graphs = int(batch_data["graphs"])
                edge_graph = packed.batch[edges.target]
                probe_field = probe.predict(batch_data["features"]).clamp(0.0, 1.0)
                tweedie_field, endpoint_rms = _tweedie_field(batch_data, diffusion, specification)
                endpoint_parts.append(endpoint_rms.cpu())
                fields = {
                    "noisy_current": batch_data["noisy_field"],
                    "frozen_linear_probe": probe_field,
                    "tweedie_self_conditioned": tweedie_field,
                    "clean_oracle": batch_data["clean_field"],
                }
                baseline_parts.append(_graph_coordinate_mse(batch_data["residual"], packed.batch, graphs).cpu())
                for name, field in fields.items():
                    metrics, auc = _field_metrics(
                        field,
                        batch_data["clean_field"],
                        batch_data["noisy_field"],
                        edge_graph,
                        graphs,
                        probe.target_mean,
                        float(specification["topology"]["hard_probability_threshold"]),
                    )
                    for key, values in metrics.items():
                        metric_parts[name][key].append(values)
                    auc_parts[name].extend(auc)
                    carrier = topology_carrier(
                        field,
                        batch_data["vector_radial"],
                        edges.target,
                        packed.batch,
                        packed.num_nodes,
                        graphs,
                    )
                    correction = torch.einsum("ncd,c->nd", carrier, coefficient.to(carrier))
                    corrected_parts[name].append(
                        _graph_coordinate_mse(
                            batch_data["residual"] - correction,
                            packed.batch,
                            graphs,
                        ).cpu()
                    )
            endpoint = torch.cat(endpoint_parts).double()
            endpoint_rows.append(
                {
                    "time": float(time_value),
                    "mean_periodic_rms_angstrom": float(endpoint.mean()),
                    "median_periodic_rms_angstrom": float(endpoint.median()),
                    "p95_periodic_rms_angstrom": float(torch.quantile(endpoint, 0.95)),
                }
            )
            baseline = torch.cat(baseline_parts).double()
            oracle_improvement = 1.0 - float(
                torch.cat(corrected_parts["clean_oracle"]).double().mean() / baseline.mean().clamp_min(1.0e-15)
            )
            for variant_index, name in enumerate(VARIANTS):
                values = {key: torch.cat(parts).double() for key, parts in metric_parts[name].items()}
                topology_rows.append(
                    {
                        "time": float(time_value),
                        "variant": name,
                        "topology_mse": float(values["mse"].mean()),
                        "explained_fraction": 1.0
                        - float(values["mse"].mean() / values["mean_mse"].mean().clamp_min(1.0e-15)),
                        "mse_improvement_over_noisy": 1.0
                        - float(values["mse"].mean() / values["noisy_mse"].mean().clamp_min(1.0e-15)),
                        "soft_jaccard": float(values["soft_jaccard"].mean()),
                        "auc": sum(auc_parts[name]) / len(auc_parts[name]),
                    }
                )
                corrected = torch.cat(corrected_parts[name]).double()
                improvement = 1.0 - float(corrected.mean() / baseline.mean().clamp_min(1.0e-15))
                interval = _bootstrap_improvement(
                    baseline,
                    corrected,
                    seed=int(diagnostic["bootstrap_seed"]) + 10 * time_index + variant_index,
                    samples=int(diagnostic["bootstrap_samples"]),
                )
                carrier_rows.append(
                    {
                        "time": float(time_value),
                        "variant": name,
                        "baseline_mse": float(baseline.mean()),
                        "corrected_mse": float(corrected.mean()),
                        "relative_improvement": improvement,
                        "oracle_gain_fraction": improvement / max(oracle_improvement, 1.0e-12),
                        "bootstrap_95_low": interval[0],
                        "bootstrap_median": interval[1],
                        "bootstrap_95_high": interval[2],
                    }
                )
    finally:
        capture.close()
    fingerprint_after = _parameter_fingerprint(runtime.model)
    focus_time = float(diagnostic["focus_time"])
    focus_topology = next(
        row for row in topology_rows if row["variant"] == "tweedie_self_conditioned" and row["time"] == focus_time
    )
    focus_carrier = next(
        row for row in carrier_rows if row["variant"] == "tweedie_self_conditioned" and row["time"] == focus_time
    )
    acceptance = specification["acceptance"]
    checks = {
        "focus_topology_auc": float(focus_topology["auc"]) >= float(acceptance["focus_topology_auc_min"]),
        "focus_topology_improves_over_noisy": float(focus_topology["mse_improvement_over_noisy"])
        >= float(acceptance["focus_topology_mse_improvement_over_noisy_min"]),
        "focus_carrier_improves_residual": float(focus_carrier["relative_improvement"])
        >= float(acceptance["focus_carrier_relative_improvement_min"]),
        "focus_carrier_retains_oracle_gain": float(focus_carrier["oracle_gain_fraction"])
        >= float(acceptance["focus_carrier_oracle_gain_fraction_min"]),
        "focus_carrier_bootstrap_supported": float(focus_carrier["bootstrap_95_low"])
        > float(acceptance["focus_carrier_bootstrap_95_low_min"]),
        "all_outputs_finite": all(
            math.isfinite(float(value))
            for rows in (topology_rows, carrier_rows, endpoint_rows)
            for row in rows
            for value in row.values()
            if not isinstance(value, str)
        ),
        "checkpoint_parameters_unchanged": fingerprint_before == fingerprint_after,
    }
    topology_predictive = checks["focus_topology_auc"] and checks["focus_topology_improves_over_noisy"]
    carrier_causal = all(
        checks[name]
        for name in (
            "focus_carrier_improves_residual",
            "focus_carrier_retains_oracle_gain",
            "focus_carrier_bootstrap_supported",
        )
    )
    if topology_predictive and carrier_causal:
        decision = "authorize_separate_staged_self_conditioning_proposal"
    elif topology_predictive:
        decision = "self_conditioned_topology_predictive_but_vector_conversion_limited"
    else:
        decision = "self_conditioned_topology_not_predictive_revisit_conditional_variance"
    result = {
        "protocol": specification["protocol"],
        "protocol_sha256": canonical_json_hash(specification),
        "decision": decision,
        "checks": checks,
        "focus_time": focus_time,
        "focus_topology": focus_topology,
        "focus_carrier": focus_carrier,
        "optimizer_steps": 0,
        "checkpoint_parameters_unchanged": fingerprint_before == fingerprint_after,
        "train_indices_sha256": canonical_json_hash(train_indices.tolist()),
        "validation_indices_sha256": canonical_json_hash(validation_indices.tolist()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_mib": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0),
        "decision_boundary": specification["decision_rule"]["boundary"],
    }
    if not checks["all_outputs_finite"] or not checks["checkpoint_parameters_unchanged"]:
        raise RuntimeError("self-conditioning audit violated a validity precondition")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "topology_recovery.csv", topology_rows)
    _write_csv(args.output / "carrier_causality.csv", carrier_rows)
    _write_csv(args.output / "endpoint_estimator.csv", endpoint_rows)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
