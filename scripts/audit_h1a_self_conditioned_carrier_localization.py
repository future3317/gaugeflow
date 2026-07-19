"""Localize the negative self-conditioned topology carrier at fixed t=0.6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from audit_h1a_latent_clean_topology import (
    CarrierAccumulator,
    FinalEdgeCapture,
    ScalarRidgeAccumulator,
    _bootstrap_improvement,
    _forward_batch,
    _graph_coordinate_mse,
    fit_carrier,
    fit_standardized_ridge,
    topology_carrier,
)
from audit_h1a_self_conditioned_topology import _select_indices, _tweedie_field
from torch import nn

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.runtime import load_tensor_free_ema_runtime

VARIANTS = (
    "clean_specific",
    "noisy_specific",
    "probe_specific",
    "tweedie_specific",
    "noisy_plus_probe",
    "noisy_plus_tweedie",
)


def _fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _carriers(
    batch_data: dict[str, Any],
    probe_field: torch.Tensor,
    tweedie_field: torch.Tensor,
) -> dict[str, torch.Tensor]:
    packed = batch_data["packed"]
    edges = batch_data["edges"]
    graphs = int(batch_data["graphs"])

    def make(field: torch.Tensor) -> torch.Tensor:
        return topology_carrier(
            field,
            batch_data["vector_radial"],
            edges.target,
            packed.batch,
            packed.num_nodes,
            graphs,
        )

    clean = make(batch_data["clean_field"])
    noisy = make(batch_data["noisy_field"])
    probe = make(probe_field)
    tweedie = make(tweedie_field)
    return {
        "clean_specific": clean,
        "noisy_specific": noisy,
        "probe_specific": probe,
        "tweedie_specific": tweedie,
        "noisy_plus_probe": torch.cat((noisy, probe), dim=1),
        "noisy_plus_tweedie": torch.cat((noisy, tweedie), dim=1),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--self-conditioning-protocol", type=Path, required=True)
    parser.add_argument("--self-conditioning-result", type=Path, required=True)
    parser.add_argument("--checkpoint-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    specification = load_json_object(args.protocol)
    if specification.get("protocol") != "h1a_self_conditioned_carrier_localization_v1":
        raise ValueError("unexpected carrier-localization protocol")
    prerequisites = specification["prerequisites"]
    frozen = (
        (args.self_conditioning_protocol, "self_conditioning_protocol_sha256"),
        (args.self_conditioning_result, "self_conditioning_result_sha256"),
        (args.checkpoint_protocol, "checkpoint_protocol_sha256"),
        (args.checkpoint_result, "checkpoint_result_sha256"),
        (args.checkpoint, "checkpoint_sha256"),
        (args.cache_root / "manifest.json", "cache_manifest_sha256"),
    )
    for path, key in frozen:
        if sha256_file(path) != str(prerequisites[key]):
            raise ValueError(f"frozen input hash mismatch: {path}")
    if load_json_object(args.self_conditioning_result).get("decision") != prerequisites["self_conditioning_decision"]:
        raise ValueError("self-conditioning result mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint_protocol = load_json_object(args.checkpoint_protocol)
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
    train_dataset = PackedAlexP1Dataset(args.cache_root, "train")
    validation_dataset = PackedAlexP1Dataset(args.cache_root, "val")
    diagnostic = specification["diagnostic"]
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
    time_value = float(diagnostic["time"])
    capture = FinalEdgeCapture(runtime.model)
    before = _fingerprint(runtime.model)
    started = time.perf_counter()
    try:
        probe_accumulator: ScalarRidgeAccumulator | None = None
        generator = torch.Generator(device=device).manual_seed(int(diagnostic["train_noise_seed"]))
        for start in range(0, train_indices.numel(), int(diagnostic["batch_size"])):
            data = _forward_batch(
                runtime,
                diffusion,
                train_dataset,
                train_indices[start : start + int(diagnostic["batch_size"])],
                time_value,
                generator,
                capture,
                specification,
                device=device,
            )
            edge_graph = data["packed"].batch[data["edges"].target]
            if probe_accumulator is None:
                probe_accumulator = ScalarRidgeAccumulator.create(data["features"].shape[-1])
            probe_accumulator.update(data["features"], data["clean_field"], edge_graph, data["graphs"])
        if probe_accumulator is None:
            raise RuntimeError("empty carrier-localization train panel")
        probe = fit_standardized_ridge(probe_accumulator, float(specification["probe"]["ridge_relative"]))

        accumulators: dict[str, CarrierAccumulator] = {}
        generator = torch.Generator(device=device).manual_seed(int(diagnostic["train_noise_seed"]))
        for start in range(0, train_indices.numel(), int(diagnostic["batch_size"])):
            data = _forward_batch(
                runtime,
                diffusion,
                train_dataset,
                train_indices[start : start + int(diagnostic["batch_size"])],
                time_value,
                generator,
                capture,
                specification,
                device=device,
            )
            probe_field = probe.predict(data["features"]).clamp(0.0, 1.0)
            tweedie_field, _ = _tweedie_field(data, diffusion, specification)
            carriers = _carriers(data, probe_field, tweedie_field)
            packed = data["packed"]
            graphs = int(data["graphs"])
            for name, carrier in carriers.items():
                if name not in accumulators:
                    accumulators[name] = CarrierAccumulator.create(carrier.shape[1])
                accumulators[name].update(carrier, data["residual"], packed.batch, graphs)
        coefficients = {
            name: fit_carrier(values, float(specification["carrier"]["ridge_relative"]))
            for name, values in accumulators.items()
        }

        validation_baseline: list[torch.Tensor] = []
        validation_corrected: dict[str, list[torch.Tensor]] = {name: [] for name in VARIANTS}
        generator = torch.Generator(device=device).manual_seed(int(diagnostic["validation_noise_seed"]))
        for start in range(0, validation_indices.numel(), int(diagnostic["batch_size"])):
            data = _forward_batch(
                runtime,
                diffusion,
                validation_dataset,
                validation_indices[start : start + int(diagnostic["batch_size"])],
                time_value,
                generator,
                capture,
                specification,
                device=device,
            )
            probe_field = probe.predict(data["features"]).clamp(0.0, 1.0)
            tweedie_field, _ = _tweedie_field(data, diffusion, specification)
            carriers = _carriers(data, probe_field, tweedie_field)
            packed = data["packed"]
            graphs = int(data["graphs"])
            validation_baseline.append(_graph_coordinate_mse(data["residual"], packed.batch, graphs).cpu())
            for name, carrier in carriers.items():
                correction = torch.einsum("ncd,c->nd", carrier, coefficients[name].to(carrier))
                validation_corrected[name].append(
                    _graph_coordinate_mse(data["residual"] - correction, packed.batch, graphs).cpu()
                )
    finally:
        capture.close()
    baseline = torch.cat(validation_baseline).double()
    acceptance = specification["acceptance"]
    rows: list[dict[str, Any]] = []
    passed: dict[str, bool] = {}
    for index, name in enumerate(VARIANTS):
        corrected = torch.cat(validation_corrected[name]).double()
        improvement = 1.0 - float(corrected.mean() / baseline.mean().clamp_min(1.0e-15))
        interval = _bootstrap_improvement(
            baseline,
            corrected,
            seed=int(diagnostic["bootstrap_seed"]) + index,
            samples=int(diagnostic["bootstrap_samples"]),
        )
        passed[name] = improvement >= float(acceptance["validation_relative_improvement_min"]) and interval[0] > float(
            acceptance["bootstrap_95_low_min"]
        )
        rows.append(
            {
                "variant": name,
                "baseline_mse": float(baseline.mean()),
                "corrected_mse": float(corrected.mean()),
                "relative_improvement": improvement,
                "bootstrap_95_low": interval[0],
                "bootstrap_median": interval[1],
                "bootstrap_95_high": interval[2],
                "passed": passed[name],
                "coefficient_norm": float(coefficients[name].norm()),
                "channels": int(coefficients[name].numel()),
            }
        )
    if passed["tweedie_specific"]:
        decision = "shared_oracle_coefficient_mismatch"
    elif passed["noisy_plus_tweedie"]:
        decision = "dual_state_linear_conversion_sufficient"
    elif passed["probe_specific"] or passed["noisy_plus_probe"]:
        decision = "hidden_topology_requires_learned_vector_conversion"
    else:
        decision = "scalar_topology_carrier_rejected_revisit_conditional_variance"
    after = _fingerprint(runtime.model)
    result = {
        "protocol": specification["protocol"],
        "protocol_sha256": canonical_json_hash(specification),
        "decision": decision,
        "passed": passed,
        "optimizer_steps": 0,
        "checkpoint_parameters_unchanged": before == after,
        "train_indices_sha256": canonical_json_hash(train_indices.tolist()),
        "validation_indices_sha256": canonical_json_hash(validation_indices.tolist()),
        "elapsed_seconds": time.perf_counter() - started,
        "decision_boundary": specification["decision_rule"]["boundary"],
    }
    if before != after:
        raise RuntimeError("carrier localization modified checkpoint parameters")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "carrier_localization.csv", rows)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
