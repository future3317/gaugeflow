"""Qualify a nonlinear state-derived all-pair Cartesian residual readout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from audit_h1a_latent_clean_topology import (
    FinalEdgeCapture,
    ScalarRidgeAccumulator,
    _bootstrap_improvement,
    _forward_batch,
    _graph_coordinate_mse,
    fit_standardized_ridge,
)
from audit_h1a_self_conditioned_carrier_localization import _fingerprint, _write_csv
from audit_h1a_self_conditioned_topology import _select_indices, _tweedie_field
from torch import nn

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.state_projection import graph_mean, sorted_segment_sum


class NonlinearPairVectorReadout(nn.Module):
    """O(3)-equivariant O(E*C) pair-to-node vector field."""

    def __init__(self, features: int, hidden: int, radial_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, radial_channels, bias=False),
        )
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0e-2)

    def forward(
        self,
        features: torch.Tensor,
        vector_radial: torch.Tensor,
        target: torch.Tensor,
        batch: torch.Tensor,
        nodes: int,
        graphs: int,
    ) -> torch.Tensor:
        coefficients = self.network(features)
        pair_vector = torch.einsum("ec,ecd->ed", coefficients, vector_radial)
        correction = sorted_segment_sum(pair_vector, target, nodes)
        degree = torch.bincount(target, minlength=nodes).clamp_min(1).to(correction)
        correction = correction / degree.sqrt().unsqueeze(-1)
        return correction - graph_mean(correction, batch, graphs)[batch]


def _to_device(batch_data: dict[str, torch.Tensor | int], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch_data.items()}


@torch.no_grad()
def _collect_batches(
    runtime: Any,
    diffusion: TensorFreeHybridDiffusion,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    generator: torch.Generator,
    capture: FinalEdgeCapture,
    specification: dict[str, Any],
    probe: Any,
    *,
    device: torch.device,
) -> list[dict[str, torch.Tensor | int]]:
    diagnostic = specification["diagnostic"]
    batches: list[dict[str, torch.Tensor | int]] = []
    for start in range(0, indices.numel(), int(diagnostic["batch_size"])):
        data = _forward_batch(
            runtime,
            diffusion,
            dataset,
            indices[start : start + int(diagnostic["batch_size"])],
            float(diagnostic["time"]),
            generator,
            capture,
            specification,
            device=device,
        )
        probe_field = probe.predict(data["features"]).clamp(0.0, 1.0)
        tweedie_field, _ = _tweedie_field(data, diffusion, specification)
        packed = data["packed"]
        edges = data["edges"]
        features = data["features"].float()
        topology_features = torch.cat((features, probe_field[:, None], tweedie_field[:, None]), dim=-1)
        base_features = torch.cat((features, features.new_zeros((features.shape[0], 2))), dim=-1)
        batches.append(
            {
                "base_features": base_features.cpu(),
                "topology_features": topology_features.cpu(),
                "vector_radial": data["vector_radial"].float().cpu(),
                "target": edges.target.cpu(),
                "batch": packed.batch.cpu(),
                "residual": data["residual"].float().cpu(),
                "nodes": int(packed.num_nodes),
                "graphs": int(data["graphs"]),
            }
        )
    return batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--localization-protocol", type=Path, required=True)
    parser.add_argument("--localization-result", type=Path, required=True)
    parser.add_argument("--checkpoint-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specification = load_json_object(args.protocol)
    if specification.get("protocol") != "h1a_nonlinear_pair_conversion_qualification_v1":
        raise ValueError("unexpected nonlinear-pair protocol")
    prerequisites = specification["prerequisites"]
    for path, key in (
        (args.localization_protocol, "localization_protocol_sha256"),
        (args.localization_result, "localization_result_sha256"),
        (args.checkpoint_protocol, "checkpoint_protocol_sha256"),
        (args.checkpoint_result, "checkpoint_result_sha256"),
        (args.checkpoint, "checkpoint_sha256"),
        (args.cache_root / "manifest.json", "cache_manifest_sha256"),
    ):
        if sha256_file(path) != str(prerequisites[key]):
            raise ValueError(f"frozen input hash mismatch: {path}")
    if load_json_object(args.localization_result).get("decision") != prerequisites["localization_decision"]:
        raise ValueError("localization decision mismatch")
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
    before = _fingerprint(runtime.model)
    capture = FinalEdgeCapture(runtime.model)
    started = time.perf_counter()
    try:
        probe_accumulator: ScalarRidgeAccumulator | None = None
        generator = torch.Generator(device=device).manual_seed(int(diagnostic["train_noise_seed"]))
        with torch.no_grad():
            for start in range(0, train_indices.numel(), int(diagnostic["batch_size"])):
                data = _forward_batch(
                    runtime,
                    diffusion,
                    train_dataset,
                    train_indices[start : start + int(diagnostic["batch_size"])],
                    float(diagnostic["time"]),
                    generator,
                    capture,
                    specification,
                    device=device,
                )
                edge_graph = data["packed"].batch[data["edges"].target]
                if probe_accumulator is None:
                    probe_accumulator = ScalarRidgeAccumulator.create(data["features"].shape[-1])
                probe_accumulator.update(
                    data["features"],
                    data["clean_field"],
                    edge_graph,
                    data["graphs"],
                )
        if probe_accumulator is None:
            raise RuntimeError("empty nonlinear-pair train panel")
        probe = fit_standardized_ridge(probe_accumulator, float(specification["probe"]["ridge_relative"]))
        train_batches = _collect_batches(
            runtime,
            diffusion,
            train_dataset,
            train_indices,
            torch.Generator(device=device).manual_seed(int(diagnostic["train_noise_seed"])),
            capture,
            specification,
            probe,
            device=device,
        )
        validation_batches = _collect_batches(
            runtime,
            diffusion,
            validation_dataset,
            validation_indices,
            torch.Generator(device=device).manual_seed(int(diagnostic["validation_noise_seed"])),
            capture,
            specification,
            probe,
            device=device,
        )
    finally:
        capture.close()
    readout = specification["readout"]
    torch.manual_seed(int(readout["seed"]))
    feature_dim = int(train_batches[0]["base_features"].shape[-1])
    radial_channels = int(train_batches[0]["vector_radial"].shape[1])
    models = {
        name: NonlinearPairVectorReadout(feature_dim, int(readout["hidden_dim"]), radial_channels).to(device)
        for name in ("base", "topology")
    }
    optimizer = torch.optim.AdamW(
        [parameter for model in models.values() for parameter in model.parameters()],
        lr=float(readout["learning_rate"]),
        weight_decay=float(readout["weight_decay"]),
    )
    order_generator = torch.Generator().manual_seed(int(readout["seed"]) + 1)
    order = torch.randperm(len(train_batches), generator=order_generator).tolist()
    cursor = 0
    loss_rows: list[dict[str, Any]] = []
    for step in range(1, int(readout["steps"]) + 1):
        if cursor == len(order):
            order = torch.randperm(len(train_batches), generator=order_generator).tolist()
            cursor = 0
        batch_data = _to_device(train_batches[order[cursor]], device)
        cursor += 1
        optimizer.zero_grad(set_to_none=True)
        losses: dict[str, torch.Tensor] = {}
        for name, model in models.items():
            correction = model(
                batch_data[f"{name}_features"],
                batch_data["vector_radial"],
                batch_data["target"],
                batch_data["batch"],
                int(batch_data["nodes"]),
                int(batch_data["graphs"]),
            )
            losses[name] = _graph_coordinate_mse(
                batch_data["residual"] - correction,
                batch_data["batch"],
                int(batch_data["graphs"]),
            ).mean()
        sum(losses.values()).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for model in models.values() for parameter in model.parameters()],
            float(readout["gradient_clip_norm"]),
        )
        optimizer.step()
        if step == 1 or step % int(readout["log_every"]) == 0:
            loss_rows.append(
                {
                    "step": step,
                    "base_loss": float(losses["base"].detach()),
                    "topology_loss": float(losses["topology"].detach()),
                    "gradient_norm_before_clip": float(gradient_norm),
                }
            )

    baseline_parts: list[torch.Tensor] = []
    corrected_parts: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    for stored in validation_batches:
        batch_data = _to_device(stored, device)
        baseline_parts.append(
            _graph_coordinate_mse(
                batch_data["residual"],
                batch_data["batch"],
                int(batch_data["graphs"]),
            )
            .detach()
            .cpu()
        )
        for name, model in models.items():
            with torch.no_grad():
                correction = model(
                    batch_data[f"{name}_features"],
                    batch_data["vector_radial"],
                    batch_data["target"],
                    batch_data["batch"],
                    int(batch_data["nodes"]),
                    int(batch_data["graphs"]),
                )
            corrected_parts[name].append(
                _graph_coordinate_mse(
                    batch_data["residual"] - correction,
                    batch_data["batch"],
                    int(batch_data["graphs"]),
                ).cpu()
            )
    baseline = torch.cat(baseline_parts).double()
    corrected = {name: torch.cat(parts).double() for name, parts in corrected_parts.items()}
    acceptance = specification["acceptance"]
    metric_rows: list[dict[str, Any]] = []
    passed: dict[str, bool] = {}
    for index, name in enumerate(("base", "topology")):
        improvement = 1.0 - float(corrected[name].mean() / baseline.mean().clamp_min(1.0e-15))
        interval = _bootstrap_improvement(
            baseline,
            corrected[name],
            seed=int(diagnostic["bootstrap_seed"]) + index,
            samples=int(diagnostic["bootstrap_samples"]),
        )
        passed[name] = improvement >= float(acceptance["validation_relative_improvement_min"]) and interval[0] > float(
            acceptance["validation_bootstrap_95_low_min"]
        )
        metric_rows.append(
            {
                "variant": name,
                "baseline_mse": float(baseline.mean()),
                "corrected_mse": float(corrected[name].mean()),
                "relative_improvement": improvement,
                "bootstrap_95_low": interval[0],
                "bootstrap_median": interval[1],
                "bootstrap_95_high": interval[2],
                "passed": passed[name],
            }
        )
    incremental_interval = _bootstrap_improvement(
        corrected["base"],
        corrected["topology"],
        seed=int(diagnostic["bootstrap_seed"]) + 10,
        samples=int(diagnostic["bootstrap_samples"]),
    )
    incremental_mean = 1.0 - float(corrected["topology"].mean() / corrected["base"].mean().clamp_min(1.0e-15))
    topology_incremental_pass = incremental_mean >= float(acceptance["topology_incremental_improvement_min"]) and float(
        incremental_interval[0]
    ) > float(acceptance["topology_incremental_bootstrap_95_low_min"])
    if passed["topology"] and topology_incremental_pass:
        decision = "qualify_time_conditioned_nonlinear_latent_pair_field"
    elif passed["base"]:
        decision = "generic_nonlinear_conversion_not_topology"
    else:
        decision = "state_derived_pair_conversion_insufficient_conditional_variance"
    after = _fingerprint(runtime.model)
    result = {
        "protocol": specification["protocol"],
        "protocol_sha256": canonical_json_hash(specification),
        "decision": decision,
        "passed": passed,
        "topology_incremental_mean": incremental_mean,
        "topology_incremental_bootstrap_95": list(map(float, incremental_interval)),
        "topology_incremental_passed": topology_incremental_pass,
        "checkpoint_parameters_unchanged": before == after,
        "optimizer_steps": int(readout["steps"]),
        "generator_optimizer_steps": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "decision_boundary": specification["decision_rule"]["boundary"],
    }
    if before != after:
        raise RuntimeError("nonlinear-pair qualification modified the generator")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "training_curve.csv", loss_rows)
    _write_csv(args.output / "validation.csv", metric_rows)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
