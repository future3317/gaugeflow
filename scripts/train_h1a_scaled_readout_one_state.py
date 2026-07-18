"""Run the frozen one-state scaled-coordinate-readout memorization test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _evaluate_noisy,
    _fixed_indices,
    _make_batch,
    _make_model,
    _train_exact_state,
)
from scripts.audit_h1a_coordinate_memorization_scaling import _checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_scaled_readout_one_state_v1":
        raise ValueError("scaled-readout one-state protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("scaled-readout one-state cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    training = protocol["training"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    dataset = PackedAlexP1Dataset(args.cache_root, str(protocol["data"]["split"]))
    indices = _fixed_indices(
        len(dataset),
        int(protocol["data"]["fixed_graphs"]),
        int(protocol["data"]["selection_seed"]),
    )
    batch_data = _make_batch(dataset, indices, device)
    blueprint = _blueprint(batch_data)
    model = _make_model(protocol, device)
    if float(model.coordinate_readout_scale) != float(
        protocol["model"]["coordinate_readout_scale"]
    ):
        raise ValueError("production coordinate readout scale changed")
    readout_names = (
        "coordinate_vector_head.weight",
        "coordinate_edge_head.2.weight",
        "coordinate_edge_head.2.bias",
    )
    initial_readout = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name in readout_names
    }
    standardizer = P1LatticeStandardizer.from_mapping(
        load_json_object(args.lattice_standardization)
    )
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training["coordinate_sigma_max"]),
        minimum_time=float(training["minimum_time"]),
        maximum_time=float(training["maximum_time"]),
    )
    args.run_root.mkdir(parents=True, exist_ok=False)
    noisy, curve = _train_exact_state(
        model,
        diffusion,
        batch_data,
        blueprint,
        protocol,
        generator=torch.Generator(device=device).manual_seed(
            int(training["noise_seed"])
        ),
        metrics_path=args.run_root / "training_metrics.jsonl",
    )
    metrics = _evaluate_noisy(
        model,
        diffusion,
        batch_data,
        blueprint,
        noisy,
        use_bf16=training["precision"] == "bf16" and device.type == "cuda",
    )
    final_parameters = dict(model.named_parameters())
    displacement = torch.sqrt(
        sum(
            (
                final_parameters[name].detach().double()
                - initial_readout[name].double()
            )
            .square()
            .sum()
            for name in readout_names
        )
    )
    checks = {
        **_checks(metrics, protocol["acceptance"]),
        "sampling_failures": metrics["sampling_failures"]
        == float(protocol["acceptance"]["sampling_failures"]),
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "seed": seed,
        "fixed_indices": indices.tolist(),
        "metrics": metrics,
        "checks": checks,
        "qualified": qualified,
        "readout_parameter_displacement_norm": float(displacement),
        "curve": curve,
        "decision": (
            "scaled_readout_one_state_qualified_freeze_16_state_test"
            if qualified
            else "scaled_readout_one_state_failed_remove_parameterization"
        ),
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
