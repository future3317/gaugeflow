"""Resolve the failed 64-state audit with fixed 1/4/16-state panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def _checks(metrics: dict[str, float], acceptance: dict[str, Any]) -> dict[str, bool]:
    return {
        "coordinate_mse": metrics["coordinate_mse"]
        <= float(acceptance["coordinate_mse_max"]),
        "explained_fraction": metrics["explained_fraction"]
        >= float(acceptance["explained_fraction_min"]),
        "low_time_endpoint": metrics["low_time_endpoint_rms_angstrom"]
        <= float(acceptance["low_time_endpoint_rms_angstrom_max"]),
        "tensor_candidates": metrics["tensor_candidates"]
        == float(acceptance["tensor_candidates"]),
    }


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
    if protocol.get("protocol") != "h1a_coordinate_memorization_scaling_audit_v1":
        raise ValueError("coordinate memorization scaling protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate memorization scaling cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    dataset = PackedAlexP1Dataset(args.cache_root, str(protocol["data"]["split"]))
    maximum = max(int(value) for value in protocol["data"]["panel_graphs"])
    indices = _fixed_indices(
        len(dataset), maximum, int(protocol["data"]["selection_seed"])
    )
    standardizer = P1LatticeStandardizer.from_mapping(
        load_json_object(args.lattice_standardization)
    )
    training = protocol["training"]
    args.run_root.mkdir(parents=True, exist_ok=False)
    panel_results: list[dict[str, Any]] = []
    for graph_count in protocol["data"]["panel_graphs"]:
        graph_count = int(graph_count)
        torch.manual_seed(int(training["seed"]))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(training["seed"]))
        batch_data = _make_batch(dataset, indices[:graph_count], device)
        blueprint = _blueprint(batch_data)
        model = _make_model(protocol, device)
        diffusion = TensorFreeHybridDiffusion(
            model,
            standardizer,
            coordinate_sigma_min=float(training["coordinate_sigma_min"]),
            coordinate_sigma_max=float(training["coordinate_sigma_max"]),
            minimum_time=float(training["minimum_time"]),
            maximum_time=float(training["maximum_time"]),
        )
        panel_root = args.run_root / f"graphs_{graph_count}"
        panel_root.mkdir()
        noisy, curve = _train_exact_state(
            model,
            diffusion,
            batch_data,
            blueprint,
            {
                "training": {
                    **training,
                    "exact_state_steps": int(training["steps"]),
                }
            },
            generator=torch.Generator(device=device).manual_seed(
                int(training["seed"]) + 1
            ),
            metrics_path=panel_root / "training_metrics.jsonl",
        )
        metrics = _evaluate_noisy(
            model,
            diffusion,
            batch_data,
            blueprint,
            noisy,
            use_bf16=training["precision"] == "bf16" and device.type == "cuda",
        )
        panel_results.append(
            {
                "graphs": graph_count,
                "indices": indices[:graph_count].tolist(),
                "metrics": metrics,
                "checks": _checks(metrics, protocol["acceptance"]),
                "curve": curve,
            }
        )
    first_qualified = all(panel_results[0]["checks"].values())
    all_qualified = all(all(value["checks"].values()) for value in panel_results)
    if not first_qualified:
        decision = "one_state_failed_inspect_forward_vector_head_loss_basis"
    elif not all_qualified:
        decision = "one_state_passed_shared_structural_memorization_scales_poorly"
    else:
        decision = "all_panels_passed_freeze_resampled_time_protocol"
    result = {
        "protocol": protocol["protocol"],
        "seed": int(training["seed"]),
        "panels": panel_results,
        "referenced_64_state_result": protocol["prerequisites"],
        "qualified": all_qualified,
        "decision": decision,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
