"""Compare real-batch reference and optimized CUDA training updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _flatten(mapping: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([mapping[name].double().reshape(-1) for name in sorted(mapping)])


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left, right, dim=0))


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_cuda_execution_optimization_v2":
        raise ValueError("snapshot comparison requires the frozen v2 protocol")
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    optimized = torch.load(args.optimized, map_location="cpu", weights_only=False)
    if reference["gradients"].keys() != optimized["gradients"].keys():
        raise ValueError("training backends produced different gradient support")
    if reference["parameters"].keys() != optimized["parameters"].keys():
        raise ValueError("training backends produced different parameter support")

    reference_loss = float(reference["selected_loss"])
    optimized_loss = float(optimized["selected_loss"])
    loss_relative = abs(optimized_loss - reference_loss) / max(abs(reference_loss), 1.0e-12)
    reference_gradient = _flatten(reference["gradients"])
    optimized_gradient = _flatten(optimized["gradients"])
    reference_parameters = _flatten(reference["parameters"])
    optimized_parameters = _flatten(optimized["parameters"])
    reference_initial = _flatten(reference["initial_parameters"])
    optimized_initial = _flatten(optimized["initial_parameters"])
    if not torch.equal(reference_initial, optimized_initial):
        raise ValueError("training backends did not start from identical parameters")
    reference_update = reference_parameters - reference_initial
    optimized_update = optimized_parameters - optimized_initial
    update_difference = optimized_parameters - reference_parameters
    reference_gradient_norm = float(torch.linalg.vector_norm(reference_gradient))
    optimized_gradient_norm = float(torch.linalg.vector_norm(optimized_gradient))
    reference_update_norm = float(torch.linalg.vector_norm(reference_update))
    optimized_update_norm = float(torch.linalg.vector_norm(optimized_update))
    result = {
        "protocol": protocol["protocol"],
        "reference_loss": reference_loss,
        "optimized_loss": optimized_loss,
        "selected_loss_relative_difference": loss_relative,
        "coordinate_prediction_cosine": _cosine(
            reference["coordinate_prediction"].double().reshape(-1),
            optimized["coordinate_prediction"].double().reshape(-1),
        ),
        "full_gradient_cosine": _cosine(reference_gradient, optimized_gradient),
        "gradient_relative_norm_difference": abs(
            optimized_gradient_norm - reference_gradient_norm
        ) / max(reference_gradient_norm, 1.0e-12),
        "parameter_update_cosine": _cosine(reference_update, optimized_update),
        "parameter_update_relative_norm_difference": abs(
            optimized_update_norm - reference_update_norm
        ) / max(reference_update_norm, 1.0e-12),
        "parameter_vector_cosine_after_one_step": _cosine(
            reference_parameters, optimized_parameters
        ),
        "parameter_update_backend_difference_norm": float(
            torch.linalg.vector_norm(update_difference)
        ),
        "finite": _finite_tree(reference) and _finite_tree(optimized),
        "reference_backend": {
            "matmul_precision": reference["matmul_precision"],
            "fused": reference["fused"],
        },
        "optimized_backend": {
            "matmul_precision": optimized["matmul_precision"],
            "fused": optimized["fused"],
        },
    }
    acceptance = protocol["equivalence"]
    checks = {
        "selected_loss": loss_relative <= float(acceptance["selected_loss_relative_difference_max"]),
        "coordinate_prediction": result["coordinate_prediction_cosine"]
        >= float(acceptance["coordinate_prediction_cosine_min"]),
        "gradient": result["full_gradient_cosine"]
        >= float(acceptance["full_gradient_cosine_min"]),
        "parameter_update_cosine": result["parameter_update_cosine"]
        >= float(acceptance["parameter_update_cosine_min"]),
        "parameter_update_norm": result["parameter_update_relative_norm_difference"]
        <= float(acceptance["parameter_update_relative_norm_difference_max"]),
        "finite": result["finite"],
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
