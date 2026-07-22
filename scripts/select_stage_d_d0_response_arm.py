"""Select the Stage-D D0 response arm under the frozen paired criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gaugeflow.file_utils import load_json_object, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _relative_change(candidate: float, reference: float) -> float:
    if reference <= 0.0:
        raise ValueError("paired D0 reference loss must be positive")
    return (candidate - reference) / reference


def select_response_arm(
    protocol: dict[str, Any],
    baseline: dict[str, Any],
    probe: dict[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared D0 rule to two provenance-matched results."""

    if protocol.get("protocol") != "stage_d_d0_response_probe_v1":
        raise ValueError("unexpected Stage-D D0 protocol")
    for name, result, expected_arm in (
        ("baseline", baseline, "baseline"),
        ("probe", probe, "probe"),
    ):
        if (
            result.get("schema") != "gaugeflow.stage_d_d0_response_arm.v1"
            or result.get("status") != "complete"
            or result.get("arm") != expected_arm
        ):
            raise ValueError(f"{name} is not a complete Stage-D D0 arm")
    provenance_fields = (
        "steps",
        "seed",
        "source_checkpoint_step",
        "source_checkpoint_sha256",
        "cache_sha256",
        "normalizer_sha256",
        "protocol_sha256",
    )
    if any(baseline.get(field) != probe.get(field) for field in provenance_fields):
        raise ValueError("Stage-D D0 arms do not share paired provenance")

    baseline_metrics = baseline.get("validation")
    probe_metrics = probe.get("validation")
    if not isinstance(baseline_metrics, dict) or not isinstance(probe_metrics, dict):
        raise ValueError("Stage-D D0 validation metrics are missing")
    metric_names = (
        "piezoelectric_loss",
        "piezoelectric_probe_loss",
        "dielectric_loss",
        "born_loss",
        "gamma_loss",
        "internal_strain_loss",
    )
    try:
        baseline_values = {name: float(baseline_metrics[name]) for name in metric_names}
        probe_values = {name: float(probe_metrics[name]) for name in metric_names}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Stage-D D0 validation metrics are invalid") from error

    other_names = (
        "dielectric_loss",
        "born_loss",
        "gamma_loss",
        "internal_strain_loss",
    )
    baseline_other_macro = sum(baseline_values[name] for name in other_names) / len(
        other_names
    )
    probe_other_macro = sum(probe_values[name] for name in other_names) / len(other_names)
    probe_improvement = -_relative_change(
        probe_values["piezoelectric_probe_loss"],
        baseline_values["piezoelectric_probe_loss"],
    )
    tensor_degradation = _relative_change(
        probe_values["piezoelectric_loss"], baseline_values["piezoelectric_loss"]
    )
    other_degradation = _relative_change(probe_other_macro, baseline_other_macro)
    thresholds = protocol.get("selection")
    if not isinstance(thresholds, dict):
        raise ValueError("Stage-D D0 selection thresholds are missing")
    checks = {
        "probe_error": probe_improvement
        >= float(thresholds["probe_error_relative_improvement_minimum"]),
        "full_piezoelectric": tensor_degradation
        <= float(thresholds["full_piezoelectric_loss_relative_degradation_maximum"]),
        "other_task_macro": other_degradation
        <= float(thresholds["other_task_macro_relative_degradation_maximum"]),
    }
    qualified = all(checks.values())
    return {
        "schema": "gaugeflow.stage_d_d0_response_selection.v1",
        "status": "complete",
        "qualified": qualified,
        "selected_arm": "probe" if qualified else "baseline",
        "paired_provenance": {
            field: baseline[field] for field in provenance_fields
        },
        "metrics": {
            "probe_error_relative_improvement": probe_improvement,
            "full_piezoelectric_loss_relative_degradation": tensor_degradation,
            "other_task_macro_relative_degradation": other_degradation,
            "baseline_other_task_macro": baseline_other_macro,
            "probe_other_task_macro": probe_other_macro,
        },
        "checks": checks,
        "thresholds": thresholds,
        "boundary": (
            "This single-seed D0 mechanism screen selects an auxiliary loss only; "
            "it is not a Stage-D predictive qualification."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D0 selection {args.output}")
    protocol = load_json_object(args.protocol)
    baseline = load_json_object(args.baseline)
    probe = load_json_object(args.probe)
    result = select_response_arm(protocol, baseline, probe)
    result["inputs"] = {
        "protocol_sha256": sha256_file(args.protocol),
        "baseline_sha256": sha256_file(args.baseline),
        "probe_sha256": sha256_file(args.probe),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
