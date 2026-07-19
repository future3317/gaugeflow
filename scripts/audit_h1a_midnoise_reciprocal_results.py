"""Independently verify the frozen H1a reciprocal-attribution artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty audit table: {path}")
    return rows


def _finite_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {key} in audit table")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    result_path = args.result_dir / "result.json"
    retrieval_path = args.result_dir / "middle_noise_retrieval.csv"
    probe_path = args.result_dir / "frozen_low_k_probe.csv"
    spectrum_path = args.result_dir / "reciprocal_spectrum.csv"
    result = load_json_object(result_path)
    retrieval = _read_csv(retrieval_path)
    probe = _read_csv(probe_path)
    spectrum = _read_csv(spectrum_path)
    expected_times = list(map(float, protocol["diagnostic"]["times"]))
    retrieval_times = [_finite_float(row, "time") for row in retrieval]
    if retrieval_times != expected_times:
        raise ValueError("retrieval times do not match the frozen protocol")
    probe_keys = [(_finite_float(row, "time"), row["band"]) for row in probe]
    expected_probe_keys = [
        (time, band)
        for time in expected_times
        for band in ("low", "high_control")
    ]
    if probe_keys != expected_probe_keys:
        raise ValueError("probe time/band rows do not match the frozen protocol")
    bin_edges = list(
        map(float, protocol["diagnostic"]["spectrum_bin_edges_inverse_angstrom"])
    )
    spectrum_keys = [
        (
            _finite_float(row, "time"),
            _finite_float(row, "q_lower"),
            _finite_float(row, "q_upper"),
        )
        for row in spectrum
    ]
    expected_spectrum_keys = [
        (time, lower, upper)
        for time in expected_times
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True)
    ]
    if spectrum_keys != expected_spectrum_keys:
        raise ValueError("spectrum rows do not match the frozen protocol")
    acceptance = protocol["acceptance"]
    retrieval_mean = sum(
        _finite_float(row, "top1_accuracy") for row in retrieval
    ) / len(retrieval)
    retrieval_margin_mean = sum(
        _finite_float(row, "mean_own_vs_best_other_margin") for row in retrieval
    ) / len(retrieval)
    retrieval_check = (
        retrieval_mean >= float(acceptance["retrieval_mean_accuracy_min"])
        and min(_finite_float(row, "top1_accuracy") for row in retrieval)
        >= float(acceptance["retrieval_each_time_accuracy_min"])
        and retrieval_margin_mean > float(acceptance["retrieval_mean_margin_min"])
    )
    low_probe = [row for row in probe if row["band"] == "low"]
    high_probe = [row for row in probe if row["band"] == "high_control"]
    low_mean = sum(
        _finite_float(row, "relative_improvement") for row in low_probe
    ) / len(low_probe)
    high_mean = sum(
        _finite_float(row, "relative_improvement") for row in high_probe
    ) / len(high_probe)
    probe_check = (
        low_mean >= float(acceptance["low_probe_mean_improvement_min"])
        and min(_finite_float(row, "relative_improvement") for row in low_probe)
        >= float(acceptance["low_probe_each_time_improvement_min"])
        and min(_finite_float(row, "bootstrap_95_low") for row in low_probe)
        >= float(acceptance["low_probe_each_time_ci_low_min"])
        and low_mean - high_mean
        >= float(acceptance["low_minus_high_probe_improvement_min"])
        and min(
            _finite_float(row, "graphs_with_modes_fraction") for row in low_probe
        )
        >= float(acceptance["low_probe_graph_coverage_min"])
    )
    by_time: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in spectrum:
        by_time[_finite_float(row, "time")].append(row)
    spectral_ratios: list[float] = []
    for time in expected_times:
        rows = by_time[time]
        low = [row for row in rows if _finite_float(row, "q_upper") <= 1.5]
        high = [row for row in rows if _finite_float(row, "q_lower") >= 2.5]
        low_ratio = sum(
            _finite_float(row, "normalized_residual_ratio") for row in low
        ) / len(low)
        high_ratio = sum(
            _finite_float(row, "normalized_residual_ratio") for row in high
        ) / len(high)
        spectral_ratios.append(low_ratio / high_ratio)
    supporting_times = sum(
        value >= float(acceptance["low_to_high_spectral_ratio_min"])
        for value in spectral_ratios
    )
    spectral_mean = sum(spectral_ratios) / len(spectral_ratios)
    spectrum_check = supporting_times >= int(
        acceptance["spectral_supporting_times_min"]
    ) and spectral_mean >= float(
        acceptance["pooled_low_to_high_spectral_ratio_min"]
    )
    recomputed_checks = {
        "middle_noise_endpoint_recoverable": retrieval_check,
        "low_k_residual_spectral_excess": spectrum_check,
        "frozen_low_k_probe_generalizes": probe_check,
    }
    recorded_checks = result["checks"]
    checks = {
        "protocol_hash_matches": result["protocol_sha256"]
        == canonical_json_hash(protocol),
        "recorded_checks_match_recomputation": recorded_checks == recomputed_checks,
        "recorded_decision_matches_checks": result["decision"]
        == (
            "authorize_separate_reciprocal_carrier_qualification"
            if all(recomputed_checks.values())
            else "do_not_implement_reciprocal_carrier"
        ),
        "recorded_metrics_match_recomputation": math.isclose(
            float(result["decision_metrics"]["retrieval_mean_accuracy"]),
            retrieval_mean,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        and math.isclose(
            float(result["decision_metrics"]["low_probe_mean_improvement"]),
            low_mean,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        and math.isclose(
            float(result["decision_metrics"]["spectral_mean_low_to_high_ratio"]),
            spectral_mean,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ),
        "no_checkpoint_or_optimizer_artifact": not any(
            path.suffix in {".pt", ".ckpt"} for path in args.result_dir.iterdir()
        ),
    }
    audit: dict[str, Any] = {
        "protocol": "h1a_midnoise_reciprocal_attribution_v1_independent_audit",
        "checks": checks,
        "qualified": all(checks.values()),
        "recomputed_scientific_checks": recomputed_checks,
        "recomputed_metrics": {
            "retrieval_mean_accuracy": retrieval_mean,
            "low_probe_mean_improvement": low_mean,
            "high_control_probe_mean_improvement": high_mean,
            "spectral_low_to_high_ratios": spectral_ratios,
            "spectral_mean_low_to_high_ratio": spectral_mean,
            "spectral_supporting_times": supporting_times,
        },
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in (result_path, retrieval_path, probe_path, spectrum_path)
        },
    }
    output = args.result_dir / "independent_audit.json"
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    if not audit["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
