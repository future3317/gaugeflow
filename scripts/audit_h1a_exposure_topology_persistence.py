"""Run the exact v2 topology audit across frozen exposure checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from gaugeflow.file_utils import load_json_object, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--topology-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-result", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("persistence audit produced no rows")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    specification = load_json_object(args.protocol)
    if specification.get("protocol") != "h1a_exposure_conditioned_topology_persistence_v1":
        raise ValueError("unexpected persistence protocol")
    prerequisites = specification["prerequisites"]
    for path, expected in (
        (args.topology_protocol, prerequisites["topology_protocol_sha256"]),
        (args.checkpoint_protocol, prerequisites["learning_curve_protocol_sha256"]),
        (args.checkpoint_result, prerequisites["learning_curve_result_sha256"]),
        (args.cache_root / "manifest.json", prerequisites["cache_manifest_sha256"]),
    ):
        if sha256_file(path) != str(expected):
            raise ValueError(f"frozen input hash mismatch: {path}")
    topology_template = load_json_object(args.topology_protocol)
    if topology_template.get("protocol") != prerequisites["topology_protocol"]:
        raise ValueError("topology template protocol mismatch")
    checkpoint_protocol = load_json_object(args.checkpoint_protocol)
    if checkpoint_protocol.get("protocol") != prerequisites["learning_curve_protocol"]:
        raise ValueError("checkpoint protocol mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    detailed_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    middle_times = set(map(float, topology_template["diagnostic"]["middle_times"]))
    for checkpoint in specification["checkpoints"]:
        step = int(checkpoint["step"])
        passes = float(checkpoint["passes"])
        checkpoint_path = args.checkpoint_root / f"checkpoint_step_{step:08d}.pt"
        if sha256_file(checkpoint_path) != str(checkpoint["sha256"]):
            raise ValueError(f"checkpoint hash mismatch at step {step}")
        derived = deepcopy(topology_template)
        derived_prerequisites = derived["prerequisites"]
        derived_prerequisites["active_checkpoint_protocol"] = checkpoint_protocol["protocol"]
        derived_prerequisites["checkpoint_protocol_sha256"] = sha256_file(
            args.checkpoint_protocol
        )
        derived_prerequisites["checkpoint_sha256"] = checkpoint["sha256"]
        derived_prerequisites["checkpoint_result_sha256"] = sha256_file(
            args.checkpoint_result
        )
        derived_protocol = args.output / f"derived_v2_step_{step:08d}.json"
        derived_protocol.write_text(
            json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checkpoint_output = args.output / f"step_{step:08d}"
        command = [
            sys.executable,
            str(Path(__file__).with_name("audit_h1a_latent_clean_topology.py")),
            "--protocol",
            str(derived_protocol),
            "--checkpoint-protocol",
            str(args.checkpoint_protocol),
            "--checkpoint-result",
            str(args.checkpoint_result),
            "--checkpoint",
            str(checkpoint_path),
            "--cache-root",
            str(args.cache_root),
            "--output",
            str(checkpoint_output),
            "--device",
            args.device,
        ]
        subprocess.run(command, check=True)
        result = load_json_object(checkpoint_output / "result.json")
        carrier_rows = _read_csv(checkpoint_output / "topology_oracle_carrier.csv")
        middle_clean = [
            row
            for row in carrier_rows
            if row["variant"] == "clean_oracle" and float(row["time"]) in middle_times
        ]
        if len(middle_clean) != len(middle_times):
            raise RuntimeError("exact v2 audit did not return every middle time")
        gains = [float(row["relative_improvement"]) for row in middle_clean]
        absolute = [
            float(row["baseline_mse"]) - float(row["corrected_mse"])
            for row in middle_clean
        ]
        positive_lower = sum(float(row["bootstrap_95_low"]) > 0.0 for row in middle_clean)
        exposure_rows.append(
            {
                "step": step,
                "passes": passes,
                "role": checkpoint["role"],
                "middle_normalized_gain": sum(gains) / len(gains),
                "middle_absolute_mse_reduction": sum(absolute) / len(absolute),
                "positive_middle_bootstrap_lower_bounds": positive_lower,
                "probe_explained_fraction": result["decision_metrics"][
                    "probe_middle_mean_explained_fraction"
                ],
                "learned_carrier_gain": result["decision_metrics"][
                    "learned_middle_mean_improvement"
                ],
                "optimizer_steps": result["optimizer_steps"],
                "parameters_unchanged": result["checkpoint_parameters_unchanged"],
            }
        )
        for row in middle_clean:
            detailed_rows.append(
                {
                    "step": step,
                    "passes": passes,
                    "time": row["time"],
                    "baseline_mse": row["baseline_mse"],
                    "corrected_mse": row["corrected_mse"],
                    "absolute_mse_reduction": float(row["baseline_mse"])
                    - float(row["corrected_mse"]),
                    "normalized_gain": row["relative_improvement"],
                    "bootstrap_95_low": row["bootstrap_95_low"],
                    "bootstrap_median": row["bootstrap_median"],
                    "bootstrap_95_high": row["bootstrap_95_high"],
                }
            )

    classified = {float(row["passes"]): row for row in exposure_rows if row["role"] == "classification"}
    early = classified[0.25]
    terminal = classified[2.0]
    terminal_gain = float(terminal["middle_normalized_gain"])
    retention = terminal_gain / max(float(early["middle_normalized_gain"]), 1.0e-12)
    supporting = int(terminal["positive_middle_bootstrap_lower_bounds"])
    if terminal_gain <= 0.05 or retention <= 0.50:
        decision = "exposure_dominant"
    elif terminal_gain >= 0.10 and retention >= 0.75 and supporting >= 2:
        decision = "topology_persistent"
    else:
        decision = "mixed"
    result = {
        "protocol": specification["protocol"],
        "decision": decision,
        "two_pass_middle_normalized_gain": terminal_gain,
        "two_pass_retention_vs_quarter_pass": retention,
        "two_pass_positive_middle_bootstrap_lower_bounds": supporting,
        "optimizer_steps": 0,
        "all_parameters_unchanged": all(bool(row["parameters_unchanged"]) for row in exposure_rows),
        "decision_boundary": specification["decision_boundary"],
    }
    _write_csv(args.output / "exposure_summary.csv", exposure_rows)
    _write_csv(args.output / "exposure_time_curves.csv", detailed_rows)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
