"""Apply the declared Pareto-minimax rule to complete Stage-C evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.stage_c_evaluation import select_pareto_minimax_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Stage-C relative step and result path as STEP=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _nested_float(value: dict[str, Any], dotted_key: str) -> float:
    current: Any = value
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"candidate lacks selection objective {dotted_key}")
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError(f"candidate objective {dotted_key} is not numeric")
    return float(current)


def _eligible(result: dict[str, Any], contract: dict[str, Any]) -> bool:
    generation = result.get("generation_retention")
    if not isinstance(generation, dict):
        return False
    finite = all(
        math.isfinite(float(value))
        for value in _numeric_leaves(result)
    )
    return (
        (finite or not bool(contract["all_metrics_finite"]))
        and
        int(generation.get("sampling_failures", -1)) == int(contract["sampling_failures"])
        and int(generation.get("terminal_masks", -1)) == int(contract["terminal_masks"])
        and float(generation.get("exact_composition_fraction", -1.0))
        == float(contract["exact_composition_fraction"])
        and float(generation.get("finite_positive_lattice_fraction", -1.0))
        == float(contract["finite_positive_lattice_fraction"])
        and float(generation.get("minimum_distance_fraction_at_0_5_angstrom", -1.0))
        == float(contract["minimum_distance_fraction_at_0_5_angstrom"])
    )


def _numeric_leaves(value: Any) -> list[int | float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in _numeric_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _numeric_leaves(child)]
    return []


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint selection {args.output}")
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "stage_c_checkpoint_selection_v1":
        raise ValueError("unexpected Stage-C selection protocol")
    declared = {int(step) for step in protocol["candidates"]["stage_c_relative_steps"]}
    paths: dict[int, Path] = {}
    for item in args.candidate:
        step_text, separator, path_text = item.partition("=")
        if not separator:
            raise ValueError("candidate must be STEP=PATH")
        step = int(step_text)
        if step in paths:
            raise ValueError("duplicate Stage-C candidate step")
        paths[step] = Path(path_text)
    if set(paths) != declared:
        raise ValueError("candidate set does not match the declared Stage-C panel")

    objective_names = list(protocol["selection"]["objectives_minimize"])
    eligibility = dict(protocol["eligibility"])
    objectives: dict[int, dict[str, float]] = {}
    evidence: dict[int, dict[str, Any]] = {}
    ineligible: list[int] = []
    for step, path in sorted(paths.items()):
        result = load_json_object(path)
        if result.get("schema") != "gaugeflow.stage_c_checkpoint_evaluation.v2":
            raise ValueError("candidate is not a complete three-panel Stage-C evaluation")
        if int(result.get("stage_c_step", -1)) != step:
            raise ValueError("candidate path and reported Stage-C step disagree")
        is_eligible = _eligible(result, eligibility)
        evidence[step] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "eligible": is_eligible,
        }
        if is_eligible:
            objectives[step] = {
                name: _nested_float(result, name) for name in objective_names
            }
        else:
            ineligible.append(step)
    selection = select_pareto_minimax_checkpoint(objectives)
    output = {
        "schema": "gaugeflow.stage_c_checkpoint_selection.v1",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "candidate_evidence": evidence,
        "ineligible_stage_c_steps": ineligible,
        "objectives": objectives,
        "selection": selection,
        "status": "operational_checkpoint_selected",
        "boundary": protocol["boundaries"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
